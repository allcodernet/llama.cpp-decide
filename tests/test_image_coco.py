import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = ("person", "car", "bicycle", "cat", "dog")


def load():
    spec = importlib.util.spec_from_file_location("image_coco", ROOT / "results-engine" / "image" / "coco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic():
    """400 images; image i has a large instance of category i % 5, every 7th one also a tiny instance of the next."""
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "coco_url": f"http://images.example/{i}.jpg", "license": 1 + i % 3,
               "width": 640, "height": 480} for i in range(1, 401)]
    anns = [{"image_id": i, "category_id": 10 + i % 5, "iscrowd": 0, "area": 5000.0} for i in range(1, 401)]
    anns += [{"image_id": i, "category_id": 10 + (i + 1) % 5, "iscrowd": 0, "area": 10.0} for i in range(1, 401, 7)]
    return {"images": images, "annotations": anns, "categories": [{"id": 10 + k, "name": n} for k, n in enumerate(CATS)],
            "licenses": [{"id": k, "name": f"Licence {k}", "url": f"http://licence.example/{k}"} for k in (1, 2, 3)]}


def test_label_rule():
    co = load()
    big = {"iscrowd": 0, "area": 3072.0}   # 1 % of 640 x 480 = 3072 > 32²
    assert co.label([], 640, 480) is False
    assert co.label([big], 640, 480) is True
    assert co.label([{**big, "area": 3071.0}], 640, 480) is None
    assert co.label([{**big, "iscrowd": 1, "area": 1e6}], 640, 480) is None
    assert co.label([{"iscrowd": 0, "area": 1024.0}], 100, 100) is True   # 1 % of 100 x 100 = 100 < 32²
    assert co.label([{"iscrowd": 0, "area": 1023.0}], 100, 100) is None


def test_select_subset_is_balanced_and_fixed():
    co = load()
    data = synthetic()
    rows = co.select_subset(data)
    assert len(rows) == 200 and len({r["image_id"] for r in rows}) == 200
    assert Counter(r["selected_for"] for r in rows) == {f"{c} {k}": 20 for c in CATS for k in ("positive", "negative")}
    assert Counter(r["split"] for r in rows) == {"calibration": 100, "test": 100}
    for r in rows:
        c, kind = r["selected_for"].split()
        assert r["labels"][c] is (kind == "positive")
        assert set(r["labels"]) == set(CATS)
    assert co.select_subset(data) == rows and co.select_subset(data, seed=1) != rows
    text = co.notice(data, rows)
    assert "CC BY 4.0" in text and "COCO Consortium" in text and "| 1 | Licence 1 | http://licence.example/1 |" in text


def test_score_and_temperature():
    co = load()
    s = co.score([(0.9, True), (0.2, False), (0.6, False)])
    assert s["n"] == 3 and math.isclose(s["accuracy"], 2 / 3)
    assert math.isclose(s["nll"], -(math.log(0.9) + math.log(0.8) + math.log(0.4)) / 3)
    assert math.isclose(co.scaled(0.8, 1.0), 0.8) and co.scaled(0.8, 2.0) < 0.8 and co.scaled(0.2, 2.0) > 0.2
    # p = 0.99 for 6 true and 4 false labels: the best scaled probability is 0.6, T = ln 99 / ln 1.5 = 11.333
    t = co.fit_temperature([0.99] * 10, [True] * 6 + [False] * 4)
    assert abs(t - math.log(99) / math.log(1.5)) < 0.02
    assert co.chat_answer(" Yes") is True and co.chat_answer("no.") is False and co.chat_answer("Maybe") is None
    assert co.chat_answer("Not sure") is None and co.chat_answer("None") is None and co.chat_answer("Yesterday") is None


def runs(tmp_path):
    """Four images (calibration 1-2, test 3-4) with decide and chat runs; image 1's car label is null."""
    labels = {1: {"person": True, "car": None}, 2: {"car": True}, 3: {"cat": True}, 4: {"dog": True}}
    rows = [{"image_id": i, "split": "calibration" if i <= 2 else "test",
             "labels": {c: labels[i].get(c, False) for c in CATS}} for i in (1, 2, 3, 4)]
    # every decide answer is right; image 1's unlabelled car gets 0.9, which would be wrong as a "no"
    dec = [{"image_id": r["image_id"], "latency_s": float(r["image_id"]), "server_ms": 10.0 * r["image_id"],
            "p_true": {c: 0.9 if r["labels"][c] in (True, None) else 0.1 for c in CATS}} for r in rows]
    odd = {(1, "car"): ("maybe", 0.9), (2, "car"): ("```", 0.8), (3, "cat"): ("Based", 0.3)}
    first = {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0}   # the first call (person); calls 2-5 take 1 s each
    chat = []
    for r in rows:
        i, fields = r["image_id"], {}
        for k, c in enumerate(CATS):
            answer, p = odd.get((i, c), ("yes", 0.95) if r["labels"][c] else ("no", 0.05))
            fields[c] = {"answer": answer, "p_yes": p, "latency_s": first[i] if k == 0 else 1.0,
                         "timings": {"prompt_n": 1100, "cache_n": 0} if k == 0 else {"prompt_n": 12, "cache_n": 1090}}
        chat.append({"image_id": i, "fields": fields, "latency_s": sum(v["latency_s"] for v in fields.values())})
    paths = {}
    for name, data in (("subset", rows), ("decide", dec), ("chat", chat)):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(x) + "\n" for x in data))
    return paths


def test_cmd_score(tmp_path):
    co = load()
    p = runs(tmp_path)
    co.cmd_score(argparse.Namespace(model="m", subset=p["subset"], decide=p["decide"], chat=p["chat"], out=tmp_path / "s.json"))
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["decide_T1"]["calibration"]["car"]["n"] == 1 and s["decide_T1"]["calibration"]["all"]["n"] == 9   # null label left out
    assert s["decide_T1"]["calibration"]["all"]["accuracy"] == 1.0 and s["decide_T1"]["test"]["all"]["n"] == 10
    assert s["chat_unclear_answers"] == 3
    assert s["chat_accuracy"]["calibration"]["car"] == {"n": 1, "accuracy": 0.0}   # "```" counts as wrong
    assert math.isclose(s["chat_accuracy"]["calibration"]["all"]["accuracy"], 8 / 9)
    assert math.isclose(s["chat_accuracy"]["test"]["all"]["accuracy"], 9 / 10)
    assert s["time_per_image_s"] == {"decide_median": 2.5, "decide_server_ms_median": 25.0, "chat_median": 6.5}


def test_cmd_readings(tmp_path):
    co = load()
    p = runs(tmp_path)
    co.cmd_readings(argparse.Namespace(model="m", subset=p["subset"], decide=p["decide"], chat=p["chat"], out=tmp_path / "r.json"))
    r = json.loads((tmp_path / "r.json").read_text())
    cal, test, both = (r["splits"][s] for s in ("calibration", "test", "all"))
    assert cal["n"] == 9 and test["n"] == 10 and both["n"] == 19
    assert cal["decide"] == {"errors": 0, "accuracy": 1.0} and both["decide"]["errors"] == 0
    assert cal["chat_strict"]["errors"] == 1 and cal["chat_p_yes"] == {"errors": 0, "accuracy": 1.0}   # "```" with p_yes 0.8
    assert test["chat_strict"]["errors"] == 1 and test["chat_p_yes"]["errors"] == 1                    # "Based" with p_yes 0.3
    assert math.isclose(both["chat_strict"]["accuracy"], 17 / 19) and math.isclose(both["chat_p_yes"]["accuracy"], 18 / 19)
    assert r["unclear_answers"] == [
        {"image_id": 1, "field": "car", "answer": "maybe", "p_yes": 0.9, "label": None, "split": "calibration"},
        {"image_id": 2, "field": "car", "answer": "```", "p_yes": 0.8, "label": True, "split": "calibration"},
        {"image_id": 3, "field": "cat", "answer": "Based", "p_yes": 0.3, "label": True, "split": "test"}]
    assert r["time_s"] == {"decide_per_image_median": 2.5, "chat_per_image_median": 6.5, "chat_per_call_median": 1.0,
                           "chat_first_call_median": 2.5, "chat_calls_2_5_median": 1.0}
    assert r["prompt_cache"] == {"first_call": {"calls": 4, "prompt_n_median": 1100, "cache_n_median": 0, "with_cache_n": 0},
                                 "calls_2_5": {"calls": 16, "prompt_n_median": 12, "cache_n_median": 1090, "with_cache_n": 16}}
