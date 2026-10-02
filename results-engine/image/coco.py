"""Sub-project 4: the COCO val2017 subset (spec 8) and its evaluation (spec 9.6); called by results-engine/image/run.sh.

  select   --annotations ZIP --out-dir D
           200 images with seed SEED from annotations/instances_val2017.json in ZIP: per category (person, car, bicycle, cat,
           dog) 20 positives and 20 negatives, every image once; split 100/100 into calibration and test ->
           D/subset.jsonl (image id, file name, COCO URL, licence id, size, labels, split) and D/NOTICE.md     [no GPU]
  download --subset F --dir DIR --sums F
           the images from their coco_url into DIR (git-ignored), their SHA-256 into --sums; when --sums exists the
           downloads must match it                                                                          [no GPU]
  decide   --url U --model P --subset F --images DIR --out F.jsonl
           one /v1/decide request per image with the five yes/no fields at T = 1 (options.temperature 1.0); one warm-up
           request first, not recorded                                                                       [server]
  chat     --url U --model P --subset F --images DIR --out F.jsonl [--cache-prompt]
           the chat baseline: one /v1/chat/completions call per field (max_tokens 1, temperature 0, top_logprobs 20,
           cache_prompt false unless --cache-prompt, thinking off), each call's server timings kept; one warm-up call
           first, not recorded                                                                               [server]
  score    --subset F --decide F --chat F --out F.json
           per field and split accuracy and NLL at T = 1 (a field is scored where its label is not null); the temperature
           fitted on the calibration half (grid of 4001 log-spaced values in [0.05, 20], NLL over all fields; reported
           only) and the NLL at it; chat accuracy (an answer that starts with neither "yes" nor "no" counts as wrong);
           median time per image                                                                            [no GPU]
  readings --subset F --decide F --chat F --out F.json
           per split and over both: errors and accuracy of decide (p_true > 0.5), of chat read strictly (the answer
           starts with "yes" or "no") and of chat read by p_yes > 0.5 (no p_yes counts as wrong); every answer that is
           neither yes nor no; median times (chat per image, per call, first call, calls 2-5); with server timings, the
           prompt cache use (prompt_n, cache_n) of the first call and of calls 2-5                          [no GPU]
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from decide_client import image_part, text_part  # noqa: E402

SEED = 20261002
CATEGORIES = ("person", "car", "bicycle", "cat", "dog")
PER_CLASS = 20
MEMBER = "annotations/instances_val2017.json"
QUESTION = "Is there a {} in this picture?"
SCHEMA = {"instructions": "Answer each question about this image.",
          "fields": {c: {"type": "bool", "description": QUESTION.format(c)} for c in CATEGORIES}}
SPLITS = ("calibration", "test")


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def label(anns, width, height):
    """spec 8: True with an instance of iscrowd 0 and area >= max(32², 1 % of the image), False without any instance of the
    category, None otherwise (not scored)."""
    if not anns:
        return False
    floor = max(32 * 32, 0.01 * width * height)
    return True if any(a["iscrowd"] == 0 and a["area"] >= floor for a in anns) else None


def select_subset(data, seed=SEED):
    cat_id = {c["name"]: c["id"] for c in data["categories"]}
    by_image = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(a)
    images = {im["id"]: im for im in data["images"]}
    labels = {i: {c: label(by_image.get(i, {}).get(cat_id[c], []), im["width"], im["height"]) for c in CATEGORIES}
              for i, im in images.items()}
    rng, used, chosen = random.Random(seed), set(), []
    for c in CATEGORIES:
        for want, kind in ((True, "positive"), (False, "negative")):
            pool = sorted(i for i in images if labels[i][c] is want and i not in used)
            for i in rng.sample(pool, PER_CLASS):
                used.add(i)
                chosen.append((i, f"{c} {kind}"))
    order = list(range(len(chosen)))
    rng.shuffle(order)
    calibration = set(order[: len(chosen) // 2])
    rows = []
    for k, (i, why) in enumerate(chosen):
        im = images[i]
        rows.append({"image_id": i, "file_name": im["file_name"], "coco_url": im["coco_url"], "license": im["license"],
                     "width": im["width"], "height": im["height"], "selected_for": why, "labels": labels[i],
                     "split": "calibration" if k in calibration else "test"})
    return rows


def notice(data, rows):
    lic = {l["id"]: l for l in data["licenses"]}
    counts = {}
    for r in rows:
        counts[r["license"]] = counts.get(r["license"], 0) + 1
    lines = ["# COCO val2017 subset: notice", "",
             "`subset.jsonl` is derived from the COCO 2017 validation annotations (`instances_val2017.json`) of the",
             "COCO Consortium (https://cocodataset.org), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
             "Changes: 200 images chosen with a fixed seed and one yes/no label per category derived from the instance",
             "annotations (`results-engine/image/coco.py`, rule in spec 8 of docs/specs/2026-10-02-image-input-design.md).",
             "", "The images are not in this repository: `coco.py download` fetches them from their `coco_url` into the",
             "git-ignored `local/` directory. Each image keeps the licence of its Flickr source, named by its `license` id:", "",
             "| licence id | name | URL | images |", "|---|---|---|---|"]
    lines += [f"| {k} | {lic[k]['name']} | {lic[k]['url']} | {counts[k]} |" for k in sorted(counts)]
    return "\n".join(lines) + "\n"


def score(pairs):
    """pairs [(p_true, label)] -> n, accuracy (p_true > 0.5 is the answer true) and mean NLL of the label (p clamped at 1e-12)."""
    return {"n": len(pairs), "accuracy": sum((p > 0.5) == y for p, y in pairs) / len(pairs),
            "nll": sum(-math.log(max(p if y else 1.0 - p, 1e-12)) for p, y in pairs) / len(pairs)}


def scaled(p, T):
    """A two-option answer at temperature T: softmax(log p / T) = sigmoid(logit(p) / T) (the engine's apply_temperature)."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return 1.0 / (1.0 + math.exp(-math.log(p / (1.0 - p)) / T))


def fit_temperature(ps, ys):
    grid = [0.05 * 400 ** (k / 4000) for k in range(4001)]   # 0.05 … 20
    return min(grid, key=lambda T: score([(scaled(p, T), y) for p, y in zip(ps, ys)])["nll"])


def cmd_select(args):
    with zipfile.ZipFile(args.annotations) as z:
        data = json.loads(z.read(MEMBER))
    rows = select_subset(data)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "subset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "NOTICE.md").write_text(notice(data, rows))
    splits = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    print(f"wrote {d / 'subset.jsonl'}: {len(rows)} images, {len({r['image_id'] for r in rows})} distinct, {splits}")


def cmd_download(args):
    rows, d = load_jsonl(args.subset), Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for r in rows:
            p = d / r["file_name"]
            if not p.exists():
                resp = client.get(r["coco_url"])
                resp.raise_for_status()
                p.write_bytes(resp.content)
    sums = "".join(f"{hashlib.sha256((d / r['file_name']).read_bytes()).hexdigest()}  {d / r['file_name']}\n" for r in rows)
    s = Path(args.sums)
    if s.exists() and s.read_text() != sums:
        sys.exit(f"{s}: the images in {d} differ from the committed checksums")
    s.write_text(sums)
    print(f"{len(rows)} images in {d}, checksums in {s}")


def cmd_decide(args):
    rows = load_jsonl(args.subset)
    client = httpx.Client(base_url=args.url, timeout=900)

    def ask(row):
        body = {"model": args.model, **SCHEMA, "states": [[image_part(Path(args.images) / row["file_name"])]],
                "options": {"temperature": 1.0}}
        t = time.perf_counter()
        r = client.post("/v1/decide", json=body)
        dt = time.perf_counter() - t
        if r.status_code != 200:
            sys.exit(f"image {row['image_id']}: {r.status_code} {r.text[:500]}")
        return r.json(), dt

    ask(rows[0])   # warm-up
    with open(args.out, "w") as f:
        for row in rows:
            j, dt = ask(row)
            res = j["results"][0]
            f.write(json.dumps({"image_id": row["image_id"], "p_true": {c: res["answers"][c]["p_true"] for c in CATEGORIES},
                                "latency_s": dt, "server_ms": j["timings"]["total_ms"], "usage": res["usage"]}) + "\n")
    print(f"wrote {args.out}: {len(rows)} images")


def chat_answer(text):
    """True for an answer that starts with "yes", False for "no", None for anything else (scored as wrong)."""
    m = re.match(r"(yes|no)\b", text.strip().lower())   # the first word exactly: "Not sure" and "None" are neither
    return None if m is None else m.group(1) == "yes"


def p_yes(top):
    """P(yes) renormalised over the yes/no tokens among the top log-probs (None when neither is listed); reported only."""
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None


def cmd_chat(args):
    rows = load_jsonl(args.subset)
    client = httpx.Client(base_url=args.url, timeout=900)

    def ask(row, c):
        body = {"model": args.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "cache_prompt": args.cache_prompt, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [image_part(Path(args.images) / row["file_name"]),
                                                          text_part(QUESTION.format(c) + " Answer yes or no.")]}]}
        t = time.perf_counter()
        r = client.post("/v1/chat/completions", json=body)
        dt = time.perf_counter() - t
        if r.status_code != 200:
            sys.exit(f"image {row['image_id']} {c}: {r.status_code} {r.text[:500]}")
        ch = r.json()["choices"][0]
        return {"answer": ch["message"]["content"], "p_yes": p_yes(ch["logprobs"]["content"][0]["top_logprobs"]), "latency_s": dt,
                "timings": r.json().get("timings")}

    ask(rows[0], CATEGORIES[0])   # warm-up
    with open(args.out, "w") as f:
        for row in rows:
            fields = {c: ask(row, c) for c in CATEGORIES}
            f.write(json.dumps({"image_id": row["image_id"], "fields": fields,
                                "latency_s": sum(v["latency_s"] for v in fields.values())}) + "\n")
    print(f"wrote {args.out}: {len(rows)} images, {len(rows) * len(CATEGORIES)} calls")


def cmd_score(args):
    rows = {r["image_id"]: r for r in load_jsonl(args.subset)}
    dec = {d["image_id"]: d for d in load_jsonl(args.decide)}
    chat = {d["image_id"]: d for d in load_jsonl(args.chat)}
    if set(dec) != set(rows) or set(chat) != set(rows):
        sys.exit("the decide or chat run does not cover the subset")

    def pairs(split, fields, value):
        return [(value(i, f), r["labels"][f]) for i, r in rows.items() for f in fields
                if r["split"] == split and r["labels"][f] is not None]

    p_dec = lambda i, f: dec[i]["p_true"][f]
    said = lambda i, f: chat_answer(chat[i]["fields"][f]["answer"])
    out = {"model": args.model, "images": len(rows), "decide_T1": {}, "chat_accuracy": {},
           "chat_unclear_answers": sum(chat_answer(v["answer"]) is None for c in chat.values() for v in c["fields"].values())}
    for split in SPLITS:
        out["decide_T1"][split] = {f: score(pairs(split, [f], p_dec)) for f in CATEGORIES}
        out["decide_T1"][split]["all"] = score(pairs(split, CATEGORIES, p_dec))
        out["chat_accuracy"][split] = {}
        for f in (*CATEGORIES, "all"):
            got = pairs(split, CATEGORIES if f == "all" else [f], said)   # None (neither yes nor no) never equals a label
            out["chat_accuracy"][split][f] = {"n": len(got), "accuracy": sum(a == y for a, y in got) / len(got)}
    cal, test = pairs("calibration", CATEGORIES, p_dec), pairs("test", CATEGORIES, p_dec)
    T = fit_temperature([p for p, _ in cal], [y for _, y in cal])
    out["fitted_temperature"] = {
        "T": T, "calibration_nll_T1": score(cal)["nll"], "calibration_nll_T": score([(scaled(p, T), y) for p, y in cal])["nll"],
        "test_nll_T1": score(test)["nll"], "test_nll_T": score([(scaled(p, T), y) for p, y in test])["nll"]}
    out["time_per_image_s"] = {"decide_median": statistics.median(d["latency_s"] for d in dec.values()),
                               "decide_server_ms_median": statistics.median(d["server_ms"] for d in dec.values()),
                               "chat_median": statistics.median(c["latency_s"] for c in chat.values())}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for split in SPLITS:
        print(split, " ".join(f"{f} {v['accuracy']:.3f}/{v['nll']:.3f}" for f, v in out["decide_T1"][split].items()),
              "| chat", f"{out['chat_accuracy'][split]['all']['accuracy']:.3f}")
    print("fitted T", f"{T:.4f}", "| time per image: decide", f"{out['time_per_image_s']['decide_median']:.2f} s, chat",
          f"{out['time_per_image_s']['chat_median']:.2f} s")
    print(f"wrote {args.out}")


def cmd_readings(args):
    rows = {r["image_id"]: r for r in load_jsonl(args.subset)}
    dec = {d["image_id"]: d for d in load_jsonl(args.decide)}
    chat = {d["image_id"]: d for d in load_jsonl(args.chat)}
    if set(dec) != set(rows) or set(chat) != set(rows):
        sys.exit("the decide or chat run does not cover the subset")

    def counts(split, right):
        """errors and accuracy over the labelled fields of a split ("all": both); right(i, f, label) -> bool"""
        got = [right(i, f, r["labels"][f]) for i, r in rows.items() for f in CATEGORIES
               if r["labels"][f] is not None and split in ("all", r["split"])]
        return len(got), {"errors": got.count(False), "accuracy": got.count(True) / len(got)}

    readings = {"decide": lambda i, f, y: (dec[i]["p_true"][f] > 0.5) == y,
                "chat_strict": lambda i, f, y: chat_answer(chat[i]["fields"][f]["answer"]) == y,
                "chat_p_yes": lambda i, f, y: chat[i]["fields"][f]["p_yes"] is not None and (chat[i]["fields"][f]["p_yes"] > 0.5) == y}
    out = {"model": args.model, "chat_file": str(args.chat), "splits": {}}
    for split in (*SPLITS, "all"):
        out["splits"][split] = {}
        for name, right in readings.items():
            out["splits"][split]["n"], out["splits"][split][name] = counts(split, right)
    out["unclear_answers"] = [{"image_id": i, "field": f, "answer": v["answer"], "p_yes": v["p_yes"], "label": rows[i]["labels"][f],
                               "split": rows[i]["split"]}
                              for i, c in chat.items() for f, v in c["fields"].items() if chat_answer(v["answer"]) is None]
    calls = [[c["fields"][f] for f in CATEGORIES] for c in chat.values()]   # in call order: the first call is CATEGORIES[0]
    out["time_s"] = {"decide_per_image_median": statistics.median(d["latency_s"] for d in dec.values()),
                     "chat_per_image_median": statistics.median(c["latency_s"] for c in chat.values()),
                     "chat_per_call_median": statistics.median(v["latency_s"] for cs in calls for v in cs),
                     "chat_first_call_median": statistics.median(cs[0]["latency_s"] for cs in calls),
                     "chat_calls_2_5_median": statistics.median(v["latency_s"] for cs in calls for v in cs[1:])}
    out["prompt_cache"] = None
    if all(v.get("timings") for cs in calls for v in cs):
        def cache(vs):
            return {"calls": len(vs), "prompt_n_median": statistics.median(v["timings"]["prompt_n"] for v in vs),
                    "cache_n_median": statistics.median(v["timings"]["cache_n"] for v in vs),
                    "with_cache_n": sum(v["timings"]["cache_n"] > 0 for v in vs)}
        out["prompt_cache"] = {"first_call": cache([cs[0] for cs in calls]), "calls_2_5": cache([v for cs in calls for v in cs[1:]])}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for split, v in out["splits"].items():
        print(split, f"n {v['n']}", " ".join(f"{k} {v[k]['errors']} wrong ({v[k]['accuracy']:.4f})" for k in readings))
    print(f"{len(out['unclear_answers'])} unclear chat answers | time: decide {out['time_s']['decide_per_image_median']:.2f} s,",
          f"chat {out['time_s']['chat_per_image_median']:.2f} s per image, first call {out['time_s']['chat_first_call_median']:.3f} s,",
          f"calls 2-5 {out['time_s']['chat_calls_2_5_median']:.3f} s | prompt cache {json.dumps(out['prompt_cache'])}")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_select)
    p = sub.add_parser("download")
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--sums", type=Path, required=True)
    p.set_defaults(func=cmd_download)
    for name, func in (("decide", cmd_decide), ("chat", cmd_chat)):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--subset", type=Path, required=True)
        p.add_argument("--images", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.set_defaults(func=func)
        if name == "chat":
            p.add_argument("--cache-prompt", action="store_true")
    for name, func in (("score", cmd_score), ("readings", cmd_readings)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--subset", type=Path, required=True)
        p.add_argument("--decide", type=Path, required=True)
        p.add_argument("--chat", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.set_defaults(func=func)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
