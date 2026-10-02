import json
import math
from pathlib import Path

import pytest

from sp3_metrics import QUEUES, compare, distributions, main, nll, score_preds, signtest

LN2 = math.log(2)

# Three tickets; hand-computed metrics below.
TEST = [
    {"text": "a", "queue": "billing", "urgency": 0, "angry": False},
    {"text": "b", "queue": "sales", "urgency": 3, "angry": True},
    {"text": "c", "queue": "technical", "urgency": 1, "angry": False},
]
PREDS = [
    {"queue": {"billing": 0.5, "technical": 0.25, "sales": 0.125, "feedback": 0.125},
     "urgency": [0.5, 0.25, 0.125, 0.125], "angry": 0.25, "latency_s": 0.1},
    {"queue": {"billing": 0.5, "technical": 0.25, "sales": 0.125, "feedback": 0.125},  # queue wrong
     "urgency": [0.0, 0.25, 0.5, 0.25], "angry": 0.875, "latency_s": 0.3},  # urgency off by one
    {"queue": {"billing": 0.125, "technical": 0.75, "sales": 0.0625, "feedback": 0.0625},
     "urgency": [0.25, 0.5, 0.125, 0.125], "angry": 0.625, "latency_s": 0.2},  # angry wrong
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_score_preds_hand_computed():
    m = score_preds(PREDS, TEST)
    assert (m["queue"], m["urgency"], m["urgency_near"], m["angry"]) == pytest.approx((2 / 3, 2 / 3, 1.0, 2 / 3))
    # NLL: mean over tickets of -ln p(label)
    assert m["nll_queue"] == pytest.approx((LN2 + 3 * LN2 - math.log(0.75)) / 3)
    assert m["nll_urgency"] == pytest.approx((LN2 + 2 * LN2 + LN2) / 3)
    assert m["nll_angry"] == pytest.approx(-(math.log(0.75) + math.log(0.875) + math.log(0.375)) / 3)
    assert m["nll_sum"] == pytest.approx(m["nll_queue"] + m["nll_urgency"] + m["nll_angry"])
    assert nll(PREDS, TEST)["sum"] == pytest.approx(m["nll_sum"])
    # ECE, 10 bins, on the probability of the predicted value.
    # queue: .5 (right) and .5 (wrong) share a bin (gap 0); .75 (right) alone -> 1/3 * .25
    assert m["ece_queue"] == pytest.approx(0.25 / 3)
    # urgency: .5, .5, .5 in one bin, 2 of 3 right -> |2/3 - 1/2|
    assert m["ece_urgency"] == pytest.approx(1 / 6)
    # angry: confidences .75 (right), .875 (right), .625 (wrong), one per bin
    assert m["ece_angry"] == pytest.approx((0.25 + 0.125 + 0.625) / 3)
    assert m["brier_angry"] == pytest.approx((0.25 ** 2 + 0.125 ** 2 + 0.625 ** 2) / 3)
    assert m["p50_ms"] == pytest.approx(200.0)


def test_nll_is_infinite_for_a_zero_label_probability():
    preds = [dict(PREDS[0], queue={"billing": 0.0, "technical": 1.0, "sales": 0.0, "feedback": 0.0})]
    assert nll(preds, TEST[:1])["queue"] == math.inf


def test_score_command_writes_score_json_with_rounds(tmp_path: Path):
    (tmp_path / "preds").mkdir()
    (tmp_path / "runs").mkdir()
    write_jsonl(tmp_path / "preds" / "x.jsonl", PREDS)
    (tmp_path / "runs" / "x.json").write_text(json.dumps({"rounds": [1]}))
    test_set = write_jsonl(tmp_path / "labels.jsonl", TEST)
    assert main(["score", "--preds-dir", str(tmp_path / "preds"), "--test-set", str(test_set)]) == 0
    out = json.loads((tmp_path / "preds" / "score.json").read_text())
    assert out["n_tickets"] == 3
    run = out["runs"]["x"]
    assert run["rounds"] == [1]
    assert run["nll_sum"] == pytest.approx(score_preds(PREDS, TEST)["nll_sum"])


def test_score_command_fails_without_preds(tmp_path: Path):
    (tmp_path / "preds").mkdir()
    test_set = write_jsonl(tmp_path / "t.jsonl", TEST)
    with pytest.raises(SystemExit):
        main(["score", "--preds-dir", str(tmp_path / "preds"), "--test-set", str(test_set)])


PREDS_A = [
    {"queue": {"billing": 0.7, "technical": 0.1, "sales": 0.1, "feedback": 0.1}, "urgency": [0.1, 0.6, 0.2, 0.1], "angry": 0.9},
    {"queue": {"billing": 0.1, "technical": 0.45, "sales": 0.4, "feedback": 0.05}, "urgency": [0.1, 0.2, 0.3, 0.4], "angry": 0.2},
]
PREDS_B = [
    {"queue": {"billing": 0.69, "technical": 0.11, "sales": 0.1, "feedback": 0.1}, "urgency": [0.1, 0.6, 0.2, 0.1], "angry": 0.9},
    # queue argmax flips inside A's 0.05 margin; angry flips outside A's 0.6 margin
    {"queue": {"billing": 0.1, "technical": 0.35, "sales": 0.5, "feedback": 0.05}, "urgency": [0.1, 0.2, 0.3, 0.4], "angry": 0.7},
]


def test_compare_preds_files(tmp_path: Path):
    a = write_jsonl(tmp_path / "a.jsonl", PREDS_A)
    b = write_jsonl(tmp_path / "b.jsonl", PREDS_B)
    fmt_a, entries_a = distributions(a)
    fmt_b, entries_b = distributions(b)
    assert (fmt_a, fmt_b) == ("preds", "preds")
    r = compare(entries_a, entries_b)
    assert r["n_tickets"] == 2
    assert r["max_abs_diff"] == pytest.approx(0.5)
    q, u, an = r["fields"]["queue"], r["fields"]["urgency"], r["fields"]["angry"]
    # queue diffs .01 .01 0 0 | 0 .1 .1 0 -> median .005
    assert q["max_abs_diff"] == pytest.approx(0.1)
    assert q["median_abs_diff"] == pytest.approx(0.005)
    assert (q["argmax_disagreements"], q["disagreements_over_margin"], q["argmax_agreement"]) == (1, 0, 0.5)
    assert q["pass"] is True
    assert (u["max_abs_diff"], u["argmax_disagreements"], u["pass"]) == (0.0, 0, True)
    assert an["median_abs_diff"] == pytest.approx(0.25)
    assert (an["argmax_disagreements"], an["disagreements_over_margin"], an["pass"]) == (1, 1, False)
    assert r["pass"] is False


def test_compare_command_max_diff_exit_code(tmp_path: Path):
    a = write_jsonl(tmp_path / "a.jsonl", PREDS_A)
    b = write_jsonl(tmp_path / "b.jsonl", PREDS_B)
    assert main(["compare", str(a), str(b), "--max-diff", "0.6"]) == 0
    assert main(["compare", str(a), str(b), "--max-diff", "0.1", "--out", str(tmp_path / "c.json")]) == 1
    out = json.loads((tmp_path / "c.json").read_text())
    assert out["formats"] == ["preds", "preds"] and out["within_bound"] is False
    assert main(["compare", str(a), str(a), "--max-diff", "1e-6"]) == 0


def dump_records(ticket, queue, urgency, p_true):
    """reference_check.py dump records (one per field) for one ticket; `nodes` is irrelevant here."""
    return [
        {"ticket": ticket, "field": "queue", "type": "choice", "keys": list(queue), "nodes": [],
         "engine": {"value": max(queue, key=queue.get), "probabilities": queue, "confidence": max(queue.values())}},
        {"ticket": ticket, "field": "urgency", "type": "score", "keys": ["0", "1", "2", "3"], "nodes": [],
         "engine": {"value": max(range(4), key=lambda i: urgency[i]), "probabilities": urgency,
                    "expected": sum(i * p for i, p in enumerate(urgency)), "confidence": max(urgency)}},
        {"ticket": ticket, "field": "angry", "type": "bool", "keys": ["true", "false"], "nodes": [],
         "engine": {"value": p_true >= 0.5, "p_true": p_true, "confidence": max(p_true, 1 - p_true)}},
    ]


def test_compare_dump_files(tmp_path: Path):
    a = write_jsonl(tmp_path / "a.jsonl",
                    dump_records(0, {"billing": 0.6, "technical": 0.2, "sales": 0.1, "feedback": 0.1}, [0.1, 0.2, 0.3, 0.4], 0.3)
                    + dump_records(1, {"billing": 0.1, "technical": 0.1, "sales": 0.1, "feedback": 0.7}, [0.4, 0.3, 0.2, 0.1], 0.8))
    b = write_jsonl(tmp_path / "b.jsonl",
                    dump_records(0, {"billing": 0.3, "technical": 0.5, "sales": 0.1, "feedback": 0.1}, [0.1, 0.2, 0.3, 0.4], 0.31)
                    + dump_records(1, {"billing": 0.1, "technical": 0.1, "sales": 0.1, "feedback": 0.7}, [0.4, 0.3, 0.2, 0.1], 0.8))
    fmt_a, entries_a = distributions(a)
    fmt_b, entries_b = distributions(b)
    assert (fmt_a, fmt_b) == ("dump", "dump")
    r = compare(entries_a, entries_b)
    assert r["n_tickets"] == 2
    assert r["max_abs_diff"] == pytest.approx(0.3)
    q, u, an = r["fields"]["queue"], r["fields"]["urgency"], r["fields"]["angry"]
    # billing .6 -> .3 flips the argmax outside A's 0.4 margin
    assert (q["argmax_disagreements"], q["disagreements_over_margin"], q["pass"]) == (1, 1, False)
    assert (u["max_abs_diff"], u["argmax_disagreements"]) == (0.0, 0)
    assert an["max_abs_diff"] == pytest.approx(0.01)
    assert an["argmax_disagreements"] == 0 and an["pass"] is True
    assert main(["compare", str(a), str(b), "--max-diff", "0.29"]) == 1
    assert main(["compare", str(a), str(b), "--max-diff", "0.31"]) == 0


def test_compare_preds_with_dump(tmp_path: Path):
    dump = write_jsonl(tmp_path / "d.jsonl", dump_records(0, dict(PREDS_A[0]["queue"]), PREDS_A[0]["urgency"], PREDS_A[0]["angry"]))
    preds = write_jsonl(tmp_path / "p.jsonl", PREDS_A[:1])
    fmt_p, entries_p = distributions(preds)
    fmt_d, entries_d = distributions(dump)
    assert (fmt_p, fmt_d) == ("preds", "dump")
    assert compare(entries_p, entries_d)["max_abs_diff"] == 0.0
    # entries that do not line up (2 tickets vs 1) are an error, not a silent partial comparison
    _, two = distributions(write_jsonl(tmp_path / "p2.jsonl", PREDS_A))
    with pytest.raises(SystemExit):
        compare(two, entries_d)


def pred_for(row, queue_right=True, urgency_right=True, angry_right=True):
    """A preds row whose argmax answers are right or wrong per field for the labelled ticket `row`."""
    q = row["queue"] if queue_right else next(k for k in QUEUES if k != row["queue"])
    level = row["urgency"] if urgency_right else (row["urgency"] + 2) % 4
    return {"queue": {k: 0.7 if k == q else 0.1 for k in QUEUES},
            "urgency": [0.7 if i == level else 0.1 for i in range(4)],
            "angry": 0.9 if row["angry"] == angry_right else 0.1}


def labels(n):
    return [{"text": str(i), "queue": QUEUES[i % 4], "urgency": i % 4, "angry": i % 2 == 0} for i in range(n)]


def test_signtest_pools_pairs():
    t1, t2 = labels(10), labels(6)
    # pair 1: queue 10-0 for A; urgency 3-1 for A (tickets 0-2 A only, ticket 3 B only)
    a1 = [pred_for(r, urgency_right=i != 3) for i, r in enumerate(t1)]
    b1 = [pred_for(r, queue_right=False, urgency_right=i >= 3) for i, r in enumerate(t1)]
    # pair 2: queue 6-0 for A
    a2 = [pred_for(r) for r in t2]
    b2 = [pred_for(r, queue_right=False) for r in t2]
    one = signtest([(a1, b1, t1)])
    assert one["n_tickets"] == 10
    assert one["fields"]["queue"]["p"] == pytest.approx(2 / 2 ** 10)
    pooled = signtest([(a1, b1, t1), (a2, b2, t2)])
    assert pooled["n_tickets"] == 16
    q, u, an = pooled["fields"]["queue"], pooled["fields"]["urgency"], pooled["fields"]["angry"]
    assert (q["a_only_right"], q["b_only_right"], q["discordant"]) == (16, 0, 16)
    assert q["p"] == pytest.approx(3.05e-05, rel=1e-3)
    assert (q["a_right"], q["b_right"]) == (16, 0)
    assert (u["a_only_right"], u["b_only_right"], u["discordant"]) == (3, 1, 4)
    assert u["p"] == pytest.approx(0.625)
    assert (an["discordant"], an["p"]) == (0, 1.0)


def test_signtest_command_pool(tmp_path: Path):
    t1, t2 = labels(10), labels(6)
    files = []
    for name, test in [("test", t1), ("train", t2)]:
        files += [write_jsonl(tmp_path / f"a-{name}.jsonl", [pred_for(r) for r in test]),
                  write_jsonl(tmp_path / f"b-{name}.jsonl", [pred_for(r, queue_right=False) for r in test]),
                  write_jsonl(tmp_path / f"t-{name}.jsonl", test)]
    out = tmp_path / "s.json"
    assert main(["signtest", "--pool", *map(str, files[:3]), "--pool", *map(str, files[3:]), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert [p["n_tickets"] for p in result["pairs"]] == [10, 6]
    assert result["pooled"]["fields"]["queue"]["p"] == pytest.approx(3.05e-05, rel=1e-3)
    # positional pair + --test-set is the single-pair form
    assert main(["signtest", str(files[0]), str(files[1]), "--test-set", str(files[2]), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["pooled"]["fields"]["queue"]["a_only_right"] == 10
