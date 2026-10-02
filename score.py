"""Score every results/preds/*.jsonl against the test labels.

Same metrics as the earlier triage experiment's scorer (data/README.md): accuracy, urg±1, ECE on the queue
confidence, Brier on angry, p50 latency; plus NLL on the queue when full distributions exist.
Usage: uv run score.py [--preds-dir DIR] [--test-set PATH]
TRIAGE_DATA: the directory of triage_test.jsonl and triage_train.jsonl (default data/, the bundled set; relative: from the
project directory); TEST_SET here and in decide_client.py, TRAIN_SET in heldout_stats.py.
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
TRIAGE_DATA = ROOT / (os.environ.get("TRIAGE_DATA") or "data")
TEST_SET = TRIAGE_DATA / "triage_test.jsonl"
QUEUES = ["billing", "technical", "sales", "feedback"]


def ece(confidences, corrects, bins=10):
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    total = 0.0
    for b in range(bins):
        members = [i for i, c in enumerate(confidences) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if members:
            accuracy = sum(corrects[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            total += len(members) / len(confidences) * abs(accuracy - confidence)
    return total


def load_test(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_file(preds_path, test):
    preds = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]
    assert len(preds) == len(test), f"{preds_path}: {len(preds)} preds for {len(test)} tickets"
    queue_pred = [max(QUEUES, key=lambda q: p["queue"].get(q, 0)) for p in preds]
    queue_ok = [pred == row["queue"] for pred, row in zip(queue_pred, test)]
    queue_conf = [max(p["queue"].values()) for p in preds]
    urgency_pred = [max(range(4), key=lambda level: p["urgency"][level]) for p in preds]
    urgency_ok = [pred == row["urgency"] for pred, row in zip(urgency_pred, test)]
    urgency_near = [abs(pred - row["urgency"]) <= 1 for pred, row in zip(urgency_pred, test)]
    angry_ok = [(p["angry"] >= 0.5) == row["angry"] for p, row in zip(preds, test)]
    brier = statistics.mean((p["angry"] - float(row["angry"])) ** 2 for p, row in zip(preds, test))
    full = all(p.get("distribution", "full") == "full" for p in preds)
    nll = None
    if full:
        nll = -statistics.mean(math.log(max(p["queue"][row["queue"]], 1e-9)) for p, row in zip(preds, test))
    return {
        "queue": statistics.mean(queue_ok),
        "urgency": statistics.mean(urgency_ok),
        "urgency_near": statistics.mean(urgency_near),
        "angry": statistics.mean(angry_ok),
        "ece_queue": ece(queue_conf, queue_ok),
        "brier_angry": brier,
        "p50_ms": statistics.median(p["latency_s"] for p in preds) * 1000,
        "nll_queue": nll,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path, default=ROOT / "results" / "preds")
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    args = ap.parse_args()
    test = load_test(args.test_set)
    header = f"{'run':34} {'queue':>6} {'urg':>6} {'urg±1':>6} {'angry':>6} {'ECE(q)':>7} {'Brier(a)':>8} {'NLL(q)':>7} {'p50 ms':>7}"
    print(header + "\n" + "-" * len(header))
    for path in sorted(args.preds_dir.glob("*.jsonl")):
        m = score_file(path, test)
        nll = f"{m['nll_queue']:7.2f}" if m["nll_queue"] is not None else f"{'—':>7}"
        print(
            f"{path.stem:34} {m['queue']:6.1%} {m['urgency']:6.1%} {m['urgency_near']:6.1%} "
            f"{m['angry']:6.1%} {m['ece_queue']:7.3f} {m['brier_angry']:8.3f} {nll} {m['p50_ms']:7.0f}"
        )


if __name__ == "__main__":
    main()
