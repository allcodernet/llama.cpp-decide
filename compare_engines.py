"""Per-ticket comparison of our engine (results-engine/preds) with the baseline fork (results/preds).

The baseline fork is a third-party llama.cpp fork with a `POST /v1/decision` endpoint, measured in an earlier phase of
this work; neither the fork nor the scripts that ran it are part of this repository.

Usage: uv run compare_engines.py

For every results-engine/preds/decide-<preset>-b<N>[-suffix].jsonl, joins it with
results/preds/decision-<preset>-b<N>.jsonl, prints the argmax disagreements per field and
writes results-engine/compare-<preset>-b<N>[-suffix].json.
"""

import json
import re
from pathlib import Path

from score import QUEUES, TEST_SET, load_test

ROOT = Path(__file__).parent
ARGMAX = {
    "queue": lambda p: max(QUEUES, key=lambda q: p["queue"][q]),
    "urgency": lambda p: max(range(4), key=lambda i: p["urgency"][i]),
    "angry": lambda p: p["angry"] >= 0.5,
}


def margin(p, field):
    probs = {"queue": list(p["queue"].values()), "urgency": p["urgency"], "angry": [p["angry"], 1 - p["angry"]]}[field]
    top = sorted(probs, reverse=True)
    return round(top[0] - top[1], 4)


def compare(ours_path, test):
    m = re.fullmatch(r"decide-(.+)-b(\d+)(-.+)?", ours_path.stem)
    baseline_path = ROOT / "results" / "preds" / f"decision-{m[1]}-b{m[2]}.jsonl"
    if not baseline_path.exists():
        print(f"{ours_path.stem:36} skipped: no baseline ({baseline_path.name})")
        return
    ours, theirs = ([json.loads(l) for l in p.read_text().splitlines() if l.strip()] for p in (ours_path, baseline_path))
    out = {"ours": str(ours_path.relative_to(ROOT)), "baseline": str(baseline_path.relative_to(ROOT)), "fields": {}}
    for field, argmax in ARGMAX.items():
        rows = [{"ticket": i, "label": row[field], "baseline": argmax(t), "ours": argmax(o),
                 "baseline_margin": margin(t, field), "ours_margin": margin(o, field)}
                for i, (o, t, row) in enumerate(zip(ours, theirs, test)) if argmax(o) != argmax(t)]
        out["fields"][field] = {"disagreements": len(rows), "tickets": rows}
        right = sum(r["ours"] == r["label"] for r in rows), sum(r["baseline"] == r["label"] for r in rows)
        print(f"{ours_path.stem:36} {field:8} {len(rows):2d} disagreements (ours right {right[0]}, baseline right {right[1]}): "
              f"{[r['ticket'] for r in rows]}")
    dest = ROOT / "results-engine" / f"compare-{ours_path.stem.removeprefix('decide-')}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")


def main():
    test = load_test(TEST_SET)
    for path in sorted((ROOT / "results-engine" / "preds").glob("decide-*.jsonl")):
        compare(path, test)


if __name__ == "__main__":
    main()
