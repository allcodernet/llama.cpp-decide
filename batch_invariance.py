"""Compare two decide_client.py preds jsonl files line-for-line with spec 8.3's statistic.

Usage: uv run batch_invariance.py <preds_a.jsonl> <preds_b.jsonl> --out <path.json>

preds_a is treated as the reference for the over-margin gate (the run with less
batching is closer to independent, mirroring reference_check.py's convention).
Reports, per field: median_abs_diff, max_abs_diff, argmax_agreement,
disagreements_over_margin, and argmax_abs_diff_median (abs diff at preds_a's argmax
option only -- arity-independent, reported not gated). Pass (spec 8.3): median <= 0.01
and zero over-margin disagreements.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

FIELDS = ["queue", "urgency", "angry"]


def field_probs(rec, field):
    if field == "queue":
        keys = sorted(rec["queue"])
        return [rec["queue"][k] for k in keys]
    if field == "urgency":
        return list(rec["urgency"])
    if field == "angry":
        p = rec["angry"]
        return [p, 1.0 - p]
    raise ValueError(field)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_a", type=Path)
    ap.add_argument("preds_b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a = [json.loads(l) for l in args.preds_a.read_text().splitlines() if l.strip()]
    b = [json.loads(l) for l in args.preds_b.read_text().splitlines() if l.strip()]
    if len(a) != len(b):
        sys.exit(f"length mismatch: {len(a)} vs {len(b)}")

    abs_diffs, argmax_agree, over_margin, argmax_diffs = {}, {}, {}, {}
    for ra, rb in zip(a, b):
        for field in FIELDS:
            pa, pb = field_probs(ra, field), field_probs(rb, field)
            diffs = [abs(x - y) for x, y in zip(pa, pb)]
            abs_diffs.setdefault(field, []).extend(diffs)
            amax_a = max(range(len(pa)), key=lambda i: pa[i])
            amax_b = max(range(len(pb)), key=lambda i: pb[i])
            agree = amax_a == amax_b
            argmax_agree.setdefault(field, []).append(agree)
            argmax_diffs.setdefault(field, []).append(abs(pa[amax_a] - pb[amax_a]))
            top2 = sorted(pa, reverse=True)
            margin = top2[0] - top2[1] if len(top2) > 1 else 1.0
            if not agree and margin > 0.15:
                over_margin[field] = over_margin.get(field, 0) + 1

    fields = {}
    for field in FIELDS:
        fields[field] = {
            "median_abs_diff": statistics.median(abs_diffs[field]),
            "max_abs_diff": max(abs_diffs[field]),
            "argmax_agreement": statistics.mean(argmax_agree[field]),
            "disagreements_over_margin": over_margin.get(field, 0),
            "argmax_abs_diff_median": statistics.median(argmax_diffs[field]),
        }
    passed = all(v["median_abs_diff"] <= 0.01 and v["disagreements_over_margin"] == 0 for v in fields.values())

    out = {"preds_a": str(args.preds_a), "preds_b": str(args.preds_b), "n": len(a), "fields": fields, "pass": passed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(fields, indent=2))
    print(f"pass: {passed}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
