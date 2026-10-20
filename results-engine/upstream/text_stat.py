"""Compare two checks.py text-run files (results-engine/image/checks.py) under spec 8.3's statistic, per field: the
median and max absolute probability difference, the answer flips, and the flips whose margin (top-1 minus top-2 of
the first file's distribution) exceeds 0.15. Pass: median <= 0.01 for every field and no flip over the margin. A request
whose response has no results on either side (an error) is skipped and listed; the files must hold the same number of
requests, and a comparison without any field fails. Equality
(checks.py text-compare) holds only between builds of the same upstream llama.cpp; a move to a newer upstream changes
its kernels, so the two builds are compared the way two GPUs or two cell placements are (README "Engine switches").

Usage: uv run results-engine/upstream/text_stat.py <a.jsonl> <b.jsonl> --out <path.json>
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "image"))
import checks  # noqa: E402


def probs(answer):
    p = answer.get("probabilities")
    if p is None:
        return [answer["p_true"], 1.0 - answer["p_true"]]
    return [p[k] for k in sorted(p)] if isinstance(p, dict) else list(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = checks.load_lines(args.a), checks.load_lines(args.b)
    if len(a) != len(b):
        sys.exit(f"{len(a)} requests in {args.a}, {len(b)} in {args.b}")
    diffs, flips, skipped = {}, [], []
    for x, y in zip(a, b):
        if not (x["response"].get("results") and y["response"].get("results")):
            skipped.append(x["label"])  # an error response on either side: nothing to compare
            continue
        for sa, sb in zip(x["response"]["results"], y["response"]["results"]):
            for field, va in sa["answers"].items():
                pa, pb = probs(va), probs(sb["answers"][field])
                diffs.setdefault(field, []).extend(abs(u - v) for u, v in zip(pa, pb))
                if va.get("value") != sb["answers"][field].get("value"):
                    top2 = sorted(pa, reverse=True)[:2]
                    flips.append({"label": x["label"], "field": field, "a": va.get("value"), "b": sb["answers"][field].get("value"),
                                  "margin": top2[0] - top2[1] if len(top2) > 1 else 1.0})
    fields = {f: {"n": len(d), "median_abs_diff": statistics.median(d), "max_abs_diff": max(d)} for f, d in diffs.items()}
    over = [f for f in flips if f["margin"] > 0.15]
    out = {"a": str(args.a), "b": str(args.b), "requests": len(a), "skipped": skipped, "fields": fields, "flips": flips,
           "flips_over_margin": len(over),
           "pass": bool(fields) and all(v["median_abs_diff"] <= 0.01 for v in fields.values()) and not over}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for f, v in fields.items():
        print(f"{f:8s} n={v['n']:5d} median {v['median_abs_diff']:.5f} max {v['max_abs_diff']:.4f}")
    print(f"flips {len(flips)} (over margin {len(over)}); skipped {len(skipped)}; pass={out['pass']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
