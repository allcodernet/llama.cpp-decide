"""Pre-registered selection rule and sign tests for the held-out prompt-variant runs (REPORT-ENGINE.md,
section "Held-out check (320 train tickets)").

Per model, reads <preds-dir>/decide-<preset>-b5<variant suffix>.jsonl for every variant in prompt_variants.py and
marks each ticket right or wrong as score.py does (queue and urgency: argmax; angry: p_true >= 0.5). Winner: most
queue tickets right, tie-break: most urgency + angry tickets right (= the mean of the two accuracies). Exact two-sided
sign tests against `default` on the discordant pairs: the winner on queue, `keywords` on all three fields.
Usage: uv run heldout_stats.py [--preds-dir DIR] [--test-set PATH] [--out PATH]
"""

import argparse
import json
import math
from pathlib import Path

from decide_client import variant_suffix
from prompt_variants import VARIANTS
from score import QUEUES, TRIAGE_DATA, load_test

ROOT = Path(__file__).parent
TRAIN_SET = TRIAGE_DATA / "triage_train.jsonl"
MODELS = ["qwen3.5-9b", "gemma-4-e4b"]
BATCH = 5
FIELDS = ["queue", "urgency", "angry"]


def correctness(preds, test):
    """Per-ticket right/wrong per field, with score.py's decision rules."""
    return {
        "queue": [max(QUEUES, key=lambda q: p["queue"].get(q, 0)) == row["queue"] for p, row in zip(preds, test)],
        "urgency": [max(range(4), key=lambda level: p["urgency"][level]) == row["urgency"] for p, row in zip(preds, test)],
        "angry": [(p["angry"] >= 0.5) == row["angry"] for p, row in zip(preds, test)],
    }


def sign_test(variant_ok, default_ok):
    """Exact two-sided sign test on the discordant pairs: twice the smaller Binomial(n, 0.5) tail, capped at 1."""
    b = sum(v and not d for v, d in zip(variant_ok, default_ok))
    c = sum(d and not v for v, d in zip(variant_ok, default_ok))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
    return {"variant_only_right": b, "default_only_right": c, "discordant": n, "p": p}


def pick_winner(right):
    """right: {variant: {field: tickets right}}. Returns (winner or None if a tie remains, the tied variants)."""
    key = {v: (r["queue"], r["urgency"] + r["angry"]) for v, r in right.items()}
    best = max(key.values())
    tied = [v for v in right if key[v] == best]
    return (tied[0] if len(tied) == 1 else None), tied


def decide(models):
    """The pre-registered rule: change the default only for one winner on both models with p < 0.05 on both."""
    winners = {m: s["winner"] for m, s in models.items()}
    first = next(iter(winners.values()))
    if first is None or any(w != first for w in winners.values()):
        return {"change_default": False, "default": "default", "reason": f"no common single winner: {winners}"}
    if first == "default":
        return {"change_default": False, "default": "default", "reason": "`default` wins on both models"}
    ps = {m: s["winner_vs_default_queue"]["p"] for m, s in models.items()}
    if all(p < 0.05 for p in ps.values()):
        return {"change_default": True, "default": first, "reason": f"`{first}` wins on both models, sign-test p {ps}"}
    return {"change_default": False, "default": "default", "reason": f"`{first}` wins on both models, but sign-test p {ps}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path, default=ROOT / "results-engine" / "heldout" / "preds")
    ap.add_argument("--test-set", type=Path, default=TRAIN_SET)
    ap.add_argument("--out", type=Path, default=ROOT / "results-engine" / "heldout" / "stats.json")
    args = ap.parse_args()
    test = load_test(args.test_set)

    models = {}
    for model in MODELS:
        ok, files = {}, {}
        for variant in VARIANTS:
            path = args.preds_dir / f"decide-{model}-b{BATCH}{variant_suffix(variant)}.jsonl"
            preds = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(preds) == len(test), f"{path}: {len(preds)} preds for {len(test)} tickets"
            ok[variant], files[variant] = correctness(preds, test), path.name
        right = {v: {f: sum(ok[v][f]) for f in FIELDS} for v in VARIANTS}
        winner, tied = pick_winner(right)
        models[model] = {
            "files": files,
            "right": right,
            "accuracy": {v: {f: right[v][f] / len(test) for f in FIELDS} for v in VARIANTS},
            "tie_break": {v: (right[v]["urgency"] + right[v]["angry"]) / (2 * len(test)) for v in VARIANTS},
            "winner": winner,
            "tied": tied,
            "winner_vs_default_queue": sign_test(ok[winner]["queue"], ok["default"]["queue"]) if winner else None,
            "sign_tests": {v: {f: sign_test(ok[v][f], ok["default"][f]) for f in FIELDS} for v in ("keywords",)},
        }
    stats = {"test_set": str(args.test_set), "n_tickets": len(test), "batch": BATCH, "models": models,
             "decision": decide(models)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n")

    for model, s in models.items():
        print(f"{model}: winner {s['winner']} (tied: {s['tied']})")
        for v in VARIANTS:
            r = s["right"][v]
            print(f"  {v:9} queue {r['queue']:3d}  urgency {r['urgency']:3d}  angry {r['angry']:3d}  / {len(test)}")
        if s["winner_vs_default_queue"]:
            print(f"  winner vs default, queue: {s['winner_vs_default_queue']}")
        for v, tests in s["sign_tests"].items():
            print("  " + v + " vs default: " + ", ".join(
                f"{f} {t['variant_only_right']}-{t['default_only_right']} p={t['p']:.3g}" for f, t in tests.items()))
    print(f"decision: {stats['decision']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
