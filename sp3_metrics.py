"""Metrics and comparisons for sub-project 3 (spec 2026-09-25-engine-improvements-design.md, sections 12.2 and 12.4).

  uv run sp3_metrics.py score --preds-dir D [--test-set T]
      For every decide_client.py preds file D/*.jsonl: queue/urgency/urg±1/angry accuracy (score.py's decision
      rules), NLL per field (mean over tickets of -ln p(label); inf when that probability is 0) and nll_sum = the
      three fields' NLLs summed, ECE per field (10 bins, on the probability of the predicted value), Brier for angry,
      p50 latency per ticket and the run's rounds (from D/../runs/<name>.json when present). Writes D/score.json and
      prints a table.

  uv run sp3_metrics.py compare A.jsonl B.jsonl [--max-diff X] [--out PATH]
      Each file is either decide_client.py preds rows (one per ticket: `queue` dict, `urgency` list, `angry` p_true)
      or reference_check.py dump records (one per ticket and field, the engine's answer under `engine`); the format is
      detected per file, so a preds file can be compared with a dump. Entries are joined by position (preds rows have
      no ids; their order is the API's) and must name the same ticket, field and option keys. Prints (and with --out
      writes) JSON: the max abs diff over all probabilities and, per field, max/median abs diff, argmax agreement and
      the batch-invariance statistic of batch_invariance.py (median abs diff over options <= 0.01 and no argmax
      disagreement where A's top-2 margin > 0.15; A is the reference). Exit 1 when the max abs diff exceeds --max-diff.

  uv run sp3_metrics.py signtest [A.jsonl B.jsonl [--test-set T]] [--pool A B T ...] [--out PATH]
      Exact two-sided sign test per field on per-ticket correctness (heldout_stats.sign_test, score.py's decision
      rules) of preds A vs preds B. The positional pair is scored on --test-set; every --pool A B T adds a pair with
      its own labels, and all pairs are pooled (e.g. test + train = 400 tickets). Prints (and with --out writes) JSON
      with each pair's result and the pooled one.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from heldout_stats import correctness, sign_test
from reference_check import engine_probs
from score import QUEUES, TEST_SET, ece, load_test

FIELDS = ["queue", "urgency", "angry"]
MARGIN = 0.15  # the statistic: argmax disagreements only count where the reference top-2 margin exceeds this
MEDIAN_BOUND = 0.01  # the statistic: per field median abs diff over options


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def label_prob(pred, field, row):
    """The probability a preds row gives to the ticket's labelled value of `field`."""
    if field == "angry":
        return pred["angry"] if row["angry"] else 1.0 - pred["angry"]
    return pred[field][row[field]]


def nll(preds, test):
    """Per field the mean over tickets of -ln p(label), and `sum` = the three fields' NLLs summed."""
    assert len(preds) == len(test), f"{len(preds)} preds for {len(test)} tickets"
    out = {}
    for f in FIELDS:
        probs = [label_prob(pred, f, row) for pred, row in zip(preds, test)]
        out[f] = statistics.mean(-math.log(p) if p > 0 else math.inf for p in probs)
    out["sum"] = sum(out[f] for f in FIELDS)
    return out


def top_confidence(pred, field):
    """The probability of the predicted value (score.py's argmax / p_true >= 0.5 rules)."""
    if field == "angry":
        return max(pred["angry"], 1.0 - pred["angry"])
    return max(pred[field].values()) if field == "queue" else max(pred[field])


def score_preds(preds, test):
    assert len(preds) == len(test), f"{len(preds)} preds for {len(test)} tickets"
    ok = correctness(preds, test)
    urgency_pred = [max(range(4), key=lambda level: p["urgency"][level]) for p in preds]
    loss = nll(preds, test)
    return {
        "queue": statistics.mean(ok["queue"]),
        "urgency": statistics.mean(ok["urgency"]),
        "urgency_near": statistics.mean(abs(u - row["urgency"]) <= 1 for u, row in zip(urgency_pred, test)),
        "angry": statistics.mean(ok["angry"]),
        **{f"nll_{k}": v for k, v in loss.items()},
        **{f"ece_{f}": ece([top_confidence(p, f) for p in preds], ok[f]) for f in FIELDS},
        "brier_angry": statistics.mean((p["angry"] - float(row["angry"])) ** 2 for p, row in zip(preds, test)),
        "p50_ms": statistics.median(p["latency_s"] for p in preds) * 1000,
    }


def distributions(path):
    """(format, entries) with one entry (ticket, field, {option key: probability}) per ticket and field."""
    rows = load_jsonl(path)
    if rows and "engine" in rows[0] and "field" in rows[0]:
        return "dump", [(r["ticket"], r["field"], dict(zip(r["keys"], engine_probs(r)))) for r in rows]
    if rows and all(f in rows[0] for f in FIELDS):
        entries = []
        for i, r in enumerate(rows):
            entries.append((i, "queue", dict(r["queue"])))
            entries.append((i, "urgency", {str(level): p for level, p in enumerate(r["urgency"])}))
            entries.append((i, "angry", {"true": r["angry"], "false": 1.0 - r["angry"]}))
        return "preds", entries
    sys.exit(f"{path}: neither decide_client.py preds rows nor reference_check.py dump records")


def compare(entries_a, entries_b):
    """Max abs diff, argmax agreement and the statistic per field; A is the reference for the margin."""
    if len(entries_a) != len(entries_b):
        sys.exit(f"entry count mismatch: {len(entries_a)} vs {len(entries_b)}")
    diffs, agree, over_margin = {}, {}, {}
    for (ta, fa, pa), (tb, fb, pb) in zip(entries_a, entries_b):
        if (ta, fa, sorted(pa)) != (tb, fb, sorted(pb)):
            sys.exit(f"entries do not line up: ticket {ta} {fa} {sorted(pa)} vs ticket {tb} {fb} {sorted(pb)}")
        keys = list(pa)
        diffs.setdefault(fa, []).extend(abs(pa[k] - pb[k]) for k in keys)
        disagree = max(keys, key=pa.get) != max(keys, key=pb.get)
        agree.setdefault(fa, []).append(not disagree)
        top2 = sorted(pa.values(), reverse=True)
        margin = top2[0] - top2[1] if len(top2) > 1 else 1.0
        over_margin[fa] = over_margin.get(fa, 0) + (disagree and margin > MARGIN)
    fields = {}
    for f in diffs:
        median = statistics.median(diffs[f])
        fields[f] = {
            "median_abs_diff": median,
            "max_abs_diff": max(diffs[f]),
            "argmax_agreement": statistics.mean(agree[f]),
            "argmax_disagreements": agree[f].count(False),
            "disagreements_over_margin": over_margin[f],
            "pass": median <= MEDIAN_BOUND and over_margin[f] == 0,
        }
    return {
        "n_tickets": len({t for t, _, _ in entries_a}),
        "max_abs_diff": max(v["max_abs_diff"] for v in fields.values()),
        "fields": fields,
        "pass": all(v["pass"] for v in fields.values()),
    }


def signtest(pairs):
    """pairs: [(preds_a, preds_b, test)]. Per field: exact two-sided sign test over the tickets of all pairs."""
    ok_a, ok_b = {f: [] for f in FIELDS}, {f: [] for f in FIELDS}
    for preds_a, preds_b, test in pairs:
        assert len(preds_a) == len(preds_b) == len(test), f"{len(preds_a)} / {len(preds_b)} preds for {len(test)} tickets"
        ca, cb = correctness(preds_a, test), correctness(preds_b, test)
        for f in FIELDS:
            ok_a[f] += ca[f]
            ok_b[f] += cb[f]
    fields = {}
    for f in FIELDS:
        t = sign_test(ok_a[f], ok_b[f])
        fields[f] = {"a_right": sum(ok_a[f]), "b_right": sum(ok_b[f]), "a_only_right": t["variant_only_right"],
                     "b_only_right": t["default_only_right"], "discordant": t["discordant"], "p": t["p"]}
    return {"n_tickets": len(ok_a["queue"]), "fields": fields}


def format_table(runs):
    width = max(len("run"), *(len(name) for name in runs))
    header = (f"{'run':{width}} {'queue':>6} {'urg':>6} {'urg±1':>6} {'angry':>6} {'NLL q':>6} {'NLL u':>6} {'NLL a':>6} "
              f"{'NLL Σ':>6} {'ECE q':>6} {'ECE u':>6} {'ECE a':>6} {'Brier a':>7} {'p50 ms':>7} {'rounds':>6}")
    lines = [header, "-" * len(header)]
    for name, m in runs.items():
        rounds = ",".join(map(str, m["rounds"])) if m["rounds"] is not None else "—"
        lines.append(
            f"{name:{width}} {m['queue']:6.1%} {m['urgency']:6.1%} {m['urgency_near']:6.1%} {m['angry']:6.1%} "
            f"{m['nll_queue']:6.3f} {m['nll_urgency']:6.3f} {m['nll_angry']:6.3f} {m['nll_sum']:6.3f} "
            f"{m['ece_queue']:6.3f} {m['ece_urgency']:6.3f} {m['ece_angry']:6.3f} {m['brier_angry']:7.3f} "
            f"{m['p50_ms']:7.1f} {rounds:>6}")
    return "\n".join(lines)


def emit(result, out):
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


def cmd_score(args):
    test = load_test(args.test_set)
    paths = sorted(args.preds_dir.glob("*.jsonl"))
    if not paths:
        sys.exit(f"no *.jsonl preds files in {args.preds_dir}")
    runs = {}
    for path in paths:
        m = score_preds(load_jsonl(path), test)
        run_json = path.parent.parent / "runs" / f"{path.stem}.json"
        m["rounds"] = json.loads(run_json.read_text()).get("rounds") if run_json.exists() else None
        runs[path.stem] = m
    out = {"preds_dir": str(args.preds_dir), "test_set": str(args.test_set), "n_tickets": len(test), "runs": runs}
    (args.preds_dir / "score.json").write_text(json.dumps(out, indent=2) + "\n")
    print(format_table(runs))
    print(f"wrote {args.preds_dir / 'score.json'}")
    return 0


def cmd_compare(args):
    fmt_a, entries_a = distributions(args.a)
    fmt_b, entries_b = distributions(args.b)
    result = {"a": str(args.a), "b": str(args.b), "formats": [fmt_a, fmt_b], **compare(entries_a, entries_b)}
    if args.max_diff is not None:
        result["max_diff_bound"] = args.max_diff
        result["within_bound"] = result["max_abs_diff"] <= args.max_diff
    emit(result, args.out)
    if args.max_diff is not None and not result["within_bound"]:
        print(f"max abs diff {result['max_abs_diff']:.3g} exceeds --max-diff {args.max_diff:g}", file=sys.stderr)
        return 1
    return 0


def cmd_signtest(args):
    triples = [tuple(t) for t in args.pool]
    if args.a is not None:
        if args.b is None:
            sys.exit("signtest: give both A and B (or --pool A B T)")
        triples.insert(0, (args.a, args.b, args.test_set))
    if not triples:
        sys.exit("signtest: give A B and/or --pool A B T")
    pairs = [(load_jsonl(a), load_jsonl(b), load_test(t)) for a, b, t in triples]
    result = {
        "pairs": [{"a": str(a), "b": str(b), "test_set": str(t), **signtest([pair])} for (a, b, t), pair in zip(triples, pairs)],
        "pooled": signtest(pairs),
    }
    emit(result, args.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sub-project 3 metrics: score, compare, signtest.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="metrics per preds file; writes <preds-dir>/score.json")
    s.add_argument("--preds-dir", type=Path, required=True)
    s.add_argument("--test-set", type=Path, default=TEST_SET)
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="max abs diff, argmax agreement and the statistic of two preds/dump files")
    c.add_argument("a", type=Path, help="reference (its top-2 margin gates argmax disagreements)")
    c.add_argument("b", type=Path)
    c.add_argument("--max-diff", type=float, default=None, help="exit 1 when the max abs diff exceeds this")
    c.add_argument("--out", type=Path, default=None)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("signtest", help="exact two-sided sign test per field on per-ticket correctness")
    t.add_argument("a", type=Path, nargs="?")
    t.add_argument("b", type=Path, nargs="?")
    t.add_argument("--test-set", type=Path, default=TEST_SET, help="labels for the positional pair")
    t.add_argument("--pool", nargs=3, type=Path, action="append", default=[], metavar=("A", "B", "T"),
                   help="another pair with its labels, pooled with the others (repeatable)")
    t.add_argument("--out", type=Path, default=None)
    t.set_defaults(func=cmd_signtest)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
