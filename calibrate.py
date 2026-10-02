"""Fit one decide temperature per model (spec docs/specs/2026-09-25-engine-improvements-design.md, section 8).

Usage: uv run calibrate.py [--models M ...] [--baseline-dir D] [--train-set T] [--test-set T] [--out-dir O]

Per model the input is the stored probabilities of the tree-mode b5 runs with decide_client.py's default prompt variant
(D/train/preds/decide-<model>-b5-keywords.jsonl and D/test/preds/..., D = the Task 0 baseline). T is scaled post hoc
exactly as the engine does it (decide-score.cpp apply_temperature: p_T(o) proportional to exp(log p(o) / T); angry as
[p_true, 1 - p_true]). The fit minimises the summed NLL of queue, urgency and angry on the train set (sp3_metrics.nll):
64 log-spaced grid points over [0.25, 4], then golden-section refinement between the best grid point's neighbours; T is
reported to 4 decimals and every "after" metric uses that value. Writes O/calibration-<model>.json: T, train and test
NLL, ECE (10 bins) and Brier (angry) before (T = 1) and after (sp3_metrics.score_preds), and the rule fixed before
looking at the test set: T goes into the model's preset in models-engine.ini as decide-temperature only if it lowers
the test-set summed NLL. This script only reports the rule's decision; it never edits models-engine.ini.
Relative paths are taken from the project directory; --train-set and --test-set default to TRIAGE_DATA's files (score.py).
"""

import argparse
import json
import math
import sys
from pathlib import Path

from decide_client import DEFAULT_VARIANT, variant_suffix
from heldout_stats import TRAIN_SET
from score import TEST_SET, load_test
from sp3_metrics import load_jsonl, nll, score_preds

ROOT = Path(__file__).parent
MODELS = ["qwen3.5-9b", "gemma-4-e4b"]
BATCH = 5
GRID_LO, GRID_HI, GRID_POINTS = 0.25, 4.0, 64
TOL = 1e-6  # golden-section stops when its bracket is narrower than this
DECIMALS = 4
METRICS = ["nll_queue", "nll_urgency", "nll_angry", "nll_sum", "ece_queue", "ece_urgency", "ece_angry", "brier_angry"]
RULE = ("the fitted T goes into the model's preset in models-engine.ini as decide-temperature only if it lowers the "
        "test-set summed NLL")


def temper(p, t):
    """The engine's apply_temperature: p_T(o) proportional to exp(log p(o) / T); T = 1 returns p; zeros stay zero."""
    if t == 1.0:
        return list(p)
    scaled = [math.log(x) / t if x > 0 else -math.inf for x in p]
    top = max(scaled)
    w = [math.exp(v - top) for v in scaled]
    z = sum(w)
    return [x / z for x in w]


def temper_preds(preds, t):
    """Copies of decide_client.py preds rows with queue, urgency and angry (p_true) at temperature T."""
    out = []
    for p in preds:
        keys = list(p["queue"])
        out.append(dict(p, queue=dict(zip(keys, temper([p["queue"][k] for k in keys], t))),
                        urgency=temper(p["urgency"], t), angry=temper([p["angry"], 1.0 - p["angry"]], t)[0]))
    return out


def grid():
    return [GRID_LO * (GRID_HI / GRID_LO) ** (i / (GRID_POINTS - 1)) for i in range(GRID_POINTS)]


def golden_section(f, a, b):
    """The minimum of a unimodal f on [a, b], to within TOL."""
    inv_phi = (math.sqrt(5) - 1) / 2
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    while b - a > TOL:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_temperature(preds, labels):
    """T minimising the summed NLL of the three fields: the best grid point, then golden-section between its neighbours
    (the summed NLL is convex in 1/T, so unimodal in T)."""
    def loss(t):
        return nll(temper_preds(preds, t), labels)["sum"]

    g = grid()
    losses = [loss(t) for t in g]
    best = min(range(len(g)), key=losses.__getitem__)
    bracket = [g[max(best - 1, 0)], g[min(best + 1, len(g) - 1)]]
    refined = golden_section(loss, *bracket)
    return {"grid": {"lo": GRID_LO, "hi": GRID_HI, "points": GRID_POINTS, "spacing": "log", "best": g[best],
                     "bracket": bracket},
            "t_refined": refined, "t": round(refined, DECIMALS)}


def metrics(preds, labels):
    m = score_preds(preds, labels)
    return {k: m[k] for k in METRICS}


def calibrate(train_preds, train_labels, test_preds, test_labels):
    """Fit T on the train rows; the metrics before (T = 1) and after on both sets; the rule's decision."""
    result = fit_temperature(train_preds, train_labels)
    t = result["t"]
    for split, preds, labels in (("train", train_preds, train_labels), ("test", test_preds, test_labels)):
        result[split] = {"before": metrics(preds, labels), "after": metrics(temper_preds(preds, t), labels)}
    before, after = result["test"]["before"]["nll_sum"], result["test"]["after"]["nll_sum"]
    result["rule"] = {"text": RULE, "test_nll_sum_before": before, "test_nll_sum_after": after, "apply": after < before}
    return result


def format_report(r):
    head = f"{'set':6} {'T':>7} {'NLL q':>6} {'NLL u':>6} {'NLL a':>6} {'NLL Σ':>6} {'ECE q':>6} {'ECE u':>6} {'ECE a':>6} {'Brier a':>7}"
    lines = [f"{r['model']}: T = {r['t']} (grid best {r['grid']['best']:.4f}, golden-section in "
             f"[{r['grid']['bracket'][0]:.4f}, {r['grid']['bracket'][1]:.4f}] -> {r['t_refined']:.6f})",
             head, "-" * len(head)]
    for split in ("train", "test"):
        for t, m in ((1.0, r[split]["before"]), (r["t"], r[split]["after"])):
            lines.append(f"{split:6} {t:7.4f} " + " ".join(f"{m[k]:6.3f}" for k in METRICS[:-1]) + f" {m['brier_angry']:7.3f}")
    rule = r["rule"]
    verdict = (f"set decide-temperature = {r['t']} in [{r['model']}]" if rule["apply"]
               else "keep the preset without decide-temperature")
    lines.append(f"rule: test summed NLL {rule['test_nll_sum_before']:.4f} -> {rule['test_nll_sum_after']:.4f}: {verdict}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit one decide temperature per model on the baseline train run (spec 8).")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--baseline-dir", type=Path, default=Path("results-engine/sp3/baseline"))
    ap.add_argument("--train-set", type=Path, default=TRAIN_SET)
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    ap.add_argument("--out-dir", type=Path, default=Path("results-engine/sp3/eval"))
    args = ap.parse_args(argv)
    train_labels, test_labels = load_test(ROOT / args.train_set), load_test(ROOT / args.test_set)
    for model in args.models:
        name = f"decide-{model}-b{BATCH}{variant_suffix(DEFAULT_VARIANT)}.jsonl"
        train_path, test_path = (args.baseline_dir / split / "preds" / name for split in ("train", "test"))
        result = {"model": model, "train_preds": str(train_path), "test_preds": str(test_path),
                  "train_set": str(args.train_set), "test_set": str(args.test_set),
                  "n_train": len(train_labels), "n_test": len(test_labels),
                  **calibrate(load_jsonl(ROOT / train_path), train_labels, load_jsonl(ROOT / test_path), test_labels)}
        out = ROOT / args.out_dir / f"calibration-{model}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(format_report(result))
        print(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
