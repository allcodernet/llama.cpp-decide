"""Sub-project 3 Task 9: evaluation analysis for run.sh in this directory (spec 2026-09-25-engine-improvements-design.md 12.4).

  uv run results-engine/sp3/eval/analysis.py flags CONFIG
      The decide_client.py switches of an evaluation configuration (full-path, after-a, after-b, debias4).
  uv run results-engine/sp3/eval/analysis.py name MODEL [CONFIG] [--temperature T]
      The run's file name (decide_client.run_name) at the effective temperature T (default 1.0; the preset run's is
      the fitted T): the Task 0 baseline's without CONFIG.
  uv run results-engine/sp3/eval/analysis.py check RUN_JSON [--temperature T]
      The run-level and every request's engine block as the run's switches imply: scoring, order_debias (the effective
      K), phases and temperature (default 1.0: every evaluation run precedes the preset change); prefix_ok; the preds row
      count. Exit 1 on a mismatch.
  uv run results-engine/sp3/eval/analysis.py summary
      For the 16 runs: the checks above; the answer keys (coverage exactly in full-path runs, order_spread exactly in
      K >= 2 runs, given exactly in dependent fields and equal to the returned values of the field's ancestors, in
      declaration order); coverage and order_spread distributions per field over test + train (400 tickets); per-field
      agreement with the Task 0 baseline over the 400 tickets (sp3_metrics.compare, the baseline as reference) and the
      number of changed field decisions; cost per run (the regression runs of sp3/integrity/, the same tree requests on the same
      build, as the tree row); the scores (score.json) per set and over test + train pooled (sp3_metrics.score_preds on
      the 400 tickets); the pooled sign tests (signtest-*.json) with Holm-adjusted p over all of them. Writes
      summary.json and summary.md (the REPORT-ENGINE.md tables). The ticket and request counts and --decide-seqs in
      summary.md's sentences are read from the label sets and the runs.
  uv run results-engine/sp3/eval/analysis.py temper IN OUT --t T
      IN's preds rows scaled to temperature T post hoc (calibrate.temper_preds), written with queue, urgency and angry
      only (a `raw` copy would still hold the T = 1 answers).
  uv run results-engine/sp3/eval/analysis.py preset MODEL
      The preset step's decision from the eval directory's calibration-MODEL.json and the presets file
      (decide_client.presets_file): prints `check T` when the rule applies and the preset holds decide-temperature = T,
      `add T` (and, on stderr, the line to add) when the rule applies and it does not, `skip T` when the rule does not
      apply. It never edits a presets file.

Settings (run.sh): MODELS (space-separated presets, default qwen3.5-9b gemma-4-e4b); EVAL_DIR (the evaluation runs and
every output above, default results-engine/sp3/eval); BASELINE_DIR (the tree baseline runs, default
results-engine/sp3/baseline); COST_DIR (the tree runs of the cost table, default BASELINE_DIR when that is set, else
results-engine/sp3/integrity/regression); relative paths from the project directory. TRIAGE_DATA: score.py.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from calibrate import temper_preds  # noqa: E402
from decide_client import QUEUES, _preset_settings, parse_args, presets_file, run_name  # noqa: E402
from heldout_stats import TRAIN_SET  # noqa: E402
from score import TEST_SET, load_test  # noqa: E402
from sp3_metrics import FIELDS, compare, distributions, load_jsonl, score_preds  # noqa: E402

SPLITS = ["test", "train"]
CONFIGS = {
    "full-path": ["--scoring", "full_path"],
    "after-a": ["--after", "urgency=queue"],
    "after-b": ["--after", "urgency=queue", "--after", "angry=queue"],
    "debias4": ["--order-debias", "4"],
}
LABELS = {"tree": "tree (baseline)", "full-path": "full-path", "after-a": "after A (urgency after queue)",
          "after-b": "after B (urgency, angry after queue)", "debias4": "debias K = 4"}
DEFAULT_COST = "results-engine/sp3/integrity/regression"
TRAIN_OFFSET = 1000  # ticket ids of the train set when pooled with the test set
MAX_LISTED = 20
LABEL_SETS = {"test": TEST_SET, "train": TRAIN_SET}
POOLED_METRICS = ["queue", "urgency", "urgency_near", "angry", "nll_queue", "nll_urgency", "nll_angry", "nll_sum",
                  "ece_queue", "ece_urgency", "ece_angry", "brier_angry"]


def settings(env):
    """(MODELS, EVAL_DIR, BASELINE_DIR, COST_DIR) from env (module docstring); an empty value counts as unset."""
    baseline = env.get("BASELINE_DIR") or "results-engine/sp3/baseline"
    cost = env.get("COST_DIR") or env.get("BASELINE_DIR") or DEFAULT_COST
    return ((env.get("MODELS") or "qwen3.5-9b gemma-4-e4b").split(), ROOT / (env.get("EVAL_DIR") or "results-engine/sp3/eval"),
            ROOT / baseline, ROOT / cost)


MODELS, EVAL, BASELINE, COST = settings(os.environ)


def name(model, config=None, temperature=1.0):
    return run_name(parse_args(["--model", model, "--batch", "5", *(CONFIGS[config] if config else [])]),
                    temperature=temperature)


def run_files(base, split, run):
    return base / split / "preds" / f"{run}.jsonl", base / split / "runs" / f"{run}.json"


def ancestors(field, after):
    out = set()
    for parent in after.get(field, []):
        out |= {parent} | ancestors(parent, after)
    return out


def level(field, after):
    return 1 + max(level(p, after) for p in after[field]) if field in after else 0


def expected_engine(switches, temperature):
    k = switches["order_debias"]
    return {"scoring": switches["scoring"], "order_debias": 1 if k < 2 else min(k, len(QUEUES)),
            "phases": 1 + max(level(f, switches["after"]) for f in FIELDS), "temperature": temperature}


def check_run(run_path, temperature=1.0):
    run = json.loads(run_path.read_text())
    preds = load_jsonl(run_path.parent.parent / "preds" / f"{run_path.stem}.jsonl")
    want = expected_engine(run["switches"], temperature)
    problems = [] if run["prefix_ok"] else ["prefix_ok false"]
    if len(preds) != run["n_tickets"]:
        problems.append(f"{len(preds)} preds rows for n_tickets {run['n_tickets']}")
    for i, block in enumerate([run["engine"]] + [r["engine"] for r in run["requests"]]):
        got = {k: block[k] for k in want}
        if got != want:
            problems.append(f"engine block {i} (0 = run level): {got} != {want}")
    return run, preds, problems


def key_problems(preds, switches):
    after, spread = switches["after"], switches["order_debias"] >= 2
    full_path = switches["scoring"] == "full_path"
    anc = {f: [a for a in FIELDS if a in ancestors(f, after)] for f in FIELDS}
    problems = []
    for i, row in enumerate(preds):
        raw = row["raw"]
        for f in FIELDS:
            answer = raw[f]
            if ("coverage" in answer) != full_path:
                problems.append(f"row {i} {f}: coverage {'missing' if full_path else 'present'}")
            if ("order_spread" in answer) != spread:
                problems.append(f"row {i} {f}: order_spread {'missing' if spread else 'present'}")
            want = [(a, raw[a]["value"]) for a in anc[f]]
            if anc[f] and list(answer.get("given", {}).items()) != want:
                problems.append(f"row {i} {f}: given {answer.get('given')} != {dict(want)}")
            if not anc[f] and "given" in answer:
                problems.append(f"row {i} {f}: given present without after")
    return problems


def quantiles(xs):
    q = statistics.quantiles(xs, n=20, method="inclusive")
    return {"n": len(xs), "min": min(xs), "p5": q[0], "p25": q[4], "median": statistics.median(xs), "p75": q[14],
            "p95": q[18], "max": max(xs), "mean": statistics.mean(xs)}


def cost(run, preds):
    reqs = run["requests"]

    def median(section, key):
        return statistics.median(r[section][key] for r in reqs)

    wall = sum(r["wall_s"] for r in reqs)
    cache = run["info"]["prefix_cache"]
    return {
        "requests": len(reqs), "p50_ms_per_ticket": statistics.median(p["latency_s"] for p in preds) * 1000,
        "wall_s": wall, "tickets_per_s": run["n_tickets"] / wall,
        **{f"{k}_median": median("timings", k) for k in ("total_ms", "prefill_ms", "prefix_ms", "scoring_ms", "decodes",
                                                          "rounds")},
        "retries": sum(r["timings"]["retries"] for r in reqs),
        **{f"{k}_median": median("usage", k) for k in ("prompt_tokens", "cached_tokens", "scored_tokens")},
        "cache_hits": sum(r["engine"]["prefix_cache"]["hits"] for r in reqs),
        "cache_misses": sum(r["engine"]["prefix_cache"]["misses"] for r in reqs),
        "cache_after_warmup": cache,
        "seqs_per_state": run["engine"]["seqs_per_state"], "states_per_round": run["engine"]["states_per_round"],
        "vram_mib_after_run": run["vram_mib_after_run"],
    }


def pooled_entries(paths):
    return [(t + offset, f, p) for (path, offset) in paths for t, f, p in distributions(path)[1]]


def pooled_score(base, run):
    """score_preds over test + train (400 tickets) of the run's preds files under `base`."""
    preds, labels = [], []
    for split in SPLITS:
        preds += load_jsonl(run_files(base, split, run)[0])
        labels += load_test(LABEL_SETS[split])
    m = score_preds(preds, labels)
    return {k: m[k] for k in POOLED_METRICS}


def holm(ps):
    """Holm step-down adjusted p-values, in the order of ps."""
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def summary():
    """The summary (summary.json) and the facts summary.md's sentences state: tickets per set and, over the train runs
    of the cost table, the request counts, batches and --decide-seqs."""
    out = {"runs": {}, "tree": {}, "pooled": {}, "distributions": {}, "agreement": {}, "signtests": {}, "problems": 0}
    facts = {"n": {split: len(load_test(LABEL_SETS[split])) for split in SPLITS}, "train_runs": []}
    for model in MODELS:
        base = name(model)
        out["tree"][model] = {}
        out["pooled"][model] = {"tree": pooled_score(BASELINE, base)}
        for split in SPLITS:
            preds_path, _ = run_files(BASELINE, split, base)
            reg_preds, reg_run = run_files(COST, split, base)
            score = json.loads((preds_path.parent / "score.json").read_text())["runs"][base]
            reg = json.loads(reg_run.read_text())
            out["tree"][model][split] = {"score": score, "cost": cost(reg, load_jsonl(reg_preds))}
            if split == "train":
                facts["train_runs"].append(reg)
        out["runs"][model], out["distributions"][model], out["agreement"][model], out["signtests"][model] = {}, {}, {}, {}
        coverage, spread = {f: [] for f in FIELDS}, {f: [] for f in FIELDS}
        for config in CONFIGS:
            run_name_ = name(model, config)
            out["runs"][model][config] = {"name": run_name_}
            for split in SPLITS:
                preds_path, run_path = run_files(EVAL, split, run_name_)
                run, preds, problems = check_run(run_path)
                if split == "train":
                    facts["train_runs"].append(run)
                problems += key_problems(preds, run["switches"])
                out["problems"] += len(problems)
                score = json.loads((preds_path.parent / "score.json").read_text())["runs"][run_name_]
                out["runs"][model][config][split] = {"problems": len(problems), "listed": problems[:MAX_LISTED],
                                                     "score": score, "cost": cost(run, preds)}
                for row in preds:
                    for f in FIELDS:
                        if "coverage" in row["raw"][f]:
                            coverage[f].append(row["raw"][f]["coverage"])
                        if "order_spread" in row["raw"][f]:
                            spread[f].append(row["raw"][f]["order_spread"])
            offsets = [(0, "test"), (TRAIN_OFFSET, "train")]
            base_entries = pooled_entries([(run_files(BASELINE, s, base)[0], o) for o, s in offsets])
            run_entries = pooled_entries([(run_files(EVAL, s, run_name_)[0], o) for o, s in offsets])
            agreement = compare(base_entries, run_entries)
            agreement["decisions"] = agreement["n_tickets"] * len(FIELDS)
            agreement["decisions_changed"] = sum(x["argmax_disagreements"] for x in agreement["fields"].values())
            agreement["decisions_changed_over_margin"] = sum(x["disagreements_over_margin"] for x in agreement["fields"].values())
            out["agreement"][model][config] = agreement
            out["pooled"][model][config] = pooled_score(EVAL, run_name_)
            sign = json.loads((EVAL / f"signtest-{run_name_.removeprefix('decide-')}.json").read_text())
            out["signtests"][model][config] = sign["pooled"]
        out["distributions"][model] = {
            "coverage (full-path)": {f: quantiles(v) for f, v in coverage.items() if v},
            "order_spread (debias K = 4)": {f: quantiles(v) for f, v in spread.items() if v},
        }
    tests = [t for m in MODELS for c in CONFIGS for t in (out["signtests"][m][c]["fields"][f] for f in FIELDS)]
    for t, adjusted in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = adjusted
    out["holm_tests"] = len(tests)
    return out, facts


def cost_source(cost_dir):
    """What the cost table's tree row is, for summary.md."""
    if cost_dir == ROOT / DEFAULT_COST:
        return "the regression run of sp3/integrity/, the same requests on this build"
    return f"the tree runs in `{cost_dir.relative_to(ROOT) if cost_dir.is_relative_to(ROOT) else cost_dir}`"


def markdown(s, facts):
    """The REPORT-ENGINE.md tables: both models per table, the tree row first."""
    rows = [(m, c) for m in MODELS for c in ["tree", *CONFIGS]]

    def get(model, config, split=None):
        entry = s["tree"][model] if config == "tree" else s["runs"][model][config]
        return entry[split] if split else entry

    def values(key):  # the train runs' values of key, "/"-joined when they differ
        return "/".join(sorted({str(r[key]) for r in facts["train_runs"]}))

    n = facts["n"]
    pooled = n["test"] + n["train"]
    lines = []
    for split in SPLITS:
        lines += [f"Scores, {split} set ({n[split]} tickets, b5, prompt `keywords`, T = 1):", "",
                  "| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for model, config in rows:
            m = get(model, config, split)["score"]
            rounds = ",".join(map(str, m["rounds"])) if m.get("rounds") else "—"
            lines.append(f"| {model} | {LABELS[config]} | {m['queue']:.1%} | {m['urgency']:.1%} | {m['urgency_near']:.1%} | "
                         f"{m['angry']:.1%} | {m['nll_queue']:.3f} | {m['nll_urgency']:.3f} | {m['nll_angry']:.3f} | {m['nll_sum']:.3f} | "
                         f"{m['ece_queue']:.3f} | {m['ece_urgency']:.3f} | {m['ece_angry']:.3f} | {m['brier_angry']:.3f} | "
                         f"{m['p50_ms']:.1f} | {rounds} |")
        lines.append("")
    lines += [f"Scores, test and train pooled ({pooled} tickets; `sp3_metrics.score_preds` on the {pooled} rows, so NLL is the mean "
              f"over the {pooled} tickets and ECE is binned over them):", "",
              "| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, config in rows:
        m = s["pooled"][model][config]
        lines.append(f"| {model} | {LABELS[config]} | {m['queue']:.1%} | {m['urgency']:.1%} | {m['urgency_near']:.1%} | "
                     f"{m['angry']:.1%} | {m['nll_queue']:.4f} | {m['nll_urgency']:.4f} | {m['nll_angry']:.4f} | "
                     f"{m['nll_sum']:.4f} | {m['ece_queue']:.3f} | {m['ece_urgency']:.3f} | {m['ece_angry']:.3f} | "
                     f"{m['brier_angry']:.3f} |")
    lines += ["", f"Exact sign tests against tree on per-ticket correctness, test and train pooled ({pooled} tickets). Cells: tickets right "
              "tree / configuration (discordant pairs: only tree right - only the configuration right, two-sided p; Holm = "
              f"Holm-adjusted over all {s['holm_tests']} tests of this table):", "",
              "| model | configuration | queue | urgency | angry |", "|---|---|---|---|---|"]
    for model, config in rows:
        if config != "tree":
            fields = s["signtests"][model][config]["fields"]
            lines.append(f"| {model} | {LABELS[config]} | " + " | ".join(
                f"{t['a_right']} / {t['b_right']} ({t['a_only_right']}-{t['b_only_right']}, p {t['p']:.2g}, Holm {t['p_holm']:.2g})"
                for t in (fields[f] for f in FIELDS)) + " |")
    lines += ["", f"Agreement with tree over the {pooled} tickets (`sp3_metrics.compare`, tree as the reference). Cells: median / max abs "
              "diff over options, argmax changes (of them where tree's top-2 margin > 0.15); decisions changed = all fields:", "",
              "| model | configuration | queue | urgency | angry | decisions changed |", "|---|---|---|---|---|---|"]
    for model, config in rows:
        if config != "tree":
            a = s["agreement"][model][config]
            lines.append(f"| {model} | {LABELS[config]} | " + " | ".join(
                f"{x['median_abs_diff']:.2g} / {x['max_abs_diff']:.2g}, {x['argmax_disagreements']} ({x['disagreements_over_margin']})"
                for x in (a["fields"][f] for f in FIELDS))
                + f" | {a['decisions_changed']} of {a['decisions']} ({a['decisions_changed_over_margin']}) |")
    for title in ("coverage (full-path)", "order_spread (debias K = 4)"):
        lines += ["", f"Distribution of {title} per field, test and train pooled ({pooled} tickets):", "",
                  "| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |", "|---|---|---|---|---|---|---|---|---|---|"]
        for model in MODELS:
            for f, q in s["distributions"][model][title].items():
                fmt = ".6f" if title.startswith("coverage") else ".4g"  # coverage sits just below 1
                lines.append(f"| {model} | {f} | " + " | ".join(
                    format(q[k], fmt) for k in ("min", "p5", "p25", "median", "p75", "p95", "max", "mean")) + " |")
    requests = "/".join(sorted({str(len(r["requests"])) for r in facts["train_runs"]}))
    lines += ["", f"Cost on the train set ({requests} requests of {values('batch')} states, `--decide-seqs {values('decide_seqs')}`; "
              f"tree = {cost_source(COST)}). Per request medians in ms; snapshots = prefix cache entries and MB after the warm-up:", "",
              "| model | configuration | p50 ms / ticket | tickets / s | vs tree | total | prefill | prefix | scoring | decodes | rounds | "
              "seqs per state | states per round | cache hits / misses | snapshots |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, config in rows:
        c, tree = get(model, config, "train")["cost"], s["tree"][model]["train"]["cost"]
        snap = c["cache_after_warmup"]
        lines.append(f"| {model} | {LABELS[config]} | {c['p50_ms_per_ticket']:.1f} | {c['tickets_per_s']:.1f} | "
                     f"{c['tickets_per_s'] / tree['tickets_per_s']:.2f}x | {c['total_ms_median']:.1f} | {c['prefill_ms_median']:.1f} | "
                     f"{c['prefix_ms_median']:.1f} | {c['scoring_ms_median']:.1f} | {c['decodes_median']:g} | {c['rounds_median']:g} | "
                     f"{c['seqs_per_state']} | {c['states_per_round']} | {c['cache_hits']} / {c['cache_misses']} | "
                     f"{snap['entries']}, {snap['bytes'] / 1e6:.1f} |")
    lines += ["", f"Checks (engine blocks, answer keys, given): {s['problems']} problems."]
    return "\n".join(lines) + "\n"


def preset_action(calibration, settings):
    """The preset step for one model (module docstring: preset): ("check" | "add" | "skip", the fitted T)."""
    t = calibration["t"]
    if not calibration["rule"]["apply"]:
        return "skip", t
    held = settings.get("decide-temperature")
    return ("check" if held is not None and float(held) == t else "add"), t


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sub-project 3 Task 9 evaluation analysis.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("flags").add_argument("config", choices=list(CONFIGS))
    n = sub.add_parser("name")
    n.add_argument("model")
    n.add_argument("config", nargs="?", choices=list(CONFIGS))
    n.add_argument("--temperature", type=float, default=1.0)
    c = sub.add_parser("check")
    c.add_argument("run", type=Path)
    c.add_argument("--temperature", type=float, default=1.0)
    sub.add_parser("summary")
    t = sub.add_parser("temper")
    t.add_argument("src", type=Path)
    t.add_argument("dst", type=Path)
    t.add_argument("--t", type=float, required=True)
    sub.add_parser("preset").add_argument("model")
    args = ap.parse_args(argv)

    if args.cmd == "flags":
        print(" ".join(CONFIGS[args.config]))
    elif args.cmd == "name":
        print(name(args.model, args.config, args.temperature))
    elif args.cmd == "check":
        run, _, problems = check_run(args.run, args.temperature)
        print(f"{args.run.name}: engine {expected_engine(run['switches'], args.temperature)}: "
              + ("ok" if not problems else f"{len(problems)} problems"), file=sys.stderr)
        for p in problems[:MAX_LISTED]:
            print("  " + p, file=sys.stderr)
        return 1 if problems else 0
    elif args.cmd == "summary":
        s, facts = summary()
        (EVAL / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
        (EVAL / "summary.md").write_text(markdown(s, facts))
        print(markdown(s, facts), end="")
        print(f"wrote {EVAL / 'summary.json'} and {EVAL / 'summary.md'}")
        return 1 if s["problems"] else 0
    elif args.cmd == "temper":
        rows = [{k: r[k] for k in FIELDS} for r in temper_preds(load_jsonl(args.src), args.t)]
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        args.dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {args.dst} ({len(rows)} rows at T = {args.t})")
    elif args.cmd == "preset":
        cal = json.loads((EVAL / f"calibration-{args.model}.json").read_text())
        settings = _preset_settings(args.model)
        action, t = preset_action(cal, settings)
        if action == "add":
            held = settings.get("decide-temperature")
            print(f"{args.model}: the rule applies; add `decide-temperature = {t}` to [{args.model}] of {presets_file()}"
                  + (f" (it holds {held})" if held is not None else "") + ", then run the preset step again", file=sys.stderr)
        elif action == "skip":
            held = settings.get("decide-temperature")
            print(f"{args.model}: the rule keeps the preset without decide-temperature"
                  + (f" (but [{args.model}] of {presets_file()} holds {held}: remove it)" if held is not None else "")
                  + "; no preset check", file=sys.stderr)
        print(action, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
