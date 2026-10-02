"""Follow-ups Task 5: notes.md for Qwen3.8-27B on the RTX 3090, every number read from a file of the output directory.

Usage: uv run results-engine/3090/notes.py [--preset P] [--dir DIR] [--seqs N] [--title TEXT]   (run.sh step `notes`)
  --preset  the preset of models-engine-3090.ini (default qwen3.8-27b)
  --dir     run.sh's OUT_DIR, which gets notes.md (default results-engine/3090; relative: from the project directory)
  --seqs    the dumps' --decide-seqs (default the preset's decide-seqs)
  --title   the model in the heading (default for qwen3.8-27b: "Qwen3.8-27B on the RTX 3090 (follow-ups Task 5)",
            else "<preset> on the RTX 3090")
"""

import argparse
import configparser
import gzip
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from calibrate import METRICS  # noqa: E402

P = "qwen3.8-27b"
FIELDS = ("queue", "urgency", "angry")
BASE = f"decide-{P}-b5-keywords"
D = HERE  # the output directory (--dir)
SEQS = None  # the dumps' --decide-seqs (--seqs)


def j(rel):
    return json.loads((D / rel).read_text())


def rows(rel):
    return [json.loads(line) for line in (D / rel).read_text().splitlines() if line.strip()]


def f4(x):
    return "—" if x is None else (f"{x:.4f}" if abs(x) >= 1e-4 or x == 0 else f"{x:.1e}")


def setup():
    run = j(f"baseline/test/runs/{BASE}.json")
    info, s = run["info"], run["preset_settings"]
    return ["## Setup", "",
            f"Engine `{run['fork_commit'][:9]}` (`engine/build` on the test machine). Preset `{P}` of `models-engine-3090.ini`: "
            f"`{run['model_file']}`, `ngl {s['ngl']}`, `fa {s['fa']}`, KV `{s['cache-type-k']}`/`{s['cache-type-v']}`, `ctx-size "
            f"{s['ctx-size']}`, `decide-seqs {run['decide_seqs']}` (`/v1/decide/info`: `decide_seqs {info['decide_seqs']}`, `n_ctx "
            f"{info['n_ctx']}`, `recurrent {str(info['memory']['recurrent']).lower()}`, server default temperature "
            f"{info['temperature']}). Data: `local/data/triage_test.jsonl` (80) and `triage_train.jsonl` (320). Every run except the preset "
            "check at T = 1 (`--ensure-t1`); the preset check runs at the preset's fitted T (Calibration below). Commands: `results-engine/3090/run.sh` (its header lists the environment).", ""]


def evaluation():
    return ["## Evaluation (`eval/summary.md`, written by `results-engine/sp3/eval/analysis.py summary`)", "",
            "Tree = this model's own baseline in `baseline/`; sign tests and Holm over this model's 12 tests only.", "",
            *(D / "eval" / "summary.md").read_text().rstrip("\n").splitlines(), ""]


def calibration():
    c = j(f"eval/calibration-{P}.json")
    lines = [f"## Calibration (`eval/calibration-{P}.json`, `calibrate.py`, spec 8)", "",
             f"Fitted on the train baseline: T = {c['t']} (grid best {c['grid']['best']:.4f}, refined {c['t_refined']:.6f}).", "",
             "| set | T | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for split in ("train", "test"):
        for t, m in ((1.0, c[split]["before"]), (c["t"], c[split]["after"])):
            lines.append(f"| {split} | {t} | " + " | ".join(f"{m[k]:.4f}" for k in METRICS) + " |")
    r = c["rule"]
    lines += ["", f"Rule: {r['text']}. Test summed NLL {r['test_nll_sum_before']:.4f} -> {r['test_nll_sum_after']:.4f}: "
              + ("apply." if r["apply"] else "do not apply (the preset keeps no decide-temperature)."), ""]
    ini = configparser.ConfigParser()
    ini.read(ROOT / "models-engine-3090.ini")
    held = ini[P].get("decide-temperature")
    lines += [f"On this machine the presets file is `models-engine-3090.ini`: `[{P}]` "
              + (f"holds `decide-temperature = {held}` (added after every run above)." if held else "holds no decide-temperature."), ""]
    preset = D / "eval" / "preset" / f"compare-{P}.json"
    if preset.exists():
        p = json.loads(preset.read_text())
        lines += [f"Preset check (`eval/preset/`): the live test run at the preset's T against the baseline test preds "
                  f"scaled post hoc: max abs diff {p['max_abs_diff']:.2e}, within 1e-9: {p['within_bound']}.", ""]
    return lines


def integrity():
    lines = ["## Integrity (`integrity/`)", "",
             f"`reference_check.py` on the first 20 test tickets, one state per line, `--decide-seqs {SEQS}`; dump with "
             "`llama-decide` (server stopped), replay through the router's `/completion` (n_probs 512; the replays are kept "
             "as `*-replay.jsonl.gz`, so `compare --offline` recomputes them). Statistic: per field median abs diff <= 0.01 "
             "and no argmax disagreement where the reference's top-2 margin > 0.15.", "",
             "| check | field | median abs diff | max abs diff | argmax agreement | disagreements over margin | pass |",
             "|---|---|---|---|---|---|---|"]
    for tag in ("tree", "fp"):
        r = j(f"integrity/reference-{tag}-{P}.json")
        for name in FIELDS:
            f = r["fields"][name]
            ok = f.get("statistic_pass", f["median_abs_diff"] <= 0.01 and f["disagreements_over_margin"] == 0)
            lines.append(f"| reference {tag} | {name} | {f4(f['median_abs_diff'])} | {f4(f['max_abs_diff'])} | "
                         f"{f['argmax_agreement']:.3g} | {f['disagreements_over_margin']} | {'PASS' if ok else 'FAIL'} |")
    b = j(f"integrity/batch-invariance-{P}.json")
    for name in FIELDS:
        f = b["fields"][name]
        ok = f["median_abs_diff"] <= 0.01 and f["disagreements_over_margin"] == 0
        lines.append(f"| batch b1 vs b5 (n {b['n']}) | {name} | {f4(f['median_abs_diff'])} | {f4(f['max_abs_diff'])} | "
                     f"{f['argmax_agreement']:.3g} | {f['disagreements_over_margin']} | {'PASS' if ok else 'FAIL'} |")
    t, fp = j(f"integrity/reference-tree-{P}.json"), j(f"integrity/reference-fp-{P}.json")
    u, gate = fp["unverifiable"], fp["coverage_gate"]
    lines += ["", f"Overall: reference tree {'PASS' if t['pass'] else 'FAIL'}; reference full-path {'PASS' if fp['pass'] else 'FAIL'} "
              "(statistic, coverage and node comparison per field, unverifiable edges); batch invariance "
              f"{'PASS' if b['pass'] else 'FAIL'}.", "",
              "Full-path detail:", "",
              "| field | engine coverage median | reference coverage median | coverage median abs diff | node median / max abs diff (values) | pass |",
              "|---|---|---|---|---|---|"]
    for name in FIELDS:
        f = fp["fields"][name]
        lines.append(f"| {name} | {f['coverage_engine_median']:.6f} | {f['coverage_reference_median']:.6f} | "
                     f"{f4(f['coverage_median_abs_diff'])} ({'PASS' if f['coverage_pass'] else 'FAIL'}) | "
                     f"{f4(f['node_median_abs_diff'])} / {f4(f['node_max_abs_diff'])} ({f['node_values']}) "
                     f"{'PASS' if f['node_pass'] else 'FAIL'} | {'PASS' if f['pass'] else 'FAIL'} |")
    lines += ["", f"Unverifiable edges: {u['unverifiable']} of {u['edges']} ({u['fraction']:.2%}; bound 5 %, each within 2x the "
              f"list's smallest probability: {u['within_bound']}). Coverage gate (median engine coverage >= {gate['threshold']}): "
              f"{'pass' if gate['pass'] else 'fail'}.", ""]
    counts = {}
    for tag in ("tree", "fp"):
        with gzip.open(D / "integrity" / f"reference-{tag}-{P}-replay.jsonl.gz", "rt") as f:
            counts[tag] = sum(json.loads(line).get("path") == "/completion" for line in f)
    lines += [f"Recorded `/completion` requests: tree {counts['tree']}, full-path {counts['fp']}. Offline recomputation "
              f"(`integrity/offline-recompute.txt`): {(D / 'integrity' / 'offline-recompute.txt').read_text().strip()}", ""]
    return lines


def latency():
    s = j("eval/summary.json")
    b5, b1 = j(f"baseline/test/runs/{BASE}.json"), j(f"latency/runs/decide-{P}-b1-keywords.json")
    lines = ["## Latency and cost", "",
             "Tree, test set (80 tickets), per ticket: wall time of a request / its tickets (p50, p90), server time "
             "(timings.total_ms / tickets); VRAM in use on the GPU after the run (desktop about 612 MiB included).", "",
             "| batch | run | p50 ms | p90 ms | server p50 ms | rounds per request | VRAM after run MiB |", "|---|---|---|---|---|---|---|"]
    for b, run, rel in ((1, b1, "latency/"), (5, b5, "baseline/test/")):
        lat, srv = run["latency_s_per_ticket"], run["server_ms_per_ticket"]
        lines.append(f"| {b} | `{rel}runs/decide-{P}-b{b}-keywords.json` | {lat['p50'] * 1000:.1f} | "
                     f"{lat['p90'] * 1000:.1f} | {srv['p50']:.1f} | {','.join(map(str, run['rounds']))} | {run['vram_mib_after_run']} |")
    tree = s["tree"][P]["train"]["cost"]
    lines += ["", f"Per mode on the train set (64 requests of 5 states, `--decide-seqs {b5['decide_seqs']}`; medians per request; from "
              "`eval/summary.json`):", "",
              "| mode | p50 ms / ticket | tickets / s | vs tree | total ms | decodes | rounds | seqs per state | states per round | VRAM after run MiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    from_summary = [("tree", tree)] + [(c, s["runs"][P][c]["train"]["cost"]) for c in s["runs"][P]]
    for mode, c in from_summary:
        lines.append(f"| {mode} | {c['p50_ms_per_ticket']:.1f} | {c['tickets_per_s']:.1f} | {c['tickets_per_s'] / tree['tickets_per_s']:.2f}x | "
                     f"{c['total_ms_median']:.1f} | {c['decodes_median']:g} | {c['rounds_median']:g} | {c['seqs_per_state']} | "
                     f"{c['states_per_round']} | {c['vram_mib_after_run']} |")
    return lines + [""]


def systemone():
    smoke = (D / "systemone" / "smoke.txt").read_text().rstrip("\n").splitlines()
    sdk = (D / "systemone" / "sdk.txt").read_text().rstrip("\n").splitlines()
    return ["## /v1/systemone", "",
            f"`systemone_smoke.py --model {P}` (`systemone/smoke.txt`), last line: `{smoke[-1]}`", "",
            f"Official SDK test, `SYSTEMONE_MODEL={P}` (`systemone/sdk.txt`), last line: `{sdk[-1]}`", ""]


def main():
    global P, BASE, D, SEQS
    ap = argparse.ArgumentParser(description="notes.md for one model of results-engine/3090/run.sh")
    ap.add_argument("--preset", default=P)
    ap.add_argument("--dir", type=Path, default=HERE)
    ap.add_argument("--seqs", type=int)
    ap.add_argument("--title")
    args = ap.parse_args()
    P, D = args.preset, ROOT / args.dir
    BASE = f"decide-{P}-b5-keywords"
    if args.seqs is None:
        ini = configparser.ConfigParser()
        ini.read(ROOT / "models-engine-3090.ini")
        args.seqs = int(ini[P].get("decide-seqs", ini["*"].get("decide-seqs")))
    SEQS = args.seqs
    title = args.title or ("Qwen3.8-27B on the RTX 3090 (follow-ups Task 5)" if P == "qwen3.8-27b" else f"{P} on the RTX 3090")
    lines = [f"# {title}", "",
             "Written by `notes.py` from the files of this directory. Measured on an RTX 3090; the sub-project 3 numbers in "
             "`REPORT-ENGINE.md` come from an RTX 3070 Ti laptop GPU.", ""]
    for part in (setup, evaluation, calibration, integrity, latency, systemone):
        lines += part()
    (D / "notes.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {D / 'notes.md'}")


if __name__ == "__main__":
    main()
