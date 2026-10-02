"""Follow-up Task 1 live checks: the 63ea2c51a build vs the new build (faster full-vocabulary log-sum-exp), on the test machine.

  uv run results-engine/followups/task-1/check.py run PRESET [--reps 5]
      With the flags of PRESET in models-engine-3090.ini (-c, -ngl, -fa, -ctk, -ctv) and --decide-seqs 64:
      1. full-path b5 (requests-fp-b5-PRESET.jsonl): REPS repetitions of old and new, alternating; answers new vs old
         max abs diff over every numeric value of `results` (bound 1e-6) and scoring time per request line, median of
         the repetitions, old vs new;
      2. tree b5 with --dump-tokens (requests-tree-b5-PRESET.jsonl): one run per build; `results` max abs diff (bound 0)
         and the `tokens` dumps identical.
      Writes fp-b5-{old,new}-PRESET.jsonl (first repetition), tree-b5-dump-{old,new}-PRESET.jsonl and PRESET.json next
      to this script. Waits up to 20 minutes for a free GPU (no llama-server, at most 1500 MiB in use), else exits 2.
"""

import argparse
import configparser
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "engine-63ea2c51a" / "build" / "bin" / "llama-decide"
NEW = ROOT / "engine" / "build" / "bin" / "llama-decide"
PRESETS = ROOT / "models-engine-3090.ini"
VRAM_MAX_MIB = 1500
WAIT_MIN = 20
ANSWER_BOUND = 1e-6


def settings(preset):
    ini = configparser.ConfigParser()
    ini.read(PRESETS)
    s = dict(ini.items("*"))
    s.update(ini.items(preset))
    return s


def command(binary, s, dump):
    cmd = [str(binary), "-m", s["model"], "--decide-seqs", "64", "-c", s["ctx-size"], "-ngl", s["ngl"], "-fa", s["fa"],
           "-ctk", s["cache-type-k"], "-ctv", s["cache-type-v"]]
    return cmd + ["--dump-tokens"] if dump else cmd


def gpu_free():
    servers = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split()
    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.split()[0])
    return not servers and used <= VRAM_MAX_MIB, f"llama-server pids {servers or 'none'}, {used} MiB in use"


def wait_gpu():
    for minute in range(WAIT_MIN + 1):
        ok, why = gpu_free()
        if ok:
            return
        print(f"GPU busy ({why}), minute {minute}", file=sys.stderr)
        if minute < WAIT_MIN:
            time.sleep(60)
    sys.exit(f"BLOCKED: GPU not free after {WAIT_MIN} minutes")


def run(cmd, requests):
    wait_gpu()
    proc = subprocess.run(cmd, input=requests.read_text(), capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]


def max_abs_diff(a, b, where="results"):
    """Max abs diff over the numeric leaves of a and b; the structure and every non-numeric leaf must be equal."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        if a != b:
            raise ValueError(f"{where}: {a!r} != {b!r}")
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    if isinstance(a, dict) and isinstance(b, dict) and list(a) == list(b):
        return max((max_abs_diff(a[k], b[k], f"{where}.{k}") for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max((max_abs_diff(x, y, f"{where}[{i}]") for i, (x, y) in enumerate(zip(a, b))), default=0.0)
    raise ValueError(f"{where}: structure differs")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def cmd_run(args):
    s = settings(args.preset)
    fp_req = HERE / f"requests-fp-b5-{args.preset}.jsonl"
    tree_req = HERE / f"requests-tree-b5-{args.preset}.jsonl"

    # 1. full-path b5, old and new alternating
    reps = {"old": [], "new": []}
    for rep in range(args.reps):
        for build, binary in (("old", OLD), ("new", NEW)):
            reps[build].append(run(command(binary, s, dump=False), fp_req))
            print(f"{args.preset} full-path rep {rep + 1} {build}: scoring_ms "
                  f"{[r['timings']['scoring_ms'] for r in reps[build][-1]]}", file=sys.stderr)
    for build in reps:
        write_jsonl(HERE / f"fp-b5-{build}-{args.preset}.jsonl", reps[build][0])
    answers_diff = max(max_abs_diff([r["results"] for r in o], [r["results"] for r in n])
                       for o in reps["old"] for n in reps["new"])
    lines = []
    for i in range(len(reps["old"][0])):
        med = {b: statistics.median(rr[i]["timings"]["scoring_ms"] for rr in reps[b]) for b in reps}
        total = {b: statistics.median(rr[i]["timings"]["total_ms"] for rr in reps[b]) for b in reps}
        lines.append({
            "line": i,
            "scored_tokens": reps["new"][0][i]["usage"]["scored_tokens"],
            "scoring_ms_old": med["old"], "scoring_ms_new": med["new"],
            "scoring_saved_ms": med["old"] - med["new"], "scoring_speedup": med["old"] / med["new"],
            "total_ms_old": total["old"], "total_ms_new": total["new"],
            "scoring_ms_all_old": [rr[i]["timings"]["scoring_ms"] for rr in reps["old"]],
            "scoring_ms_all_new": [rr[i]["timings"]["scoring_ms"] for rr in reps["new"]],
        })

    # 2. tree b5 with --dump-tokens, one run per build
    tree = {b: run(command(binary, s, dump=True), tree_req) for b, binary in (("old", OLD), ("new", NEW))}
    for build in tree:
        write_jsonl(HERE / f"tree-b5-dump-{build}-{args.preset}.jsonl", tree[build])
    tree_diff = max_abs_diff([r["results"] for r in tree["old"]], [r["results"] for r in tree["new"]])
    tokens_equal = [r["tokens"] for r in tree["old"]] == [r["tokens"] for r in tree["new"]]

    out = {
        "preset": args.preset,
        "old": str(OLD.relative_to(ROOT)), "new": str(NEW.relative_to(ROOT)),
        "command": command(NEW, s, dump=False)[1:],
        "reps": args.reps,
        "full_path": {
            "requests": fp_req.name,
            "answers_max_abs_diff": answers_diff,
            "answers_pass": answers_diff <= ANSWER_BOUND,
            "scoring_ms_old_sum_of_medians": sum(l["scoring_ms_old"] for l in lines),
            "scoring_ms_new_sum_of_medians": sum(l["scoring_ms_new"] for l in lines),
            "lines": lines,
        },
        "tree": {
            "requests": tree_req.name,
            "results_max_abs_diff": tree_diff,
            "tokens_dump_identical": tokens_equal,
            "pass": tree_diff == 0.0 and tokens_equal,
        },
    }
    (HERE / f"{args.preset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"preset": args.preset, "answers_max_abs_diff": answers_diff,
                      "scoring_ms_old_new": [(l["scoring_ms_old"], l["scoring_ms_new"]) for l in lines],
                      "tree_max_abs_diff": tree_diff, "tree_tokens_identical": tokens_equal}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("preset")
    r.add_argument("--reps", type=int, default=5)
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
