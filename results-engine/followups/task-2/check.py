"""Follow-up Task 2 live checks: the 63ea2c51a build ("old") vs the build with Task 2 ("new"), and the new fault modes.

  uv run results-engine/followups/task-2/check.py requests PRESET...            (laptop: needs the triage test set)
      writes next to this script, per preset: aba-requests-P.jsonl (keywords, default, keywords), lru-abca-requests-P.jsonl
      (keywords, default, framing, keywords), both 5 states per line as sub-project 3 Task 4; requests-tree-b5-P.jsonl
      (20 tickets, 5 per line, tree); fault-requests-P.jsonl: R0 (plain, 5 states), R1-after (R0 + urgency after queue),
      R1-debias (R0 + order_debias 2).
  uv run results-engine/followups/task-2/check.py sequences PRESET --seqs N
      CLI, old and new: A/B/A at --decide-prefix-cache 8 and A,B,C,A at 2: per line (hits, misses, entries) old vs new
      (pass: equal) and answers max abs diff; tree b5 with --dump-tokens: results max abs diff and tokens dump (pass: 0, equal).
      Writes seq-P.json and the responses (seq-{aba,abca,tree}-{old,new}-P.jsonl).
  uv run results-engine/followups/task-2/check.py info PRESET --seqs N
      server (-np 1), old and new: POST A, B, A, then GET /v1/decide/info; pass: 3 x 200 and the info bodies equal.
      Writes info-P.json.
  uv run results-engine/followups/task-2/check.py fault MODE                   (Qwen2.5-0.5B, --decide-seqs 64)
      MODE cell-estimate / lines-join (R1 = R1-after) or tail-mismatch (R1 = R1-debias). Server F (LLAMA_DECIDE_FAULT=MODE,
      new build) and server N (no fault, new build) each get R0, R1, GET info, R2 (= R0's body). Pass: on F R0 200, R1 500
      `engine` whose message starts with the path's text, info lists slot 0 only and valid, R2 200; on N all 200; R2 F vs
      N max abs diff 0 (else reported with the statistic). Writes fault-MODE.json.
Server and CLI flags come from models-engine-3090.ini (-c, -ngl, -fa, -ctk, -ctv; the server also --jinja,
--reasoning off, -np 1), on 127.0.0.1:8097. Before every model run: no llama-server and at most 1500 MiB VRAM in use,
else wait up to 20 minutes and exit 2 (BLOCKED). Servers are stopped by PID.
"""

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

BUILDS = {"old": ROOT / "engine-63ea2c51a" / "build" / "bin", "new": ROOT / "engine" / "build" / "bin"}
PRESETS = ROOT / "models-engine-3090.ini"
PORT = 8097
VRAM_MAX_MIB = 1500
WAIT_MIN = 20
FAULT_TEXT = {
    "cell-estimate": "cell estimate exceeded",
    "lines-join": "answer lines change the tokenisation",
    "tail-mismatch": "catalogue variant 1 renders another chat template tail",
}


def settings(preset):
    ini = configparser.ConfigParser()
    ini.read(PRESETS)
    s = dict(ini.items("*"))
    s.update(ini.items(preset))
    return s


def model_flags(s):
    return ["-m", s["model"], "-c", s["ctx-size"], "-ngl", s["ngl"], "-fa", s["fa"],
            "-ctk", s["cache-type-k"], "-ctv", s["cache-type-v"]]


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


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


# ---- requests (laptop)

def cmd_requests(args):
    from decide_client import TEST_SET, build_request

    texts = [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()]
    for p in args.presets:
        write_jsonl(HERE / f"aba-requests-{p}.jsonl", [build_request(p, texts[:5], v) for v in ("keywords", "default", "keywords")])
        write_jsonl(HERE / f"lru-abca-requests-{p}.jsonl",
                    [build_request(p, texts[:5], v) for v in ("keywords", "default", "framing", "keywords")])
        write_jsonl(HERE / f"requests-tree-b5-{p}.jsonl", [build_request(p, texts[i : i + 5]) for i in range(0, 20, 5)])
        write_jsonl(HERE / f"fault-requests-{p}.jsonl", [
            build_request(p, texts[:5]),
            build_request(p, texts[:5], after={"urgency": ["queue"]}),
            build_request(p, texts[:5], order_debias=2),
        ])
        print(f"wrote the request files of {p}")


# ---- CLI sequences

def run_cli(build, s, seqs, extra, requests):
    wait_gpu()
    cmd = [str(BUILDS[build] / "llama-decide"), *model_flags(s), "--decide-seqs", str(seqs), *extra]
    proc = subprocess.run(cmd, input=Path(requests).read_text(), capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]


def cache_counts(resps):
    return [[r["engine"]["prefix_cache"][k] for k in ("hits", "misses", "entries")] for r in resps]


def cmd_sequences(args):
    s = settings(args.preset)
    out = {"preset": args.preset, "decide_seqs": args.seqs, "flags": model_flags(s)}
    ok = True
    for name, req, cache in (("aba", f"aba-requests-{args.preset}.jsonl", 8), ("abca", f"lru-abca-requests-{args.preset}.jsonl", 2)):
        r = {b: run_cli(b, s, args.seqs, ["--decide-prefix-cache", str(cache)], HERE / req) for b in BUILDS}
        for b in r:
            write_jsonl(HERE / f"seq-{name}-{b}-{args.preset}.jsonl", r[b])
        counts = {b: cache_counts(r[b]) for b in r}
        diff = max_abs_diff([x["results"] for x in r["old"]], [x["results"] for x in r["new"]])
        same = counts["old"] == counts["new"]
        ok = ok and same
        out[name] = {"requests": req, "prefix_cache": cache, "hits_misses_entries_old": counts["old"],
                     "hits_misses_entries_new": counts["new"], "counts_equal": same, "answers_max_abs_diff": diff,
                     "cached_tokens_old": [x["usage"]["cached_tokens"] for x in r["old"]],
                     "cached_tokens_new": [x["usage"]["cached_tokens"] for x in r["new"]]}
    req = f"requests-tree-b5-{args.preset}.jsonl"
    r = {b: run_cli(b, s, args.seqs, ["--dump-tokens"], HERE / req) for b in BUILDS}
    for b in r:
        write_jsonl(HERE / f"seq-tree-{b}-{args.preset}.jsonl", r[b])
    tdiff = max_abs_diff([x["results"] for x in r["old"]], [x["results"] for x in r["new"]])
    tok_eq = [x["tokens"] for x in r["old"]] == [x["tokens"] for x in r["new"]]
    eng_eq = [x["engine"] for x in r["old"]] == [x["engine"] for x in r["new"]]
    rounds_eq = [x["timings"]["rounds"] for x in r["old"]] == [x["timings"]["rounds"] for x in r["new"]]
    out["tree"] = {"requests": req, "results_max_abs_diff": tdiff, "tokens_dump_identical": tok_eq,
                   "engine_identical": eng_eq, "rounds_identical": rounds_eq}
    out["pass"] = ok and tdiff == 0.0 and tok_eq and eng_eq and rounds_eq
    (HERE / f"seq-{args.preset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("preset", "pass")} | {n: (out[n]["hits_misses_entries_old"], out[n]["hits_misses_entries_new"],
                                                                  out[n]["answers_max_abs_diff"]) for n in ("aba", "abca")}
                     | {"tree": out["tree"]}))


# ---- servers

class Server:
    def __init__(self, build, s, seqs, fault=None):
        wait_gpu()
        env = dict(os.environ)
        env.pop("LLAMA_DECIDE_FAULT", None)
        if fault:
            env["LLAMA_DECIDE_FAULT"] = fault
        self.cmd = [str(BUILDS[build] / "llama-server"), *model_flags(s), "--jinja", "--reasoning", "off", "-np", "1",
                    "--decide-seqs", str(seqs), "--host", "127.0.0.1", "--port", str(PORT)]
        self.log = Path(f"/tmp/t2-server-{build}-{fault or 'none'}.log")
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, env=env, stdout=self.log.open("w"), stderr=subprocess.STDOUT)
        import httpx

        self.client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=600)
        for _ in range(240):
            if self.proc.poll() is not None:
                sys.exit(f"server exited ({self.proc.returncode}); log {self.log}")
            try:
                if self.client.get("/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        sys.exit(f"server not ready after 120 s; log {self.log}")

    def post(self, body):
        r = self.client.post("/v1/decide", json=body)
        return {"status": r.status_code, "body": r.json()}

    def info(self):
        r = self.client.get("/v1/decide/info")
        return {"status": r.status_code, "body": r.json()}

    def stop(self):
        self.client.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def cmd_info(args):
    s = settings(args.preset)
    bodies = load(HERE / f"aba-requests-{args.preset}.jsonl")
    out = {"preset": args.preset, "decide_seqs": args.seqs}
    for b in BUILDS:
        srv = Server(b, s, args.seqs)
        try:
            posts = [srv.post(x) for x in bodies]
            info = srv.info()
        finally:
            srv.stop()
        out[b] = {"command": srv.cmd[1:], "statuses": [p["status"] for p in posts],
                  "prefix_cache": [p["body"].get("engine", {}).get("prefix_cache") for p in posts], "info": info}
    out["info_equal"] = out["old"]["info"] == out["new"]["info"]
    out["pass"] = out["info_equal"] and all(st == 200 for b in BUILDS for st in out[b]["statuses"]) and out["new"]["info"]["status"] == 200
    (HERE / f"info-{args.preset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"preset": args.preset, "pass": out["pass"], "info_equal": out["info_equal"],
                      "prefixes_new": out["new"]["info"]["body"].get("prefixes"), "prefix_cache_new": out["new"]["info"]["body"].get("prefix_cache")}))


def cmd_fault(args):
    preset = "qwen2.5-0.5b"
    s = settings(preset)
    r0, r1_after, r1_debias = load(HERE / f"fault-requests-{preset}.jsonl")
    r1 = r1_debias if args.mode == "tail-mismatch" else r1_after
    runs = {}
    for name, fault in (("F", args.mode), ("N", None)):
        srv = Server("new", s, 64, fault)
        try:
            runs[name] = {"command": srv.cmd[1:], "fault": fault, "R0": srv.post(r0), "R1": srv.post(r1), "info": srv.info(), "R2": srv.post(r0)}
        finally:
            srv.stop()
    from sp3_metrics import compare

    def entries(resp):
        out = []
        for i, res in enumerate(resp["results"]):
            a = res["answers"]
            out.append((i, "queue", dict(a["queue"]["probabilities"])))
            out.append((i, "urgency", {str(k): p for k, p in enumerate(a["urgency"]["probabilities"])}))
            out.append((i, "angry", {"true": a["angry"]["p_true"], "false": 1.0 - a["angry"]["p_true"]}))
        return out

    F, N = runs["F"], runs["N"]
    err = F["R1"]["body"].get("error", {})
    prefixes = F["info"]["body"].get("prefixes", [])
    checks = {
        "F_R0_200": F["R0"]["status"] == 200,
        "F_R1_500_engine": F["R1"]["status"] == 500 and err.get("code") == "engine",
        "F_R1_message": str(err.get("message", "")).startswith(FAULT_TEXT[args.mode]),
        "F_info_slot0_only_valid": F["info"]["status"] == 200 and [p["slot"] for p in prefixes] == [0] and prefixes[0]["valid"],
        "F_R2_200": F["R2"]["status"] == 200,
        "N_all_200": all(N[k]["status"] == 200 for k in ("R0", "R1", "R2")),
    }
    r2 = {}
    if checks["F_R2_200"] and checks["N_all_200"]:
        r2["max_abs_diff"] = max_abs_diff(F["R2"]["body"]["results"], N["R2"]["body"]["results"])
        if r2["max_abs_diff"] != 0.0:
            r2["statistic"] = compare(entries(N["R2"]["body"]), entries(F["R2"]["body"]))
        r2["prefix_cache_F"] = F["R2"]["body"]["engine"]["prefix_cache"]
        r2["prefix_cache_N"] = N["R2"]["body"]["engine"]["prefix_cache"]
    out = {"mode": args.mode, "preset": preset, "R1": "R1-debias (order_debias 2)" if args.mode == "tail-mismatch" else "R1-after (urgency after queue)",
           "checks": checks, "R2_F_vs_N": r2, "runs": runs, "pass": all(checks.values()) and r2.get("max_abs_diff") is not None}
    (HERE / f"fault-{args.mode}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "checks": checks, "F_R1": [F["R1"]["status"], err], "F_info_prefixes": prefixes,
                      "R2_F_vs_N": {k: v for k, v in r2.items() if k != "statistic"},
                      "N_R1_status": N["R1"]["status"], "pass": out["pass"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requests")
    r.add_argument("presets", nargs="+")
    r.set_defaults(func=cmd_requests)
    for name, fn in (("sequences", cmd_sequences), ("info", cmd_info)):
        p = sub.add_parser(name)
        p.add_argument("preset")
        p.add_argument("--seqs", type=int, required=True)
        p.set_defaults(func=fn)
    f = sub.add_parser("fault")
    f.add_argument("mode", choices=list(FAULT_TEXT))
    f.set_defaults(func=cmd_fault)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
