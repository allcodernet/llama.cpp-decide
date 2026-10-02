"""Diagnosis of the 500 in gemma-4-e4b-vision's held check (a diagnostic added after Task 6); outputs go to
integrity/gemma-4-e4b-vision/held-control/. Every command records what it saw and exits 0 (no pass key: a diagnosis, not a check).

  server --url U --model P --image JPG --chat none|text|image --out F.json
      optionally a chat request resident in the slot (text only or with the image), /slots, a decide probe whose 400 budget
      text gives held, prefix, tail, branches and the limit, then a text state sized to fit exactly; its answer, /slots, info
  bisect --url U --model P --lo N --hi N --steps K --out F.json
      on an idle server: one-state requests of "cat cat …" sized by /tokenize, halving [lo, hi] (lo answers 200, hi does not)
  cli --model GGUF --preset P --words N --out F.json [--no-mmproj]
      llama-decide (the preset's arguments, without --mmproj and the image-token keys with --no-mmproj) on one state of N words;
      its answer and the KV cache lines of its log
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checks as ck  # noqa: E402
from checks import ONE_FIELD, body_for, budget_parts, image_part, info, post, rc, sized_text, text_part  # noqa: E402

KV_LINE = ("KV cache, size", "llama_kv_cache: size", "n_swa ", "n_seq_max", "n_ctx ", "n_ubatch", "failed to find", "swa_full",
           "kv_unified")


def save(out, result):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("summary", "steps", "boundary")}, indent=1)[:3000])


def slots(client, model):
    r = client.get("/slots", params={"model": model})
    return [{k: s.get(k) for k in ("id", "n_ctx", "is_processing", "n_prompt_tokens", "n_prompt_tokens_processed")}
            for s in r.json()] if r.status_code == 200 else {"status": r.status_code, "text": r.text[:300]}


def short(r):
    """status, error and usage of a /v1/decide answer"""
    resp = r["response"]
    return {"status": r["status"], "error": resp.get("error"), "usage": resp.get("usage"),
            "state_usage": [x.get("usage") for x in resp.get("results", [])]}


def cmd_server(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    out = {"model": args.model, "chat_kind": args.chat}
    if args.chat != "none":
        content = [text_part("Describe this picture in one word.")]
        if args.chat == "image":
            content.append(image_part(args.image))
        chat = client.post("/v1/chat/completions", json={
            "model": args.model, "max_tokens": 1, "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]})
        out["chat"] = {"status": chat.status_code, "usage": chat.json().get("usage")}
    out["slots_before"] = slots(client, args.model)
    probe = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, ["cat" + " cat" * 19999]))
    out["probe"] = short(probe)
    b = budget_parts((probe["response"].get("error") or {}).get("message"))
    out["budget"] = b
    size = b["limit"] - b["held"] - b["prefix"] - b["tail"] - b["branches"]
    out["size"] = size
    out["sized"] = short(post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [sized_text(client, args.model, size)])))
    out["slots_after"] = slots(client, args.model)
    i = info(client, args.model)
    out["info"] = {k: i.get(k) for k in ("n_ctx", "n_batch", "decide_seqs", "n_parallel", "kv_unified", "memory", "prefixes", "vision")}
    out["summary"] = {"chat": args.chat, "held": b["held"], "size": size, "sized_status": out["sized"]["status"],
                      "sized_error": (out["sized"]["error"] or {}).get("message")}
    save(args.out, out)


def cmd_bisect(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    steps = []

    def run(n):
        r = short(post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [sized_text(client, args.model, n)])))
        steps.append({"state_tokens": n, "status": r["status"], "error": (r["error"] or {}).get("message")})
        return r["status"] == 200

    lo, hi = args.lo, args.hi
    ends = {"lo answers 200": run(lo), "hi answers 200": run(hi)}
    if ends["lo answers 200"] and not ends["hi answers 200"]:
        for _ in range(args.steps):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            if run(mid):
                lo = mid
            else:
                hi = mid
    i = info(client, args.model)
    save(args.out, {"model": args.model, "ends": ends, "steps": steps, "boundary": {"largest_200": lo, "smallest_not_200": hi},
                    "info": {k: i.get(k) for k in ("n_ctx", "n_batch", "decide_seqs", "n_parallel", "kv_unified", "memory", "prefixes")}})


def cmd_cli(args):
    cmd = rc.decide_command(args.model, args.preset, int(ck._preset_settings(args.preset)["decide-seqs"]), dump=False)
    if args.no_mmproj:
        keep, skip = [], False
        for a in cmd:
            if skip:
                skip = False
            elif a in ("--mmproj", "--image-min-tokens", "--image-max-tokens"):
                skip = True
            else:
                keep.append(a)
        cmd = keep
    body = body_for(args.preset, ONE_FIELD, ["cat" + " cat" * (args.words - 1)])
    proc = subprocess.run(cmd + ["-v"] if args.verbose else cmd, input=json.dumps(body) + "\n", capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    resp = json.loads(lines[-1]) if lines else None
    err = ck.last_json(proc.stderr)
    save(args.out, {"command": cmd, "words": args.words, "returncode": proc.returncode,
                    "usage": resp and resp.get("usage"), "state_usage": resp and [x.get("usage") for x in resp.get("results", [])],
                    "error": err, "kv_lines": [l for l in proc.stderr.splitlines() if any(k in l for k in KV_LINE)][:60],
                    "summary": {"no_mmproj": args.no_mmproj, "words": args.words, "returncode": proc.returncode,
                                "error": (err or {}).get("error")}})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("server")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--chat", choices=["none", "text", "image"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_server)
    p = sub.add_parser("bisect")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--lo", type=int, required=True)
    p.add_argument("--hi", type=int, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_bisect)
    p = sub.add_parser("cli")
    p.add_argument("--model", required=True, help="the GGUF")
    p.add_argument("--preset", required=True)
    p.add_argument("--words", type=int, required=True)
    p.add_argument("--no-mmproj", action="store_true")
    p.add_argument("--verbose", action="store_true", help="pass -v to llama-decide")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cli)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
