"""Sub-project 3 Task 4 live checks (spec 12.3 prefix cache and fault hook): request lines and response summaries.

  uv run results-engine/sp3/checks/task-4/checks.py requests --preset P --n N --per-request K [--variants V1,V2,...] OUT
      N test tickets as request lines of K states each (decide_client.build_request); with --variants, one line per
      variant (cycling), all with the first K tickets: e.g. keywords,default,keywords is the A/B/A sequence.
  uv run results-engine/sp3/checks/task-4/checks.py aba RESP.jsonl OUT.json [--cache on|off]
      A/B/A: line 3 vs line 1 (max abs diff <= 1e-3 and no argmax disagreement; cache on: line 3 hits = 1, else
      hits = misses = 0 and line 3 = line 1 exactly), plus usage/timings/prefix_cache of every line.
  uv run results-engine/sp3/checks/task-4/checks.py sequence REQ.jsonl RESP.jsonl OUT.json
      per line: prefix_cache, cached_tokens, prompt_tokens, prefix_ms; every repeated request line vs its first
      occurrence (max abs diff, argmax disagreements).
  uv run results-engine/sp3/checks/task-4/checks.py compare REF.jsonl RUN.jsonl OUT.json [--min-retries R] [--max-diff X]
      every answer of RUN vs REF (responses in the same order) with sp3_metrics.compare (the statistic); per RUN line
      timings.retries, rounds and states_per_round; pass = statistic, every line retries >= R, max abs diff <= X.
  uv run results-engine/sp3/checks/task-4/checks.py server-twice --preset P --n K --url URL OUT.json
      rc1-twice server: POST /v1/decide (first request, expect 500), GET /v1/decide/info, POST the same request again.
  uv run results-engine/sp3/checks/task-4/checks.py server-post --preset P --n K --url URL OUT.jsonl
      one POST /v1/decide of the same request (a normal run to compare the rc1-twice follow-up with).
  uv run results-engine/sp3/checks/task-4/checks.py server-moved --preset P --n K --url URL OUT.json
      A/B/A on a server with -np 1 where a /completion between B and the second A leaves chat cells in the KV
      cache, so the second A (restored, or rebuilt with the cache off) and its trunks land on other cells than the
      first. Reports third vs first (KV layout changed, so not held to 1e-3) and writes the three responses to
      OUT.responses.jsonl; `compare` then sets the restored third answer against the rebuilt one.
  uv run results-engine/sp3/checks/task-4/checks.py restore-fault FAULT.jsonl CACHE8.jsonl CACHE0.jsonl OUT.json
      A/B/A under LLAMA_DECIDE_FAULT=restore: third request hits 0, misses >= 1, one entry fewer than the normal
      cache-8 run, and every answer equal (max abs diff 0) to the cache-0 run.
  uv run results-engine/sp3/checks/task-4/checks.py server-aba-info --preset P --n K --url URL OUT.json
      POST A, B, A (keywords, default, keywords), then GET /v1/decide/info; pass: 3 x 200 and `prefixes` = slot 0 only.
  uv run results-engine/sp3/checks/task-4/checks.py server-budget --preset P --n K --url URL OUT.json
      POST A, GET info, POST B whose one state holds every test ticket (a 400 budget over n_ctx), GET info; pass: the
      400 and slot 0 still valid with A's hash, nothing else resident, no cache entry made.
  uv run results-engine/sp3/checks/task-4/checks.py server-flags --preset P --n K --url URL OUT.json
      a server started with a non-default --decide-temperature: GET /v1/decide/info, then POST /v1/decide (expect
      the 500 "not implemented yet: temperature" until Task 7).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from decide_client import TEST_SET, build_request  # noqa: E402
from sp3_metrics import compare  # noqa: E402


def tickets(n):
    return [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()][:n]


def entries(responses):
    """sp3_metrics entries (ticket, field, {key: p}) over every result of every response, in order."""
    out, i = [], 0
    for resp in responses:
        for res in resp["results"]:
            a = res["answers"]
            out.append((i, "queue", dict(a["queue"]["probabilities"])))
            out.append((i, "urgency", {str(k): p for k, p in enumerate(a["urgency"]["probabilities"])}))
            out.append((i, "angry", {"true": a["angry"]["p_true"], "false": 1.0 - a["angry"]["p_true"]}))
            i += 1
    return out


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def line_summary(resp):
    return {"usage": resp["usage"], "timings": resp["timings"], "engine": resp["engine"]}


def cmd_requests(args):
    texts = tickets(args.n)
    if args.variants:
        lines = [build_request(args.preset, texts[: args.per_request], v) for v in args.variants.split(",")]
    else:
        lines = [build_request(args.preset, texts[i : i + args.per_request]) for i in range(0, len(texts), args.per_request)]
    Path(args.out).write_text("".join(json.dumps(l) + "\n" for l in lines))
    print(f"wrote {args.out} ({len(lines)} lines)")


def cmd_aba(args):
    r = load(args.resp)
    if len(r) != 3:
        sys.exit(f"{args.resp}: expected 3 response lines, got {len(r)}")
    cmp = compare(entries([r[0]]), entries([r[2]]))
    pc = r[2]["engine"]["prefix_cache"]
    if args.cache == "on":
        cache_ok = pc["hits"] == 1 and r[2]["usage"]["cached_tokens"] > 0
    else:
        cache_ok = pc["hits"] == 0 and pc["misses"] == 0 and cmp["max_abs_diff"] == 0.0
    agree = all(f["argmax_disagreements"] == 0 for f in cmp["fields"].values())
    out = {
        "resp": args.resp,
        "cache": args.cache,
        "third_vs_first": cmp,
        "lines": [line_summary(x) for x in r],
        "pass": cmp["max_abs_diff"] <= 1e-3 and agree and cache_ok,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"max_abs_diff": cmp["max_abs_diff"], "argmax_agree": agree, "third_prefix_cache": pc,
                      "cached_tokens": [x["usage"]["cached_tokens"] for x in r],
                      "prefix_ms": [x["timings"]["prefix_ms"] for x in r], "pass": out["pass"]}))


def cmd_sequence(args):
    reqs, resps = load(args.req), load(args.resp)
    if len(reqs) != len(resps):
        sys.exit(f"{len(reqs)} request lines, {len(resps)} responses")
    first, repeats = {}, []
    for i, q in enumerate(reqs):
        key = json.dumps(q, sort_keys=True)
        if key in first:
            cmp = compare(entries([resps[first[key]]]), entries([resps[i]]))
            repeats.append({"line": i, "first": first[key], "max_abs_diff": cmp["max_abs_diff"],
                            "argmax_disagreements": sum(f["argmax_disagreements"] for f in cmp["fields"].values())})
        else:
            first[key] = i
    out = {
        "lines": [{"prefix_cache": r["engine"]["prefix_cache"], "cached_tokens": r["usage"]["cached_tokens"],
                   "prompt_tokens": r["usage"]["prompt_tokens"], "prefix_ms": r["timings"]["prefix_ms"]} for r in resps],
        "repeats": repeats,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["repeats"]))


def cmd_compare(args):
    ref, run = load(args.ref), load(args.run)
    cmp = compare(entries(ref), entries(run))
    retries = [x["timings"]["retries"] for x in run]
    out = {
        "ref": args.ref,
        "run": args.run,
        **cmp,
        "run_lines": [{"retries": x["timings"]["retries"], "rounds": x["timings"]["rounds"],
                       "states_per_round": x["engine"]["states_per_round"], "decodes": x["timings"]["decodes"]} for x in run],
        "ref_lines": [{"retries": x["timings"]["retries"], "rounds": x["timings"]["rounds"],
                       "states_per_round": x["engine"]["states_per_round"]} for x in ref],
    }
    ok = cmp["pass"] and all(r >= args.min_retries for r in retries)
    if args.max_diff is not None:
        out["max_diff_bound"] = args.max_diff
        ok = ok and cmp["max_abs_diff"] <= args.max_diff
    out["pass_all"] = ok
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_tickets": cmp["n_tickets"], "max_abs_diff": cmp["max_abs_diff"], "statistic": cmp["pass"],
                      "retries": retries, "pass_all": ok}))


def _request(preset, n):
    return build_request(preset, tickets(n))


def cmd_server_twice(args):
    body = _request(args.preset, args.n)
    with httpx.Client(base_url=args.url, timeout=600) as c:
        first = c.post("/v1/decide", json=body)
        info = c.get("/v1/decide/info")
        nxt = c.post("/v1/decide", json=body)
    prefixes = info.json().get("prefixes", [])
    only_slot0 = all(p["slot"] == 0 for p in prefixes) and all(p["valid"] for p in prefixes)
    out = {
        "request_states": args.n,
        "first": {"status": first.status_code, "body": first.json()},
        "info_after_first": {"status": info.status_code, "body": info.json()},
        "next": {"status": nxt.status_code, "body": nxt.json()},
        "pass": first.status_code == 500 and info.status_code == 200 and only_slot0 and nxt.status_code == 200,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    # the follow-up response alone, for `compare` against a normal run
    Path(args.out).with_suffix(".next.jsonl").write_text(json.dumps(nxt.json()) + "\n")
    print(json.dumps({"first": [first.status_code, first.json().get("error")], "prefixes": prefixes,
                      "next": nxt.status_code, "pass": out["pass"]}))


def cmd_server_post(args):
    body = _request(args.preset, args.n)
    with httpx.Client(base_url=args.url, timeout=600) as c:
        r = c.post("/v1/decide", json=body)
    if r.status_code != 200:
        sys.exit(f"POST /v1/decide: {r.status_code} {r.text[:500]}")
    Path(args.out).write_text(json.dumps(r.json()) + "\n")
    print(f"wrote {args.out}")


def cmd_restore_fault(args):
    fault, normal, off = load(args.fault), load(args.cache8), load(args.cache0)
    cmp = compare(entries(off), entries(fault))
    pc, pc_normal = fault[2]["engine"]["prefix_cache"], normal[2]["engine"]["prefix_cache"]
    out = {
        "fault": args.fault, "cache8": args.cache8, "cache0": args.cache0,
        "vs_cache0": cmp,
        "third_prefix_cache": pc,
        "normal_third_prefix_cache": pc_normal,
        "lines": [line_summary(x) for x in fault],
        "pass": pc["hits"] == 0 and pc["misses"] >= 1 and pc["entries"] == pc_normal["entries"] - 1 and cmp["max_abs_diff"] == 0.0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"third_prefix_cache": pc, "normal_entries": pc_normal["entries"], "max_abs_diff_vs_cache0": cmp["max_abs_diff"],
                      "pass": out["pass"]}))


def cmd_server_aba_info(args):
    texts = tickets(args.n)
    a, b = build_request(args.preset, texts, "keywords"), build_request(args.preset, texts, "default")
    with httpx.Client(base_url=args.url, timeout=600) as c:
        rs = [c.post("/v1/decide", json=x) for x in (a, b, a)]
        info = c.get("/v1/decide/info")
    prefixes = info.json()["prefixes"]
    out = {
        "status": [r.status_code for r in rs],
        "lines": [line_summary(r.json()) for r in rs],
        "info_after": info.json(),
        "pass": all(r.status_code == 200 for r in rs) and [p["slot"] for p in prefixes] == [0] and prefixes[0]["valid"],
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "third_prefix_cache": rs[2].json()["engine"]["prefix_cache"],
                      "prefixes": prefixes, "prefix_cache": info.json()["prefix_cache"], "pass": out["pass"]}))


def cmd_server_budget(args):
    texts = tickets(args.n)
    a = build_request(args.preset, texts, "keywords")
    b = build_request(args.preset, [" ".join(tickets(80))], "default")
    with httpx.Client(base_url=args.url, timeout=600) as c:
        ra = c.post("/v1/decide", json=a)
        before = c.get("/v1/decide/info").json()
        rb = c.post("/v1/decide", json=b)
        after = c.get("/v1/decide/info").json()
    slot0 = before["prefixes"][0] if before["prefixes"] else {}
    out = {
        "a_status": ra.status_code,
        "info_before": {"prefix": before["prefix"], "prefixes": before["prefixes"], "prefix_cache": before["prefix_cache"]},
        "b": {"status": rb.status_code, "body": rb.json()},
        "info_after": {"prefix": after["prefix"], "prefixes": after["prefixes"], "prefix_cache": after["prefix_cache"]},
        "pass": ra.status_code == 200 and rb.status_code == 400 and rb.json()["error"]["code"] == "budget"
                and after["prefixes"] == [slot0] and slot0.get("valid") is True and after["prefix_cache"]["entries"] == 0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"b": [rb.status_code, rb.json().get("error")], "before": before["prefixes"], "after": after["prefixes"],
                      "entries_after": after["prefix_cache"]["entries"], "pass": out["pass"]}))


def cmd_server_moved(args):
    texts = tickets(args.n)
    a, b = build_request(args.preset, texts, "keywords"), build_request(args.preset, texts, "default")
    filler = " ".join(tickets(20))   # a few hundred tokens of ticket text for the chat slot
    with httpx.Client(base_url=args.url, timeout=600) as c:
        r1 = c.post("/v1/decide", json=a).json()
        r2 = c.post("/v1/decide", json=b).json()
        comp = c.post("/completion", json={"prompt": filler, "n_predict": 1, "cache_prompt": True})
        r3 = c.post("/v1/decide", json=a).json()
        info = c.get("/v1/decide/info").json()
    cmp = compare(entries([r1]), entries([r3]))
    agree = all(f["argmax_disagreements"] == 0 for f in cmp["fields"].values())
    out = {
        "completion": {"status": comp.status_code, "tokens_evaluated": comp.json().get("tokens_evaluated")},
        "third_vs_first": cmp,
        "third_argmax_agrees": agree,
        "lines": [line_summary(x) for x in (r1, r2, r3)],
        "info_after": info,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    Path(args.out).with_suffix(".responses.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (r1, r2, r3)))
    print(json.dumps({"completion_tokens": out["completion"]["tokens_evaluated"], "max_abs_diff": cmp["max_abs_diff"],
                      "argmax_agree": agree, "third_prefix_cache": r3["engine"]["prefix_cache"]}))


def cmd_server_flags(args):
    body = _request(args.preset, args.n)
    with httpx.Client(base_url=args.url, timeout=600) as c:
        info = c.get("/v1/decide/info")
        dec = c.post("/v1/decide", json=body)
    i = info.json()
    out = {
        "info": {"status": info.status_code, "body": i},
        "decide": {"status": dec.status_code, "body": dec.json()},
        "pass": info.status_code == 200 and dec.status_code == 500
                and dec.json()["error"]["message"] == "not implemented yet: temperature",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"temperature": i.get("temperature"), "prefix_cache": i.get("prefix_cache"),
                      "decide": [dec.status_code, dec.json().get("error")], "pass": out["pass"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("requests")
    q.add_argument("--preset", required=True)
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--per-request", type=int, required=True)
    q.add_argument("--variants", default="")
    q.add_argument("out")
    a = sub.add_parser("aba")
    a.add_argument("resp")
    a.add_argument("out")
    a.add_argument("--cache", choices=["on", "off"], default="on")
    rf = sub.add_parser("restore-fault")
    for name in ("fault", "cache8", "cache0", "out"):
        rf.add_argument(name)
    sq = sub.add_parser("sequence")
    sq.add_argument("req")
    sq.add_argument("resp")
    sq.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("run")
    c.add_argument("out")
    c.add_argument("--min-retries", type=int, default=0)
    c.add_argument("--max-diff", type=float, default=None)
    for name in ("server-twice", "server-post", "server-flags", "server-moved", "server-aba-info", "server-budget"):
        s = sub.add_parser(name)
        s.add_argument("--preset", required=True)
        s.add_argument("--n", type=int, required=True)
        s.add_argument("--url", default="http://127.0.0.1:8097")
        s.add_argument("out")
    args = ap.parse_args()
    {"requests": cmd_requests, "aba": cmd_aba, "sequence": cmd_sequence, "compare": cmd_compare,
     "server-twice": cmd_server_twice, "server-post": cmd_server_post, "server-flags": cmd_server_flags,
     "server-moved": cmd_server_moved, "restore-fault": cmd_restore_fault, "server-aba-info": cmd_server_aba_info,
     "server-budget": cmd_server_budget}[args.cmd](args)


if __name__ == "__main__":
    main()
