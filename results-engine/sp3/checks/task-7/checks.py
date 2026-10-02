"""Sub-project 3 Task 7 live checks (spec 7, 8, 12.3): request lines, server drivers and response checks.
run.sh in this directory runs every check; `checks.py notes` writes notes.md from the outputs.

  requests --preset P --n N --per-request K [--skip S] [--debias K] [--temperature T] [--scoring S] [--after F=A[,A]]...
           [--rotate V/K] [--one-state] OUT
      N test tickets (after the first S) as request lines of K states each (decide_client.build_request, prompt variant
      `default`); --rotate V/K declares every choice field's options in the order of catalogue variant V of K (spec 7.1);
      --one-state joins the N tickets into one state.
  same REF RUN OUT [--ignore-engine K,..] [--ignore-tokens K,..] [--ignore-entry K,..]
      per line: results and usage equal, engine and the tokens dump equal apart from the listed keys; max abs diff.
  compare REF RUN OUT [--max-diff X]
      every answer of RUN vs REF with sp3_metrics.compare (the statistic).
  k2 RESP OUT
      three lines: K = 0 with the choice options in variant 1's order, K = 0, K = 2 (--dump-tokens): the K = 2 answers vs the
      per-key mean of the two K = 0 answers under the statistic, and the K = 2 hits, prefixes, dump variants, order_spread.
  k2fp RESP OUT
      the k2 sequence in full-path mode: the per-variant distributions of the K = 2 dump vs the K = 0 request with the same
      catalogue (variant 1 restored into slot 1 from the snapshot of slot 0), and the K = 2 answers vs the K = 0 mean.
  posthoc T1.jsonl T2.jsonl OUT --t T
      line by line: the T answers equal the T = 1 answers scaled post hoc (p_T ~ p^(1/T)): probabilities, expected,
      confidence, p_true within 1e-9, values equal; coverage, order_spread, given equal; engine.temperature 1 and T.
  dumpcheck RESP OUT --t T
      debias + full-path dump: per state, field and variant the distribution and coverage recomputed from the dumped rows
      (Task 5 recompute_field), their mean, order_spread and mean coverage, then temperature T: equal to the answers within
      1e-9; dependent entries carry the ancestors' returned values as given and lines.
  allswitch REQ RESP OUT
      the full-path + after + K + T request: answers sum to 1, coverage and order_spread on every answer, given on every
      dependent answer (= the ancestors' values), engine block (scoring, order_debias, temperature, phases, seqs_per_state).
  cells ERR.json K2RESP OUT --k K
      the per-state cell budget error of a K-variant request: need = held + the K prefixes + K x (state + tail + branches),
      the prefixes as long as those of the K = 2 dump K2RESP (last line).
  retries RESP OUT [--min M]
      timings.retries of every line >= M, with rounds and states_per_round.
  info RESP OUT --expect S:R,...
      --info lines: schema.seqs_per_state and states_per_round per line as expected.
  k4 RESP OUT --k K
      repeated K-variant requests: order_debias, sums, order_spread; second line hits K - 1; second vs first (statistic).
  server-twice --preset P --n N --url URL OUT       (rc1-twice server) K = 2 request (500), GET info, K = 1 request
  server-post --preset P --n N --debias K --url URL OUT.jsonl
  server-temp --preset P --n N --url URL OUT        GET info, POST decide (no temperature), decide (temperature 1), systemone
  server-temp-check A.json B.json OUT --t T         B (--decide-temperature T) vs A (default): post-hoc scaling
  server-budget --preset P --n N --url URL OUT       plain request, info, all-switch request (400 budget), info
  notes                                             writes notes.md from the outputs in this directory
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from decide_client import TEST_SET, build_request  # noqa: E402
from sp3_metrics import compare  # noqa: E402
from systemone_smoke import triage_ticket0  # noqa: E402

TOL = 1e-9
BUDGET_44_28 = ("schema needs 44 sequences per state (K_eff 4 × (1 trunk + 10 branch sequences, full_path)); "
                "--decide-seqs 32 leaves 28 after 4 prefix slots")


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def tickets(n):
    return [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()][:n]


def task5():
    """The checks.py of task-5/ (recompute_field: spec 4.1 from dumped full-path rows)."""
    spec = importlib.util.spec_from_file_location("task5_checks", HERE.parent / "task-5" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dist(ans):
    """An answer's distribution keyed like sp3_metrics: choice keys, score levels as strings, bool true/false."""
    if "p_true" in ans:
        return {"true": ans["p_true"], "false": 1.0 - ans["p_true"]}
    p = ans["probabilities"]
    return dict(p) if isinstance(p, dict) else {str(k): v for k, v in enumerate(p)}


def entries(responses):
    """sp3_metrics entries (ticket, field, {key: p}) over every result of every response, in order."""
    out, i = [], 0
    for resp in responses:
        for res in resp["results"]:
            for name, a in res["answers"].items():
                out.append((i, name, dist(a)))
            i += 1
    return out


def temper(p, T):
    """Spec 8 post hoc: p_T proportional to p^(1/T); zeros stay zero."""
    lp = [math.log(x) / T if x > 0 else -math.inf for x in p]
    m = max(lp)
    w = [math.exp(v - m) if v != -math.inf else 0.0 for v in lp]
    z = sum(w)
    return [x / z for x in w]


def jev_confidence(p):
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1.0) / (n - 1.0)))


def ancestors(fields):
    """name -> its ancestors in declaration order (transitive closure of `after`)."""
    names = list(fields)

    def close(n, seen):
        for a in fields[n].get("after", []):
            if a not in seen:
                seen.add(a)
                close(a, seen)
        return seen

    return {n: [m for m in names if m in close(n, set())] for n in names}


def jtext(v):
    return json.dumps(v, ensure_ascii=False)


# ---------------------------------------------------------------- request lines


def cmd_requests(args):
    texts = tickets(args.skip + args.n)[args.skip:]
    if args.one_state:
        texts = ["\n\n".join(texts)]
    lines = []
    for i in range(0, len(texts), args.per_request):
        body = build_request(args.preset, texts[i : i + args.per_request])
        opts = body["options"]
        opts["scoring"] = args.scoring
        opts["order_debias"] = args.debias
        if args.temperature is not None:
            opts["temperature"] = args.temperature
        for spec in args.after:
            name, parents = spec.split("=")
            body["fields"][name]["after"] = parents.split(",")
        if args.rotate:
            v, k = (int(x) for x in args.rotate.split("/"))
            for f in body["fields"].values():
                if f["type"] == "choice":
                    keys = list(f["options"])
                    s = v * len(keys) // k
                    order = keys[s:] + keys[:s]
                    f["options"] = {key: f["options"][key] for key in order} if isinstance(f["options"], dict) else order
        lines.append(body)
    Path(args.out).write_text("".join(json.dumps(l) + "\n" for l in lines))
    print(f"wrote {args.out} ({len(lines)} lines; scoring {args.scoring}, order_debias {args.debias}, "
          f"temperature {args.temperature}, after {args.after}, rotate {args.rotate})")


# ---------------------------------------------------------------- equality and the statistic


def _split(s):
    return set(s.split(",")) if s else set()


def _strip_tokens(tok, ignore_tokens, ignore_entry):
    t = {k: v for k, v in tok.items() if k not in ignore_tokens}
    t["states"] = [{**st, "branches": [{k: v for k, v in b.items() if k not in ignore_entry} for b in st["branches"]]}
                   for st in tok["states"]]
    return t


def cmd_same(args):
    ref, run = load(args.ref), load(args.run)
    ig_eng, ig_tok, ig_ent = _split(args.ignore_engine), _split(args.ignore_tokens), _split(args.ignore_entry)
    problems = [] if len(ref) == len(run) else [f"{len(ref)} reference lines, {len(run)} run lines"]
    for li, (a, b) in enumerate(zip(ref, run)):
        for key in ("results", "usage"):
            if a[key] != b[key]:
                problems.append(f"line {li}: {key} differs")
        ea = {k: v for k, v in a["engine"].items() if k not in ig_eng}
        eb = {k: v for k, v in b["engine"].items() if k not in ig_eng}
        if ea != eb:
            problems.append(f"line {li}: engine differs: {ea} vs {eb}")
        if ("tokens" in a) != ("tokens" in b):
            problems.append(f"line {li}: tokens in one file only")
        elif "tokens" in a and _strip_tokens(a["tokens"], ig_tok, ig_ent) != _strip_tokens(b["tokens"], ig_tok, ig_ent):
            problems.append(f"line {li}: tokens dump differs")
    cmp = compare(entries(ref), entries(run))
    out = {"ref": args.ref, "run": args.run, "ignored": {"engine": sorted(ig_eng), "tokens": sorted(ig_tok), "entry": sorted(ig_ent)},
           "lines": len(run), "max_abs_diff": cmp["max_abs_diff"], "problems": problems,
           "pass": not problems and cmp["max_abs_diff"] == 0.0}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("lines", "max_abs_diff", "problems", "pass")}))


def statistic(ref_entries, run_entries):
    cmp = compare(ref_entries, run_entries)
    return {"n_tickets": cmp["n_tickets"], "max_abs_diff": cmp["max_abs_diff"],
            "fields": {f: {k: v[k] for k in ("median_abs_diff", "max_abs_diff", "argmax_disagreements", "disagreements_over_margin", "pass")}
                       for f, v in cmp["fields"].items()},
            "pass": cmp["pass"]}


def cmd_compare(args):
    st = statistic(entries(load(args.ref)), entries(load(args.run)))
    out = {"ref": args.ref, "run": args.run, **st}
    ok = st["pass"]
    if args.max_diff is not None:
        out["max_diff_bound"] = args.max_diff
        ok = ok and st["max_abs_diff"] <= args.max_diff
    out["pass_all"] = ok
    dump_json(args.out, out)
    print(json.dumps({"max_abs_diff": st["max_abs_diff"], "medians": {f: v["median_abs_diff"] for f, v in st["fields"].items()},
                      "statistic": st["pass"], "pass_all": ok}))


# ---------------------------------------------------------------- K = 2 vs two K = 0 requests


def cmd_k2(args):
    rot, plain, k2 = load(args.resp)
    ref = []
    for i, (ra, pa) in enumerate(zip(rot["results"], plain["results"])):
        for name in pa["answers"]:
            dp, dr = dist(pa["answers"][name]), dist(ra["answers"][name])
            ref.append((i, name, {k: (dp[k] + dr[k]) / 2 for k in dp}))
    st = statistic(ref, entries([k2]))
    problems, spread_diff = [], 0.0
    if [r["engine"]["order_debias"] for r in (rot, plain, k2)] != [1, 1, 2]:
        problems.append(f"order_debias {[r['engine']['order_debias'] for r in (rot, plain, k2)]}")
    pc = k2["engine"]["prefix_cache"]
    if pc["hits"] != 1 or pc["misses"] != 0:
        problems.append(f"K = 2 prefix_cache {pc}")
    prefixes = k2["tokens"]["prefixes"]
    if len(prefixes) != 2 or prefixes[0] != plain["tokens"]["prefix"] or prefixes[1] != rot["tokens"]["prefix"]:
        problems.append("K = 2 tokens.prefixes are not [plain prefix, rotated prefix]")
    if k2["tokens"]["prefix"] != prefixes[0]:
        problems.append("tokens.prefix is not variant 0")
    if k2["usage"]["cached_tokens"] != len(prefixes[0]) + len(prefixes[1]):
        problems.append(f"cached_tokens {k2['usage']['cached_tokens']}, prefixes {len(prefixes[0])} + {len(prefixes[1])}")
    for si, (st_k2, st_p) in enumerate(zip(k2["tokens"]["states"], plain["tokens"]["states"])):
        per_v = {}
        for b in st_k2["branches"]:
            per_v.setdefault(b.get("variant"), []).append({k: v for k, v in b.items() if k != "variant"})
        plain_entries = [{k: v for k, v in b.items() if k != "variant"} for b in st_p["branches"]]
        if sorted(per_v) != [0, 1] or any(v != plain_entries for v in per_v.values()):
            problems.append(f"state {si}: K = 2 entries per variant differ from the K = 0 entries (variants {sorted(per_v)})")
    for i, (res, ra, pa) in enumerate(zip(k2["results"], rot["results"], plain["results"])):
        for name, a in res["answers"].items():
            if "order_spread" not in a or "order_spread" in pa["answers"][name]:
                problems.append(f"state {i} {name}: order_spread only on K = 2 answers expected")
                continue
            dp, dr = dist(pa["answers"][name]), dist(ra["answers"][name])
            spread_diff = max(spread_diff, abs(a["order_spread"] - max(abs(dp[k] - dr[k]) for k in dp)))
    out = {"resp": args.resp, "states": len(k2["results"]), "statistic": st,
           "k2_engine": k2["engine"], "k2_usage": k2["usage"], "k2_timings": k2["timings"],
           "order_spread_vs_k0_max_abs_diff": spread_diff, "problems": problems,
           "pass_structure": not problems, "pass": not problems and st["pass"]}
    dump_json(args.out, out)
    print(json.dumps({"medians": {f: v["median_abs_diff"] for f, v in st["fields"].items()}, "max_abs_diff": st["max_abs_diff"],
                      "statistic": st["pass"], "hits": pc["hits"], "problems": problems}))


def cmd_k2fp(args):
    """The k2 sequence in full-path mode: per-variant distributions from the K = 2 dump against the K = 0 request with the
    same catalogue (variant 0: plain, resident slot 0; variant 1: rotated, restored into slot 1), and the K = 2 answer
    against the mean of the two K = 0 answers."""
    t5 = task5()
    rot, plain, k2 = load(args.resp)
    per_v, spread_diff = {0: [], 1: []}, 0.0
    for si, (res, st) in enumerate(zip(k2["results"], k2["tokens"]["states"])):
        groups = {}
        for b in st["branches"]:
            groups.setdefault((b["field"], b["variant"]), []).append(b)
        for name, ans in res["answers"].items():
            keys = list(dist(ans))
            probs = {v: t5.recompute_field(groups[(name, v)])[0] for v in (0, 1)}
            for v in (0, 1):
                per_v[v].append((si, name, dict(zip(keys, probs[v]))))
            spread_diff = max(spread_diff, abs(ans["order_spread"] - max(abs(a - b) for a, b in zip(probs[0], probs[1]))))
    ref = []
    for i, (ra, pa) in enumerate(zip(rot["results"], plain["results"])):
        for name in pa["answers"]:
            dp, dr = dist(pa["answers"][name]), dist(ra["answers"][name])
            ref.append((i, name, {k: (dp[k] + dr[k]) / 2 for k in dp}))
    out = {"resp": args.resp, "k2_engine": k2["engine"],
           "variant0_vs_plain_k0": statistic(entries([plain]), per_v[0]),
           "variant1_vs_rotated_k0": statistic(entries([rot]), per_v[1]),
           "k2_vs_mean_of_k0": statistic(ref, entries([k2])),
           "order_spread_vs_dump_max_abs_diff": spread_diff}
    out["pass"] = out["k2_vs_mean_of_k0"]["pass"] and spread_diff <= TOL and k2["engine"]["prefix_cache"]["hits"] == 1
    dump_json(args.out, out)
    print(json.dumps({k: (v["max_abs_diff"], {f: x["median_abs_diff"] for f, x in v["fields"].items()}, v["pass"])
                      for k, v in out.items() if isinstance(v, dict) and "fields" in v} | {"spread_diff": spread_diff, "pass": out["pass"]}))


# ---------------------------------------------------------------- temperature


def tempered_answer(ans, T):
    """The answer the engine gives at temperature T, computed from its T = 1 answer."""
    if "p_true" in ans:
        pt = temper([ans["p_true"], 1.0 - ans["p_true"]], T)[0]
        return {"value": pt >= 0.5, "p_true": pt, "confidence": max(pt, 1.0 - pt)}
    p = ans["probabilities"]
    keys = list(p) if isinstance(p, dict) else None
    q = temper(list(p.values()) if keys else list(p), T)
    out = {"value": ans["value"], "probabilities": dict(zip(keys, q)) if keys else q, "confidence": max(q)}
    if "expected" in ans:
        out["expected"] = sum(i * x for i, x in enumerate(q))
    return out


def answer_diff(want, got):
    """max abs diff over the numbers of `want`; None when a value or a key disagrees."""
    worst = 0.0
    for k, v in want.items():
        if k == "value" or isinstance(v, (str, bool)):
            if got.get(k) != v:
                return None
        elif isinstance(v, dict):
            if list(v) != list(got[k]):
                return None
            worst = max([worst] + [abs(v[x] - got[k][x]) for x in v])
        elif isinstance(v, list):
            worst = max([worst] + [abs(a - b) for a, b in zip(v, got[k])])
        else:
            worst = max(worst, abs(v - got[k]))
    return worst


def cmd_posthoc(args):
    t1, t2 = load(args.t1), load(args.t2)
    problems, worst, n = [], 0.0, 0
    if len(t1) != len(t2):
        problems.append(f"{len(t1)} vs {len(t2)} lines")
    for li, (a, b) in enumerate(zip(t1, t2)):
        if a["engine"]["temperature"] != 1.0 or b["engine"]["temperature"] != args.t:
            problems.append(f"line {li}: engine.temperature {a['engine']['temperature']} / {b['engine']['temperature']}")
        for si, (ra, rb) in enumerate(zip(a["results"], b["results"])):
            for name, ans in ra["answers"].items():
                got = rb["answers"][name]
                d = answer_diff(tempered_answer(ans, args.t), got)
                if d is None:
                    problems.append(f"line {li} state {si} {name}: value or keys differ")
                    continue
                worst = max(worst, d)
                n += 1
                for k in ("coverage", "order_spread", "given"):
                    if ans.get(k) != got.get(k):
                        problems.append(f"line {li} state {si} {name}: {k} {ans.get(k)} vs {got.get(k)}")
    out = {"t1": args.t1, "t2": args.t2, "T": args.t, "answers": n, "max_abs_diff": worst, "problems": problems,
           "pass": not problems and n > 0 and worst <= TOL}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("answers", "max_abs_diff", "problems", "pass")}))


# ---------------------------------------------------------------- debias + full-path dump, all switches


def cmd_dumpcheck(args):
    t5 = task5()
    resps = load(args.resp)
    worst = {"probabilities": 0.0, "coverage": 0.0, "order_spread": 0.0}
    problems, n_fields, n_variants = [], 0, set()
    for li, r in enumerate(resps):
        K = r["engine"]["order_debias"]
        for si, (res, st) in enumerate(zip(r["results"], r["tokens"]["states"])):
            groups, n_rows = {}, 0
            for b in st["branches"]:
                groups.setdefault((b["field"], b["variant"]), []).append(b)
                n_rows += len(b["rows"])
            if n_rows != res["usage"]["scored_tokens"]:
                problems.append(f"line {li} state {si}: {n_rows} rows dumped, scored_tokens {res['usage']['scored_tokens']}")
            for name, ans in res["answers"].items():
                per_v = []
                for v in range(K):
                    probs, cov, _ = t5.recompute_field(groups[(name, v)])
                    per_v.append((probs, cov))
                n_variants.add(len(per_v))
                mean = [sum(p[o] for p, _ in per_v) / K for o in range(len(per_v[0][0]))]
                spread = max(max(p[o] for p, _ in per_v) - min(p[o] for p, _ in per_v) for o in range(len(mean)))
                cov = sum(c for _, c in per_v) / K
                want = temper(mean, args.t)
                got = [ans["p_true"], 1.0 - ans["p_true"]] if "p_true" in ans else (
                    list(ans["probabilities"].values()) if isinstance(ans["probabilities"], dict) else ans["probabilities"])
                worst["probabilities"] = max([worst["probabilities"]] + [abs(a - b) for a, b in zip(want, got)])
                worst["coverage"] = max(worst["coverage"], abs(cov - ans["coverage"]))
                if K >= 2:
                    worst["order_spread"] = max(worst["order_spread"], abs(spread - ans["order_spread"]))
                elif "order_spread" in ans:
                    problems.append(f"line {li} state {si} {name}: order_spread with K = 1")
                n_fields += 1
                for b in st["branches"]:
                    if b["field"] == name and ans.get("given") != b.get("given"):
                        problems.append(f"line {li} state {si} {name}: entry given {b.get('given')}, answer given {ans.get('given')}")
                        break
    out = {"resp": args.resp, "T": args.t, "fields_checked": n_fields, "variants": sorted(n_variants), "max_abs_diff": worst,
           "problems": problems, "pass": not problems and n_fields > 0 and max(worst.values()) <= TOL}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("fields_checked", "variants", "max_abs_diff", "problems", "pass")}))


def cmd_allswitch(args):
    reqs, resps = load(args.req), load(args.resp)
    problems, worst_sum, n = [], 0.0, 0
    for li, (req, r) in enumerate(zip(reqs, resps)):
        fields, opts, eng = req["fields"], req["options"], r["engine"]
        anc = ancestors(fields)
        want_eng = {"scoring": opts["scoring"], "order_debias": opts["order_debias"],
                    "temperature": opts.get("temperature", 1.0), "phases": 2}
        for k, v in want_eng.items():
            if eng.get(k) != v:
                problems.append(f"line {li}: engine.{k} {eng.get(k)}, want {v}")
        for si, res in enumerate(r["results"]):
            a = res["answers"]
            for name, ans in a.items():
                n += 1
                p = list(dist(ans).values())
                worst_sum = max(worst_sum, abs(sum(p) - 1.0))
                for key in ("coverage", "order_spread"):
                    if key not in ans:
                        problems.append(f"line {li} state {si} {name}: no {key}")
                want_given = {g: a[g]["value"] for g in anc[name]} or None
                if ans.get("given") != want_given:
                    problems.append(f"line {li} state {si} {name}: given {ans.get('given')}, want {want_given}")
    out = {"req": args.req, "resp": args.resp, "answers": n, "max_abs_sum_minus_1": worst_sum,
           "engine": [r["engine"] for r in resps], "timings": [r["timings"] for r in resps], "usage": [r["usage"] for r in resps],
           "problems": problems, "pass": not problems and n > 0 and worst_sum <= 1e-12}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("answers", "max_abs_sum_minus_1", "problems", "pass")}))


def cmd_cells(args):
    err = json.loads(Path(args.err).read_text())["error"]
    msg = err["message"]
    m = re.fullmatch(r"state 0 needs (\d+) cells \(held (\d+), prefix (\d+), K_eff (\d+) \u00d7 \(state (\d+), tail (\d+), "
                     r"branches (\d+)\)\), n_ctx - 16 is (\d+)", msg)
    k2 = load(args.k2)[-1]
    prefix = sum(len(x) for x in k2["tokens"]["prefixes"])
    ok = False
    if m:
        need, held, pre, k, state, tail, branches, cap = (int(x) for x in m.groups())
        ok = (err["code"] == "budget" and k == args.k and pre == prefix and need == held + pre + k * (state + tail + branches)
              and need > cap)
    out = {"err": args.err, "error": err, "prefix_tokens_of_the_k2_dump": prefix, "pass": ok}
    dump_json(args.out, out)
    print(json.dumps(out))


def cmd_retries(args):
    resps = load(args.resp)
    retries = [r["timings"]["retries"] for r in resps]
    out = {"resp": args.resp, "min": args.min, "retries": retries, "rounds": [r["timings"]["rounds"] for r in resps],
           "states_per_round": [r["engine"]["states_per_round"] for r in resps], "order_debias": [r["engine"]["order_debias"] for r in resps],
           "pass": bool(resps) and all(x >= args.min for x in retries)}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("retries", "rounds", "states_per_round", "pass")}))


def cmd_info(args):
    resps = load(args.resp)
    want = [tuple(int(x) for x in e.split(":")) for e in args.expect.split(",")]
    got = [(r["schema"]["seqs_per_state"], r["schema"]["states_per_round"]) for r in resps]
    out = {"resp": args.resp, "want": want, "got": got, "temperature": [r["temperature"] for r in resps],
           "prefixes": [r["prefixes"] for r in resps], "pass": got == want}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("want", "got", "pass")}))


def cmd_k4(args):
    resps = load(args.resp)
    problems, worst_sum = [], 0.0
    for li, r in enumerate(resps):
        if r["engine"]["order_debias"] != args.k:
            problems.append(f"line {li}: order_debias {r['engine']['order_debias']}")
        for si, res in enumerate(r["results"]):
            for name, ans in res["answers"].items():
                worst_sum = max(worst_sum, abs(sum(dist(ans).values()) - 1.0))
                if "order_spread" not in ans:
                    problems.append(f"line {li} state {si} {name}: no order_spread")
    pcs = [r["engine"]["prefix_cache"] for r in resps]
    if len(resps) >= 2 and (pcs[1]["hits"] != args.k - 1 or pcs[1]["misses"] != 0):
        problems.append(f"line 1 prefix_cache {pcs[1]}, want {args.k - 1} hits")
    st = statistic(entries(resps[:1]), entries(resps[1:2])) if len(resps) >= 2 else None
    out = {"resp": args.resp, "K": args.k, "max_abs_sum_minus_1": worst_sum, "prefix_cache": pcs,
           "engine": [r["engine"] for r in resps], "timings": [r["timings"] for r in resps], "usage": [r["usage"] for r in resps],
           "second_vs_first": st, "problems": problems, "pass": not problems and worst_sum <= 1e-12}
    dump_json(args.out, out)
    print(json.dumps({"prefix_cache": pcs, "max_abs_sum_minus_1": worst_sum,
                      "second_vs_first_max": st and st["max_abs_diff"], "problems": problems, "pass": out["pass"]}))


# ---------------------------------------------------------------- server drivers


def _body(preset, n, debias, **extra):
    body = build_request(preset, tickets(n))
    body["options"]["order_debias"] = debias
    body["options"].update(extra)
    return body


def cmd_server_twice(args):
    with httpx.Client(base_url=args.url, timeout=600) as c:
        first = c.post("/v1/decide", json=_body(args.preset, args.n, 2))
        info = c.get("/v1/decide/info")
        nxt = c.post("/v1/decide", json=_body(args.preset, args.n, 1))
    prefixes = info.json().get("prefixes", [])
    out = {"request_states": args.n,
           "first": {"status": first.status_code, "body": first.json()},
           "info_after_first": {"status": info.status_code, "body": info.json()},
           "next": {"status": nxt.status_code, "engine": nxt.json().get("engine"), "timings": nxt.json().get("timings")},
           "pass": first.status_code == 500 and first.json()["error"]["message"] == "no KV cell space (rc=1)" and info.status_code == 200
                   and [p["slot"] for p in prefixes] == [0] and prefixes[0]["valid"] and nxt.status_code == 200}
    dump_json(args.out, out)
    Path(args.out).with_suffix(".next.jsonl").write_text(json.dumps(nxt.json()) + "\n")
    print(json.dumps({"first": [first.status_code, first.json().get("error")], "prefixes": prefixes, "next": nxt.status_code,
                      "pass": out["pass"]}))


def cmd_server_post(args):
    with httpx.Client(base_url=args.url, timeout=600) as c:
        r = c.post("/v1/decide", json=_body(args.preset, args.n, args.debias))
    if r.status_code != 200:
        sys.exit(f"POST /v1/decide: {r.status_code} {r.text[:500]}")
    Path(args.out).write_text(json.dumps(r.json()) + "\n")
    print(f"wrote {args.out}")


def cmd_server_temp(args):
    body = _body(args.preset, args.n, 0)
    body_t1 = _body(args.preset, args.n, 0, temperature=1.0)
    with httpx.Client(base_url=args.url, timeout=600) as c:
        info = c.get("/v1/decide/info")
        dec = c.post("/v1/decide", json=body)
        dec_t1 = c.post("/v1/decide", json=body_t1)
        sys1 = c.post("/v1/systemone", json=triage_ticket0(args.preset))
    out = {"info": info.json(), "decide": dec.json(), "decide_t1": dec_t1.json(), "systemone": sys1.json(),
           "status": [info.status_code, dec.status_code, dec_t1.status_code, sys1.status_code]}
    dump_json(args.out, out)
    print(json.dumps({"status": out["status"], "temperature": info.json().get("temperature"),
                      "engine_temperature": [dec.json()["engine"]["temperature"], dec_t1.json()["engine"]["temperature"]]}))


def systemone_tempered(ans, T):
    """A /v1/systemone answer at temperature T from its T = 1 answer (decide_to_systemone on the tempered decide answer)."""
    if ans["type"] == "noul":
        return {"noul": temper([ans["noul"], 1.0 - ans["noul"]], T)[0]}
    keys = list(ans["probabilities"])
    q = temper([ans["probabilities"][k] for k in keys], T)
    out = {"probabilities": dict(zip(keys, q)), "confidence": jev_confidence(q)}
    if ans["type"] == "choice":
        out["choice"] = ans["choice"]
    else:
        out["score"] = sum(i * x for i, x in enumerate(q))
    return out


def cmd_server_temp_check(args):
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    problems, worst = [], {"decide": 0.0, "decide_t1": 0.0, "systemone": 0.0}
    if a["status"] != [200] * 4 or b["status"] != [200] * 4:
        problems.append(f"status {a['status']} / {b['status']}")
    if a["info"]["temperature"] != 1.0 or b["info"]["temperature"] != args.t:
        problems.append(f"info temperature {a['info']['temperature']} / {b['info']['temperature']}")
    if [a["decide"]["engine"]["temperature"], b["decide"]["engine"]["temperature"], b["decide_t1"]["engine"]["temperature"]] != [1.0, args.t, 1.0]:
        problems.append("engine.temperature is not 1 / T / 1")
    for ra, rb in zip(a["decide"]["results"], b["decide"]["results"]):
        for name, ans in ra["answers"].items():
            d = answer_diff(tempered_answer(ans, args.t), rb["answers"][name])
            if d is None:
                problems.append(f"decide {name}: value or keys differ")
            else:
                worst["decide"] = max(worst["decide"], d)
    for ra, rb in zip(a["decide_t1"]["results"], b["decide_t1"]["results"]):
        for name, ans in ra["answers"].items():
            d = answer_diff(ans, rb["answers"][name])
            if d is None:
                problems.append(f"decide_t1 {name}: differs")
            else:
                worst["decide_t1"] = max(worst["decide_t1"], d)
    for qid, ans in a["systemone"]["answers"].items():
        d = answer_diff(systemone_tempered(ans, args.t), b["systemone"]["answers"][qid])
        if d is None:
            problems.append(f"systemone {qid}: differs")
        else:
            worst["systemone"] = max(worst["systemone"], d)
    out = {"a": args.a, "b": args.b, "T": args.t, "max_abs_diff": worst, "problems": problems,
           "pass": not problems and max(worst.values()) <= TOL}
    dump_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("max_abs_diff", "problems", "pass")}))


def cmd_server_budget(args):
    plain = _body(args.preset, args.n, 0)
    allsw = _body(args.preset, args.n, 4, scoring="full_path", temperature=2.0)
    for name in ("urgency", "angry"):
        allsw["fields"][name]["after"] = ["queue"]
    with httpx.Client(base_url=args.url, timeout=600) as c:
        first = c.post("/v1/decide", json=plain)
        info1 = c.get("/v1/decide/info")
        sw = c.post("/v1/decide", json=allsw)
        info2 = c.get("/v1/decide/info")
    err = sw.json().get("error", {})
    out = {"plain_status": first.status_code, "info_before": info1.json(), "allswitch": {"status": sw.status_code, "body": sw.json()},
           "info_after": info2.json(),
           "pass": first.status_code == 200 and sw.status_code == 400 and err.get("code") == "budget" and err.get("message") == BUDGET_44_28
                   and info1.json()["prefixes"] == info2.json()["prefixes"] and info1.json()["prefix_cache"] == info2.json()["prefix_cache"]}
    dump_json(args.out, out)
    print(json.dumps({"plain": first.status_code, "allswitch": [sw.status_code, err], "prefixes_before": info1.json()["prefixes"],
                      "prefixes_after": info2.json()["prefixes"], "pass": out["pass"]}))


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("requests")
    q.add_argument("--preset", required=True)
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--per-request", type=int, required=True)
    q.add_argument("--skip", type=int, default=0)
    q.add_argument("--debias", type=int, default=0)
    q.add_argument("--temperature", type=float, default=None)
    q.add_argument("--scoring", default="tree")
    q.add_argument("--after", action="append", default=[])
    q.add_argument("--rotate", default=None)
    q.add_argument("--one-state", action="store_true")
    q.add_argument("out")
    s = sub.add_parser("same")
    s.add_argument("ref")
    s.add_argument("run")
    s.add_argument("out")
    s.add_argument("--ignore-engine", default="")
    s.add_argument("--ignore-tokens", default="")
    s.add_argument("--ignore-entry", default="")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("run")
    c.add_argument("out")
    c.add_argument("--max-diff", type=float, default=None)
    k = sub.add_parser("k2")
    k.add_argument("resp")
    k.add_argument("out")
    kf = sub.add_parser("k2fp")
    kf.add_argument("resp")
    kf.add_argument("out")
    p = sub.add_parser("posthoc")
    p.add_argument("t1")
    p.add_argument("t2")
    p.add_argument("out")
    p.add_argument("--t", type=float, required=True)
    d = sub.add_parser("dumpcheck")
    d.add_argument("resp")
    d.add_argument("out")
    d.add_argument("--t", type=float, required=True)
    a = sub.add_parser("allswitch")
    a.add_argument("req")
    a.add_argument("resp")
    a.add_argument("out")
    ce = sub.add_parser("cells")
    ce.add_argument("err")
    ce.add_argument("k2")
    ce.add_argument("out")
    ce.add_argument("--k", type=int, required=True)
    rt = sub.add_parser("retries")
    rt.add_argument("resp")
    rt.add_argument("out")
    rt.add_argument("--min", type=int, default=1)
    i = sub.add_parser("info")
    i.add_argument("resp")
    i.add_argument("out")
    i.add_argument("--expect", required=True)
    k4 = sub.add_parser("k4")
    k4.add_argument("resp")
    k4.add_argument("out")
    k4.add_argument("--k", type=int, required=True)
    for name in ("server-twice", "server-temp", "server-budget", "server-post"):
        sp = sub.add_parser(name)
        sp.add_argument("--preset", required=True)
        sp.add_argument("--n", type=int, required=True)
        sp.add_argument("--url", required=True)
        sp.add_argument("out")
        if name == "server-post":
            sp.add_argument("--debias", type=int, default=0)
    tc = sub.add_parser("server-temp-check")
    tc.add_argument("a")
    tc.add_argument("b")
    tc.add_argument("out")
    tc.add_argument("--t", type=float, required=True)
    sub.add_parser("notes")
    args = ap.parse_args()
    {"requests": cmd_requests, "same": cmd_same, "compare": cmd_compare, "k2": cmd_k2, "k2fp": cmd_k2fp, "posthoc": cmd_posthoc,
     "dumpcheck": cmd_dumpcheck, "allswitch": cmd_allswitch, "cells": cmd_cells, "retries": cmd_retries, "info": cmd_info, "k4": cmd_k4,
     "server-twice": cmd_server_twice, "server-post": cmd_server_post, "server-temp": cmd_server_temp,
     "server-temp-check": cmd_server_temp_check, "server-budget": cmd_server_budget, "notes": cmd_notes}[args.cmd](args)


MODELS3 = ("qwen2.5-0.5b", "qwen3.5-4b", "gemma-4-e4b")
OUTPUT_BUILD, C1_BUILD = "8daa120b5", "3a43e2e62"   # the engine commits whose build produced the outputs (c1-*: C1_BUILD)
# `git log --format='%h %s' -3 OUTPUT_BUILD` in the engine/ that built the outputs
PINNED_LOG = """8daa120b5 decide: option-order debiasing and temperature
3a43e2e62 decide: one value rule for answers, given and lines; keep join-check detail
694ae5934 decide: sequential fields (after)"""


def _j(name):
    return json.loads((HERE / name).read_text())


def _g(x):
    """a number in the notes: 3 significant digits (0 stays 0)."""
    return "0" if x == 0 else f"{x:.3g}"


def _meds(st):
    return " / ".join(_g(v["median_abs_diff"]) for v in st["fields"].values())


def _stat(st):
    return f"medians {_meds(st)}, max {_g(st['max_abs_diff'])}, statistic {'PASS' if st['pass'] else 'FAIL'}"


def _engine_log():
    """(the engine log up to OUTPUT_BUILD, a note): from engine/ when it has that commit; an engine/ built from
    patches/decide/ (git am: the same trees, other commit hashes) gets PINNED_LOG and a note instead."""
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT / "engine"), "log", "--format=%h %s", "-3", OUTPUT_BUILD], capture_output=True,
                       text=True)
    if r.returncode == 0:
        return r.stdout.strip(), ""
    return PINNED_LOG, (f" (pinned: engine/ has no commit {OUTPUT_BUILD}; a build from patches/decide/ has the same code; "
                        "the trees differ only in comments and test text, REPORT-ENGINE.md, Setup)")


def cmd_notes(args):
    log, log_note = _engine_log()
    fp05 = _j("k2fp-check-qwen2.5-0.5b.json")
    bi05 = json.loads((ROOT / "results-engine" / "batch-invariance-qwen2.5-0.5b.json").read_text())
    rows, extra = [], []
    k1 = _j("k1-vs-k0-qwen2.5-0.5b.json")
    rows.append(("K = 1 vs K = 0, Qwen2.5-0.5B, 20 tickets b5 (bound 1e-6)", "`k1-vs-k0-*`",
                 f"max abs diff {_g(k1['max_abs_diff'])}; results, usage and engine blocks identical: {'PASS' if k1['pass'] else 'FAIL'}"))
    for m in MODELS3:
        d = _j(f"k2-check-{m}.json")
        pc = d["k2_engine"]["prefix_cache"]
        rows.append((f"K = 2 vs the per-key mean of rotated K = 0 and K = 0 (5 states), {m}", f"`k2-{m}.jsonl`, `k2-check-{m}.json`",
                     f"{_stat(d['statistic'])} (queue / urgency / angry); K = 2 `hits` {pc['hits']}, `misses` {pc['misses']}, "
                     f"`cached_tokens` {d['k2_usage']['cached_tokens']}; prefixes, dump variants, `order_spread`: "
                     f"{'ok' if d['pass_structure'] else d['problems']}"))
    for m in ("qwen2.5-0.5b", "qwen3.5-4b"):
        t = _j(f"server-rc1-twice-{m}.json")
        c = _j(f"server-rc1-twice-next-vs-normal-{m}.json")
        info = t["info_after_first"]["body"]
        rows.append((f"`rc1-twice` server, K = 2 first request, {m}", f"`server-rc1-twice-{m}.json`, `server-rc1-twice-next-vs-normal-{m}.json`",
                     f"first {t['first']['status']} `{t['first']['body']['error']['message']}`; info right after: `prefixes` slots "
                     f"{[x['slot'] for x in info['prefixes']]} (valid {[x['valid'] for x in info['prefixes']]}), cache entries "
                     f"{info['prefix_cache']['entries']}; next K = 1 request {t['next']['status']}, vs a normal server {_stat(c)}"))
    k4 = _j("k4-check-qwen3.5-9b.json")
    e0, e1 = k4["engine"]
    rows.append(("K = 4, Qwen3.5-9B, `--decide-seqs 21`, the same 5 states twice", "`k4-qwen3.5-9b.jsonl`, `k4-check-qwen3.5-9b.json`",
                 f"`seqs_per_state` {e0['seqs_per_state']}, `states_per_round` {e0['states_per_round']}; `prefix_cache` hits/misses "
                 f"{e0['prefix_cache']['hits']}/{e0['prefix_cache']['misses']} then {e1['prefix_cache']['hits']}/{e1['prefix_cache']['misses']}; "
                 f"max abs(sum - 1) {_g(k4['max_abs_sum_minus_1'])}; second vs first max abs diff {_g(k4['second_vs_first']['max_abs_diff'])}; "
                 f"`prefix_ms` {k4['timings'][0]['prefix_ms']} then {k4['timings'][1]['prefix_ms']}"))
    t2 = _j("t2-vs-t1-posthoc-qwen2.5-0.5b.json")
    rows.append(("T = 2 vs the T = 1 answers scaled post hoc, Qwen2.5-0.5B, 20 tickets b5 (bound 1e-9)", "`t2-vs-t1-posthoc-*`",
                 f"{t2['answers']} answers: max abs diff {_g(t2['max_abs_diff'])} over probabilities, `expected`, `confidence`, `p_true`; "
                 f"values equal: {'PASS' if t2['pass'] else 'FAIL'}"))
    st = _j("server-temp-check-qwen2.5-0.5b.json")
    b = _j("server-temp-2-qwen2.5-0.5b.json")
    rows.append(("server `--decide-temperature 2` vs a default server, same request sequence (Qwen2.5-0.5B)", "`server-temp-*`",
                 f"info `temperature` {b['info']['temperature']}; `/v1/decide` without `temperature`: `engine.temperature` "
                 f"{b['decide']['engine']['temperature']}, post-hoc max abs diff {_g(st['max_abs_diff']['decide'])}; with `temperature: 1`: "
                 f"{b['decide_t1']['engine']['temperature']}, vs the default server {_g(st['max_abs_diff']['decide_t1'])}; `/v1/systemone` vs "
                 f"post-hoc scaling of the default server's {_g(st['max_abs_diff']['systemone'])}: {'PASS' if st['pass'] else 'FAIL'}"))
    a = _j("allswitch-check-qwen2.5-0.5b.json")
    d2, d1 = _j("allswitch-t2-dumpcheck-qwen2.5-0.5b.json"), _j("allswitch-t1-dumpcheck-qwen2.5-0.5b.json")
    ph = _j("allswitch-t2-vs-t1-posthoc-qwen2.5-0.5b.json")
    e = a["engine"][0]
    rows.append(("every switch: full-path + urgency, angry after queue + K = 4 + T = 2, Qwen2.5-0.5B `--decide-seqs 64`, 5 states",
                 "`allswitch-*`",
                 f"200; engine {e['scoring']}, `order_debias` {e['order_debias']}, `temperature` {e['temperature']}, `phases` {e['phases']}, "
                 f"`seqs_per_state` {e['seqs_per_state']}, `states_per_round` {e['states_per_round']}; {a['answers']} answers sum to 1 within "
                 f"{_g(a['max_abs_sum_minus_1'])}, each with `coverage` and `order_spread`, dependents with `given` = the queue value: "
                 f"{'PASS' if a['pass'] else 'FAIL'}; per-variant recomputation from the dump (mean, `order_spread`, mean coverage, then T): "
                 f"max abs diff {_g(max(d2['max_abs_diff'].values()))} (T = 2), {_g(max(d1['max_abs_diff'].values()))} (T = 1); T = 2 vs the T = 1 "
                 f"request scaled post hoc {_g(ph['max_abs_diff'])}, `coverage`, `order_spread`, `given` equal"))
    bu = _j("server-budget-allswitch-qwen3.5-4b.json")

    def slots(info):
        return ", ".join(f"slot {x['slot']} {x['hash']} {x['tokens']} tokens valid {x['valid']}" for x in info["prefixes"]) + \
            f", cache entries {info['prefix_cache']['entries']}"

    rows.append(("the same request on Qwen3.5-4B, server `--decide-seqs 32`, after a plain request", "`server-budget-allswitch-qwen3.5-4b.json`",
                 f"plain {bu['plain_status']}; all-switch {bu['allswitch']['status']} `{bu['allswitch']['body']['error']['code']}`: "
                 f"`{bu['allswitch']['body']['error']['message']}`; info before: {slots(bu['info_before'])}; after: {slots(bu['info_after'])}: "
                 f"{'PASS' if bu['pass'] else 'FAIL'}"))

    c1t, rt = _j("c1-tree-regression-qwen2.5-0.5b.json"), _j("tree-regression-qwen2.5-0.5b.json")
    extra.append(("shared value rule: tree dump vs the Task 0 baseline (bound 1e-6)", "`c1-tree-regression-*`",
                  f"max abs diff {_g(c1t['max_abs_diff'])}"))
    for mode in ("tree", "fp"):
        c = _j(f"c1-after-{mode}-b5-vs-task6-qwen2.5-0.5b.json")
        extra.append((f"shared value rule: the task-6/ `after` {mode} b5 run again", f"`c1-after-{mode}-*`",
                       f"results, usage, engine and dump identical to the task-6/ file, max abs diff {_g(c['max_abs_diff'])}"))
    extra.append(("final build: tree dump vs the Task 0 baseline", "`tree-regression-*`", f"max abs diff {_g(rt['max_abs_diff'])}"))
    for mode in ("tree", "fp"):
        c = _j(f"regress-after-{mode}-b5-vs-task6-qwen2.5-0.5b.json")
        extra.append((f"final build: the task-6/ `after` {mode} b5 run again", f"`regress-after-{mode}-*`",
                       f"identical to the task-6/ file apart from the new keys (engine `order_debias`, `temperature`; `tokens.prefixes`; entry "
                       f"`variant`), max abs diff {_g(c['max_abs_diff'])}"))
    for m in MODELS3:
        d = _j(f"k2fp-check-{m}.json")
        extra.append((f"K = 2 sequence in full-path mode (per-variant rows in the dump), {m}", f"`k2fp-*-{m}*`",
                      f"variant 0 (slot 0, resident) vs plain K = 0: {_stat(d['variant0_vs_plain_k0'])}; variant 1 (slot 1, restored from "
                      f"slot 0's snapshot) vs rotated K = 0: {_stat(d['variant1_vs_rotated_k0'])}; K = 2 vs the K = 0 mean: "
                      f"{_stat(d['k2_vs_mean_of_k0'])}; `order_spread` vs the dump {_g(d['order_spread_vs_dump_max_abs_diff'])}; hits "
                      f"{d['k2_engine']['prefix_cache']['hits']}"))
    inf = _j("info-check-qwen2.5-0.5b.json")
    extra.append(("`--info` with a body: K = 0, 2, 4 tree and K = 4 full-path + after + T = 2 (`--decide-seqs 64`)", "`info-*`",
                  f"(`seqs_per_state`, `states_per_round`) = {inf['got']}: {'PASS' if inf['pass'] else 'FAIL'}"))
    ce = _j("cells-k2-check-qwen2.5-0.5b.json")
    extra.append(("per-state cell check of a K = 2 request (one state of all 80 tickets, `-c 4096`)", "`cells-k2-*`",
                  f"{ce['error']['code']}: `{ce['error']['message']}`; need = held + prefixes ({ce['prefix_tokens_of_the_k2_dump']} as in the K = 2 "
                  f"dump) + 2 x (state + tail + branches): {'PASS' if ce['pass'] else 'FAIL'}"))

    rr = _j("fault-rc1-k2-b5-seqs17-qwen2.5-0.5b.retries.json")
    rc = _j("fault-rc1-k2-vs-normal-qwen2.5-0.5b.json")
    extra.append(("`LLAMA_DECIDE_FAULT=rc1`, K = 2, 20 tickets b5, `--decide-seqs 17` (1 state per round) vs a normal run", "`fault-rc1-k2-*`, `k2-b5-seqs17-*`",
                  f"retries per request {rr['retries']}, rounds {rr['rounds']}; answers max abs diff {_g(rc['max_abs_diff'])}: "
                  f"{'PASS' if rr['pass'] and rc['pass_all'] else 'FAIL'}"))

    cost = []
    for m in MODELS3:
        for name, x in zip(("rotated K = 0 (prefix built)", "K = 0 (prefix built)", "K = 2 (slot 0 resident, slot 1 restored)"),
                           load(HERE / f"k2-{m}.jsonl")):
            t = x["timings"]
            cost.append(f"| {m} | {name} | {t['prefix_ms']:.1f} | {t['prefill_ms'] - t['prefix_ms']:.1f} | {t['scoring_ms']:.1f} | "
                        f"{t['total_ms']:.1f} | {t['rounds']} | {t['decodes']} | {x['usage']['prompt_tokens']} | {x['usage']['cached_tokens']} |")

    def table(rs):
        return "\n".join(["| check | files | result |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rs])

    text = f"""# Sub-project 3 Task 7 live checks (option-order debiasing and temperature)

Written by `checks.py notes` from the check files of this directory; `run.sh` produced every file (commands per check in
its functions) with the engine build of {OUTPUT_BUILD} (the `c1-*` files: {C1_BUILD}), except the second-build logs
`rebuild-build.log`, `rebuild-ctest.log` and `rebuild-test-engine.log` (build 5e9e030d3, model-free). Engine log up to {OUTPUT_BUILD}{log_note}:
```
{log}
```
CLI = `engine/build/bin/llama-decide -c 4096 -ngl 99 -fa on`, KV f16 for qwen2.5-0.5b, q8_0 otherwise (`models-engine.ini`);
`--decide-seqs` 64 (Qwen2.5-0.5B, Gemma 4 E4B), 32 (Qwen3.5-4B), 21 (Qwen3.5-9B). Servers: the same flags plus `-np 1 --jinja
--port 8097`. Requests: `decide_client.build_request` (prompt variant `default`) on the first test tickets; b5 = lines of 5
states. "The statistic": per field median abs diff <= 0.01 and no argmax disagreement over a 0.15 reference margin
(`sp3_metrics.compare`); medians are given as queue / urgency / angry.

## Checks

{table(rows)}

## Extra checks

{table(extra)}

Qwen2.5-0.5B `angry` in the per-variant full-path comparisons (medians {_g(fp05['variant0_vs_plain_k0']['fields']['angry']['median_abs_diff'])}, {_g(fp05['variant1_vs_rotated_k0']['fields']['angry']['median_abs_diff'])}) is the size of that model's known
batch-composition effect on `angry` (tree b5 vs b1: {_g(bi05['fields']['angry']['median_abs_diff'])}, sub-project 1, `results-engine/batch-invariance-qwen2.5-0.5b.json`; Tasks 5, 6); Qwen3.5-4B and Gemma 4 E4B are the
binding models for the statistic here.

## Cost of K = 2 (the three lines of the K = 2 check, 5 states each; ms)

| model | line | prefix_ms | trunks (prefill - prefix) | scoring_ms | total_ms | rounds | decodes | prompt_tokens | cached_tokens |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(cost) + "\n"
    (HERE / "notes.md").write_text(text)
    print(f"wrote {HERE / 'notes.md'}")


if __name__ == "__main__":
    main()
