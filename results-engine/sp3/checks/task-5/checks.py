"""Sub-project 3 Task 5 live checks (spec 4, 12.3 full-path items): request lines and response checks.

  uv run results-engine/sp3/checks/task-5/checks.py requests --preset P --n N --per-request K [--scoring S] OUT
      N test tickets as request lines of K states each (decide_client.build_request, prompt variant `default`),
      options.scoring = S (default full_path).
  uv run results-engine/sp3/checks/task-5/checks.py sums RESP.jsonl OUT.json
      full-path responses: per field, the probabilities sum to 1 (|sum - 1| <= 1e-6) and coverage is in (0, 1 + 1e-6];
      engine.scoring = full_path; coverage min/median/max per field; engine and usage numbers per line.
  uv run results-engine/sp3/checks/task-5/checks.py dumpcheck RESP.jsonl OUT.json
      responses with --dump-tokens: from the full-path entries alone (tokens, rows, logp) recompute every option's
      log P (spec 4.1), the distribution and the coverage, and compare with the engine's answers (max abs diff <= 1e-9);
      every trie node is read once per state and field (distinct node ids, = results[].usage.scored_tokens).
  uv run results-engine/sp3/checks/task-5/checks.py split B5.jsonl B1.jsonl OUT.json --row-cap C [--n-batch N]
      row-cap split: B5 = full-path responses of 5-state requests run with --dump-tokens, B1 = the same states one per
      request. Per B5 line: the branch decodes (timings.decodes - rounds - 1 when the line built the prefix), the chunks
      predicted from the dumped branch sequences (flush at N tokens or C rows) and the sequences a chunk boundary splits.
      Pass: B5 answers match B1 under the statistic and every B5 line has >= 2 branch decodes.
  uv run results-engine/sp3/checks/task-5/checks.py rowcompare SPLIT.jsonl WHOLE.jsonl OUT.json --row-cap C [--n-batch N]
      the same requests with --dump-tokens, SPLIT with a branch phase cut by the row cap, WHOLE in one decode: per dumped
      row (state, field, node), the max abs diff in probability of its values (children, end, merged); reported for all
      rows and for the rows of split sequences that were decoded after the chunk boundary (a row read at the wrong batch
      index would differ by about 1, batch noise by about 0.01).
  uv run results-engine/sp3/checks/task-5/checks.py compare REF.jsonl RUN.jsonl OUT.json [--max-diff X]
      every answer of RUN vs REF (responses in the same order) with sp3_metrics.compare (the statistic).
  uv run results-engine/sp3/checks/task-5/checks.py rounds RESP.jsonl OUT.json --spr S
      engine.states_per_round == S and timings.rounds == ceil(states / S) on every line.
  uv run results-engine/sp3/checks/task-5/checks.py sale-requests OUT_DIR
      a choice field ["sale", "sales"] (tree mode rejects it, spec 4.2): OUT_DIR/sale-full_path.jsonl and sale-tree.jsonl.
  uv run results-engine/sp3/checks/task-5/checks.py sale-check FULL_OUT.jsonl TREE_ERR.txt OUT.json
      full-path: 200 lines whose q distribution sums to 1 with coverage in (0, 1]; tree: the 400 invalid_schema naming both keys.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from decide_client import TEST_SET, build_request  # noqa: E402
from sp3_metrics import compare  # noqa: E402

COVERAGE_EPS = 1e-6  # coverage is not clamped; float rounding can put it just above 1


def tickets(n):
    return [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()][:n]


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


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


def answer_probs(ans):
    """The answer's distribution as a list in declaration order (bool: [p_true, 1 - p_true])."""
    if "p_true" in ans:
        return [ans["p_true"], 1.0 - ans["p_true"]]
    p = ans["probabilities"]
    return list(p.values()) if isinstance(p, dict) else list(p)


def cmd_requests(args):
    texts = tickets(args.n)
    lines = []
    for i in range(0, len(texts), args.per_request):
        body = build_request(args.preset, texts[i : i + args.per_request])
        body["options"]["scoring"] = args.scoring
        lines.append(body)
    Path(args.out).write_text("".join(json.dumps(l) + "\n" for l in lines))
    print(f"wrote {args.out} ({len(lines)} lines, scoring {args.scoring})")


def cmd_sums(args):
    resps = load(args.resp)
    worst_sum, bad, coverage = 0.0, [], {}
    for li, r in enumerate(resps):
        if r["engine"]["scoring"] != "full_path":
            bad.append(f"line {li}: engine.scoring {r['engine']['scoring']}")
        for si, res in enumerate(r["results"]):
            for name, ans in res["answers"].items():
                p = answer_probs(ans)
                if "probabilities" in ans:
                    worst_sum = max(worst_sum, abs(sum(p) - 1.0))
                if not all(0.0 <= x <= 1.0 for x in p):
                    bad.append(f"line {li} state {si} {name}: probability outside [0, 1]")
                c = ans.get("coverage")
                if c is None or not (0.0 < c <= 1.0 + COVERAGE_EPS):
                    bad.append(f"line {li} state {si} {name}: coverage {c}")
                else:
                    coverage.setdefault(name, []).append(c)
    out = {
        "resp": args.resp,
        "lines": len(resps),
        "states": sum(len(r["results"]) for r in resps),
        "max_abs_sum_minus_1": worst_sum,
        "coverage": {f: {"min": min(v), "median": statistics.median(v), "max": max(v), "n": len(v)} for f, v in coverage.items()},
        "engine": [r["engine"] for r in resps],
        "usage": [r["usage"] for r in resps],
        "timings": [r["timings"] for r in resps],
        "problems": bad,
        "pass": worst_sum <= 1e-6 and not bad,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"states": out["states"], "max_abs_sum_minus_1": worst_sum, "coverage": out["coverage"],
                      "problems": bad[:5], "pass": out["pass"]}))


def _stem_len(branches):
    """Stem length of one field's full-path entries: option 0 reads the root at the stem's last token."""
    first = next(b for b in branches if b["option"] == 0)
    return first["rows"][0] + 1


def _log_add(a, b):
    hi, lo = max(a, b), min(a, b)
    return hi if hi == -math.inf else hi + math.log1p(math.exp(lo - hi))


def _num(x):
    return -math.inf if x is None else x  # the engine writes -inf as null


def recompute_field(branches):
    """spec 4.1 from one field's dumped entries: (probabilities in option order, coverage, node ids read)."""
    stem = _stem_len(branches)
    by_prefix, rows_by_node = {}, {}
    for b in branches:
        for idx, row in zip(b["rows"], b["logp"]):
            key = tuple(b["tokens"][: idx + 1])
            if key in by_prefix or row["node"] in rows_by_node:
                raise ValueError(f"node {row['node']} read twice")
            by_prefix[key] = row["node"]
            rows_by_node[row["node"]] = row
    logp = []
    for b in sorted(branches, key=lambda x: x["option"]):
        toks = b["tokens"]
        if tuple(toks[:stem]) != tuple(branches[0]["tokens"][:stem]):
            raise ValueError("options disagree on the stem")
        path = toks[stem:]
        lp = 0.0
        for i, t in enumerate(path):
            parent = rows_by_node[by_prefix[tuple(toks[: stem + i])]]
            child_lp = _num(parent["children"][str(t)])
            if i + 1 < len(path):
                lp += child_lp
            else:
                leaf = rows_by_node[by_prefix[tuple(toks[: stem + i + 1])]]
                lp += _log_add(child_lp + _num(leaf["end"]), _num(parent.get("merged", {}).get(str(t))))
        logp.append(lp)
    log_z = -math.inf
    for v in logp:
        log_z = _log_add(log_z, v)
    return [math.exp(v - log_z) for v in logp], math.exp(log_z), sorted(rows_by_node)


def cmd_dumpcheck(args):
    resps = load(args.resp)
    worst, problems, n_fields = 0.0, [], 0
    for li, r in enumerate(resps):
        for si, (res, st) in enumerate(zip(r["results"], r["tokens"]["states"])):
            by_field = {}
            for b in st["branches"]:
                if b.get("mode") != "full_path":
                    problems.append(f"line {li} state {si}: entry mode {b.get('mode')}")
                by_field.setdefault(b["field"], []).append(b)
            n_rows = 0
            for name, ans in res["answers"].items():
                probs, cov, nodes = recompute_field(by_field[name])
                n_rows += len(nodes)
                if nodes != list(range(len(nodes))):
                    problems.append(f"line {li} state {si} {name}: nodes read {nodes}")
                eng = answer_probs(ans)
                d = max([abs(a - b) for a, b in zip(probs, eng)] + [abs(cov - ans["coverage"])])
                worst = max(worst, d)
                n_fields += 1
            if n_rows != res["usage"]["scored_tokens"]:
                problems.append(f"line {li} state {si}: {n_rows} rows dumped, scored_tokens {res['usage']['scored_tokens']}")
    out = {"resp": args.resp, "fields_checked": n_fields, "max_abs_diff": worst, "problems": problems,
           "pass": worst <= 1e-9 and not problems}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


def predicted_chunks(state_dumps, n_batch, row_cap):
    """decode_chunks over the branch phase of one round: entries in dump order (state, field, option); returns the
    chunk count and every sequence a chunk boundary splits (state, field, option, tokens before / after the boundary)."""
    seqs = [(si, b) for si, st in enumerate(state_dumps) for b in st["branches"]]
    chunks, tokens, rows, split = 0, 0, 0, []
    total = sum(len(b["tokens"]) for _, b in seqs)
    seen = 0
    for si, b in seqs:
        row_at = set(b["rows"])
        for i in range(len(b["tokens"])):
            tokens += 1
            rows += i in row_at
            seen += 1
            if tokens == n_batch or rows == row_cap or seen == total:
                chunks += 1
                if i + 1 < len(b["tokens"]):
                    split.append({"state": si, "field": b["field"], "option": b["option"], "tokens_before": i + 1,
                                  "tokens_after": len(b["tokens"]) - i - 1})
                tokens, rows = 0, 0
    return chunks, split


def _row_probs(row):
    """(kind, key) -> probability for one dumped row."""
    out = {}
    for kind in ("children", "merged"):
        for tok, v in row.get(kind, {}).items():
            out[(kind, tok)] = math.exp(_num(v))
    if "end" in row:
        out[("end", "")] = math.exp(_num(row["end"]))
    return out


def cmd_rowcompare(args):
    split, whole = load(args.split), load(args.whole)
    worst_all, after, n_rows = 0.0, [], 0
    for li, (a, b) in enumerate(zip(split, whole)):
        # rows of `a` decoded after a chunk boundary inside their own sequence
        late = set()
        seqs = [(si, br) for si, st in enumerate(a["tokens"]["states"]) for br in st["branches"]]
        total, seen, tokens, rows = sum(len(br["tokens"]) for _, br in seqs), 0, 0, 0
        for si, br in seqs:
            cut = None
            for i in range(len(br["tokens"])):
                tokens += 1
                rows += i in br["rows"]
                seen += 1
                if tokens == args.n_batch or rows == args.row_cap or seen == total:
                    tokens, rows = 0, 0
                    if i + 1 < len(br["tokens"]):
                        cut = i
            if cut is not None:
                for idx, row in zip(br["rows"], br["logp"]):
                    if idx > cut:
                        late.add((si, br["field"], row["node"]))
        vals = {}
        for tag, resp in (("a", a), ("b", b)):
            for si, st in enumerate(resp["tokens"]["states"]):
                for br in st["branches"]:
                    for row in br["logp"]:
                        vals.setdefault((si, br["field"], row["node"]), {})[tag] = _row_probs(row)
        for key, v in vals.items():
            d = max(abs(v["a"][k] - v["b"][k]) for k in v["a"])
            n_rows += 1
            worst_all = max(worst_all, d)
            if key in late:
                after.append({"line": li, "state": key[0], "field": key[1], "node": key[2], "max_abs_diff": d,
                              "values_split": {f"{k[0]}:{k[1]}": p for k, p in v["a"].items()},
                              "values_whole": {f"{k[0]}:{k[1]}": p for k, p in v["b"].items()}})
    out = {"split": args.split, "whole": args.whole, "row_cap": args.row_cap, "rows_compared": n_rows,
           "max_abs_diff_all_rows": worst_all, "rows_after_boundary": after,
           "max_abs_diff_after_boundary": max((x["max_abs_diff"] for x in after), default=None)}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"rows_compared": n_rows, "max_abs_diff_all_rows": worst_all, "rows_after_boundary": len(after),
                      "max_abs_diff_after_boundary": out["max_abs_diff_after_boundary"]}))


def cmd_split(args):
    b5, b1 = load(args.b5), load(args.b1)
    cmp = compare(entries(b1), entries(b5))
    lines, ok_decodes = [], True
    for r in b5:
        built = r["usage"]["cached_tokens"] == 0  # the prefix was decoded by this request (one decode: < n_batch tokens)
        branch_decodes = r["timings"]["decodes"] - r["timings"]["rounds"] - (1 if built else 0)
        chunks, split = predicted_chunks(r["tokens"]["states"], args.n_batch, args.row_cap)
        ok_decodes = ok_decodes and branch_decodes >= 2
        lines.append({"states": len(r["results"]), "rounds": r["timings"]["rounds"], "decodes": r["timings"]["decodes"],
                      "prefix_built": built, "branch_decodes": branch_decodes, "predicted_branch_chunks": chunks,
                      "rows_read": r["usage"]["scored_tokens"], "split_sequences": split,
                      "states_per_round": r["engine"]["states_per_round"]})
    out = {"b5": args.b5, "b1": args.b1, "row_cap": args.row_cap, "n_batch": args.n_batch, "b5_vs_b1": cmp, "b5_lines": lines,
           "pass": cmp["pass"] and ok_decodes}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"statistic": cmp["pass"], "max_abs_diff": cmp["max_abs_diff"],
                      "medians": {f: v["median_abs_diff"] for f, v in cmp["fields"].items()},
                      "branch_decodes": [l["branch_decodes"] for l in lines],
                      "predicted": [l["predicted_branch_chunks"] for l in lines],
                      "split_sequences": [len(l["split_sequences"]) for l in lines], "pass": out["pass"]}))


def cmd_compare(args):
    ref, run = load(args.ref), load(args.run)
    cmp = compare(entries(ref), entries(run))
    out = {"ref": args.ref, "run": args.run, **cmp}
    ok = cmp["pass"]
    if args.max_diff is not None:
        out["max_diff_bound"] = args.max_diff
        ok = ok and cmp["max_abs_diff"] <= args.max_diff
    out["pass_all"] = ok
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_tickets": cmp["n_tickets"], "max_abs_diff": cmp["max_abs_diff"], "statistic": cmp["pass"], "pass_all": ok}))


def cmd_rounds(args):
    resps = load(args.resp)
    lines, ok = [], True
    for r in resps:
        n = len(r["results"])
        good = r["engine"]["states_per_round"] == args.spr and r["timings"]["rounds"] == math.ceil(n / args.spr)
        ok = ok and good
        lines.append({"states": n, "states_per_round": r["engine"]["states_per_round"], "rounds": r["timings"]["rounds"],
                      "seqs_per_state": r["engine"]["seqs_per_state"], "decodes": r["timings"]["decodes"], "ok": good})
    out = {"resp": args.resp, "spr": args.spr, "lines": lines, "pass": ok}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"lines": [(l["states"], l["states_per_round"], l["rounds"]) for l in lines], "pass": ok}))


SALE_STATE = "Hi, I saw your spring sale ad. Does the discount also apply to annual plans?"


def cmd_sale_requests(args):
    out_dir = Path(args.out_dir)
    for scoring in ("full_path", "tree"):
        body = {"instructions": "Answer about the ticket.",
                "fields": {"q": {"type": "choice", "description": "Which word fits the ticket?", "options": ["sale", "sales"]}},
                "states": [SALE_STATE], "options": {"scoring": scoring}}
        (out_dir / f"sale-{scoring}.jsonl").write_text(json.dumps(body) + "\n")
    print(f"wrote {out_dir}/sale-full_path.jsonl, sale-tree.jsonl")


def cmd_sale_check(args):
    full = load(args.full)
    err = json.loads(Path(args.tree_err).read_text().strip().splitlines()[-1])["error"]
    q = full[0]["results"][0]["answers"]["q"]
    s = sum(q["probabilities"].values())
    full_ok = abs(s - 1.0) <= 1e-6 and 0.0 < q["coverage"] <= 1.0 + COVERAGE_EPS and full[0]["engine"]["scoring"] == "full_path"
    tree_ok = err["code"] == "invalid_schema" and '"sale"' in err["message"] and '"sales"' in err["message"] and err.get("field") == "q"
    out = {"full_path": {"answer": q, "sum": s, "engine": full[0]["engine"]}, "tree_error": err, "pass": full_ok and tree_ok}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"full_path": q, "tree_error": err, "pass": out["pass"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("requests")
    q.add_argument("--preset", required=True)
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--per-request", type=int, required=True)
    q.add_argument("--scoring", default="full_path")
    q.add_argument("out")
    s = sub.add_parser("sums")
    s.add_argument("resp")
    s.add_argument("out")
    d = sub.add_parser("dumpcheck")
    d.add_argument("resp")
    d.add_argument("out")
    sp = sub.add_parser("split")
    sp.add_argument("b5")
    sp.add_argument("b1")
    sp.add_argument("out")
    sp.add_argument("--row-cap", type=int, required=True)
    sp.add_argument("--n-batch", type=int, default=2048)
    rc = sub.add_parser("rowcompare")
    rc.add_argument("split")
    rc.add_argument("whole")
    rc.add_argument("out")
    rc.add_argument("--row-cap", type=int, required=True)
    rc.add_argument("--n-batch", type=int, default=2048)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("run")
    c.add_argument("out")
    c.add_argument("--max-diff", type=float, default=None)
    r = sub.add_parser("rounds")
    r.add_argument("resp")
    r.add_argument("out")
    r.add_argument("--spr", type=int, required=True)
    sr = sub.add_parser("sale-requests")
    sr.add_argument("out_dir")
    sc = sub.add_parser("sale-check")
    sc.add_argument("full")
    sc.add_argument("tree_err")
    sc.add_argument("out")
    args = ap.parse_args()
    {"requests": cmd_requests, "sums": cmd_sums, "dumpcheck": cmd_dumpcheck, "split": cmd_split, "rowcompare": cmd_rowcompare,
     "compare": cmd_compare,
     "rounds": cmd_rounds, "sale-requests": cmd_sale_requests, "sale-check": cmd_sale_check}[args.cmd](args)


if __name__ == "__main__":
    main()
