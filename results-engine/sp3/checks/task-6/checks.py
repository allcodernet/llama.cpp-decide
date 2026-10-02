"""Sub-project 3 Task 6 live checks (spec 5, 12.3 `after` items): request lines and response checks.

  uv run results-engine/sp3/checks/task-6/checks.py requests --preset P --n N --per-request K [--scoring S]
                                                    [--after F=A[,A]]... [--tag-field] OUT
      N test tickets as request lines of K states each (decide_client.build_request, prompt variant `default`),
      options.scoring = S (default tree), field F after the fields A. --tag-field adds a choice field "tag" with the options
      "a" and "<b" after urgency (tree mode quote-splits it: its first children ' "' and ' "<' share a prefix).
  uv run results-engine/sp3/checks/task-6/checks.py after REQ.jsonl RESP.jsonl OUT.json
      responses run with --dump-tokens: engine.phases = 1 + the deepest level; every dependent answer's `given` holds the
      values its ancestors returned for that state (transitive, declaration order), other answers have none; every dump
      entry carries its field's level as `phase`, entries come phase by phase; every dependent entry's `given` equals the
      answer's and its `lines` is the ancestors' stem text + JSON value + ",\n"; level-0 entries have neither; answers sum
      to 1 (full-path: coverage in (0, 1]).
  uv run results-engine/sp3/checks/task-6/checks.py detok REQ.jsonl RESP.jsonl OUT.json --url URL
      a server with the same model: /detokenize of every dependent entry's tokens starts with its lines and is lines + stem
      text + the option's value (full-path) or a prefix of that for an option through the node (tree); /tokenize of the
      lines tells whether the entry's tokens start with the lines' own tokens (else a token spans the lines and the key).
  uv run results-engine/sp3/checks/task-6/checks.py records REQ.jsonl RESP.jsonl --preset P --tag T
      a tree-mode response run with --dump-tokens as reference_check.py dump records (one per state and field, ticket = state
      index over all lines; prompt = prefix + state + tail + the entry's tokens, which hold the lines for a dependent field),
      written to results-engine/reference-T-P-dump.jsonl for `reference_check.py compare --tag T`.
  uv run results-engine/sp3/checks/task-6/checks.py compare REF.jsonl RUN.jsonl OUT.json [--max-diff X] [--fields F,G]
      every answer of RUN vs REF with sp3_metrics.compare (the statistic); --fields restricts the fields compared.
  uv run results-engine/sp3/checks/task-6/checks.py retries RESP.jsonl OUT.json [--min M]
      timings.retries per line; pass when every line has >= M (M = 0: exactly 0 on every line).

Model-free, from committed outputs only:
  uv run results-engine/sp3/checks/task-6/checks.py dump-diff REF.jsonl RUN.jsonl OUT.json
      the same requests with --dump-tokens on two builds, RUN adding `phase` to the dump entries: results, usage and the
      tokens dump without `phase` identical, the phases found, the engine keys RUN adds, decodes per line.
  uv run results-engine/sp3/checks/task-6/checks.py records-compare A-dump.jsonl B-dump.jsonl OUT.json --fields F[,G]
      reference_check.py dump records of the same tickets: for the fields F, the records whose node prompt tokens are
      identical in A and B, and the engine distributions compared with sp3_metrics.compare (A is the reference).
  uv run results-engine/sp3/checks/task-6/checks.py cost OUT.json RESP.jsonl...
      b5 responses: per file the medians over the lines after the first (prefix resident) of scoring_ms, total_ms, decodes,
      prompt_tokens and scored_tokens, with engine.phases and states_per_round.
  uv run results-engine/sp3/checks/task-6/checks.py lines REQ.jsonl AFTER.jsonl PLAIN.jsonl OUT.json
                                                    [--budget-plain ERR --budget-after ERR]
      AFTER and PLAIN: the same requests with and without `after` (same model and mode) run with --dump-tokens. Every
      dependent entry ends with the tokens of its lines-free entry (same field and node or option); the tokens before them
      are the lines. Full-path, one ancestor: the lines start with the ancestor's entry for the answered option without its
      last token (the closing quote), the rest is the text after the value. Per line, prompt_tokens(AFTER) -
      prompt_tokens(PLAIN) = the lines tokens of all dependent entries. With the budget messages of the first request (spec
      5.4 cell check): branches(AFTER) - branches(PLAIN) per dependent sequence and its slack over the longest exact lines.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from decide_client import TEST_SET, build_request  # noqa: E402
from reference_check import engine_probs  # noqa: E402
from sp3_metrics import compare  # noqa: E402

COVERAGE_EPS = 1e-6
TAG_FIELD = {"type": "choice", "description": "Answer a when the ticket names a price or an amount, <b otherwise.",
             "options": ["a", "<b"]}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def jtext(v):
    """JSON text of a value as the engine writes it (nlohmann: UTF-8, no ASCII escaping)."""
    return json.dumps(v, ensure_ascii=False)


def stem_text(name):
    return "  " + jtext(name) + ": "


def option_values(fdef):
    """The answer values of a field's options in declaration order: keys, level indices or true/false."""
    if fdef["type"] == "choice":
        return list(fdef["options"])
    if fdef["type"] == "score":
        return list(range(len(fdef["levels"])))
    return [True, False]


def ancestors(fields):
    """name -> ancestors in declaration order (transitive closure of `after`), and name -> level."""
    names = list(fields)
    anc, level = {}, {}

    def close(n, seen):
        for a in fields[n].get("after", []):
            if a not in seen:
                seen.add(a)
                close(a, seen)
        return seen

    def lvl(n):
        if n not in level:
            level[n] = 1 + max((lvl(a) for a in fields[n].get("after", [])), default=-1)
        return level[n]

    for n in names:
        s = close(n, set())
        anc[n] = [m for m in names if m in s]
        lvl(n)
    return anc, level


def answer_value(ans):
    return ans["value"]


def cmd_requests(args):
    texts = [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()][: args.n]
    lines = []
    for i in range(0, len(texts), args.per_request):
        body = build_request(args.preset, texts[i : i + args.per_request])
        body["options"]["scoring"] = args.scoring
        if args.tag_field:
            f = body["fields"]
            body["fields"] = {"queue": f["queue"], "urgency": f["urgency"], "tag": dict(TAG_FIELD), "angry": f["angry"]}
        for spec in args.after:
            name, parents = spec.split("=")
            body["fields"][name]["after"] = parents.split(",")
        lines.append(body)
    Path(args.out).write_text("".join(json.dumps(l) + "\n" for l in lines))
    print(f"wrote {args.out} ({len(lines)} lines, scoring {args.scoring}, after {args.after}, tag field {args.tag_field})")


def cmd_after(args):
    reqs, resps = load(args.req), load(args.resp)
    problems, n_dep_entries, n_states, worst_sum, per_field_entries = [], 0, 0, 0.0, {}
    if len(reqs) != len(resps):
        problems.append(f"{len(reqs)} requests, {len(resps)} responses")
    for li, (req, r) in enumerate(zip(reqs, resps)):
        fields = req["fields"]
        anc, level = ancestors(fields)
        want_phases = 1 + max(level.values())
        if r["engine"].get("phases") != want_phases:
            problems.append(f"line {li}: engine.phases {r['engine'].get('phases')}, want {want_phases}")
        for si, (res, st) in enumerate(zip(r["results"], r["tokens"]["states"])):
            n_states += 1
            a = res["answers"]
            given = {n: [(g, answer_value(a[g])) for g in anc[n]] for n in fields}
            lines = {n: "".join(stem_text(g) + jtext(answer_value(a[g])) + ",\n" for g in anc[n]) for n in fields}
            for n in fields:
                got = a[n].get("given")
                if anc[n]:
                    if got is None or list(got.items()) != given[n]:
                        problems.append(f"line {li} state {si} {n}: given {got}, want {dict(given[n])}")
                elif got is not None:
                    problems.append(f"line {li} state {si} {n}: level-0 answer has given {got}")
                if "probabilities" in a[n]:
                    p = a[n]["probabilities"]
                    worst_sum = max(worst_sum, abs(sum(p.values() if isinstance(p, dict) else p) - 1.0))
                elif not 0.0 <= a[n]["p_true"] <= 1.0:
                    problems.append(f"line {li} state {si} {n}: p_true {a[n]['p_true']}")
                if r["engine"]["scoring"] == "full_path" and not (0.0 < a[n].get("coverage", -1) <= 1.0 + COVERAGE_EPS):
                    problems.append(f"line {li} state {si} {n}: coverage {a[n].get('coverage')}")
            last_phase = 0
            seen = set()
            for b in st["branches"]:
                n = b["field"]
                seen.add(n)
                per_field_entries.setdefault(n, set()).add(sum(1 for x in st["branches"] if x["field"] == n))
                if b.get("phase") != level[n]:
                    problems.append(f"line {li} state {si} {n}: entry phase {b.get('phase')}, level {level[n]}")
                if b.get("phase", 0) < last_phase:
                    problems.append(f"line {li} state {si} {n}: entry of phase {b.get('phase')} after phase {last_phase}")
                last_phase = max(last_phase, b.get("phase", 0))
                if anc[n]:
                    n_dep_entries += 1
                    if b.get("given") is None or list(b["given"].items()) != given[n]:
                        problems.append(f"line {li} state {si} {n}: entry given {b.get('given')}, want {dict(given[n])}")
                    if b.get("lines") != lines[n]:
                        problems.append(f"line {li} state {si} {n}: entry lines {b.get('lines')!r}, want {lines[n]!r}")
                elif "given" in b or "lines" in b:
                    problems.append(f"line {li} state {si} {n}: level-0 entry has given/lines")
            if seen != set(fields):
                problems.append(f"line {li} state {si}: entries for {sorted(seen)}, fields {sorted(fields)}")
    out = {"req": args.req, "resp": args.resp, "lines": len(resps), "states": n_states, "dependent_entries": n_dep_entries,
           "entries_per_field": {k: sorted(v) for k, v in per_field_entries.items()},
           "phases": [r["engine"].get("phases") for r in resps], "max_abs_sum_minus_1": worst_sum,
           "engine": [r["engine"] for r in resps], "timings": [r["timings"] for r in resps], "usage": [r["usage"] for r in resps],
           "problems": problems, "pass": not problems and worst_sum <= 1e-6 and n_dep_entries > 0}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"states": n_states, "dependent_entries": n_dep_entries, "entries_per_field": out["entries_per_field"],
                      "phases": out["phases"], "max_abs_sum_minus_1": worst_sum, "problems": problems[:5], "pass": out["pass"]}))


def cmd_detok(args):
    reqs, resps = load(args.req), load(args.resp)
    client = httpx.Client(base_url=args.url, timeout=60)
    problems, checked, kept = [], 0, {}
    for li, (req, r) in enumerate(zip(reqs, resps)):
        fields = req["fields"]
        anc, _ = ancestors(fields)
        for si, st in enumerate(r["tokens"]["states"]):
            for b in st["branches"]:
                n = b["field"]
                if not anc[n]:
                    continue
                text = client.post("/detokenize", json={"tokens": b["tokens"]}).json()["content"]
                vals = [jtext(v) for v in option_values(fields[n])]
                full = [b["lines"] + stem_text(n) + v for v in vals]
                if b["mode"] == "full_path":
                    ok = text == full[b["option"]]
                else:
                    through = sorted({o for opts in b["options"] for o in opts})
                    ok = any(full[o].startswith(text) for o in through)
                ok = ok and text.startswith(b["lines"])
                if not ok:
                    problems.append(f"line {li} state {si} {n}: detokenized {text!r}")
                lt = client.post("/tokenize", json={"content": b["lines"], "add_special": False, "parse_special": False}).json()["tokens"]
                key = f"{n}: {b['lines']!r}"
                kept.setdefault(key, set()).add(b["tokens"][: len(lt)] == lt)
                checked += 1
    out = {"req": args.req, "resp": args.resp, "entries_checked": checked,
           "lines_tokens_kept_as_prefix": {k: sorted(v) for k, v in kept.items()}, "problems": problems,
           "pass": not problems and checked > 0}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"entries_checked": checked, "kept": {k: sorted(v) for k, v in kept.items()}, "problems": problems[:5],
                      "pass": out["pass"]}))


def record_keys(fdef):
    if fdef["type"] == "choice":
        return list(fdef["options"])
    if fdef["type"] == "score":
        return [str(i) for i in range(len(fdef["levels"]))]
    return ["true", "false"]


def cmd_records(args):
    reqs, resps = load(args.req), load(args.resp)
    records, ticket = [], 0
    for req, r in zip(reqs, resps):
        prefix = r["tokens"]["prefix"]
        for res, st in zip(r["results"], r["tokens"]["states"]):
            for name, fdef in req["fields"].items():
                nodes = [{"node": b["node"], "prompt_tokens": prefix + st["state"] + st["tail"] + b["tokens"],
                          "children": b["children"], "options": b["options"], "phase": b["phase"], "lines": b.get("lines", "")}
                         for b in st["branches"] if b["field"] == name]
                records.append({"ticket": ticket, "field": name, "type": fdef["type"], "keys": record_keys(fdef),
                                "engine": res["answers"][name], "nodes": nodes})
            ticket += 1
    out = ROOT / "results-engine" / f"reference-{args.tag}-{args.preset}-dump.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in records))
    print(f"wrote {out} ({len(records)} records, {ticket} states)")


def entries(responses, only=None):
    """sp3_metrics entries (ticket, field, {key: p}) over every result of every response, in order."""
    out, i = [], 0
    for resp in responses:
        for res in resp["results"]:
            for name, a in res["answers"].items():
                if only and name not in only:
                    continue
                if "p_true" in a:
                    out.append((i, name, {"true": a["p_true"], "false": 1.0 - a["p_true"]}))
                elif isinstance(a["probabilities"], dict):
                    out.append((i, name, dict(a["probabilities"])))
                else:
                    out.append((i, name, {str(k): p for k, p in enumerate(a["probabilities"])}))
            i += 1
    return out


def cmd_compare(args):
    only = set(args.fields.split(",")) if args.fields else None
    cmp = compare(entries(load(args.ref), only), entries(load(args.run), only))
    out = {"ref": args.ref, "run": args.run, **cmp}
    ok = cmp["pass"]
    if args.max_diff is not None:
        out["max_diff_bound"] = args.max_diff
        ok = ok and cmp["max_abs_diff"] <= args.max_diff
    out["pass_all"] = ok
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n_tickets": cmp["n_tickets"], "max_abs_diff": cmp["max_abs_diff"],
                      "medians": {f: v["median_abs_diff"] for f, v in cmp["fields"].items()}, "statistic": cmp["pass"],
                      "pass_all": ok}))


def cmd_retries(args):
    resps = load(args.resp)
    retries = [r["timings"]["retries"] for r in resps]
    ok = all(x >= args.min for x in retries) if args.min > 0 else all(x == 0 for x in retries)
    out = {"resp": args.resp, "min": args.min, "retries": retries, "rounds": [r["timings"]["rounds"] for r in resps],
           "states_per_round": [r["engine"]["states_per_round"] for r in resps], "decodes": [r["timings"]["decodes"] for r in resps],
           "pass": ok}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("retries", "rounds", "states_per_round", "pass")}))


def cmd_dump_diff(args):
    ref, run = load(args.ref), load(args.run)
    phases, tokens_same = set(), len(ref) == len(run)
    for x, y in zip(ref, run):
        t = json.loads(json.dumps(y["tokens"]))
        for st in t["states"]:
            for b in st["branches"]:
                phases.add(b.pop("phase", None))
        tokens_same = tokens_same and x["tokens"] == t
    out = {"ref": args.ref, "run": args.run,
           "results_identical": len(ref) == len(run) and all(x["results"] == y["results"] for x, y in zip(ref, run)),
           "tokens_identical_without_phase": tokens_same,
           "phases_in_dump": sorted(phases, key=lambda v: (v is None, v or 0)),   # null: an entry without phase
           "new_engine_keys": sorted({k for x, y in zip(ref, run) for k in set(y["engine"]) - set(x["engine"])}),
           "usage_identical": len(ref) == len(run) and all(x["usage"] == y["usage"] for x, y in zip(ref, run)),
           "decodes": [[x["timings"]["decodes"], y["timings"]["decodes"]] for x, y in zip(ref, run)]}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("ref", "run")}))


def cmd_records_compare(args):
    fields = args.fields.split(",")
    a = [r for r in load(args.a) if r["field"] in fields]
    b = [r for r in load(args.b) if r["field"] in fields]
    if [(r["ticket"], r["field"]) for r in a] != [(r["ticket"], r["field"]) for r in b]:
        sys.exit("the records of A and B do not line up")
    same = sum([n["prompt_tokens"] for n in x["nodes"]] == [n["prompt_tokens"] for n in y["nodes"]] for x, y in zip(a, b))

    def ents(rs):
        return [(r["ticket"], r["field"], dict(zip(r["keys"], engine_probs(r)))) for r in rs]

    cmp = compare(ents(a), ents(b))
    out = {"a": args.a, "b": args.b, "fields": fields, "records": len(a), "prompts_identical": same, **cmp}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"records": len(a), "prompts_identical": same, "max_abs_diff": cmp["max_abs_diff"],
                      "fields": {f: (v["median_abs_diff"], v["max_abs_diff"], v["argmax_disagreements"]) for f, v in cmp["fields"].items()}}))


def cmd_cost(args):
    runs = {}
    for path in args.resp:
        rs = load(path)[1:]

        def med(key, part):
            return statistics.median(r[part][key] for r in rs)

        runs[Path(path).stem] = {"lines": len(rs), "scoring_ms": med("scoring_ms", "timings"), "total_ms": med("total_ms", "timings"),
                                 "decodes": med("decodes", "timings"), "prompt_tokens": med("prompt_tokens", "usage"),
                                 "scored_tokens": med("scored_tokens", "usage"), "phases": rs[0]["engine"]["phases"],
                                 "states_per_round": rs[0]["engine"]["states_per_round"]}
    out = {"what": "b5 responses, medians over the lines after the first (prefix resident)", "runs": runs}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    for name, v in runs.items():
        print(name, json.dumps(v))


def budget_branches(path):
    """`branches N` of the spec 5.4 per-state cell check message in a budget error file."""
    msg = json.loads(Path(path).read_text().strip().splitlines()[-1])["error"]["message"]
    return int(re.search(r"branches (\d+)", msg).group(1))


def entry_key(b):
    return (b["field"], b["mode"], b["node"] if b["mode"] == "tree" else b["option"])


def cmd_lines(args):
    reqs, after, plain = load(args.req), load(args.after), load(args.plain)
    free = {entry_key(b): b["tokens"] for r in plain for st in r["tokens"]["states"] for b in st["branches"]}
    counts, terminators, not_kept, split_mismatch, extra, n_dep = {}, {}, [], 0, [], None
    for li, (req, r, p) in enumerate(zip(reqs, after, plain)):
        fields, line_total = req["fields"], 0
        for si, st in enumerate(r["tokens"]["states"]):
            deps = [b for b in st["branches"] if "given" in b]
            n_dep = len(deps) if n_dep is None else n_dep
            for b in deps:
                lf = free[entry_key(b)]
                if b["tokens"][len(b["tokens"]) - len(lf):] != lf or len(b["tokens"]) <= len(lf):
                    not_kept.append(f"line {li} state {si} {entry_key(b)}")
                    continue
                lines_tok = b["tokens"][: len(b["tokens"]) - len(lf)]
                counts[len(lines_tok)] = counts.get(len(lines_tok), 0) + 1
                line_total += len(lines_tok)
                if b["mode"] != "full_path" or len(b["given"]) != 1:
                    continue
                (g, value), = b["given"].items()
                o = option_values(fields[g]).index(value)
                anc = next(x for x in st["branches"] if x["field"] == g and x["option"] == o)["tokens"][:-1]
                if lines_tok[: len(anc)] != anc:
                    split_mismatch += 1
                    continue
                key = json.dumps(lines_tok[len(anc):])
                terminators[key] = terminators.get(key, 0) + 1
        extra.append([r["usage"]["prompt_tokens"] - p["usage"]["prompt_tokens"], line_total])
    budget = None
    if args.budget_plain and args.budget_after:
        bp, ba = budget_branches(args.budget_plain), budget_branches(args.budget_after)
        per_seq = (ba - bp) / n_dep
        budget = {"branches_plain": bp, "branches_after": ba, "dependent_sequences_per_state": n_dep,
                  "reserved_lines_tokens_per_sequence": per_seq, "max_exact_lines_tokens": max(counts),
                  "slack_per_sequence": per_seq - max(counts)}
    out = {"req": args.req, "after": args.after, "plain": args.plain, "mode": after[0]["engine"]["scoring"],
           "dependent_entries": sum(counts.values()) + len(not_kept), "lines_free_tokens_kept": not not_kept,
           "not_kept": not_kept[:5], "lines_token_counts": {str(k): v for k, v in sorted(counts.items())},
           "tokens_after_the_value": terminators or None, "split_mismatch": split_mismatch,
           "prompt_tokens_after_minus_plain_vs_lines_tokens": extra,
           "prompt_tokens_extra_equal": all(a == b for a, b in extra), "budget": budget,
           "pass": not not_kept and split_mismatch == 0 and all(a == b for a, b in extra) and (budget is None or budget["slack_per_sequence"] >= 0)}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("mode", "dependent_entries", "lines_free_tokens_kept", "lines_token_counts",
                                          "tokens_after_the_value", "split_mismatch", "prompt_tokens_extra_equal", "budget", "pass")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("requests")
    q.add_argument("--preset", required=True)
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--per-request", type=int, required=True)
    q.add_argument("--scoring", default="tree")
    q.add_argument("--after", action="append", default=[])
    q.add_argument("--tag-field", action="store_true")
    q.add_argument("out")
    a = sub.add_parser("after")
    a.add_argument("req")
    a.add_argument("resp")
    a.add_argument("out")
    d = sub.add_parser("detok")
    d.add_argument("req")
    d.add_argument("resp")
    d.add_argument("out")
    d.add_argument("--url", required=True)
    rd = sub.add_parser("records")
    rd.add_argument("req")
    rd.add_argument("resp")
    rd.add_argument("--preset", required=True)
    rd.add_argument("--tag", required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("run")
    c.add_argument("out")
    c.add_argument("--max-diff", type=float, default=None)
    c.add_argument("--fields", default=None)
    r = sub.add_parser("retries")
    r.add_argument("resp")
    r.add_argument("out")
    r.add_argument("--min", type=int, default=1)
    dd = sub.add_parser("dump-diff")
    dd.add_argument("ref")
    dd.add_argument("run")
    dd.add_argument("out")
    rc = sub.add_parser("records-compare")
    rc.add_argument("a")
    rc.add_argument("b")
    rc.add_argument("out")
    rc.add_argument("--fields", required=True)
    co = sub.add_parser("cost")
    co.add_argument("out")
    co.add_argument("resp", nargs="+")
    ln = sub.add_parser("lines")
    ln.add_argument("req")
    ln.add_argument("after")
    ln.add_argument("plain")
    ln.add_argument("out")
    ln.add_argument("--budget-plain", default=None)
    ln.add_argument("--budget-after", default=None)
    args = ap.parse_args()
    {"requests": cmd_requests, "after": cmd_after, "detok": cmd_detok, "records": cmd_records, "compare": cmd_compare,
     "retries": cmd_retries, "dump-diff": cmd_dump_diff, "records-compare": cmd_records_compare, "cost": cmd_cost,
     "lines": cmd_lines}[args.cmd](args)


if __name__ == "__main__":
    main()
