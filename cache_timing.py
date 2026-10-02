"""Prefix cache timing (spec 12.3): A/B alternations with --decide-prefix-cache 0 vs 8.

  uv run cache_timing.py --model <gguf> --preset <name> --out PATH [--alternations 20] [--states 5]
                         [--decide-seqs S] [--test-set T]
  uv run cache_timing.py --resummarize PATH [PATH ...]

(server STOPPED) Schema A is the triage request with the `keywords` prompt variant (decide_client's default), schema B
the `default` variant: other instructions and descriptions, so another prefix (the A/B/A pair of results-engine/sp3/checks/task-4/). Both carry the first
--states test tickets. The request lines A, B, A, B, ... (--alternations pairs) run through llama-decide twice, with
--decide-prefix-cache 0 and with 8 (reference_check.decide_command: the preset's -c/-ngl/-fa/cache types; --decide-seqs
defaults to the preset's decide-seqs, as the server). Per run and schema: the medians over the schema's requests of
timings.prefill_ms, prefix_ms and total_ms, and the prefix cache hits and misses summed. The two prefixes differ in
length, so the two schemas' timings form separate groups and a median over both would describe neither: there is none.
Writes PATH (both runs with every request's schema and timings) and prints the table.

--resummarize recomputes the summaries of existing timing files from their per-request timings, labelled by the line
order (A, B, A, B, ...), and rewrites them; no model run.
"""

import argparse
import json
import statistics
from pathlib import Path

import reference_check
from decide_client import TEST_SET, build_request

CACHE_SIZES = (0, 8)
TIMINGS = ("prefill_ms", "prefix_ms", "total_ms")
SCHEMAS = {"A": "keywords", "B": "default"}  # schema -> prompt variant


def alternation_lines(preset, texts, alternations):
    """A, B, A, B, ...: the same states under two prompt variants, i.e. two prefixes."""
    a = build_request(preset, texts, "keywords")
    b = build_request(preset, texts, "default")
    return [a, b] * alternations


def alternation_schemas(alternations):
    """The schema of each line of alternation_lines, in order."""
    return ["A", "B"] * alternations


def summary_of(per_request):
    """Per schema the medians of the timings and the hits and misses summed; over all requests only the counts."""
    def counts(ps):
        return {"requests": len(ps), "hits": sum(p["hits"] for p in ps), "misses": sum(p["misses"] for p in ps)}

    per_schema = {}
    for schema in sorted({p["schema"] for p in per_request}):
        ps = [p for p in per_request if p["schema"] == schema]
        per_schema[schema] = {**counts(ps), "medians": {k: statistics.median(p[k] for p in ps) for k in TIMINGS}}
    return {**counts(per_request), "per_schema": per_schema, "per_request": per_request}


def summarize(responses, schemas):
    """One run's responses, schemas[i] the schema of response i."""
    if len(responses) != len(schemas):
        raise ValueError(f"{len(responses)} responses, {len(schemas)} schema labels")
    return summary_of([{"schema": schema, **{k: r["timings"][k] for k in TIMINGS},
                        "hits": r["engine"]["prefix_cache"]["hits"], "misses": r["engine"]["prefix_cache"]["misses"]}
                       for r, schema in zip(responses, schemas)])


def resummarize(data):
    """A timing file with its summaries recomputed from the per-request timings, which are in line order (files
    written before the per-schema summaries have no labels and pooled medians; those medians are dropped)."""
    schemas = alternation_schemas(data["alternations"])
    for size, run in data["runs"].items():
        per_request = run["per_request"]
        if len(per_request) != len(schemas):
            raise ValueError(f"run {size}: {len(per_request)} requests, {len(schemas)} lines")
        for i, (p, schema) in enumerate(zip(per_request, schemas)):
            if p.get("schema", schema) != schema:
                raise ValueError(f"run {size}: request {i} labelled {p['schema']}, line {i} is {schema}")
        data["runs"][size] = summary_of([{"schema": schema, **{k: v for k, v in p.items() if k != "schema"}}
                                         for p, schema in zip(per_request, schemas)])
    return data


def table(runs):
    lines = [f"{'schema':>6} {'cache':>5} {'prefill_ms':>10} {'prefix_ms':>9} {'total_ms':>9} {'hits':>5} {'misses':>6}"]
    for schema in SCHEMAS:
        for size, run in runs.items():
            s = run["per_schema"][schema]
            m = s["medians"]
            lines.append(f"{schema:>6} {size:>5} {m['prefill_ms']:10.1f} {m['prefix_ms']:9.1f} {m['total_ms']:9.1f} "
                         f"{s['hits']:5d} {s['misses']:6d}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prefix cache timing: A/B alternations with --decide-prefix-cache 0 vs 8.")
    ap.add_argument("--model")
    ap.add_argument("--preset")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--alternations", type=int, default=20)
    ap.add_argument("--states", type=int, default=5, help="test tickets per request (default 5)")
    ap.add_argument("--decide-seqs", type=int, default=None, help="default: the preset's decide-seqs")
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    ap.add_argument("--resummarize", type=Path, nargs="+", metavar="PATH",
                    help="recompute the summaries of existing timing files from their per-request timings (no model run)")
    args = ap.parse_args(argv)
    if args.resummarize:
        for path in args.resummarize:
            data = resummarize(json.loads(path.read_text()))
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"{path}\n{table(data['runs'])}")
        return
    missing = [f"--{name}" for name in ("model", "preset", "out") if getattr(args, name) is None]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))

    seqs = args.decide_seqs or int(reference_check._engine_settings(args.preset)["decide-seqs"])
    texts = [json.loads(l)["text"] for l in args.test_set.read_text().splitlines() if l.strip()][: args.states]
    lines = alternation_lines(args.preset, texts, args.alternations)
    runs = {}
    for size in CACHE_SIZES:
        cmd = reference_check.decide_command(args.model, args.preset, seqs, dump=False) + ["--decide-prefix-cache", str(size)]
        runs[size] = summarize(reference_check.run_decide(cmd, lines), alternation_schemas(args.alternations))

    out = {"preset": args.preset, "model": args.model, "decide_seqs": seqs, "alternations": args.alternations,
           "states": len(texts), "schemas": SCHEMAS, "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(table(runs))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
