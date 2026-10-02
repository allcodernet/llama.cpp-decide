"""Sub-project 3 Task 8 extra check (added after Task 7): `after` + order_debias 2 with >= 2 states per round.

  uv run results-engine/sp3/integrity/checks.py debias-run --model GGUF --preset P --decide-seqs S --scoring MODE --tag T
                                                           [--n 20] [--batch 5]
      (server STOPPED) the first N test tickets (decide_client.build_request, prompt variant `default`) with urgency and
      angry after queue, options.order_debias 2 and options.scoring MODE, as B-state and as 1-state request lines, through
      `llama-decide --dump-tokens` (reference_check.decide_command: the preset's -c/-ngl/-fa/cache types). Writes
      debias-T-P-requests-b<B|1>.jsonl and debias-T-P-b<B|1>.jsonl (responses) next to this script.
  uv run results-engine/sp3/integrity/checks.py debias-check --preset P --tag T [--batch 5] [--binding]
      From those files only: (1) the B-state lines ran more than one state in a round (engine.states_per_round >= 2 and
      fewer rounds than states); (2) their answers against the 1-state answers (the reference) under the statistic
      (sp3_metrics.compare); (3) both runs: every dependent answer and dump entry (every variant) has `given` = the
      ancestors' returned values (the averaged answers) and `lines` built from them (reference_check.dependent_problems);
      (4) full-path dumps: per state and field, each variant's distribution and coverage recomputed from its dumped rows
      (spec 4.1), then the variant mean against the answer's probabilities, order_spread = max over options of (max -
      min over variants) and coverage = the variant mean (spec 7.2; T = 1 here), max abs diff <= 1e-9; also counted:
      queue answers whose value differs from variant 0's argmax (where lines from variant 0 would be wrong). Writes
      debias-T-P.json; pass = (1) and (3) and (4) and, with --binding, (2).
  uv run results-engine/sp3/integrity/checks.py rowcap-rows --preset P --tag T --row-cap C [--batch 5]
      A B-state full-path reference (reference_check.py dump --batch B and compare --tag T), one round per line: every
      dumped row numbered in decode order within its round (states, fields, options, rows); row k is read in chunk
      (k - 1) // C of the branch phase (spec 4.4: a chunk ends when its requested rows reach the row cap, here
      llama_n_seq_max = --decide-seqs + 1). Per chunk: the rows, the sequences a chunk boundary cuts, and the node
      comparison (engine vs replay, in probability) of its values. Writes rowcap-rows-T-P.json.
  uv run results-engine/sp3/integrity/checks.py chunk-answers --preset P [--tag fp-b5] [--b1-tag fp] [--split-tag rowcap]
                                                              [--nosplit-tag s84]
      The fields grouped by the row-cap chunk that read their rows (numbered as rowcap-rows, with the row cap and batch
      of split-<split-tag>-P.json). Per field, the max abs diff over options of: the B-state reference's answer against
      its replay; the one-state reference's (reference-<b1-tag>-P.json) against its replay; and, from the answers of
      `reference_check.py split` at the split row cap (<split-tag>) and at a row cap that holds a whole round
      (<nosplit-tag>), B-state vs one-state within each run and split vs unsplit B-state. Median and max per chunk, and
      the three fields farthest from their replay. Writes chunk-answers-P.json.
  uv run results-engine/sp3/integrity/checks.py request-identity --old REV --new REV
      decide_client.py at two commits: every request body the regression sends without switches (the warm-up and the b5
      lines of the test and train sets, prompt variant keywords, for qwen3.5-9b and gemma-4-e4b) and the output name,
      built by both; identical bytes required. Writes regression-request-identity.json.
  uv run results-engine/sp3/integrity/checks.py offline-check
      For every reference-T-P-replay.jsonl.gz here: its dump and replay copied to a temporary directory, then
      `reference_check.py compare --offline` there (no server); the result must equal the committed reference-T-P.json
      byte for byte. Writes offline-recompute.json.
  uv run results-engine/sp3/integrity/checks.py run-params
      run-params.json: per output file of this directory the run.sh step, runner, preset, model, --decide-seqs, b (states
      per request line), mode (options.scoring), other request options and the command as run.sh runs it (model paths,
      dumps, replays and the steps' flags read from run.sh; model-free outputs: command and inputs). Every output file
      must be listed, and where an output records its own parameters they must agree with run.sh.
  uv run results-engine/sp3/integrity/checks.py notes
      notes.md from the result files of this directory and run-params.json (every number read from a file named in the
      text).
"""

import argparse
import configparser
import gzip
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import cache_timing  # noqa: E402
import reference_check as rc  # noqa: E402
from decide_client import TEST_SET, build_request, chunks  # noqa: E402
from sp3_metrics import compare  # noqa: E402

AFTER = {"urgency": ["queue"], "angry": ["queue"]}
RECOMPUTE_BOUND = 1e-9


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def debias_body(preset, texts, scoring):
    body = build_request(preset, texts)
    body["options"]["scoring"] = scoring
    body["options"]["order_debias"] = 2
    for name, parents in AFTER.items():
        body["fields"][name]["after"] = list(parents)
    return body


def cmd_debias_run(args):
    texts = [json.loads(l)["text"] for l in TEST_SET.read_text().splitlines() if l.strip()][: args.n]
    for batch in (args.batch, 1):
        bodies = [debias_body(args.preset, b, args.scoring) for b in chunks(texts, batch)]
        (HERE / f"debias-{args.tag}-{args.preset}-requests-b{batch}.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bodies))
        responses = rc.run_decide(rc.decide_command(args.model, args.preset, args.decide_seqs), bodies)
        (HERE / f"debias-{args.tag}-{args.preset}-b{batch}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses))
        print(f"b{batch}: {len(responses)} lines, rounds {[r['timings']['rounds'] for r in responses]}, "
              f"states_per_round {responses[0]['engine']['states_per_round']}")


def dump_records(requests, responses):
    """reference_check.build_records of a run (answers, ancestors, per-variant entries kept by the helper below)."""
    fields = requests[0]["fields"]
    return rc.build_records(responses, fields, rc.ancestors_of(fields))


def variant_entries(responses):
    """(state, field) -> {variant: [full-path entries]} over every state of every line, in order."""
    out, i = {}, 0
    for r in responses:
        for st in r["tokens"]["states"]:
            for b in st["branches"]:
                out.setdefault((i, b["field"]), {}).setdefault(b.get("variant", 0), []).append(b)
            i += 1
    return out


def entry_problems(requests, responses):
    """Every dump entry of every variant: given/lines from the returned (averaged) ancestor answers."""
    records = dump_records(requests, responses)
    by_state_field = variant_entries(responses)
    problems, checked = [], 0
    for rec in records:
        for entries in by_state_field[rec["ticket"], rec["field"]].values():
            parts = [{k: e[k] for k in ("lines", "given") if k in e} for e in entries]
            key = "entries" if rec.get("mode") == "full_path" else "nodes"
            problems += rc.dependent_problems({**rec, key: parts})
            checked += len(parts) if rec["ancestors"] else 0
    return problems, checked


def recompute(requests, responses):
    """Full-path: per state and field the variant mean, order_spread and mean coverage from the dumped rows."""
    fields = requests[0]["fields"]
    keys = {name: rc.field_keys(f) for name, f in fields.items()}
    worst, n, value_not_variant0 = 0.0, 0, 0
    i = 0
    per_state = variant_entries(responses)
    for r in responses:
        for res in r["results"]:
            for name, ans in res["answers"].items():
                variants = per_state[i, name]
                dists, covs = [], []
                for v in sorted(variants):
                    def eng(kind, node, tok, rows=rc.field_nodes(variants[v])[1]):
                        row = rows[node]
                        return rc._num(row["end"] if kind == "end" else row[kind][str(tok)])
                    p, c = rc._normalise(rc.option_log_probs(variants[v], eng), f"state {i} {name} variant {v}")
                    dists.append(p)
                    covs.append(c)
                mean = [sum(d[o] for d in dists) / len(dists) for o in range(len(keys[name]))]
                spread = max(max(d[o] for d in dists) - min(d[o] for d in dists) for o in range(len(keys[name])))
                engine = rc.engine_probs({"type": fields[name]["type"], "keys": keys[name], "engine": ans})
                diffs = [abs(a - b) for a, b in zip(mean, engine)]
                diffs += [abs(spread - ans["order_spread"]), abs(sum(covs) / len(covs) - ans["coverage"])]
                worst = max(worst, max(diffs))
                n += 1
                if name == "queue" and keys[name][max(range(len(dists[0])), key=dists[0].__getitem__)] != ans["value"]:
                    value_not_variant0 += 1
            i += 1
    return {"answers": n, "max_abs_diff": worst, "queue_value_not_variant0_argmax": value_not_variant0,
            "pass": worst <= RECOMPUTE_BOUND}


def answer_entries(responses):
    return rc.answer_entries(responses)


def cmd_debias_check(args):
    stem = f"debias-{args.tag}-{args.preset}"
    req_b, resp_b = load(HERE / f"{stem}-requests-b{args.batch}.jsonl"), load(HERE / f"{stem}-b{args.batch}.jsonl")
    req_1, resp_1 = load(HERE / f"{stem}-requests-b1.jsonl"), load(HERE / f"{stem}-b1.jsonl")
    states = [len(r["results"]) for r in resp_b]
    rounds = [r["timings"]["rounds"] for r in resp_b]
    multi = resp_b[0]["engine"]["states_per_round"] >= 2 and all(rd < s for rd, s in zip(rounds, states) if s > 1)
    stat = compare(answer_entries(resp_1), answer_entries(resp_b))
    problems_b, checked_b = entry_problems(req_b, resp_b)
    problems_1, checked_1 = entry_problems(req_1, resp_1)
    full_path = resp_b[0]["engine"]["scoring"] == "full_path"
    re_b = recompute(req_b, resp_b) if full_path else None
    re_1 = recompute(req_1, resp_1) if full_path else None
    out = {
        "preset": args.preset, "tag": args.tag, "engine": resp_b[0]["engine"],
        "batched": {"states": states, "rounds": rounds, "states_per_round": resp_b[0]["engine"]["states_per_round"],
                    "more_than_one_state_per_round": multi},
        "batched_vs_b1": stat, "binding": args.binding,
        "given_lines": {"batched_entries": checked_b, "b1_entries": checked_1, "problems": (problems_b + problems_1)[:20],
                        "pass": not problems_b and not problems_1 and checked_b > 0},
        "recompute": None if not full_path else {"batched": re_b, "b1": re_1},
    }
    ok = multi and out["given_lines"]["pass"] and (not full_path or (re_b["pass"] and re_1["pass"]))
    out["pass"] = ok and (stat["pass"] or not args.binding)
    (HERE / f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"states_per_round": out["batched"]["states_per_round"], "rounds": rounds, "multi": multi,
                      "medians": {f: v["median_abs_diff"] for f, v in stat["fields"].items()}, "statistic": stat["pass"],
                      "given_lines": out["given_lines"]["pass"], "entries": [checked_b, checked_1],
                      "recompute": out["recompute"], "pass": out["pass"]}, indent=2))


def row_chunks(records, row_cap, batch):
    """A B-state full-path dump, one round per line: {(ticket, field, node): the chunk that reads the node's row} and the
    options whose rows two chunks share (cut sequences). Rows are numbered in decode order within a round (states,
    fields in declaration order, options, rows); row k is read in chunk (k - 1) // row_cap of the branch phase."""
    first_field = records[0]["field"]
    chunk_of, cut, rows = {}, [], 0
    for rec in records:
        if rec["field"] == first_field and rec["ticket"] % batch == 0:
            rows = 0  # the first state of a line: a new round
        for e in sorted(rec["entries"], key=lambda x: x["option"]):
            chunks = set()
            for row in e["logp"]:
                rows += 1
                chunk_of[rec["ticket"], rec["field"], row["node"]] = (rows - 1) // row_cap
                chunks.add((rows - 1) // row_cap)
            if len(chunks) > 1:
                cut.append({"ticket": rec["ticket"], "field": rec["field"], "option": e["option"]})
    return chunk_of, cut


def cmd_rowcap_rows(args):
    stem = f"reference-{args.tag}-{args.preset}"
    detail = {(c["ticket"], c["field"]): c for c in json.loads((HERE / f"{stem}.json").read_text())["detail"]}
    chunk_of, cut = row_chunks(load(HERE / f"{stem}-dump.jsonl"), args.row_cap, args.batch)
    diffs = {}
    for (ticket, field), c in detail.items():
        for p in c["nodes"]:
            if not p["both_zero"]:
                diffs.setdefault(chunk_of[ticket, field, p["node"]], []).append(p["abs_diff"])
    n_rows = {}
    for c in chunk_of.values():
        n_rows[c] = n_rows.get(c, 0) + 1
    out = {"preset": args.preset, "tag": args.tag, "row_cap": args.row_cap, "batch": args.batch,
           "rows_per_chunk": {str(c): n for c, n in sorted(n_rows.items())}, "sequences_cut": cut,
           "node_comparison_per_chunk": {str(c): {"values": len(v), "median_abs_diff": statistics.median(v),
                                                  "max_abs_diff": max(v)} for c, v in sorted(diffs.items())}}
    (HERE / f"rowcap-rows-{args.tag}-{args.preset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def _list_diff(a, b):
    if len(a) != len(b):
        raise ValueError(f"{len(a)} vs {len(b)} options")
    return max(abs(x - y) for x, y in zip(a, b))


def _dict_diff(a, b):
    if list(a) != list(b):
        raise ValueError(f"keys {list(a)} vs {list(b)}")
    return max(abs(a[k] - b[k]) for k in a)


CHUNK_MEASURES = ("b5_vs_replay", "b1_vs_replay", "b5_vs_b1", "b5_vs_b5_unsplit", "b5_unsplit_vs_b1_unsplit")


def cmd_chunk_answers(args):
    stem = f"reference-{args.tag}-{args.preset}"
    runs = {t: _j(f"split-{t}-{args.preset}.json") for t in (args.split_tag, args.nosplit_tag)}
    split = runs[args.split_tag]
    chunk_of, _ = row_chunks(load(HERE / f"{stem}-dump.jsonl"), split["row_cap"], split["batch"])
    held = {}  # (ticket, field) -> the chunks that read its rows
    for (ticket, field, _), c in chunk_of.items():
        held.setdefault((ticket, field), set()).add(c)
    ref = {(c["ticket"], c["field"]): c for c in _j(f"{stem}.json")["detail"]}
    ref1 = {(c["ticket"], c["field"]): c for c in _j(f"reference-{args.b1_tag}-{args.preset}.json")["detail"]}

    def answers(tag, batch):  # (state, field) -> {key: probability}, states numbered over all lines as in the dumps
        return {(i, n): a for i, n, a in rc.answer_entries(load(HERE / f"split-{tag}-{args.preset}-full_path-b{batch}.jsonl"))}

    b5, b1 = answers(args.split_tag, split["batch"]), answers(args.split_tag, 1)
    w5, w1 = answers(args.nosplit_tag, split["batch"]), answers(args.nosplit_tag, 1)
    measures = {
        "b5_vs_replay": lambda k: _list_diff(ref[k]["engine"], ref[k]["reference"]),
        "b1_vs_replay": lambda k: _list_diff(ref1[k]["engine"], ref1[k]["reference"]),
        "b5_vs_b1": lambda k: _dict_diff(b5[k], b1[k]),
        "b5_vs_b5_unsplit": lambda k: _dict_diff(b5[k], w5[k]),
        "b5_unsplit_vs_b1_unsplit": lambda k: _dict_diff(w5[k], w1[k]),
    }
    groups = {}
    for k, cs in sorted(held.items()):
        groups.setdefault("+".join(str(c) for c in sorted(cs)), []).append(k)
    per_chunk = {}
    for g, keys in sorted(groups.items()):
        per_chunk[g] = {"fields": len(keys)}
        for name in CHUNK_MEASURES:
            v = [measures[name](k) for k in keys]
            per_chunk[g][name] = {"median": statistics.median(v), "max": max(v)}
    far = sorted(held, key=measures["b5_vs_replay"], reverse=True)[:3]
    out = {"preset": args.preset, "reference": f"{stem}.json", "b1_reference": f"reference-{args.b1_tag}-{args.preset}.json",
           "runs": {t: {"decide_seqs": s["decide_seqs"], "row_cap": s["row_cap"], "batch": s["batch"],
                        "branch_decodes": [l["branch_decodes"] for l in s["full_path"]["lines"]]} for t, s in runs.items()},
           "split_tag": args.split_tag, "nosplit_tag": args.nosplit_tag,
           # the dumped answers against the split run's (the same requests, b5 and b1): 0 when the runs agree
           "reference_vs_split_run_max_abs_diff": max(_list_diff(ref[k]["engine"], list(b5[k].values())) for k in held),
           "b1_reference_vs_split_run_max_abs_diff": max(_list_diff(ref1[k]["engine"], list(b1[k].values())) for k in held),
           "per_chunk": per_chunk,
           "farthest_from_replay": [{"ticket": k[0], "field": k[1], "chunks": sorted(held[k]), "keys": list(b5[k]),
                                     "replay": ref[k]["reference"], "b5": list(b5[k].values()),
                                     "b5_unsplit": list(w5[k].values()), "b1": list(b1[k].values()),
                                     "b5_vs_replay": measures["b5_vs_replay"](k),
                                     "b5_unsplit_vs_replay": _list_diff(list(w5[k].values()), ref[k]["reference"])}
                                    for k in far]}
    (HERE / f"chunk-answers-{args.preset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "farthest_from_replay"}, indent=2))


def cmd_request_identity(args):
    import importlib.util
    import subprocess
    import tempfile

    def module(rev):
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:decide_client.py"], capture_output=True, text=True,
                             check=True).stdout
        path = Path(tempfile.mkdtemp()) / "decide_client.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location(f"decide_client_{rev}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    old, new = module(args.old), module(args.new)
    sets = {"test": TEST_SET, "train": TEST_SET.parent / "triage_train.jsonl"}
    runs, identical = [], True
    for preset in ("qwen3.5-9b", "gemma-4-e4b"):
        for name, path in sets.items():
            texts = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
            new_args = new.parse_args(["--model", preset, "--batch", "5", "--test-set", str(path)])
            old_bodies = [old.build_request(preset, [texts[0]], "keywords")] + [
                old.build_request(preset, b, "keywords") for b in old.chunks(texts, 5)]
            new_bodies = [new.request_body(new_args, [texts[0]])] + [new.request_body(new_args, b) for b in new.chunks(texts, 5)]
            same = [json.dumps(a) == json.dumps(b) for a, b in zip(old_bodies, new_bodies)]
            name_new = new.run_name(new_args)
            ok = len(old_bodies) == len(new_bodies) and all(same) and name_new == f"decide-{preset}-b5-keywords"
            identical = identical and ok
            runs.append({"preset": preset, "set": name, "requests": len(new_bodies), "identical": ok, "run_name": name_new})
    out = {"old": args.old, "new": args.new, "runs": runs, "identical": identical}
    (HERE / "regression-request-identity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def cmd_offline_check(args):
    import shutil
    import subprocess
    import tempfile

    results = []
    for replay in sorted(HERE.glob("reference-*-replay.jsonl.gz")):
        stem = replay.name[: -len("-replay.jsonl.gz")]
        dump = json.loads((HERE / f"{stem}-dump.jsonl").read_text().splitlines()[0])
        preset = next(p for p in ("qwen2.5-0.5b", "qwen3.5-4b", "qwen3.5-9b", "gemma-4-e4b") if stem.endswith("-" + p))
        tag = stem[len("reference-"): -len("-" + preset)]
        with tempfile.TemporaryDirectory() as tmp:
            for name in (f"{stem}-dump.jsonl", replay.name):
                shutil.copy(HERE / name, tmp)
            subprocess.run([sys.executable, str(ROOT / "reference_check.py"), "compare", "--preset", preset, "--tag", tag,
                            "--out-dir", tmp, "--offline"], check=True, capture_output=True)
            same = (Path(tmp) / f"{stem}.json").read_bytes() == (HERE / f"{stem}.json").read_bytes()
        results.append({"result": f"{stem}.json", "mode": dump.get("mode", "tree"), "identical": same})
    out = {"results": results, "all_identical": all(r["identical"] for r in results)}
    (HERE / "offline-recompute.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


MODELS = ["qwen2.5-0.5b", "qwen3.5-4b", "gemma-4-e4b", "qwen3.5-9b"]  # their --decide-seqs: run-params.json
# The engine build of every run in this directory: the sources of 5e9e030d3 (every engine source older than the
# binaries, the engine tree clean). The binaries were linked 15 s before that commit was made, so their build info
# (engine/build/common/build-info.cpp, printed by --version) names its parent 8daa120b5.
ENGINE = {"commit": "5e9e030d3", "subject": "decide: clamp free sequences in the budget message; cell message helper",
          "build_info_commit": "8daa120b5"}
# The presets of every run (the router's models and settings; the CLI's -c/-ngl/-fa/cache types through
# reference_check.decide_command): models-engine.ini as committed, unchanged until after the last run here.
PRESETS = {"file": "models-engine.ini", "blob": "ccfeeb1376dc1f30b09b4492c5b3861310301e57", "committed_in": "b527dbd"}
RUN_PARAMS = "run-params.json"
TOOLS = {"checks.py", "run.sh"}  # this directory's inputs, not outputs
OUT = "results-engine/sp3/integrity"
BATCH_INVARIANCE = "results-engine/batch-invariance-qwen2.5-0.5b.json"  # the 0.5B tree b5 vs b1 of sub-project 1
FIELDS3 = ("queue", "urgency", "angry")


def _j(name):
    return json.loads((HERE / name).read_text())


def _f(x, digits=4):
    return f"{x:.{digits}f}" if abs(x) >= 10 ** -digits or x == 0 else f"{x:.1e}"


def _reference_rows(result):
    rows = []
    for name in FIELDS3:
        v = result["fields"][name]
        rows.append(f"{name}: statistic {_f(v['median_abs_diff'])} / {_f(v['max_abs_diff'])} / {v['disagreements_over_margin']} "
                    f"{'PASS' if v['statistic_pass'] else 'FAIL'}; coverage {_f(v['coverage_median_abs_diff'])} "
                    f"{'PASS' if v['coverage_pass'] else 'FAIL'}; nodes {_f(v['node_median_abs_diff'])} / "
                    f"{_f(v['node_max_abs_diff'])} ({v['node_values']}) {'PASS' if v['node_pass'] else 'FAIL'}")
    return "<br>".join(rows)


def end_rows(preset, tag="fp"):
    """A full-path reference's END values (spec 4.1: the END sum of the row after a path's last token): the largest
    difference, with its field, the piece of the path token into its row and that token's engine probability at the row
    before; and the largest change of an option probability when every replayed END value replaces the engine's (every
    other value the engine's)."""
    recs = {(r["ticket"], r["field"]): r for r in load(HERE / f"reference-{tag}-{preset}-dump.jsonl")}
    with gzip.open(HERE / f"reference-{tag}-{preset}-replay.jsonl.gz", "rt") as f:
        pieces = json.loads(f.read().splitlines()[-1])["pieces"]
    top, moved = None, 0.0
    for c in _j(f"reference-{tag}-{preset}.json")["detail"]:
        rec = recs[c["ticket"], c["field"]]
        stem, rows, node_at = rc.field_nodes(rec["entries"])
        into = {}  # node -> (the path token into its row, that token's engine probability at the row before)
        for e in rec["entries"]:
            t = e["tokens"]
            for i in range(stem, len(t)):
                before = rows[node_at[tuple(t[:i])]]
                into[node_at[tuple(t[: i + 1])]] = (t[i], math.exp(rc._num(before["children"][str(t[i])])))
        replayed = {q["node"]: q["p_replay"] for q in c["nodes"] if q["kind"] == "end"}
        for q in c["nodes"]:
            if q["kind"] == "end" and not q["both_zero"] and (top is None or q["abs_diff"] > top["abs_diff"]):
                tok, p = into[q["node"]]
                top = {"abs_diff": q["abs_diff"], "field": c["field"], "p_token": p,
                       "piece": bytes.fromhex(pieces[str(tok)]).decode("utf-8", "replace")}

        def value(kind, node, tok, swap=False, rows=rows, replayed=replayed):
            if swap and kind == "end":
                return math.log(replayed[node]) if replayed[node] > 0 else -math.inf
            row = rows[node]
            return rc._num(row["end"] if kind == "end" else row[kind][str(tok)])

        engine, _ = rc._normalise(rc.option_log_probs(rec["entries"], value), "engine rows")
        swapped, _ = rc._normalise(rc.option_log_probs(rec["entries"], lambda k, n, t: value(k, n, t, True)), "END replayed")
        moved = max(moved, _list_diff(engine, swapped))
    return {**top, "option_moved": moved}


def _presets_ini():
    import subprocess

    ini = configparser.ConfigParser()
    ini.read_string(subprocess.run(["git", "-C", str(ROOT), "cat-file", "-p", PRESETS["blob"]], capture_output=True,
                                   text=True, check=True).stdout)
    return ini


def run_params():
    """{output file: its run parameters} from run.sh (module docstring: run-params), checked for coverage and against
    the outputs that record their own parameters."""
    sh = (HERE / "run.sh").read_text()
    steps = dict(re.findall(r"^step_(\w+)\(\) \{\n(.*?)^\}", sh, re.M | re.S))
    # run.sh's path variables (VAR=${VAR:-default}, or VAR=${VAR:-${OUTER:-default}}): the environment's values of VAR,
    # then OUTER, else the default with $HOME expanded
    def path_variable(v):
        m = re.search(rf"^{v}=\$\{{{v}:-(?:\$\{{(\w+):-)?([^{{}}\s]+)\}}+$", sh, re.M)
        return (os.environ.get(v) or (m.group(1) and os.environ.get(m.group(1)))
                or m.group(2).replace("$HOME", str(Path.home())))

    paths = {v: path_variable(v) for v in ("DATA", "LMSTUDIO_MODELS")}

    def expand(word):  # a shell word from run.sh: quotes stripped, $DATA / $LMSTUDIO_MODELS substituted
        word = word.strip('"')
        for v, value in paths.items():
            word = word.replace("${" + v + "}", value).replace("$" + v, value)
        return word

    gguf = {k: expand(v) for k, v in re.findall(r"^\s+(\S+)\) echo (\S+) ;;$", sh, re.M)}  # model_of
    url = re.search(r"^URL=(\S+)\$PORT$", sh, re.M).group(1) + re.search(r"^PORT=(\d+)$", sh, re.M).group(1)
    ini = _presets_ini()

    def preset_seqs(preset):  # a preset's decide-seqs: its section, else the router's global [*] section
        return int(ini[preset].get("decide-seqs", ini["*"]["decide-seqs"]))

    cli, server = "llama-decide (server stopped)", "llama-server (serve-engine.sh router)"
    files = {}

    def entry(step, command, runner=None, preset=None, model=None, decide_seqs=None, b=None, mode=None, **extra):
        return {"step": step, "runner": runner, "preset": preset, "model": model, "decide_seqs": decide_seqs, "b": b,
                "mode": mode, **extra, "command": command}

    def add(names, params):
        for name in names:
            if name in files:
                raise ValueError(f"{name} listed twice")
            files[name] = params

    def loop(step):  # the presets of a step's `for p in ...` loop
        return re.search(r"for p in ([^;]+); do", steps[step]).group(1).split()

    # dumps (step dumps; the regression's 0.5B tree dump): `dump PRESET TAG ARGS` over reference_check.py dump defaults
    n = re.search(r'--n (\d+) --tag "\$tag"', sh).group(1)
    dumps = {}
    for preset, tag, rest in re.findall(r"^\s+dump (\S+) (\S+)(.*)$", sh, re.M):
        words = rest.split()
        pairs = list(zip(words[::2], words[1::2]))
        flags = {"--scoring": "tree", "--decide-seqs": "9", "--batch": "1", **{f: v for f, v in pairs if f != "--after"}}
        dumps[preset, tag] = entry(
            "regression" if tag == "regression" else "dumps",
            f"uv run reference_check.py dump --model {gguf[preset]} --preset {preset} --n {n} --tag {tag} --out-dir {OUT}"
            + "".join(" " + w for w in words), cli, preset, gguf[preset], int(flags["--decide-seqs"]), int(flags["--batch"]),
            flags["--scoring"], after=[v for f, v in pairs if f == "--after"] or None)
        add([f"reference-{tag}-{preset}-dump.jsonl"], dumps[preset, tag])

    # replays: reference_check.py compare of a dump through the router (b, mode and after are the dump's)
    replays = [(p, t) for ts, p in re.findall(r"for t in ([^;]+); do compare (\S+) \$t; done", sh) for t in ts.split()]
    for preset, tag in replays + re.findall(r"^\s+compare (\S+) (\S+)$", sh, re.M):
        d = dumps[preset, tag]
        add([f"reference-{tag}-{preset}.json", f"reference-{tag}-{preset}-replay.jsonl.gz"], entry(
            "replays", f"uv run reference_check.py compare --preset {preset} --tag {tag} --out-dir {OUT} --url {url}",
            server, preset, ini[preset]["model"], preset_seqs(preset), d["b"], d["mode"], after=d["after"],
            inputs=[f"reference-{tag}-{preset}-dump.jsonl"]))

    # split and the nosplit diagnostic: reference_check.py split runs full-path and tree, b5 (its default) and b1 lines
    for step in ("split", "nosplit"):
        seqs, n_states, tag = re.search(r"--decide-seqs (\d+) --n (\d+) --tag (\S+)", steps[step]).groups()
        for preset in loop(step):
            command = (f"uv run reference_check.py split --model {gguf[preset]} --preset {preset} --decide-seqs {seqs} "
                       f"--n {n_states} --tag {tag} --out-dir {OUT}")
            for mode in ("full_path", "tree"):
                for b in (5, 1):
                    add([f"split-{tag}-{preset}-{mode}-b{b}.jsonl"],
                        entry(step, command, cli, preset, gguf[preset], int(seqs), b, mode))
            add([f"split-{tag}-{preset}.json"],
                entry(step, command, cli, preset, gguf[preset], int(seqs), [5, 1], ["full_path", "tree"]))
    for preset in loop("nosplit"):
        add([f"chunk-answers-{preset}.json"], entry(
            "nosplit", f"uv run {OUT}/checks.py chunk-answers --preset {preset}",
            inputs=[f"reference-fp-b5-{preset}-dump.jsonl", f"reference-fp-b5-{preset}.json", f"reference-fp-{preset}.json",
                    f"split-rowcap-{preset}.json", f"split-s84-{preset}.json", "their full_path b5 and b1 lines"]))

    # cache timing: cache_timing.py (--states default 5, --decide-seqs default the preset's, tree requests)
    alternations = re.search(r"--alternations (\d+)", steps["cache"]).group(1)
    for preset in loop("cache"):
        out = f"{OUT}/cache-timing-{preset}.json"
        add([f"cache-timing-{preset}.json"], entry(
            "cache", f"uv run cache_timing.py --model {gguf[preset]} --preset {preset} --alternations {alternations} --out {out}",
            cli, preset, gguf[preset], preset_seqs(preset), 5, "tree", prompt_variants=cache_timing.SCHEMAS,
            prefix_cache=list(cache_timing.CACHE_SIZES),
            summaries=f"uv run cache_timing.py --resummarize {out} (second pass: from the stored per-request timings)"))

    # the extra after + order_debias 2 check: checks.py debias-run (b5, its default, and b1 lines), debias-check
    after = [f"{name}={','.join(parents)}" for name, parents in AFTER.items()]
    binding = {(p, t): bool(x) for p, t, x in re.findall(r"debias-check --preset (\S+) --tag (\S+)( --binding)?", steps["debias"])}
    for model_of, preset, seqs, mode, tag in re.findall(
            r'debias-run --model "\$\(model_of (\S+)\)" --preset (\S+) --decide-seqs (\d+) --scoring (\S+) --tag (\S+)',
            steps["debias"]):
        command = (f"uv run {OUT}/checks.py debias-run --model {gguf[model_of]} --preset {preset} --decide-seqs {seqs} "
                   f"--scoring {mode} --tag {tag}")
        runs = []
        for b in (5, 1):
            runs += [f"debias-{tag}-{preset}-requests-b{b}.jsonl", f"debias-{tag}-{preset}-b{b}.jsonl"]
            add(runs[-2:], entry("debias", command, cli, preset, gguf[model_of], int(seqs), b, mode, order_debias=2,
                                 after=after))
        add([f"debias-{tag}-{preset}.json"], entry(
            "debias", f"uv run {OUT}/checks.py debias-check --preset {preset} --tag {tag}"
                      + (" --binding" if binding[preset, tag] else ""), inputs=runs))

    # regression: decide_client.py through a freshly started router per run, then sp3_metrics.py compare
    body = steps["regression"]
    batch = int(re.search(r"decide_client\.py --model \$p --batch (\d+)", body).group(1))
    test_set = expand(re.search(r'--test-set "([^"]+)"', body).group(1))
    for preset in loop("regression"):
        for s in re.search(r"local sets=\(([^)]*)\)", body).group(1).split():
            stem = f"decide-{preset}-b{batch}-keywords"
            add([f"regression/{s}/preds/{stem}.jsonl", f"regression/{s}/runs/{stem}.json"], entry(
                "regression", f"uv run decide_client.py --model {preset} --batch {batch} --test-set "
                              f"{test_set.replace('$s', s)} --out-dir {OUT}/regression/{s}",
                server + ", freshly started", preset, ini[preset]["model"], preset_seqs(preset), batch, "tree",
                prompt_variant="keywords"))
            add([f"regression-{s}-{preset}.json"], entry(
                "regression", f"uv run sp3_metrics.py compare results-engine/sp3/baseline/{s}/preds/{stem}.jsonl "
                              f"{OUT}/regression/{s}/preds/{stem}.jsonl --max-diff 1e-6 --out {OUT}/regression-{s}-{preset}.json",
                inputs=[f"regression/{s}/preds/{stem}.jsonl"]))
    add(["regression-dump-qwen2.5-0.5b.json"], entry(
        "regression", "uv run sp3_metrics.py compare results-engine/sp3/baseline/dump-qwen2.5-0.5b.jsonl "
                      f"{OUT}/reference-regression-qwen2.5-0.5b-dump.jsonl --max-diff 1e-6 --out {OUT}/regression-dump-qwen2.5-0.5b.json",
        inputs=["reference-regression-qwen2.5-0.5b-dump.jsonl"]))
    ident = _j("regression-request-identity.json")
    add(["regression-request-identity.json"], entry(
        "regression", f"uv run {OUT}/checks.py request-identity --old {ident['old']} --new {ident['new']}",
        inputs=["decide_client.py at both commits"]))

    # model-free, run by hand
    for preset in sorted(p for p, t in dumps if t == "fp-b5"):
        w = _j(f"rowcap-rows-fp-b5-{preset}.json")
        add([f"rowcap-rows-fp-b5-{preset}.json"], entry(
            "model-free", f"uv run {OUT}/checks.py rowcap-rows --preset {preset} --tag {w['tag']} --row-cap {w['row_cap']}",
            inputs=[f"reference-fp-b5-{preset}-dump.jsonl", f"reference-fp-b5-{preset}.json"]))
    add(["offline-recompute.json"], entry("model-free", f"uv run {OUT}/checks.py offline-check",
                                          inputs=sorted(f for f in files if f.endswith("-replay.jsonl.gz"))))
    add(["notes.md"], entry("model-free", f"uv run {OUT}/checks.py notes", inputs=["the files it names", RUN_PARAMS]))
    add([RUN_PARAMS], entry("model-free", f"uv run {OUT}/checks.py run-params",
                            inputs=["run.sh", f"{PRESETS['file']} (blob {PRESETS['blob'][:7]})"]))
    add(["gpu-log.txt"], entry(
        "all", f"cp $LOGS/gpu.log {OUT}/gpu-log.txt",
        note="the lines run.sh's gpu_ready and router_load wrote at each step start and model load (VRAM, temperature); "
             "the last two, marked (diagnostic split s84, ...), were written by hand after the runs"))

    present = {f.relative_to(HERE).as_posix() for f in HERE.rglob("*") if f.is_file() and "__pycache__" not in f.parts}
    missing, stale = sorted(present - TOOLS - files.keys()), sorted(files.keys() - present - {RUN_PARAMS})
    if missing or stale:
        raise SystemExit(f"run-params: outputs without parameters {missing}; parameters without an output {stale}")
    _crosscheck(files)
    return files


def _closure(direct, field):
    """A field's ancestors under `after` (FIELD -> parents), transitively."""
    seen, todo = set(), list(direct.get(field, []))
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            todo += direct.get(x, [])
    return seen


def _crosscheck(files):
    """Where an output records its own run parameters, they must equal those reconstructed from run.sh. Not recorded
    by any output, so from run.sh alone: the dumps' --decide-seqs and b, and the replays' and regression's router
    settings beyond decide_seqs; the debias runs' --decide-seqs only as consistent with the engine's states per round."""
    def expect(name, what, got, want):
        if got != want:
            raise SystemExit(f"run-params: {name}: {what} is {got!r}, run.sh gives {want!r}")

    for name, p in files.items():
        if name.startswith("split-") and name.endswith(".jsonl"):
            first = json.loads((HERE / name).read_text().splitlines()[0])
            expect(name, "scoring", first["engine"]["scoring"], p["mode"])
            expect(name, "states per line", len(first["results"]), p["b"])
        elif name.startswith("split-") and name.endswith(".json"):
            s = _j(name)
            for key, want in (("decide_seqs", p["decide_seqs"]), ("batch", p["b"][0]), ("model", p["model"]),
                              ("preset", p["preset"])):
                expect(name, key, s[key], want)
        elif name.startswith("cache-timing-"):
            c = _j(name)
            for key, want in (("decide_seqs", p["decide_seqs"]), ("states", p["b"]), ("model", p["model"]),
                              ("schemas", p["prompt_variants"])):
                expect(name, key, c[key], want)
        elif "/runs/" in name:
            r = _j(name)
            for key, want in (("decide_seqs", p["decide_seqs"]), ("batch", p["b"]), ("preset", p["preset"]),
                              ("prompt_variant", p["prompt_variant"])):
                expect(name, key, r[key], want)
            expect(name, "fork_commit", r["fork_commit"][: len(ENGINE["commit"])], ENGINE["commit"])
        elif name.startswith("debias-") and "-requests-" in name:
            for body in load(HERE / name):
                expect(name, "scoring", body["options"]["scoring"], p["mode"])
                expect(name, "order_debias", body["options"]["order_debias"], p["order_debias"])
                expect(name, "after", sorted(f"{f}={','.join(v['after'])}" for f, v in body["fields"].items()
                                             if v.get("after")), sorted(p["after"]))
                expect(name, "states per line", len(body["states"]), p["b"])
        elif name.startswith("debias-") and name.endswith(".jsonl"):  # the responses
            for r in load(HERE / name):
                e = r["engine"]
                expect(name, "scoring", e["scoring"], p["mode"])
                expect(name, "order_debias", e["order_debias"], p["order_debias"])
                expect(name, "states per line", len(r["results"]), p["b"])
                # --decide-seqs D: the engine fits (D - K) // seqs_per_state states in a round (K = order_debias)
                expect(name, "states_per_round", e["states_per_round"],
                       (p["decide_seqs"] - p["order_debias"]) // e["seqs_per_state"])
        elif name.endswith("-dump.jsonl"):
            records = load(HERE / name)
            expect(name, "mode", sorted({r.get("mode", "tree") for r in records}), [p["mode"]])
            direct = dict((f, ps.split(",")) for f, ps in (a.split("=") for a in p["after"] or []))
            expect(name, "ancestors", {r["field"]: sorted(r["ancestors"]) for r in records if r.get("ancestors")},
                   {f: sorted(_closure(direct, f)) for f in direct})
            expect(name, "records with ancestors", any("ancestors" in r for r in records), bool(direct))
        elif name.startswith("reference-") and name.endswith(".json"):
            expect(name, "preset", _j(name)["preset"], p["preset"])


def cmd_run_params(args):
    files = run_params()
    out = {"engine": ENGINE, "presets": PRESETS,
           "notes": ["Commands as run.sh runs them (paths from the project directory, $I expanded). CLI runs: llama-decide "
                     "with the preset's -c/-ngl/-fa/cache types (reference_check.decide_command), the server stopped. "
                     "Server runs: the serve-engine.sh router on port 8097 with the presets above; their decide_seqs is "
                     "the preset's (the replays' /completion requests do not use it).",
                     "b: states per request line; mode: options.scoring; after: dependent fields (FIELD=PARENT).",
                     "The nosplit step and the model-free commands were run by hand; nosplit is a run.sh step since.",
                     "Checked against the outputs that record their parameters (checks.py _crosscheck): split, cache "
                     "timing, regression runs, debias requests and responses, dump modes and `after` ancestors, replay "
                     "presets. Recorded by no output, so from run.sh alone: the dumps' --decide-seqs and b; the debias "
                     "runs' --decide-seqs only as consistent with the engine's states per round."],
           "files": dict(sorted(files.items()))}
    (HERE / RUN_PARAMS).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {HERE / RUN_PARAMS}: {len(files)} files")


def cmd_notes(args):
    rp = _j(RUN_PARAMS)["files"]
    L = ["# Sub-project 3 Task 8: integrity runs (spec 12.2, 12.3)", ""]
    L += [f"Engine `{ENGINE['commit']} {ENGINE['subject']}`: llama-decide and llama-server built from its sources (their build",
          f"info, `--version`, names the parent `{ENGINE['build_info_commit']}`: they were linked 15 s before the commit, with the",
          f"same sources). Run parameters per output file: `{RUN_PARAMS}`. Tools: `reference_check.py` (dump, compare, split;",
          "compare keeps every request with its answer in `reference-<tag>-<preset>-replay.jsonl.gz`, and `--offline` recomputes a",
          "result from that file alone), `cache_timing.py`, and in this directory `checks.py` (the extra `after` + debias check,",
          "`rowcap-rows`, `chunk-answers`, `request-identity`, `offline-check`, `run-params`, `notes`) and `run.sh` (steps dumps, replays, split,",
          "cache, debias, regression, in that order, then the diagnostic step nosplit). Requests: the triage schema of `decide_client.build_request` (prompt variant `default`) on",
          "the first 20 test tickets; b1 = one state per request line, b5 = five. KV cache as the presets of `models-engine.ini`",
          "(f16 for qwen2.5-0.5b, q8_0 otherwise); dumps with `llama-decide`, the server stopped; replays through the preset",
          "router (`serve-engine.sh`, port 8097), `/completion` with `n_probs` 512 and `cache_prompt` off. \"The statistic\" = per",
          "field median abs diff over options <= 0.01 and no argmax disagreement where the reference's top-2 margin > 0.15",
          "(`sp3_metrics.compare`, with the replay or the one-state run as the reference). Cell format below: median / max /",
          "disagreements over the margin; nodes: median / max (values compared).", ""]

    # 1. full-path reference
    L += ["## 1. Full-path reference, b1 (spec 12.3)", "",
          "| model (`--decide-seqs`) | per field | unverifiable edges | pass | file |", "|---|---|---|---|---|"]
    for preset in MODELS:
        r = _j(f"reference-fp-{preset}.json")
        u = r["unverifiable"]
        L.append(f"| {preset} ({rp[f'reference-fp-{preset}-dump.jsonl']['decide_seqs']}) | {_reference_rows(r)} | {u['unverifiable']} of {u['edges']} | "
                 f"{'PASS' if r['pass'] else 'FAIL'} | `reference-fp-{preset}.json` |")
    tb1, tb5 = _j("reference-tree-b1-qwen2.5-0.5b.json"), _j("reference-tree-b5-qwen2.5-0.5b.json")
    fp05 = _j("reference-fp-qwen2.5-0.5b.json")
    L += ["", "Qwen3.5-4B, Gemma 4 E4B and Qwen3.5-9B are binding. Qwen2.5-0.5B beside its tree-mode reference on the same 20",
          "states at 64 sequences (`reference-tree-b1-qwen2.5-0.5b.json`, the tree replay of `reference_check.py`, renormalised",
          "children): " + ", ".join(f"{n} full-path {_f(fp05['fields'][n]['median_abs_diff'])} vs tree "
                                     f"{_f(tb1['fields'][n]['median_abs_diff'])}" for n in FIELDS3) +
          f" (tree `pass` {tb1['pass']}). Its known `angry` batch noise (sub-project 1, `{BATCH_INVARIANCE}`: "
          f"{_f(json.loads((ROOT / BATCH_INVARIANCE).read_text())['fields']['angry']['median_abs_diff'])} b5 vs b1) is of the "
          "size of both.", ""]
    gate = {p: _j(f"reference-fp-{p}.json")["coverage_gate"] for p in ("qwen3.5-9b", "gemma-4-e4b")}
    L += ["Coverage sanity gate (median engine coverage per field >= 0.8): " + "; ".join(
        f"{p} " + " / ".join(_f(g["median_engine_coverage"][n]) for n in FIELDS3) + (" PASS" if g["pass"] else " FAIL")
        for p, g in gate.items()) + " (queue / urgency / angry).", ""]
    L += ["Unverifiable edges (a path token absent from the replay's top-512): none on any model or run (`unverifiable` in every",
          "`reference-*.json`), so the forced-token check reserved for that case was not needed. Engine rows recomputed with spec 4.1",
          "reproduce the answers within " + _f(max(_j(f"reference-{t}-{p}.json")["fields"][n]["engine_recompute_max_abs_diff"]
                                                     for t, p in (("fp", m) for m in MODELS) for n in FIELDS3), 1) +
          " (`engine_recompute_max_abs_diff`).", ""]
    ex = {p: _j(f"reference-fp-{p}.json") for p in MODELS}
    worst = {}
    for p, r in ex.items():
        pairs = [(abs(q["p_replay"] - q["p_engine"]), q["kind"], c["field"]) for c in r["detail"] for q in c["nodes"]]
        worst[p] = max(pairs)
    ends = {p: end_rows(p) for p in MODELS}
    L += ["The largest node differences: " + "; ".join(f"{p} {_f(w[0])} ({w[2]} {w[1]})" for p, w in worst.items()) + ".",
          "The largest END differences (in parentheses: the field; the path token into the row; that token's engine",
          "probability at the row before): " + "; ".join(f"{p} {_f(e['abs_diff'])} ({e['field']}; `{e['piece']}`; "
                                                         f"{_f(e['p_token'], 1)})" for p, e in ends.items()) + ".",
          "These rows follow the lone closing quote of a queue value, a non-canonical continuation (the models write the",
          "merged `\",`), and enter P(o) only multiplied by that token's probability: with every replayed END value in place",
          "of the engine's (spec 4.1 on the engine's rows otherwise) no option probability moves by more than " +
          ", ".join(f"{_f(e['option_moved'], 1)} ({p})" for p, e in ends.items()) + ". END/MERGED pairs that are -inf on",
          "both sides (digits and ` true`/` false` have no MERGED tokens) are counted, not compared: " +
          f"{sum(v['node_both_zero'] for v in ex['qwen3.5-9b']['fields'].values())} per 20-state run.", ""]

    # 2. b5 extras and chunk rows
    L += ["## 2. Extra: b5 references, rows after the row-cap chunk boundary", "",
          "| model | per field | chunk rows (0 / 1) | nodes chunk 0 | nodes chunk 1 | sequences cut | pass | files |",
          "|---|---|---|---|---|---|---|---|"]
    for preset in ("qwen2.5-0.5b", "gemma-4-e4b"):
        r, w = _j(f"reference-fp-b5-{preset}.json"), _j(f"rowcap-rows-fp-b5-{preset}.json")
        c0, c1 = w["node_comparison_per_chunk"]["0"], w["node_comparison_per_chunk"]["1"]
        L.append(f"| {preset} | {_reference_rows(r)} | {w['rows_per_chunk']['0']} / {w['rows_per_chunk']['1']} | "
                 f"{_f(c0['median_abs_diff'])} / {_f(c0['max_abs_diff'])} ({c0['values']}) | {_f(c1['median_abs_diff'])} / "
                 f"{_f(c1['max_abs_diff'])} ({c1['values']}) | {len(w['sequences_cut'])} | {'PASS' if r['pass'] else 'FAIL'} | "
                 f"`reference-fp-b5-{preset}.json`, `rowcap-rows-fp-b5-{preset}.json` |")
    sp = {p: _j(f"split-rowcap-{p}.json") for p in ("qwen2.5-0.5b", "gemma-4-e4b")}
    shape = {(s["row_cap"], s["decide_seqs"], s["batch"], l["rows_read"]) for s in sp.values() for l in s["full_path"]["lines"]}
    assert len(shape) == 1, shape  # both models: the same row cap and rows per round
    cap, seqs, batch, per_round = shape.pop()
    cas = {p: _j(f"chunk-answers-{p}.json") for p in sp}
    both = sum(v["fields"] for ca in cas.values() for g, v in ca["per_chunk"].items() if "+" in g)
    L += ["", f"Row cap {cap} (`--decide-seqs {seqs}` + the slot sequence), {per_round // batch} rows per state, {per_round} per",
          f"{batch}-state round: rows {cap + 1}-{per_round} of each round were read from the second branch decode with",
          "chunk-relative batch indices (two branch decodes per b5 line in `split-rowcap-*.json`). Fields with rows in both",
          f"chunks: {both}, so the boundary falls between two fields and cuts no sequence (Task 5 cut sequences mid-way at 60",
          "and 68). Tree b5 beside the 0.5B: " + ", ".join(f"{n} {_f(tb5['fields'][n]['median_abs_diff'])}" for n in FIELDS3) +
          " (`reference-tree-b5-qwen2.5-0.5b.json`).", ""]
    runs = {p: ca["runs"] for p, ca in cas.items()}
    L += ["Answers by the chunk that read their field's rows (`chunk-answers-<preset>.json`; per field the max abs diff over",
          "options, median / max over the chunk's fields). b5 split and b1: `reference_check.py split` at " +
          " and ".join(sorted({f"{r['rowcap']['decide_seqs']} sequences (branch decodes per b5 line {r['rowcap']['branch_decodes']})"
                               for r in runs.values()})) + "; b5 unsplit and its b1: the same at " +
          " and ".join(sorted({f"{r['s84']['decide_seqs']} sequences, where a round's rows fit one chunk (branch decodes "
                               f"{r['s84']['branch_decodes']}" for r in runs.values()})) +
          "; `split-s84-*.json`, a diagnostic run after the regression on the same build). The dumped b5 and b1 answers equal",
          "the split run's (max abs diff " + ", ".join(
              _f(max(ca["reference_vs_split_run_max_abs_diff"], ca["b1_reference_vs_split_run_max_abs_diff"])) + f" {p}"
              for p, ca in cas.items()) + ").", "",
          "| model | chunk (fields) | b5 vs replay | b1 vs replay | b5 vs b1 | b5 split vs b5 unsplit | b5 unsplit vs b1 |",
          "|---|---|---|---|---|---|---|"]
    for p, ca in cas.items():
        for g, v in ca["per_chunk"].items():
            L.append(f"| {p} | {g} ({v['fields']}) | " +
                     " | ".join(f"{_f(v[m]['median'])} / {_f(v[m]['max'])}" for m in CHUNK_MEASURES) + " |")
    far = cas["gemma-4-e4b"]["farthest_from_replay"]
    L += ["", "Between the split and the unsplit run, chunk-1 fields move about as much as chunk-0 fields (medians " + "; ".join(
        f"{p} {_f(ca['per_chunk']['1']['b5_vs_b5_unsplit']['median'])} vs {_f(ca['per_chunk']['0']['b5_vs_b5_unsplit']['median'])}"
        for p, ca in cas.items()) + "). The Gemma fields farthest from their replay (rows in chunk " +
          ", ".join(sorted({"+".join(map(str, f["chunks"])) for f in far})) + ") are about as far without the split: " + "; ".join(
        f"ticket {f['ticket']} `{f['field']}` {_f(f['b5_vs_replay'])} split, {_f(f['b5_unsplit_vs_replay'])} unsplit, "
        f"{_f(_list_diff(f['b1'], f['replay']))} b1" for f in far) + ".", ""]

    # 3. after reference
    L += ["## 3. After reference: full-path, urgency and angry after queue, b1 (spec 12.3)", "",
          "| model (`--decide-seqs`) | per field | unverifiable | given/lines | pass | file |", "|---|---|---|---|---|---|"]
    for preset in ("qwen2.5-0.5b", "qwen3.5-4b"):
        seqs = rp[f"reference-fp-after-{preset}-dump.jsonl"]["decide_seqs"]
        r = _j(f"reference-fp-after-{preset}.json")
        d = r["dependent"]
        L.append(f"| {preset} ({seqs}) | {_reference_rows(r)} | {r['unverifiable']['unverifiable']} of {r['unverifiable']['edges']} | "
                 f"{d['dependent_records']} records, {d['dependent_entries']} entries, {len(d['problems'])} problems | "
                 f"{'PASS' if r['pass'] else 'FAIL'} | `reference-fp-after-{preset}.json` |")
    L += ["", "The dependent entries' tokens hold the lines, so their replay prompts do; every dependent answer and entry has",
          "`given` = the queue value the state returned and `lines` = `  \"queue\": <value>,\\n` built from it. Qwen3.5-4B is binding.",
          "The 4B's largest END difference (as in section 1): " + (lambda e: f"{_f(e['abs_diff'])} ({e['field']}; `{e['piece']}`; "
          f"{_f(e['p_token'], 1)}); the replayed END values move no option probability by more than {_f(e['option_moved'], 1)}.")(
              end_rows("qwen3.5-4b", "fp-after")), ""]

    # 4. split
    L += ["## 4. Row-cap split: full-path b5 vs b1 at 64 sequences, tree b5 vs b1 beside (spec 12.3)", "",
          "| model (`--decide-seqs`) | branch decodes per b5 line | b1 lines unsplit | " +
          " | ".join(f"{n}: full-path / tree" for n in FIELDS3) + " | pass | file |", "|---|---|---|---|---|---|---|---|"]
    for preset, tag in (("gemma-4-e4b", "rowcap"), ("gemma-4-e4b", "s84"), ("qwen2.5-0.5b", "rowcap"), ("qwen2.5-0.5b", "s84")):
        s = _j(f"split-{tag}-{preset}.json")
        cells = []
        for n in FIELDS3:
            a, b = s["side_by_side"][n]["full_path"], s["side_by_side"][n]["tree"]
            cells.append(f"{_f(a['median_abs_diff'])} {'PASS' if a['pass'] else 'FAIL'} / {_f(b['median_abs_diff'])} "
                         f"{'PASS' if b['pass'] else 'FAIL'}")
        verdict = ("PASS" if s["pass"] else "FAIL") if tag == "rowcap" else "diagnostic (no split)"
        L.append(f"| {preset} ({s['decide_seqs']}) | {[l['branch_decodes'] for l in s['full_path']['lines']]} (rounds "
                 f"{[l['rounds'] for l in s['full_path']['lines']]}, rows {s['full_path']['lines'][0]['rows_read']}) | "
                 f"{s['full_path']['single_unsplit']} | " + " | ".join(cells) + f" | {verdict} | `split-{tag}-{preset}.json` |")
    a64, a84 = (_j(f"split-{t}-qwen2.5-0.5b.json")["side_by_side"]["angry"] for t in ("rowcap", "s84"))
    L += ["", "Gemma 4 E4B is binding. The rows at 84 sequences are the diagnostic of section 2 (a round's rows in one chunk).",
          f"The 0.5B's full-path `angry` median is {_f(a64['full_path']['median_abs_diff'])} with the split and "
          f"{_f(a84['full_path']['median_abs_diff'])} without it, beside tree mode's {_f(a64['tree']['median_abs_diff'])} and "
          f"{_f(a84['tree']['median_abs_diff'])} (no split at either count): the batch-composition effect, not the split.", ""]

    # 5. cache timing
    L += ["## 5. Prefix cache timing: 20 A/B alternations (spec 12.3)", "",
          "| model | schema (prompt variant) | `--decide-prefix-cache` | prefill_ms | prefix_ms | total_ms | hits / misses | file |",
          "|---|---|---|---|---|---|---|---|"]
    for preset in ("qwen3.5-9b", "gemma-4-e4b"):
        c = _j(f"cache-timing-{preset}.json")
        for schema, variant in c["schemas"].items():
            for size in ("0", "8"):
                s = c["runs"][size]["per_schema"][schema]
                m = s["medians"]
                L.append(f"| {preset} | {schema} (`{variant}`) | {size} | {m['prefill_ms']:.1f} | {m['prefix_ms']:.1f} | "
                         f"{m['total_ms']:.1f} | {s['hits']} / {s['misses']} | `cache-timing-{preset}.json` |")
    c = _j("cache-timing-qwen3.5-9b.json")
    L += ["", f"Medians per schema over its {c['runs']['0']['per_schema']['A']['requests']} requests of a run (A and B "
          f"alternate; {c['states']} states each, `--decide-seqs` {c['decide_seqs']}, `llama-decide`). The two prefixes differ in",
          "length, so the schemas' timings form two groups: a median over both, as first reported, describes neither.",
          "With the cache the first A and B miss and every later request restores its prefix (after snapshotting the other one).", ""]

    # 6. extra debias + after
    L += ["## 6. Extra (Task 7): `after` + order_debias 2 with more than one state per round", "",
          "| run | states per round, rounds per b5 line | b5 vs b1 medians (q / u / a) | statistic | given/lines entries (b5, b1) | "
          "recomputation | pass | file |", "|---|---|---|---|---|---|---|---|"]
    for tag, preset, what in (("tree", "qwen3.5-4b", "Qwen3.5-4B tree, `--decide-seqs {}` (binding)"),
                              ("fp", "qwen2.5-0.5b", "Qwen2.5-0.5B full-path, `--decide-seqs {}`")):
        what = what.format(rp[f"debias-{tag}-{preset}-b5.jsonl"]["decide_seqs"])
        d = _j(f"debias-{tag}-{preset}.json")
        st = d["batched_vs_b1"]["fields"]
        rec = d["recompute"]
        rec_txt = "n/a (a tree dump has no per-variant values)" if rec is None else (
            f"{rec['batched']['answers']} + {rec['b1']['answers']} answers, max {_f(max(rec['batched']['max_abs_diff'], rec['b1']['max_abs_diff']), 1)}; "
            f"queue value != variant 0's argmax in {rec['batched']['queue_value_not_variant0_argmax']} states")
        L.append(f"| {what} | {d['batched']['states_per_round']}, {d['batched']['rounds']} | " +
                 " / ".join(_f(st[n]["median_abs_diff"]) for n in FIELDS3) + f" | {'PASS' if d['batched_vs_b1']['pass'] else 'FAIL'} | "
                 f"{d['given_lines']['batched_entries']}, {d['given_lines']['b1_entries']} ({len(d['given_lines']['problems'])} problems) | "
                 f"{rec_txt} | {'PASS' if d['pass'] else 'FAIL'} | `debias-{tag}-{preset}.json` |")
    fp, tr = _j("debias-fp-qwen2.5-0.5b.json")["engine"], _j("debias-tree-qwen3.5-4b.json")["engine"]
    s4, s05 = rp["debias-tree-qwen3.5-4b-b5.jsonl"]["decide_seqs"], rp["debias-fp-qwen2.5-0.5b-b5.jsonl"]["decide_seqs"]
    L += ["", f"Full-path K = 2 needs {fp['seqs_per_state']} sequences per state (`seqs_per_state` of the 0.5B run), one state per "
          f"round at {s4} sequences on the 4B; the averages, order_spread and coverage are therefore recomputed on the 0.5B at "
          f"{s05} sequences ({fp['states_per_round']} states per round), and the 4B binding check runs in tree mode "
          f"({tr['seqs_per_state']} sequences per state, {tr['states_per_round']} states per round).", ""]

    # 7. regression
    L += ["## 7. Regression (spec 12.2), the last planned run, each on a freshly started router", ""]
    ident = _j("regression-request-identity.json")
    L += [f"`decide_client.py` at `{ident['new']}` (the commit used); its requests without switches are byte-identical to those "
          f"of `{ident['old']}` (the Task 0 baseline commit): {sum(r['requests'] for r in ident['runs'])} request bodies and the "
          f"output names, `identical` {ident['identical']} (`regression-request-identity.json`).", "",
          "| run | max abs diff vs the Task 0 baseline | within 1e-6 | file |", "|---|---|---|---|"]
    for s in ("test", "train"):
        for preset in ("qwen3.5-9b", "gemma-4-e4b"):
            r = _j(f"regression-{s}-{preset}.json")
            L.append(f"| {preset} {s} (b5, keywords) | {r['max_abs_diff']:.3g} | {r['within_bound']} | `regression-{s}-{preset}.json` |")
    r = _j("regression-dump-qwen2.5-0.5b.json")
    seqs = rp["reference-regression-qwen2.5-0.5b-dump.jsonl"]["decide_seqs"]
    L.append(f"| qwen2.5-0.5b tree dump (`reference_check.py dump`, 20 states, `--decide-seqs` {seqs}) | {r['max_abs_diff']:.3g} | "
             f"{r['within_bound']} | `regression-dump-qwen2.5-0.5b.json` |")
    L += ["", "The regression was the last planned run; the no-split diagnostic of section 2 (`split-s84-*`, `llama-decide` only, "
          "no server) followed it on the same build.", ""]

    # 8. offline
    o = _j("offline-recompute.json")
    L += ["## 8. Offline recomputation", "",
          f"Every replay request and answer is saved (`reference-*-replay.jsonl.gz`: the prompt token ids and parameters, the",
          f"generated token, the listed ids and log-probs, the listed pieces); `reference_check.py compare --offline` recomputes each",
          f"result from its dump and replay alone: {sum(r['identical'] for r in o['results'])} of {len(o['results'])} results "
          f"byte-identical to the committed ones (`offline-recompute.json`).", ""]

    # 9. run parameters and the GPU log
    L += ["## 9. Run parameters and GPU log", "",
          f"`{RUN_PARAMS}`: per output file ({len(rp)}) the `run.sh` step, runner, preset, model, `--decide-seqs`, b, mode and "
          "command, reconstructed from `run.sh` by `checks.py run-params` and checked against the outputs that record their own "
          "parameters (the dumps' `--decide-seqs` and b are recorded by none: see its `notes`).", "",
          "`gpu-log.txt`: VRAM and temperature at each step start and model load as `run.sh` wrote them; its last two lines, "
          "marked `(diagnostic split s84, ...)`, were written by hand.", ""]
    (HERE / "notes.md").write_text("\n".join(L))
    print(f"wrote {HERE / 'notes.md'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("debias-run")
    r.add_argument("--model", required=True)
    r.add_argument("--preset", required=True)
    r.add_argument("--decide-seqs", type=int, required=True)
    r.add_argument("--scoring", choices=["tree", "full_path"], required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--batch", type=int, default=5)
    c = sub.add_parser("debias-check")
    c.add_argument("--preset", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--batch", type=int, default=5)
    c.add_argument("--binding", action="store_true", help="the statistic is part of pass")
    w = sub.add_parser("rowcap-rows")
    w.add_argument("--preset", required=True)
    w.add_argument("--tag", required=True)
    w.add_argument("--row-cap", type=int, required=True)
    w.add_argument("--batch", type=int, default=5)
    a = sub.add_parser("chunk-answers")
    a.add_argument("--preset", required=True)
    a.add_argument("--tag", default="fp-b5")
    a.add_argument("--b1-tag", default="fp")
    a.add_argument("--split-tag", default="rowcap")
    a.add_argument("--nosplit-tag", default="s84")
    q = sub.add_parser("request-identity")
    q.add_argument("--old", required=True)
    q.add_argument("--new", required=True)
    sub.add_parser("offline-check")
    sub.add_parser("run-params")
    sub.add_parser("notes")
    args = ap.parse_args()
    {"debias-run": cmd_debias_run, "debias-check": cmd_debias_check, "rowcap-rows": cmd_rowcap_rows,
     "chunk-answers": cmd_chunk_answers, "request-identity": cmd_request_identity, "run-params": cmd_run_params, "offline-check": cmd_offline_check, "notes": cmd_notes}[args.cmd](args)


if __name__ == "__main__":
    main()
