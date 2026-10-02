"""Sub-project 4: result tables for REPORT-ENGINE.md from the committed files under results-engine/image/ (no GPU).
Usage: uv run results-engine/image/notes.py   -> results-engine/image/notes.md; exits 1 when a result file is missing."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

O = Path(__file__).resolve().parent
VISION = ("qwen3.8-27b-vision", "smolvlm-500m", "gemma-4-e4b-vision", "qwen3.6-35b-a3b-vision", "gemma-3-4b-vision")
REF_TAGS = ("tree-one", "tree-multi", "fp", "after", "debias2", "t0.5")   # checks.py REF_TAGS
BINDING = "qwen3.8-27b-vision"   # every integrity file of the binding model must exist; the others may miss some (the integrity step)
MODEL = "qwen3.8-27b-vision"     # the COCO evaluation (spec 9.6)


def load(path):
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())


def jsonl(path):
    if not path.exists():
        sys.exit(f"missing {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mark(ok):
    return "pass" if ok else "**FAIL**"


def worst_median(r):
    return max((v["median_abs_diff"] for v in r.get("fields", {}).values() if "median_abs_diff" in v), default=float("nan"))


def over_margin(r):
    return sum(v.get("disagreements_over_margin", 0) for v in r.get("fields", {}).values())


def answer(r):
    """'<status> <code>' of a recorded HTTP answer ({"status", "response"})."""
    err = (r.get("response") or {}).get("error") or {}
    return f"{r['status']} {err['code']}" if err.get("code") else str(r["status"])


def regression():
    out = ["## Text regression, old vs new build (spec 9.2)", "",
           "Old build `engine-49785a95d` (before this work) against the new build, same preset, same request sequence on a "
           "fresh router each.", "",
           "| preset | run | requests | max abs diff | differences | result |", "|---|---|---|---|---|---|"]
    rows = [("qwen3.8-27b", "Task 5", O / "regression" / "qwen3.8-27b.json"),
            ("qwen3.8-27b-vision", "Task 5", O / "regression" / "qwen3.8-27b-vision.json")]
    new = load(O / "swa" / "regression" / "qwen3.8-27b-new.info.json")["server_build"]["engine_commit"]
    rows.append(("qwen3.8-27b", f"Task 6b, new build `{new[:9]}`", O / "swa" / "regression" / "qwen3.8-27b.json"))
    for p, run, f in rows:
        r = load(f)
        out.append(f"| {p} | {run} | {r['requests']} | {r['max_abs_diff']} | {len(r['differences'])} | {mark(r['pass'])} |")
    ti = O / "text-integrity" / "integrity"
    tree, fp = (load(ti / f"reference-{t}-qwen3.8-27b.json") for t in ("tree", "fp"))
    batch = load(ti / "batch-invariance-qwen3.8-27b.json")
    if not (ti / "offline-recompute.txt").exists():
        sys.exit(f"missing {ti / 'offline-recompute.txt'}")
    out += ["", f"Existing integrity scripts on the new build (qwen3.8-27b, 20 tickets): tree reference {mark(tree['pass'])}, "
            f"full-path reference {mark(fp['pass'])}, batch invariance {mark(batch['pass'])}, offline recompute "
            f"`{(ti / 'offline-recompute.txt').read_text().strip()}`."]
    return out


def no_vision():
    out = ["## No vision (spec 9.3)", "",
           "| preset | info, both 400s, log, serving | text answers equal the run without the rejected requests | CLI 400 |",
           "|---|---|---|---|"]
    for p in ("qwen2.5-0.5b", "qwen3.8-27b"):
        d = O / "no-vision"
        w, c, cli = load(d / f"{p}-with.json"), load(d / f"{p}.json"), load(d / f"cli-{p}.json")
        out.append(f"| {p} | {mark(w['pass'])} | {mark(c['pass'])} (max abs diff {c['max_abs_diff']}) | {mark(cli['pass'])} |")
    return out


def result(d, p, name):
    """The JSON of d/name, or None when a non-binding preset has no such file (its failures.txt or left-out.txt says why)."""
    if p == BINDING or (d / name).exists():
        return load(d / name)
    return None


def held_cell(h):
    if h is None:
        return "missing (see failures.txt)"
    b, old = h.get("budget") or {}, h.get("old_held", "?")
    if "held" not in b:
        return f"{mark(h['pass'])} (no budget answer; see held-new.json)"
    return f"{mark(h['pass'])} ({old} → {b['held']})" if h.get("applicable") else f"{mark(h['pass'])}, n/a ({old} = {b['held']})"


HEAD = ["preset", *REF_TAGS, "batch", "retry", "malformed (engine unchanged)", "held cells (old → new)"]


def integrity_row(d, p):
    """One row of the integrity table for preset p with result directory d."""
    if (d / "SKIPPED.txt").exists():
        return f"| {p} | skipped: {(d / 'SKIPPED.txt').read_text().splitlines()[0]} |" + " |" * (len(HEAD) - 2)
    left_out = (d / "left-out.txt").read_text() if (d / "left-out.txt").exists() else ""
    cells = []
    for t in REF_TAGS:
        if t + ":" in left_out.split():
            cells.append("left out (left-out.txt)")
            continue
        r = result(d, p, f"reference-{t}-{p}.json")
        cells.append("missing (see failures.txt)" if r is None else f"{mark(r['pass'])} ({worst_median(r):.2g}; {over_margin(r)})")
    for n in ("batch", "retry"):
        r = result(d, p, f"{n}.json")
        cells.append("missing (see failures.txt)" if r is None else mark(r["pass"]))
    m, mc = result(d, p, "malformed.json"), result(d, p, "malformed-compare.json")
    cells.append("missing (see failures.txt)" if m is None or mc is None else f"{mark(m['pass'])} ({mark(mc['pass'])})")
    cells.append(held_cell(result(d, p, "held-new.json")))
    return f"| {p} | " + " | ".join(cells) + " |"


def held_answers(root):
    """The sized state's answer on the old and the new build, per preset with both held files under root."""
    held = []
    for p in VISION:
        d = root / p
        if (d / "held-old.json").exists() and (d / "held-new.json").exists():
            o, n = load(d / "held-old.json"), load(d / "held-new.json")
            held.append(f"{p} {answer(o['sized'])} → {answer(n['sized'])}")
    return ["", "Held cells, answer to the sized state, old build → new build: "
            + ("; ".join(held) if held else "no held pair recorded") + "."]


def gate_detail(d, p="gemma-3-4b-vision"):
    """After a failed gate: ticket 1's dog answers across two layouts and the engine against itself (p is non-binding,
    so a missing file is named, not fatal)."""
    if not d.exists() or (d / "SKIPPED.txt").exists():
        return []
    out = []
    refs = {t: result(d, p, f"reference-{t}-{p}.json") for t in ("tree-multi", "fp")}
    by_tag = {t: x for t, r in refs.items() if r for x in r["detail"] if x["ticket"] == 1 and x["field"] == "dog"}
    if len(by_tag) == 2:
        out += ["", f"Ticket 1, P(dog = true): reference {by_tag['tree-multi']['reference'][0]:.3f} (tree-multi) / "
                    f"{by_tag['fp']['reference'][0]:.3f} (fp); engine {by_tag['tree-multi']['engine'][0]:.3f} (tree-multi) / "
                    f"{by_tag['fp']['engine'][0]:.3f} (fp)."]
    rt, bt = result(d, p, "retry.json"), result(d, p, "batch.json")
    retry = "retry missing (see failures.txt)" if rt is None else \
        f"retry (rc1 vs clean) max abs diff {rt['statistic']['max_abs_diff']:.3f}"
    batch = "batch missing (see failures.txt)" if bt is None else \
        f"batch together max abs diff {bt['together']['max_abs_diff']:.3f}, mixed {bt['mixed']['max_abs_diff']:.3f}"
    return out + ["", f"Engine against itself on gemma-3-4b: {retry}; {batch}."]


def integrity():
    out = ["## Integrity of image states (spec 9.4)", "",
           "Reference cells: result (worst per-field median abs diff; argmax disagreements over the margin, all fields). "
           "Held cells: n/a where cells equal positions.", "",
           "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    out += [integrity_row(O / "integrity" / p, p) for p in VISION]
    fails = []
    for p in VISION:
        f = O / "integrity" / p / "failures.txt"
        if f.exists():
            fails.append(f"{p}: " + "; ".join(x.strip() for x in f.read_text().splitlines() if x.strip()))
    out += ["", "Kept failures (`failures.txt`): " + (" | ".join(fails) if fails else "none") + "."]
    out += held_answers(O / "integrity")
    g = load(O / "gate.json")
    if g["pass"]:
        out += ["", "Non-causal gate (gemma-3-4b, spec 9.5): pass; non-causal projectors stay enabled."]
    else:
        off = load(O / "gate-off.json")
        live = (f"not checked live (gemma unavailable: {off['skipped']})" if "skipped" in off
                else f"check of the rebuilt engine: {mark(off['pass'])}, info `{json.dumps(off['vision'])}`")
        out += ["", f"Non-causal gate (gemma-3-4b, spec 9.5): **failed** ({', '.join(k for k, v in g['checks'].items() if not v)}); "
                f"non-causal projectors are switched off ({live})."]
        out += gate_detail(O / "integrity" / "gemma-3-4b-vision")
    u = O / "integrity" / "gemma-3-4b-vision" / "ubatch.json"
    if u.exists():
        out += ["", f"n_ubatch rule (gemma-3-4b, `-ub 128`): {mark(load(u)['pass'])}."]
    return out


def swa():
    out = ["## Cell budget and the sliding-window cache (spec 5.3 rev 5, Task 6b)", "",
           "| preset | n_ctx | info `memory.swa_cells` | SWA cache in llama.cpp's log | result |", "|---|---|---|---|---|"]
    d = O / "swa"
    for p in ("gemma-4-e4b", "gemma-4-e4b-16k", "gemma-4-e4b-16k-swa-full", "gemma-3-4b-vision"):
        r = load(d / f"info-{p}.json")
        out.append(f"| {p} | {r['n_ctx']} | {r['memory']['swa_cells']} | {', '.join(map(str, r['logged_swa_cells']))} | {mark(r['pass'])} |")
    out += ["", "| run | held | prefix | K | limit | largest state answered 200 | +1 token | any 500 | result |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in ("gemma-4-e4b-16k-idle", "gemma-4-e4b-16k-debias2", "gemma-4-e4b-16k-chat", "gemma-4-e4b-16k-swa-full-idle",
                 "gemma-3-4b-vision-idle"):
        r = load(d / f"{name}.json")
        b, big = r["probe"]["budget"], r["largest_admitted"]
        over = next(s for s in r["sweep"] if s["state_tokens"] == big + 1)
        over_ans = f"{over['status']} {(over.get('error') or {}).get('code', '')}".strip()
        any500 = "yes" if any(s["status"] == 500 for s in r["sweep"]) else "no"
        out.append(f"| {name} | {b['held']} | {b['prefix']} | {b['K']} | {b['limit']} | {big} | {over_ans} | {any500} | {mark(r['pass'])} |")
    pre = load(d / "info-gemma-4-e4b-16k.json")["prefix"]
    full = load(d / "info-gemma-4-e4b-16k-swa-full.json")["prefix"]
    out += ["", f"Prefix of {pre['words']} words: gemma-4-e4b-16k exit {pre['returncode']}, "
            f"`{(pre.get('error') or {}).get('message')}`; gemma-4-e4b-16k-swa-full exit {full['returncode']}, "
            f"prefix {full['info_prefix_tokens']} tokens."]
    rnd = load(d / "gemma-4-e4b-16k-idle.json")["round"]
    out += ["", "Round sizing (gemma-4-e4b-16k, three states in one request; gate: each round of the split equals the same "
            "composition sent on its own):", "",
            "| case | state tokens | need of 2 / 3 states | limit | status, rounds, retries | gate: round 1 vs pair, round 2 vs alone "
            "(max abs diff) | recorded, pair vs each alone: median / max abs diff, argmax disagreements |",
            "|---|---|---|---|---|---|---|"]
    for case, c in rnd.items():
        m, gt, pa = c["multi"], c["gate"], c["pair_vs_alone"]["fields"]["animal"]
        out.append(f"| {case} | {', '.join(map(str, c['state_tokens']))} | {c['need_2']} / {c['need_3']} | {c['limit']} | "
                   f"{m['status']}, {m['rounds']}, {m['retries']} | {gt['first_two_vs_pair_max_abs_diff']}, "
                   f"{gt['third_vs_alone_max_abs_diff']}: {mark(gt['pass'])} | {pa['median_abs_diff']:.4f} / "
                   f"{pa['max_abs_diff']:.4f}, {pa['argmax_disagreements']} |")
    out += ["", "Controls for the pair against each text alone (`swa/control/`, two states in one request vs each alone, "
            "answers repeated twice):", "",
            "| control | sizes | state 0: max abs diff | state 1: max abs diff, argmax disagreements | repeat equals first run |",
            "|---|---|---|---|---|"]
    probs = {}
    for name in ("gemma-4-e4b-16k-full", "gemma-4-e4b-16k-swa-full-full", "gemma-4-e4b-16k-full-old", "gemma-4-e4b-16k-quarter"):
        r = load(d / "control" / f"{name}.json")
        s0, s1 = (x["fields"]["animal"] for x in r["repeats"][0]["per_state"])
        rep = all(v["max_abs_diff"] == 0 for x in r["repeat_vs_0"] for v in x.values())
        out.append(f"| {name} | {', '.join(map(str, r['sizes']))} | {s0['max_abs_diff']:.4f} | {s1['max_abs_diff']:.4f}, "
                   f"{s1['argmax_disagreements']} | {'yes' if rep else 'no'} |")
        probs[name] = (r["repeats"][0]["pair_probs"], r["repeats"][0]["alone_probs"])
    same = probs["gemma-4-e4b-16k-full"] == probs["gemma-4-e4b-16k-swa-full-full"] == probs["gemma-4-e4b-16k-full-old"]
    out.append(f"\nPair and alone answers equal bit for bit on the limited preset, its `--swa-full` twin and the old build: "
               f"{'yes' if same else 'no'}.")
    return out


def yes_no(text):
    """coco.chat_answer: True for an answer that starts with "yes", False for "no", None otherwise."""
    m = re.match(r"(yes|no)\b", text.strip().lower())
    return None if m is None else m.group(1) == "yes"


def chat_runs(a, b):
    """The two chat runs compared per field: (fields, first tokens that differ, parsed answers that differ, fields whose
    p_yes lies on the other side of 0.5 or is missing in one run)."""
    b = {r["image_id"]: r for r in b}
    n = tokens = parsed = side = 0
    for ra in a:
        for f, va in ra["fields"].items():
            vb = b[ra["image_id"]]["fields"][f]
            n += 1
            tokens += va["answer"] != vb["answer"]
            parsed += yes_no(va["answer"]) != yes_no(vb["answer"])
            pa, pb = va["p_yes"], vb["p_yes"]
            side += (pa is None) != (pb is None) or (pa is not None and pb is not None and (pa > 0.5) != (pb > 0.5))
    return n, tokens, parsed, side


def unclear(u):
    right = sum(1 for x in u if x["p_yes"] is not None and (x["p_yes"] > 0.5) == x["label"])
    labels = Counter("false" if x["label"] is False else "true" if x["label"] is True else "null" for x in u)
    answers = ", ".join(f'"{a}" {k}' for a, k in Counter(x["answer"] for x in u).most_common())
    return (f"{len(u)} (labels: {', '.join(f'{k} {v}' for k, v in sorted(labels.items()))}; p_yes on the label's side of 0.5: "
            f"{right}; answers: {answers})")


def coco():
    d = O / "coco"
    s = load(d / f"scores-{MODEL}.json")
    r1, r2 = load(d / f"chat-readings-{MODEL}.json"), load(d / f"chat-cached-readings-{MODEL}.json")
    out = ["## COCO val2017 subset, Qwen3.8-27B (spec 9.6)", "",
           "One decide run and two chat runs (run 1 with `cache_prompt` false, run 2 with `cache_prompt` true; otherwise the "
           "same requests). Chat: one question per call, `max_tokens` 1, temperature 0, thinking off.", "",
           "| split | field | n | accuracy (decide, T = 1) | NLL (T = 1) | accuracy (chat run 1, strict) |",
           "|---|---|---|---|---|---|"]
    for split in ("calibration", "test"):
        for f, v in s["decide_T1"][split].items():
            out.append(f"| {split} | {f} | {v['n']} | {v['accuracy']:.3f} | {v['nll']:.3f} | {s['chat_accuracy'][split][f]['accuracy']:.3f} |")
    out += ["", "Errors (wrong answers) per reading. Strict: the chat answer must start with yes or no. `p_yes`: the yes/no "
            "log-probabilities among the top 20, renormalised, read at 0.5.", "",
            "| split | n | decide | chat strict, run 1 / run 2 | chat `p_yes`, run 1 / run 2 |", "|---|---|---|---|---|"]
    for split in ("calibration", "test", "all"):
        a, b = r1["splits"][split], r2["splits"][split]
        assert a["decide"] == b["decide"], "both readings must use the same decide run"
        out.append(f"| {split} | {a['n']} | {a['decide']['errors']} ({a['decide']['accuracy']:.4f}) | "
                   f"{a['chat_strict']['errors']} / {b['chat_strict']['errors']} | {a['chat_p_yes']['errors']} / {b['chat_p_yes']['errors']} |")
    n, tokens, parsed, side = chat_runs(jsonl(d / f"chat-{MODEL}.jsonl"), jsonl(d / f"chat-cached-{MODEL}.jsonl"))
    out += ["", f"Chat answers that are neither yes nor no: run 1 {unclear(r1['unclear_answers'])}; run 2 "
            f"{unclear(r2['unclear_answers'])}.",
            "", f"Run 1 against run 2, per field ({n}): first tokens differ {tokens}, parsed yes/no/other differs {parsed}, "
            f"`p_yes` on the other side of 0.5 {side}."]
    ft = s["fitted_temperature"]
    out += ["", f"Temperature fitted on the calibration half: T = {ft['T']:.4f} (calibration NLL {ft['calibration_nll_T1']:.4f} → "
            f"{ft['calibration_nll_T']:.4f}, test NLL {ft['test_nll_T1']:.4f} → {ft['test_nll_T']:.4f}); reported only, no preset change.",
            "", "Time (medians, client wall-clock around each POST, requests one at a time):", "",
            "| condition | per image | per call | first call | calls 2-5 |", "|---|---|---|---|---|",
            f"| decide, one request with five fields | {s['time_per_image_s']['decide_median']:.3f} s (server "
            f"{s['time_per_image_s']['decide_server_ms_median']:.1f} ms) | | | |"]
    for label, r in (("chat run 1, `cache_prompt` false", r1), ("chat run 2, `cache_prompt` true", r2)):
        t = r["time_s"]
        out.append(f"| {label} | {t['chat_per_image_median']:.3f} s | {t['chat_per_call_median']:.3f} s | "
                   f"{t['chat_first_call_median']:.3f} s | {t['chat_calls_2_5_median']:.3f} s |")
    pc = r2["prompt_cache"]["calls_2_5"]
    out.append(f"\nRun 2, calls 2-5: {pc['calls']} calls, median `prompt_n` {pc['prompt_n_median']:.0f}, median `cache_n` "
               f"{pc['cache_n_median']:.0f}, calls with `cache_n` > 0: {pc['with_cache_n']}.")
    return out


def final_engine():
    """spec rev 6 and the binding checks again on the final engine (results-engine/image/final/)."""
    d = O / "final"
    new = load(d / "regression" / "qwen3.8-27b-new.info.json")["server_build"]["engine_commit"]
    reg, smoke, lim = load(d / "regression" / "qwen3.8-27b.json"), load(d / f"smoke-{BINDING}.json"), load(d / f"limits-{BINDING}.json")
    out = ["## Final engine (spec rev 6)", "",
           f"On the final engine `{new[:9]}`: the text regression against the old build, the server smoke, the limits of rev 6 "
           f"and every integrity check of the binding model (`final/`).", "",
           "| check | result |", "|---|---|",
           f"| text regression qwen3.8-27b, old vs new: {reg['requests']} requests, max abs diff {reg['max_abs_diff']}, "
           f"{len(reg['differences'])} differences | {mark(reg['pass'])} |",
           f"| smoke {BINDING}: {sum(c['pass'] for c in smoke['checks'])} of {len(smoke['checks'])} checks | "
           f"{mark(all(c['pass'] for c in smoke['checks']))} |"]
    out += [f"| limits {BINDING}: {name} | {mark(ok)} |" for name, ok in lim["checks"].items()]
    early = [f"over after image {k} (text {e['text_tokens']} tokens): {e['status']} {e['error'].get('code', '')}"
             for k in (1, 2) if (e := lim.get(f"over after image {k}"))]
    out += ["", f"Early budget check: one image has {lim.get('image_cells')} cells, the largest state the budget admits is "
            f"{lim.get('largest_state')}; " + "; ".join(early) + "."]
    i = d / "integrity" / BINDING
    m = load(i / "malformed.json")
    out += ["", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD), integrity_row(i, BINDING), "",
            f"Malformed cases: {sum(m['checks'].values())} of {len(m['checks'])} checks pass "
            f"({len(m['cases'])} requests, the aspect-ratio cases among them)."]
    f = i / "failures.txt"
    out += ["", "Failures (`failures.txt`): " + ("; ".join(x.strip() for x in f.read_text().splitlines() if x.strip())
                                                 if f.exists() else "none") + "."]
    return out


def main():
    lines = ["# Sub-project 4 (image input): results", "", "Generated by `uv run results-engine/image/notes.py` from the files in "
             "this directory.", ""]
    for section in (regression, no_vision, integrity, swa, coco, final_engine):
        lines += section() + [""]
    (O / "notes.md").write_text("\n".join(lines))
    print(f"wrote {O / 'notes.md'}")


if __name__ == "__main__":
    main()
