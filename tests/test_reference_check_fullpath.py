"""reference_check.py full-path, `after` and row-cap split checks (spec 4.1, 5.2, 12.3) on synthetic dump records and
responses with hand-computed values; tree-mode dump records and comparisons keep their format."""

import json
import math
import subprocess
import sys

import httpx
import pytest

import reference_check as rc

LN = math.log
PROMPT_PREFIX = [1000, 1001, 1002]  # prefix + state + tail of the synthetic state


# A synthetic full-path field "q" (choice ["ab", "cd"]): stem [100, 101]; option 0 = [10, 1] (the value token, then the
# closing quote 1), option 1 = [11] (one token 'cd"'). Nodes: 0 root (row after 101), 1 after 10, 2 leaf of option 0,
# 3 leaf of option 1. The root row is read in option 0's sequence only (spec 4.3).
def engine_rows(quote_p=0.001):
    """The engine's rows: root children 10 (0.5) and 11 (0.2), MERGED['cd"'] 0.25 for the leaf child 11; node 1: the
    quote at quote_p, MERGED['"'] 0.9; END 0.5 and 0.4 on the leaves."""
    return {
        0: {"node": 0, "children": {"10": LN(0.5), "11": LN(0.2)}, "merged": {"11": LN(0.25)}},
        1: {"node": 1, "children": {"1": LN(quote_p)}, "merged": {"1": LN(0.9)}},
        2: {"node": 2, "end": LN(0.5)},
        3: {"node": 3, "end": LN(0.4)},
    }


def engine_answer(quote_p=0.001):
    """spec 4.1 on engine_rows: P(ab) = 0.5 (quote_p 0.5 + 0.9), P(cd) = 0.2 x 0.4 + 0.25."""
    p0, p1 = 0.5 * (quote_p * 0.5 + 0.9), 0.2 * 0.4 + 0.25
    z = p0 + p1
    return {"value": "ab", "probabilities": {"ab": p0 / z, "cd": p1 / z}, "confidence": p0 / z, "coverage": z}


def fp_record(rows, answer, ticket=0):
    entries = [
        {"option": 0, "tokens": [100, 101, 10, 1], "rows": [1, 2, 3], "logp": [rows[0], rows[1], rows[2]]},
        {"option": 1, "tokens": [100, 101, 11], "rows": [2], "logp": [rows[3]]},
    ]
    return {"ticket": ticket, "field": "q", "type": "choice", "keys": ["ab", "cd"], "engine": answer, "mode": "full_path",
            "prompt_prefix": PROMPT_PREFIX, "entries": entries}


# The replay's top lists (token, piece, probability) per prompt after PROMPT_PREFIX; the last entry is the smallest
# probability ("the 512th"). The quote token 1 is in none of them (an unverifiable edge; its piece comes from /detokenize).
REPLAY = {
    (100, 101): [(10, b"ab", 0.52), (11, b'cd"', 0.18), (12, b'cd",', 0.20), (13, b'cd"\n', 0.05), (14, b'cd"x', 0.02),
                 (15, b"}", 0.01)],
    (100, 101, 10): [(20, b'",', 0.85), (21, b'",&', 0.04), (22, b'"}', 0.02), (23, b'"a', 0.01), (24, b" ", 0.005)],
    (100, 101, 10, 1): [(33, b"x", 0.5), (30, b",", 0.3), (31, b"\n", 0.1), (32, b" }", 0.05), (34, b",&", 0.02)],
    (100, 101, 11): [(30, b",", 0.35), (36, b"a}", 0.1), (35, b"\t", 0.05), (37, b"", 0.01)],
}
PIECES = {1: '"'}


def fake_server(replay, pieces, prefix=PROMPT_PREFIX):
    """An httpx client whose /completion answers with the replay list of the prompt and /detokenize with `pieces`."""
    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/completion":
            prompt = body["prompt"]
            assert prompt[: len(prefix)] == prefix
            assert (body["n_predict"], body["n_probs"], body["cache_prompt"], body["post_sampling_probs"]) == (1, 512, False, False)
            top = replay[tuple(prompt[len(prefix):])]
            return httpx.Response(200, json={"completion_probabilities": [{"id": top[0][0], "top_logprobs": [
                {"id": t, "token": s.decode(errors="replace"), "bytes": list(s), "logprob": math.log(p)} for t, s, p in top]}]})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": "".join(pieces[t] for t in body["tokens"])})
        return httpx.Response(404)
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("w, want", [
    (b",", True), (b"}", True), (b" ,", True), (b"\n", True), (b" \t\r\n", True), (b",&", True), (b" }x", True),
    (b"", False), (b"x,", False), (b" x", False), (b'"', False), (b'",', False),
])
def test_ends_value_is_the_engine_predicate(w, want):
    assert rc.ends_value(w) is want


def test_merged_membership():
    assert rc.in_merged(b'",&', b'"')      # the remainder ',&' ends a value (its first non-space character is ',')
    assert rc.in_merged(b'",\n', b'"')
    assert rc.in_merged(b'" \n', b'"')
    assert not rc.in_merged(b'"', b'"')    # empty remainder
    assert not rc.in_merged(b'"a', b'"')
    assert not rc.in_merged(b'x",', b'"')  # does not start with the base
    assert not rc.in_merged(b",", b"")     # empty base


def test_full_path_reference_hand_computed():
    rec = fp_record(engine_rows(), engine_answer())
    out = rc.compare_full_path([rec], fake_server(REPLAY, PIECES), "p")
    d = out["detail"][0]
    # spec 4.1 on the replay's values. Option 0: root child 10 (0.52) x (quote x END + MERGED['"']); the quote is not in
    # node 1's list, so its engine value 0.001 is used; END = ',' '\n' ' }' ',&' = 0.47; MERGED['"'] = '",' '",&' '"}' = 0.91.
    ref0 = 0.52 * (0.001 * (0.3 + 0.1 + 0.05 + 0.02) + (0.85 + 0.04 + 0.02))
    # Option 1: root child 11 (0.18) x END (',' '\t' = 0.40) + MERGED['cd"'] at the root ('cd",' 'cd"\n' = 0.25).
    ref1 = 0.18 * (0.35 + 0.05) + (0.20 + 0.05)
    eng0, eng1 = 0.5 * (0.001 * 0.5 + 0.9), 0.2 * 0.4 + 0.25
    assert ref0 == pytest.approx(0.4734444) and ref1 == pytest.approx(0.322)
    assert d["coverage_reference"] == pytest.approx(ref0 + ref1)
    assert d["reference"] == pytest.approx([ref0 / (ref0 + ref1), ref1 / (ref0 + ref1)])
    assert d["coverage_engine"] == pytest.approx(eng0 + eng1)
    assert d["engine"] == pytest.approx([eng0 / (eng0 + eng1), eng1 / (eng0 + eng1)])
    assert d["engine_recompute_max_abs_diff"] < 1e-12  # the dumped rows reproduce the engine's answer
    assert d["argmax_agree"]

    f = out["fields"]["q"]
    # node comparison in probability, the unverifiable quote edge left out: child 10 |0.5-0.52|, child 11 |0.2-0.18|,
    # MERGED['cd"'] 0, MERGED['"'] |0.9-0.91|, END node 2 |0.5-0.47|, END node 3 0 -> median 0.015, max 0.03
    assert f["node_values"] == 6 and f["node_both_zero"] == 0
    assert f["node_median_abs_diff"] == pytest.approx(0.015)
    assert f["node_max_abs_diff"] == pytest.approx(0.03)
    assert not f["node_pass"]
    assert f["end_merged_replay_excess_max"] == pytest.approx(0.01)  # MERGED['"'] 0.91 in the replay vs 0.9
    assert f["coverage_median_abs_diff"] == pytest.approx((ref0 + ref1) - (eng0 + eng1))  # 0.0152
    assert f["coverage_pass"]
    diff = ref0 / (ref0 + ref1) - eng0 / (eng0 + eng1)  # 0.0181, the same for both options
    assert f["median_abs_diff"] == pytest.approx(diff) and f["max_abs_diff"] == pytest.approx(diff)
    assert f["disagreements_over_margin"] == 0 and not f["statistic_pass"]
    assert not f["pass"]

    u = out["unverifiable"]
    assert (u["edges"], u["unverifiable"]) == (3, 1)  # root: 10, 11; node 1: the quote
    edge = u["detail"][0]
    assert (edge["ticket"], edge["field"], edge["node"], edge["token"]) == (0, "q", 1, 1)
    assert edge["p_engine"] == pytest.approx(0.001) and edge["p_min"] == pytest.approx(0.005)
    assert edge["ratio"] == pytest.approx(0.2) and edge["within_bound"]  # 0.001 <= 2 x 0.005
    assert u["fraction"] == pytest.approx(1 / 3) and not u["pass"]  # over 5 %
    assert not out["pass"]
    assert "dependent" not in out


def test_full_path_reference_exact_replay_passes():
    # every engine value is in the replay's lists (one MERGED/END token each), so reference = engine
    replay = {
        (100, 101): [(10, b"ab", 0.5), (12, b'cd",', 0.25), (11, b'cd"', 0.2)],
        (100, 101, 10): [(20, b'",', 0.9), (1, b'"', 0.001)],
        (100, 101, 10, 1): [(30, b",", 0.5)],
        (100, 101, 11): [(30, b",", 0.4)],
    }
    rec = fp_record(engine_rows(), engine_answer())
    out = rc.compare_full_path([rec], fake_server(replay, {}), "p")  # every piece is in a list: no /detokenize
    f = out["fields"]["q"]
    assert f["median_abs_diff"] < 1e-12 and f["node_max_abs_diff"] < 1e-12 and f["coverage_max_abs_diff"] < 1e-12
    assert f["statistic_pass"] and f["coverage_pass"] and f["node_pass"] and f["pass"]
    assert (out["unverifiable"]["edges"], out["unverifiable"]["unverifiable"]) == (3, 0) and out["unverifiable"]["pass"]
    assert out["pass"]
    # sanity gate: median engine coverage 0.78025 < 0.8 (reported, not part of pass)
    assert out["coverage_gate"] == {"threshold": 0.8, "median_engine_coverage": {"q": pytest.approx(0.78025)}, "pass": False}


def test_unverifiable_edge_over_the_bound():
    # the engine gives the quote 0.02 > 2 x 0.005 (the smallest probability of node 1's list)
    rec = fp_record(engine_rows(quote_p=0.02), engine_answer(quote_p=0.02))
    out = rc.compare_full_path([rec], fake_server(REPLAY, PIECES), "p")
    edge = out["unverifiable"]["detail"][0]
    assert edge["ratio"] == pytest.approx(4.0) and not edge["within_bound"]
    assert not out["unverifiable"]["within_bound"] and not out["unverifiable"]["pass"]
    ref0, ref1 = 0.52 * (0.02 * 0.47 + 0.91), 0.18 * 0.40 + 0.25  # the reference takes the engine's 0.02 for that edge
    assert out["detail"][0]["reference"] == pytest.approx([ref0 / (ref0 + ref1), ref1 / (ref0 + ref1)])


def test_unverifiable_edge_between_one_and_two_times_the_smallest_listed():
    # 0.008 is above the list's smallest probability 0.005 but within twice it: allowed
    rec = fp_record(engine_rows(quote_p=0.008), engine_answer(quote_p=0.008))
    out = rc.compare_full_path([rec], fake_server(REPLAY, PIECES), "p")
    edge = out["unverifiable"]["detail"][0]
    assert edge["ratio"] == pytest.approx(1.6) and edge["within_bound"] and out["unverifiable"]["within_bound"]


def test_unverifiable_cap_is_five_percent_of_a_models_edges():
    edge = {"within_bound": True, "ratio": 1.5}
    assert rc.unverifiable_summary([edge], 20)["pass"]  # 1 of 20 = 5 %
    assert not rc.unverifiable_summary([edge, edge], 20)["pass"]  # 10 %
    assert not rc.unverifiable_summary([dict(edge, within_bound=False)], 100)["pass"]
    empty = rc.unverifiable_summary([], 20)
    assert empty["pass"] and empty["fraction"] == 0 and empty["max_ratio"] is None


def test_merged_null_pairs_are_counted_not_compared():
    # a leaf child without MERGED tokens: the engine writes null (-inf) and the replay finds none either
    rows = {0: {"node": 0, "children": {"5": LN(0.6), "6": LN(0.4)}, "merged": {"5": None, "6": None}},
            1: {"node": 1, "end": LN(0.9)}, 2: {"node": 2, "end": LN(0.8)}}
    rec = {"ticket": 0, "field": "u", "type": "score", "keys": ["0", "1"], "mode": "full_path", "prompt_prefix": PROMPT_PREFIX,
           "engine": {"value": 0, "probabilities": [0.54 / 0.86, 0.32 / 0.86], "coverage": 0.86},
           "entries": [{"option": 0, "tokens": [200, 5], "rows": [0, 1], "logp": [rows[0], rows[1]]},
                       {"option": 1, "tokens": [200, 6], "rows": [1], "logp": [rows[2]]}]}
    replay = {(200,): [(5, b"0", 0.6), (6, b"1", 0.4)], (200, 5): [(30, b",", 0.9)], (200, 6): [(30, b",", 0.8)]}
    out = rc.compare_full_path([rec], fake_server(replay, {}), "p")
    f = out["fields"]["u"]
    assert f["node_values"] == 4 and f["node_both_zero"] == 2  # children 5, 6 and two END; the two null MERGED pairs
    assert f["node_max_abs_diff"] < 1e-12 and f["pass"]
    assert out["detail"][0]["reference"] == pytest.approx([0.54 / 0.86, 0.32 / 0.86])


# spec 5.2: lines(F) = over the ancestors in declaration order: stem text + JSON value + ",\n"
ANC = {"queue": "sales", "urgency": 2, "angry": True, "topic": 'say "hi"'}
LINES = '  "queue": "sales",\n  "urgency": 2,\n  "angry": true,\n  "topic": "say \\"hi\\"",\n'


def dependent(**changes):
    rec = {"ticket": 3, "field": "tag", "engine": {"value": "a", "given": dict(ANC)}, "mode": "full_path",
           "ancestors": dict(ANC), "entries": [{"option": 0, "lines": LINES, "given": dict(ANC)},
                                               {"option": 1, "lines": LINES, "given": dict(ANC)}]}
    rec.update(changes)
    return rec


def test_dependent_given_and_lines():
    assert rc.expected_lines(ANC) == LINES
    level0 = {"ticket": 3, "field": "queue", "engine": {"value": "sales"}, "mode": "full_path", "ancestors": {},
              "entries": [{"option": 0}, {"option": 1}]}
    tree = {"ticket": 3, "field": "tag", "engine": {"value": "a", "given": dict(ANC)}, "ancestors": dict(ANC),
            "nodes": [{"node": 0, "lines": LINES, "given": dict(ANC)}]}
    assert rc.dependent_problems(dependent()) == []
    assert rc.dependent_problems(level0) == [] and rc.dependent_problems(tree) == []
    check = rc.dependent_check([level0, dependent(), tree])
    assert check["pass"] and check["dependent_records"] == 2 and check["dependent_entries"] == 3

    wrong_value = dependent(entries=[{"option": 0, "lines": LINES, "given": dict(ANC, queue="billing")},
                                     {"option": 1, "lines": LINES, "given": dict(ANC)}])
    assert len(rc.dependent_problems(wrong_value)) == 1
    reordered = dict(reversed(list(ANC.items())))
    assert len(rc.dependent_problems(dependent(entries=[{"option": 0, "lines": LINES, "given": reordered}]))) == 1
    assert len(rc.dependent_problems(dependent(entries=[{"option": 0, "lines": LINES.replace("sales", "billing"),
                                                         "given": dict(ANC)}]))) == 1
    assert len(rc.dependent_problems(dependent(entries=[{"option": 0}]))) == 2  # neither given nor lines
    assert len(rc.dependent_problems(dependent(engine={"value": "a", "given": {"queue": "sales"}}))) == 1
    assert len(rc.dependent_problems(dict(level0, entries=[{"option": 0, "lines": "x", "given": {}}]))) == 1
    assert len(rc.dependent_problems(dict(level0, engine={"value": "sales", "given": {}}))) == 1
    assert "ticket 3 tag" in rc.dependent_problems(wrong_value)[0]
    assert not rc.dependent_check([level0, wrong_value])["pass"]
    assert rc.dependent_check([{"ticket": 0, "field": "q", "nodes": []}]) is None  # no --after: no check


def test_ancestors_are_transitive_in_declaration_order():
    fields = {"d": {"after": ["c", "a"]}, "a": {}, "b": {"after": ["a"]}, "c": {"after": ["b"]}, "e": {}}
    assert rc.ancestors_of(fields) == {"d": ["a", "b", "c"], "a": [], "b": ["a"], "c": ["a", "b"], "e": []}


def test_parse_after():
    assert rc.parse_after(["urgency=queue", "angry=queue,urgency"]) == {"urgency": ["queue"], "angry": ["queue", "urgency"]}
    for bad in (["urgency"], ["urgency="], ["=queue"], ["angry=queue,"]):
        with pytest.raises(SystemExit):
            rc.parse_after(bad)


def test_parse_after_rejects_a_repeated_field_and_unknown_names():
    for bad in (["urgency=queue", "urgency=angry"], ["urgency=team"], ["mood=queue"]):
        with pytest.raises(SystemExit):
            rc.parse_after(bad)
    assert rc.parse_after(["u=q"], ["q", "u"]) == {"u": ["q"]}  # a --schema request names its own fields


@pytest.mark.parametrize("after, message", [(["u=q", "u=q"], "given twice"), (["u=x"], "unknown field 'x'"),
                                            (["urgency=queue"], "unknown field 'urgency'")])
def test_dump_rejects_a_repeated_or_unknown_after_field(monkeypatch, tmp_path, after, message):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA))  # fields q and u
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: pytest.fail("llama-decide must not run"))
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m", "--preset", "p", "--schema", str(schema),
                                      "--tag", "t", "--out-dir", str(tmp_path), *[x for a in after for x in ("--after", a)]])
    with pytest.raises(SystemExit) as e:
        rc.main()
    assert message in str(e.value.code)


SCHEMA = {"instructions": "Answer.", "fields": {"q": {"type": "choice", "description": "Q?", "options": ["ab", "cd"]},
                                                "u": {"type": "bool", "description": "U?"}}}
ANSWER_Q = {"value": "ab", "probabilities": {"ab": 0.6, "cd": 0.4}, "confidence": 0.6, "coverage": 0.9}
ANSWER_U = {"value": True, "p_true": 0.7, "confidence": 0.7, "coverage": 0.95, "given": {"q": "ab"}}
LINES_Q = '  "q": "ab",\n'
FP_BRANCHES = [
    {"field": "q", "mode": "full_path", "option": 0, "tokens": [100, 101, 10, 7], "rows": [1, 2, 3], "variant": 0, "phase": 0,
     "logp": [{"node": 0, "children": {"10": -0.5, "11": -1.0}, "merged": {"11": None}},
              {"node": 1, "children": {"7": -9.0}, "merged": {"7": -0.1}}, {"node": 2, "end": -0.2}]},
    {"field": "q", "mode": "full_path", "option": 1, "tokens": [100, 101, 11], "rows": [2], "variant": 0, "phase": 0,
     "logp": [{"node": 3, "end": -0.3}]},
    {"field": "u", "mode": "full_path", "option": 0, "tokens": [60, 61, 200, 201, 20], "rows": [3, 4], "variant": 0, "phase": 1,
     "logp": [{"node": 0, "children": {"20": -0.4, "21": -1.2}, "merged": {"20": None, "21": None}}, {"node": 1, "end": -0.01}],
     "lines": LINES_Q, "given": {"q": "ab"}},
    {"field": "u", "mode": "full_path", "option": 1, "tokens": [60, 61, 200, 201, 21], "rows": [4], "variant": 0, "phase": 1,
     "logp": [{"node": 2, "end": -0.02}], "lines": LINES_Q, "given": {"q": "ab"}},
]


def response(branches, answers, prefix=(1, 2)):
    return {"object": "decide", "results": [{"answers": answers}], "usage": {}, "timings": {}, "engine": {},
            "tokens": {"prefix": list(prefix), "prefixes": [list(prefix)],
                       "states": [{"state": [3, 4], "tail": [5], "branches": branches}]}}


def run_dump(monkeypatch, tmp_path, stdout, *extra):
    """reference_check.py dump on one ticket ("x") with SCHEMA and a faked llama-decide; returns (cmd, stdin lines)."""
    test_set = tmp_path / "t.jsonl"
    test_set.write_text(json.dumps({"text": "x"}) + "\n")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    seen = []

    def fake_run(cmd, input=None, **kwargs):
        seen.append((cmd, input))
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m.gguf", "--preset", "p", "--n", "1",
                                      "--test-set", str(test_set), "--schema", str(schema), "--out-dir", str(tmp_path), *extra])
    rc.main()
    cmd, stdin = seen[-1]
    return cmd, [json.loads(l) for l in stdin.splitlines()]


def test_dump_full_path_after_records(monkeypatch, tmp_path):
    stdout = json.dumps(response(FP_BRANCHES, {"q": ANSWER_Q, "u": ANSWER_U})) + "\n"
    cmd, bodies = run_dump(monkeypatch, tmp_path, stdout, "--scoring", "full_path", "--after", "u=q", "--tag", "t",
                           "--decide-seqs", "64")
    assert "--dump-tokens" in cmd and cmd[cmd.index("--decide-seqs") + 1] == "64"
    assert bodies == [{"instructions": "Answer.", "fields": {"q": SCHEMA["fields"]["q"], "u": dict(SCHEMA["fields"]["u"], after=["q"])},
                       "states": ["x"], "options": {"scoring": "full_path"}}]
    records = [json.loads(l) for l in (tmp_path / "reference-t-p-dump.jsonl").read_text().splitlines()]
    strip = lambda b, *extra: {k: b[k] for k in ("option", "tokens", "rows", "logp", *extra)}  # noqa: E731
    assert records == [
        {"ticket": 0, "field": "q", "type": "choice", "keys": ["ab", "cd"], "engine": ANSWER_Q, "mode": "full_path",
         "prompt_prefix": [1, 2, 3, 4, 5], "entries": [strip(b) for b in FP_BRANCHES[:2]], "ancestors": {}},
        {"ticket": 0, "field": "u", "type": "bool", "keys": ["true", "false"], "engine": ANSWER_U, "mode": "full_path",
         "prompt_prefix": [1, 2, 3, 4, 5], "entries": [strip(b, "lines", "given") for b in FP_BRANCHES[2:]],
         "ancestors": {"q": "ab"}},
    ]
    assert rc.dependent_check(records)["pass"]


def test_dump_numbers_states_over_batched_lines(monkeypatch, tmp_path):
    test_set = tmp_path / "t.jsonl"
    test_set.write_text("".join(json.dumps({"text": t}) + "\n" for t in ("a", "b", "c")))
    two = response(FP_BRANCHES[:2], {"q": ANSWER_Q})
    two["results"].append({"answers": {"q": dict(ANSWER_Q, value="cd")}})
    two["tokens"]["states"].append({"state": [9], "tail": [5], "branches": FP_BRANCHES[:2]})
    one = response(FP_BRANCHES[:2], {"q": ANSWER_Q})
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"instructions": "Answer.", "fields": {"q": SCHEMA["fields"]["q"]}}))
    seen = []
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, input=None, **kw: seen.append(input) or subprocess.CompletedProcess(
        cmd, 0, json.dumps(two) + "\n" + json.dumps(one) + "\n", ""))
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m", "--preset", "p", "--n", "3", "--batch", "2",
                                      "--test-set", str(test_set), "--schema", str(schema), "--scoring", "full_path",
                                      "--tag", "t", "--out-dir", str(tmp_path)])
    rc.main()
    assert [json.loads(l)["states"] for l in seen[0].splitlines()] == [["a", "b"], ["c"]]
    records = [json.loads(l) for l in (tmp_path / "reference-t-p-dump.jsonl").read_text().splitlines()]
    assert [(r["ticket"], r["engine"]["value"], r["prompt_prefix"]) for r in records] == [
        (0, "ab", [1, 2, 3, 4, 5]), (1, "cd", [1, 2, 9, 5]), (2, "ab", [1, 2, 3, 4, 5])]
    assert all("ancestors" not in r for r in records)


@pytest.mark.parametrize("extra", [["--scoring", "full_path"], ["--after", "urgency=queue"], ["--batch", "5"],
                                   ["--decide-seqs", "64"], ["--schema", "tests/data/decide-multibranch.json"]])
def test_dump_needs_a_tag_beside_the_default_dump(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: pytest.fail("llama-decide must not run"))
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m", "--preset", "p", "--out-dir", str(tmp_path), *extra])
    with pytest.raises(SystemExit) as e:
        rc.main()
    assert "--tag is required" in str(e.value.code)


TREE_BRANCHES = [
    {"field": "q", "mode": "tree", "node": 0, "tokens": [100, 101], "children": [10, 11], "options": [[0], [1]], "variant": 0, "phase": 0},
    {"field": "u", "mode": "tree", "node": 0, "tokens": [60, 61], "children": [20, 21], "options": [[0], [1]], "variant": 0, "phase": 0},
]


def test_dump_tree_records_keep_their_format(monkeypatch, tmp_path):
    answers = {"q": ANSWER_Q, "u": {"value": True, "p_true": 0.7, "confidence": 0.7}}
    cmd, bodies = run_dump(monkeypatch, tmp_path, json.dumps(response(TREE_BRANCHES, answers)) + "\n", "--tag", "t")
    assert bodies == [{"instructions": "Answer.", "fields": SCHEMA["fields"], "states": ["x"]}]  # no options, as before
    expected = [
        {"ticket": 0, "field": "q", "type": "choice", "keys": ["ab", "cd"], "engine": ANSWER_Q,
         "nodes": [{"node": 0, "prompt_tokens": [1, 2, 3, 4, 5, 100, 101], "children": [10, 11], "options": [[0], [1]]}]},
        {"ticket": 0, "field": "u", "type": "bool", "keys": ["true", "false"], "engine": answers["u"],
         "nodes": [{"node": 0, "prompt_tokens": [1, 2, 3, 4, 5, 60, 61], "children": [20, 21], "options": [[0], [1]]}]},
    ]
    # a tree dump's records, byte for byte as before (--schema needs a --tag)
    assert (tmp_path / "reference-t-p-dump.jsonl").read_text() == "".join(json.dumps(r) + "\n" for r in expected)


def test_compare_tree_records_as_before(monkeypatch, tmp_path):
    rec = {"ticket": 0, "field": "q", "type": "choice", "keys": ["ab", "cd"], "engine": {"probabilities": {"ab": 0.7, "cd": 0.3}},
           "nodes": [{"node": 0, "prompt_tokens": PROMPT_PREFIX + [100], "children": [10, 11], "options": [[0], [1]]}]}
    (tmp_path / "reference-t-p-dump.jsonl").write_text(json.dumps(rec) + "\n")
    client = fake_server({(100,): [(10, b"a", 0.3), (11, b"c", 0.1), (12, b"z", 0.05)]}, {})
    monkeypatch.setattr(rc.httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "compare", "--preset", "p", "--tag", "t", "--out-dir", str(tmp_path)])
    rc.main()
    out = json.loads((tmp_path / "reference-t-p.json").read_text())
    assert list(out) == ["preset", "tag", "n_records", "fields", "pass", "detail"]
    # children renormalised: 0.3 / 0.4 = 0.75, 0.25 against the engine's 0.7 / 0.3
    assert out["fields"]["q"] == {"median_abs_diff": pytest.approx(0.05), "max_abs_diff": pytest.approx(0.05),
                                  "argmax_agreement": 1, "disagreements_over_margin": 0,
                                  "argmax_abs_diff_median": pytest.approx(0.05)}
    assert out["detail"][0]["reference"] == pytest.approx([0.75, 0.25]) and out["pass"] is False


def test_compare_writes_the_full_path_result(monkeypatch, tmp_path):
    rec = fp_record(engine_rows(), engine_answer())
    (tmp_path / "reference-t-p-dump.jsonl").write_text(json.dumps(rec) + "\n")
    client = fake_server(REPLAY, PIECES)
    monkeypatch.setattr(rc.httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "compare", "--preset", "p", "--tag", "t", "--out-dir", str(tmp_path)])
    rc.main()
    out = json.loads((tmp_path / "reference-t-p.json").read_text())
    assert (out["preset"], out["tag"], out["mode"], out["n_records"], out["pass"]) == ("p", "t", "full_path", 1, False)
    assert out["fields"]["q"]["node_median_abs_diff"] == pytest.approx(0.015)


def decide_response(states, p_true, decodes, rounds=1, cached=100, rows=2):
    return {"results": [{"answers": {"angry": {"value": p > 0.5, "p_true": p}}} for p in p_true],
            "timings": {"decodes": decodes, "rounds": rounds}, "usage": {"cached_tokens": cached, "scored_tokens": rows * states},
            "engine": {"states_per_round": states}}


def test_split_summary_side_by_side():
    # full-path: two 2-state lines (the first built its prefix: 4 decodes = prefix + trunk + 2 branch chunks)
    fp_b = [decide_response(2, [0.10, 0.50], decodes=4, cached=0), decide_response(2, [0.30, 0.90], decodes=3)]
    fp_1 = [decide_response(1, [p], decodes=d, cached=c) for p, d, c in ((0.101, 3, 0), (0.52, 2, 100), (0.303, 2, 100), (0.904, 2, 100))]
    tree_b = [decide_response(2, [0.10, 0.50], decodes=3, cached=0), decide_response(2, [0.30, 0.90], decodes=2)]
    tree_1 = [decide_response(1, [p], decodes=2) for p in (0.10, 0.49, 0.30, 0.95)]
    out = rc.split_summary(fp_b, fp_1, tree_b, tree_1)
    assert [l["branch_decodes"] for l in out["full_path"]["lines"]] == [2, 2]
    assert out["full_path"]["split"] and out["tree"]["branch_decodes"] == [1, 1]
    assert out["full_path"]["single_branch_decodes"] == [1, 1, 1, 1] and out["full_path"]["single_unsplit"]
    # |diffs| (each counted for true and false): full-path .001 .02 .003 .004 -> median .0035; tree 0 .01 0 .05 -> .005
    side = out["side_by_side"]["angry"]
    assert side["full_path"]["median_abs_diff"] == pytest.approx(0.0035) and side["full_path"]["pass"]
    assert side["full_path"]["max_abs_diff"] == pytest.approx(0.02)
    assert side["tree"]["median_abs_diff"] == pytest.approx(0.005) and side["tree"]["max_abs_diff"] == pytest.approx(0.05)
    assert out["pass"]
    # a one-state reference line that split its branch phase is not the unsplit reference
    fp_1_split = fp_1[:3] + [decide_response(1, [0.904], decodes=3)]
    assert not rc.split_summary(fp_b, fp_1_split, tree_b, tree_1)["pass"]
    # one full-path line decoded its branch phase in one chunk: no split
    fp_b[1] = decide_response(2, [0.30, 0.90], decodes=2)
    assert not rc.split_summary(fp_b, fp_1, tree_b, tree_1)["pass"]
    # two rounds per line: the check expects one round per batched request
    fp_b[1] = decide_response(2, [0.30, 0.90], decodes=5, rounds=2)
    assert not rc.split_summary(fp_b, fp_1, tree_b, tree_1)["full_path"]["split"]


def test_replay_is_recorded_and_served_offline(tmp_path):
    # every /completion and /detokenize answer is kept, and compare can run from that file alone
    rec = fp_record(engine_rows(), engine_answer())
    live = rc.RecordingClient(fake_server(REPLAY, PIECES))
    out_live = rc.compare_full_path([rec], live, "p")
    path = tmp_path / "replay.jsonl.gz"
    live.write(path)
    assert sum(1 for c in live.calls if c["path"] == "/completion") == 4 and any(c["path"] == "/detokenize" for c in live.calls)
    offline = rc.offline_client(path)
    out_offline = rc.compare_full_path([rec], offline, "p")
    assert json.dumps(out_offline, sort_keys=True) == json.dumps(out_live, sort_keys=True)
    with pytest.raises(SystemExit):  # a prompt that was never replayed is not invented
        rc.top_logprobs(offline, "p", [1, 2, 3], {})


def test_recorded_lists_keep_float32_logprobs_and_the_generated_token(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"completion_probabilities": [{"id": 7, "logprob": -0.25, "top_logprobs": [
            {"id": 7, "token": "�", "bytes": [226, 128], "logprob": -0.25},
            {"id": 9, "token": "x", "bytes": [120], "logprob": -3.4028234663852886e38}]}]})
    live = rc.RecordingClient(httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler)))
    body = {"model": "p", "prompt": [5, 6], "n_predict": 1, "n_probs": 512, "logit_bias": [[7, 100]]}
    live.post("/completion", json=body)
    path = tmp_path / "r.jsonl.gz"
    live.write(path)
    got = rc.offline_client(path).post("/completion", json=body).json()["completion_probabilities"][0]
    assert (got["id"], got["logprob"]) == (7, -0.25)
    assert [(t["id"], bytes(t["bytes"]), t["logprob"]) for t in got["top_logprobs"]] == [
        (7, b"\xe2\x80", -0.25), (9, b"x", -3.4028234663852886e38)]
