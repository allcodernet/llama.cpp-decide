import json
import subprocess

import pytest

import cache_timing


def timed(prefill, prefix, total, hits, misses):
    return {"timings": {"prefill_ms": prefill, "prefix_ms": prefix, "scoring_ms": 1.0, "total_ms": total},
            "engine": {"prefix_cache": {"capacity": 8, "entries": 2, "hits": hits, "misses": misses}}}


def test_alternation_lines_alternate_two_prefixes():
    lines = cache_timing.alternation_lines("qwen2.5-0.5b", ["t1", "t2"], 3)
    assert len(lines) == 6
    a, b = lines[0], lines[1]
    assert a["instructions"] != b["instructions"]  # keywords vs default prompt: two prefixes
    assert lines == [a, b] * 3
    # the schema labels of the summaries follow this order: A = the keywords variant, B = the default variant
    assert cache_timing.alternation_schemas(3) == ["A", "B"] * 3
    assert a == cache_timing.build_request("qwen2.5-0.5b", ["t1", "t2"], cache_timing.SCHEMAS["A"])
    assert b == cache_timing.build_request("qwen2.5-0.5b", ["t1", "t2"], cache_timing.SCHEMAS["B"])
    assert a["states"] == b["states"] == ["t1", "t2"]
    assert a["options"] == {"scoring": "tree", "order_debias": 0}


def test_summarize_medians_per_schema_and_cache_counts():
    # A and B have prefixes of different lengths: two timing groups, so medians per schema and none over both
    runs = [timed(300, 200, 400, 0, 1), timed(210, 95, 310, 0, 1),  # A, B: the first of each misses
            timed(120, 6, 212, 1, 0), timed(118, 5, 208, 1, 0),  # A, B
            timed(122, 7, 216, 1, 0), timed(117, 4, 205, 1, 0)]  # A, B
    s = cache_timing.summarize(runs, ["A", "B"] * 3)
    assert (s["requests"], s["hits"], s["misses"]) == (6, 4, 2)
    a, b = s["per_schema"]["A"], s["per_schema"]["B"]
    assert a["medians"] == {"prefill_ms": 122, "prefix_ms": 7, "total_ms": 216}
    assert b["medians"] == {"prefill_ms": 118, "prefix_ms": 5, "total_ms": 208}
    assert (a["requests"], a["hits"], a["misses"]) == (3, 2, 1) and (b["requests"], b["hits"], b["misses"]) == (3, 2, 1)
    assert "medians" not in s  # the pooled prefill median would be 121, between the groups
    assert s["per_request"][2] == {"schema": "A", "prefill_ms": 120, "prefix_ms": 6, "total_ms": 212, "hits": 1, "misses": 0}
    with pytest.raises(ValueError):
        cache_timing.summarize(runs, ["A", "B"])


def test_resummarize_labels_stored_requests_in_alternation_order(tmp_path):
    # a file written before the per-schema summaries: per-request timings in line order (A, B, A, B), pooled medians
    names = ("prefill_ms", "prefix_ms", "total_ms", "hits", "misses")
    per_request = [dict(zip(names, t)) for t in [(300, 200, 400, 0, 0), (210, 95, 310, 0, 0), (290, 190, 390, 0, 0),
                                                 (200, 90, 300, 0, 0)]]
    stored = {"preset": "p", "alternations": 2, "schemas": {"A": "keywords", "B": "default"},
              "runs": {"0": {"requests": 4, "medians": {"prefill_ms": 250, "prefix_ms": 142.5, "total_ms": 350},
                             "hits": 0, "misses": 0, "per_request": per_request}}}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(stored))
    cache_timing.main(["--resummarize", str(path)])  # no model run
    new = json.loads(path.read_text())
    run = new["runs"]["0"]
    assert [p["schema"] for p in run["per_request"]] == ["A", "B", "A", "B"]
    assert [{k: v for k, v in p.items() if k != "schema"} for p in run["per_request"]] == per_request  # timings kept
    assert run["per_schema"]["A"]["medians"] == {"prefill_ms": 295, "prefix_ms": 195, "total_ms": 395}
    assert run["per_schema"]["B"]["medians"] == {"prefill_ms": 205, "prefix_ms": 92.5, "total_ms": 305}
    assert "medians" not in run and (run["requests"], run["hits"], run["misses"]) == (4, 0, 0) and new["preset"] == "p"
    cache_timing.main(["--resummarize", str(path)])  # idempotent
    assert json.loads(path.read_text()) == new
    stored["alternations"] = 3  # the labels come from the alternation count: a mismatch is an error
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError):
        cache_timing.main(["--resummarize", str(path)])


def test_main_runs_both_cache_sizes(monkeypatch, tmp_path):
    test_set = tmp_path / "t.jsonl"
    test_set.write_text("".join(json.dumps({"text": f"t{i}"}) + "\n" for i in range(3)))
    calls = []

    def fake_run(cmd, input=None, **kwargs):
        calls.append((cmd, input))
        bodies = [json.loads(l) for l in input.splitlines()]
        size = int(cmd[cmd.index("--decide-prefix-cache") + 1])
        # schema A (the first line's body) has the longer prefix
        out = [timed(100 + size, (50 if b == bodies[0] else 30) if size == 0 else 5, 200, 0 if size == 0 else 1, 0)
               for b in bodies]
        return subprocess.CompletedProcess(cmd, 0, "".join(json.dumps(r) + "\n" for r in out), "")

    monkeypatch.setattr(cache_timing.reference_check.subprocess, "run", fake_run)
    out_path = tmp_path / "timing.json"
    cache_timing.main(["--model", "m.gguf", "--preset", "qwen2.5-0.5b", "--alternations", "2", "--states", "2",
                       "--test-set", str(test_set), "--out", str(out_path)])
    assert [c[c.index("--decide-prefix-cache") + 1] for c, _ in calls] == ["0", "8"]
    assert all("--dump-tokens" not in c and c[c.index("--decide-seqs") + 1] == "21" for c, _ in calls)  # the preset's
    bodies = [json.loads(l) for l in calls[0][1].splitlines()]
    assert len(bodies) == 4 and bodies[0]["states"] == ["t0", "t1"] and bodies[0] != bodies[1] and bodies[0] == bodies[2]
    out = json.loads(out_path.read_text())
    assert out["decide_seqs"] == 21 and out["alternations"] == 2
    assert out["schemas"] == {"A": "keywords", "B": "default"}
    zero, eight = out["runs"]["0"]["per_schema"], out["runs"]["8"]["per_schema"]
    assert (zero["A"]["medians"]["prefix_ms"], zero["B"]["medians"]["prefix_ms"]) == (50, 30)
    assert (eight["A"]["medians"]["prefix_ms"], eight["B"]["medians"]["prefix_ms"]) == (5, 5)
    assert (out["runs"]["8"]["hits"], out["runs"]["0"]["hits"]) == (4, 0)
