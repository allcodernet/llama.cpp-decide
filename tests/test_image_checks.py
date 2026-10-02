import base64
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("image_checks", ROOT / "results-engine" / "image" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_json_diff_skips_timings_and_finds_differences():
    ck = load()
    a = {"label": "x", "status": 200, "response": {"results": [{"answers": {"angry": {"p_true": 0.25}}}], "timings": {"total_ms": 1.0},
                                                   "engine": {"prefix_cache": {"hits": 1}}}}
    b = json.loads(json.dumps(a))
    b["response"]["timings"]["total_ms"] = 2.0
    assert ck.json_diff(a, b) == (0.0, [])
    b["response"]["results"][0]["answers"]["angry"]["p_true"] = 0.5
    b["response"]["engine"]["prefix_cache"]["hits"] = 2
    worst, diffs = ck.json_diff(a, b)
    assert worst == 1.0 and len(diffs) == 2
    assert ck.json_diff({"a": 1}, {"b": 1})[1] and ck.json_diff([1], [1, 2])[1] and ck.json_diff(True, 1)[1]


def test_text_requests_sequence(tmp_path):
    ck = load()
    t = tmp_path / "t.jsonl"
    t.write_text("".join(json.dumps({"text": f"ticket {i}"}) + "\n" for i in range(12)))
    seq = ck.text_requests("p", t, 2)
    assert [label for label, _ in seq] == ["tree"] * 3 + ["full_path"] * 3 + ["after"] * 3 + ["debias2"] * 3 + ["t0.5"] * 3 + ["cache"] * 3
    assert [len(b["states"]) for _, b in seq[:3]] == [5, 5, 2]
    assert seq[3][1]["options"]["scoring"] == "full_path" and seq[9][1]["options"]["order_debias"] == 2
    assert seq[6][1]["fields"]["urgency"]["after"] == ["queue"] and seq[6][1]["fields"]["angry"]["after"] == ["queue"]
    assert seq[12][1]["options"]["temperature"] == 0.5
    assert seq[15][1]["instructions"] != seq[16][1]["instructions"]  # the cache segment alternates two prompt variants
    assert len(ck.text_requests("p", t, 2, limit=5)) == 6


def test_malformed_cases_and_byte_helpers():
    ck = load()
    jpg = (b"\xff\xd8" + b"\xff\xe0" + (16).to_bytes(2, "big") + bytes(14)
           + b"\xff\xc0" + (17).to_bytes(2, "big") + bytes([8, 0, 2, 0, 4, 3]) + bytes(9) + b"\xff\xd9")
    assert ck.jpeg_sof(jpg) == 20
    h = ck.png_header(8000, 4001)
    assert len(h) == 33 and h[:8] == b"\x89PNG\r\n\x1a\n" and h[12:16] == b"IHDR"
    assert int.from_bytes(h[16:20], "big") == 8000 and int.from_bytes(h[20:24], "big") == 4001
    cases = ck.malformed_cases(jpg)
    assert len(cases) == 29 and len({name for name, _, _ in cases}) == 29
    ratio = [states[0][0]["image_url"]["url"] for name, states, _ in cases if "(aspect ratio)" in name]
    sizes = [ck.struct.unpack(">II", base64.b64decode(u.split(",", 1)[1])[16:24]) for u in ratio]
    assert sizes == [(201, 1), (16000, 2), (16000000, 2)] and all(w * h <= 32000000 for w, h in sizes)   # the pixel limit passes
    by = {name: (states, field) for name, states, field in cases}
    states, field = by["33 images in a request"]
    assert field == "states[8]" and len(states) == 9 and sum(len(s) for s in states) == 36
    assert by["empty part list"][1] == "states[1]" and by["12-bit JPEG (pixel decoding, pass 3)"][1] == "states[1]"
    twelve = base64.b64decode(by["12-bit JPEG (pixel decoding, pass 3)"][0][1][0]["image_url"]["url"].split(",", 1)[1])
    assert twelve[24] == 12 and twelve[:24] == jpg[:24] and twelve[25:] == jpg[25:]
    safe = by["URL-safe base64"][0][0][0]["image_url"]["url"]
    assert "_" in safe and "/" not in safe.split(",", 1)[1]
    assert all(field.startswith("states[") for _, _, field in cases)
    assert by[";base64 without the comma"][0][0][0]["image_url"]["url"].startswith("data:image/png;base64iVBOR")
    assert "17 parts" not in by   # spec rev 4: no part limit


def test_solid_png_is_a_decodable_png():
    ck = load()
    p = ck.solid_png(3, 2)
    assert p[:8] == b"\x89PNG\r\n\x1a\n" and p[12:16] == b"IHDR" and ck.struct.unpack(">II", p[16:24]) == (3, 2)
    assert p[24] == 8 and p[25] == 0   # 8-bit gray
    n = ck.struct.unpack(">I", p[33:37])[0]
    assert p[37:41] == b"IDAT" and ck.zlib.decompress(p[41:41 + n]) == b"\x00\x80\x80\x80" * 2
    assert p.endswith(b"IEND\xaeB`\x82")


def test_log_window_checks():
    ck = load()
    proxy = "0.01.798.097 I srv  proxy_reques: proxying request to model qwen2.5-0.5b on port 57511"
    errors, checks = ck.log_window_checks([proxy, proxy, proxy])
    assert errors == [] and all(checks.values())
    errors, checks = ck.log_window_checks([])   # a wrong offset leaves an empty window: it must not pass
    assert errors == [] and not all(checks.values())
    errors, checks = ck.log_window_checks([proxy])
    assert not all(checks.values())
    errors, checks = ck.log_window_checks([proxy, "srv  decide: engine error: abort", proxy])
    assert errors == ["srv  decide: engine error: abort"] and not all(checks.values())


def test_sized_answer_ok_rejects_a_500():
    ck = load()
    assert ck.sized_answer_ok({"status": 200, "response": {"results": []}})
    assert ck.sized_answer_ok({"status": 400, "response": {"error": {"code": "budget", "message": "state 0 needs …"}}})
    assert not ck.sized_answer_ok({"status": 500, "response": {"error": {"code": "engine", "message": "no KV cell space (rc=1)"}}})
    assert not ck.sized_answer_ok({"status": 400, "response": {"error": {"code": "invalid_states"}}})


def test_budget_parts_reads_both_limit_forms():
    ck = load()
    # decide-debias.cpp state_cells_message: K = 1 with n_ctx the bound (the text before 0043)
    assert ck.budget_parts("state 0 needs 3446 cells (held 0, prefix 248, state 3176, tail 6, branches 16), n_ctx - 16 is 2032") == {
        "needs": 3446, "held": 0, "prefix": 248, "K": 1, "state": 3176, "tail": 6, "branches": 16, "limit": 2032, "swa_cells": None}
    # K_eff 2 with image cells
    assert ck.budget_parts("state 1 needs 2790 cells (held 50, prefix 496, K_eff 2 × (state 1100 (image cells 1082), tail 6, "
                           "branches 16)), n_ctx - 16 is 1000") == {
        "needs": 2790, "held": 50, "prefix": 496, "K": 2, "state": 1100, "tail": 6, "branches": 16, "limit": 1000, "swa_cells": None}
    # spec 5.3 rev 5: the SWA cache is the bound
    assert ck.budget_parts("state 0 needs 9218 cells (held 0, prefix 53, state 9154, tail 7, branches 4), the sliding-window cache "
                           "holds 9216 cells (limit 9200); start with --swa-full to use n_ctx") == {
        "needs": 9218, "held": 0, "prefix": 53, "K": 1, "state": 9154, "tail": 7, "branches": 4, "limit": 9200, "swa_cells": 9216}
    assert ck.budget_parts("no KV cell space (rc=1)") is None


def test_largest_state_fills_the_limit():
    ck = load()
    b = {"held": 16, "prefix": 53, "K": 1, "tail": 7, "branches": 4, "limit": 9200}
    assert ck.largest_state(b) == 9200 - 16 - 53 - 7 - 4
    b2 = {"held": 0, "prefix": 106, "K": 2, "tail": 7, "branches": 4, "limit": 9200}
    m = ck.largest_state(b2)   # held + prefix + K (state + tail + branches) <= limit, and one token more is over it
    assert 106 + 2 * (m + 11) <= 9200 < 106 + 2 * (m + 1 + 11)


def test_held_budget_matches_old_allows_the_swa_limit():
    ck = load()
    old = {"prefix": 53, "tail": 7, "branches": 4, "limit": 16368}
    assert ck.held_budget_matches(dict(old, swa_cells=None), old)
    assert ck.held_budget_matches(dict(old, limit=9200, swa_cells=9216), old)
    assert not ck.held_budget_matches(dict(old, limit=9200, swa_cells=None), old)
    assert not ck.held_budget_matches(dict(old, limit=9100, swa_cells=9216), old)
    assert not ck.held_budget_matches(dict(old, prefix=54, swa_cells=None), old)


def test_answers_match_applies_the_statistic_state_by_state():
    ck = load()
    fields = ["animal"]

    def res(*dists):
        return [{"answers": {"animal": {"probabilities": dict(zip(("cat", "dog", "bird"), d))}}} for d in dists]

    a = res((0.5, 0.3, 0.2), (0.1, 0.1, 0.8))
    same = ck.answers_match(a, a, fields)
    assert same["pass"] and same["max_abs_diff"] == 0
    near = ck.answers_match(a, res((0.504, 0.297, 0.199), (0.1, 0.103, 0.797)), fields)
    assert near["pass"] and 0 < near["max_abs_diff"] < 0.01
    # an argmax flip where the reference's top-2 margin is 0.7 fails, even with every other state equal
    assert not ck.answers_match(a, res((0.5, 0.3, 0.2), (0.1, 0.8, 0.1)), fields)["pass"]
    # a per-field median abs diff over 0.01 fails without an argmax flip
    assert not ck.answers_match(a, res((0.55, 0.25, 0.2), (0.15, 0.05, 0.8)), fields)["pass"]


def test_server_build_names_the_binary(tmp_path):
    ck = load()
    import subprocess
    eng = tmp_path / "engine-x"
    binary = eng / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'ggml_cuda_init: found 1 CUDA devices' >&2\necho 'version: 42 (abc1234)' >&2\n"
                      "echo 'built with GNU 15 for Linux x86_64' >&2\n")
    binary.chmod(0o755)
    b = ck.server_build(str(binary))
    assert b == {"path": str(binary), "version": ["version: 42 (abc1234)", "built with GNU 15 for Linux x86_64"], "engine_commit": None}
    git = ["git", "-C", str(eng), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(eng)], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.run(["git", "-C", str(eng), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert ck.server_build(str(binary))["engine_commit"] == head


def test_round_split_gate_requires_exact_equality():
    ck = load()
    fields = ["animal"]

    def res(*dists):
        return [{"answers": {"animal": {"probabilities": dict(zip(("cat", "dog", "bird"), d))}}} for d in dists]

    multi = res((0.5, 0.3, 0.2), (0.1, 0.1, 0.8), (0.3, 0.3, 0.4))
    g = ck.round_split_gate(multi, res((0.5, 0.3, 0.2), (0.1, 0.1, 0.8)), res((0.3, 0.3, 0.4)), fields)
    assert g["pass"] and g["first_two_vs_pair_max_abs_diff"] == 0 and g["third_vs_alone_max_abs_diff"] == 0
    # a difference far inside the statistic still fails the exact gate, in either part
    g = ck.round_split_gate(multi, res((0.5, 0.3, 0.2), (0.1, 0.1000001, 0.7999999)), res((0.3, 0.3, 0.4)), fields)
    assert not g["pass"] and 0 < g["first_two_vs_pair_max_abs_diff"] < 1e-6
    g = ck.round_split_gate(multi, res((0.5, 0.3, 0.2), (0.1, 0.1, 0.8)), res((0.3, 0.3001, 0.3999)), fields)
    assert not g["pass"] and g["third_vs_alone_max_abs_diff"] > 0
