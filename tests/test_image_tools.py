import base64
import gzip
import hashlib
import json
import math
import subprocess
import sys

import httpx
import pytest

import decide_client
import reference_check
from decide_client import image_part, state_from_spec, text_part

# the 2 x 3 PNG fixture of engine/tests/test-decide-image.cpp
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAIAAAA2iEnWAAAAGElEQVR4nAXBAQEAAAjDIG7/zhOEoHJiPUfdBv3wKgpGAAAAAElFTkSuQmCC")
JPG_HEAD = b"\xff\xd8\xff\xe0" + bytes(20)  # image_part looks at the magic bytes only
FIELDS = {"cat": {"type": "bool", "description": "d"}, "animal": {"type": "choice", "description": "a", "options": ["cat", "dog"]}}


def image_dump(sha, variants=2, temperature=0.5):
    """One --dump-tokens response with one image state (text part, image part), tree branches for cat and animal per variant."""
    branches = []
    for v in range(variants):
        branches.append({"field": "cat", "mode": "tree", "node": 0, "tokens": [8], "children": [11, 12], "options": [[0], [1]],
                         "variant": v, "phase": 0, "pieces_hex": ["2063"]})
        branches.append({"field": "animal", "mode": "tree", "node": 0, "tokens": [9], "children": [21, 22], "options": [[0], [1]],
                         "variant": v, "phase": 0, "pieces_hex": ["2061"]})
    return {"results": [{"answers": {"cat": {"value": True, "p_true": 0.49 / 0.58},
                                     "animal": {"value": "cat", "probabilities": {"cat": 0.5, "dog": 0.5}}}}],
            "engine": {"temperature": temperature},
            "tokens": {"prefix": [1], "prefix_texts": ["<s0>", "<s1>"][:variants], "tail_text": "<t>{\n",
                       "states": [{"parts": [{"type": "text", "tokens": [5]},
                                             {"type": "image", "sha256": sha, "n_tokens": 10, "n_pos": 3, "chunks": []}],
                                   "tail": [7], "branches": branches}]}}


def test_image_part_uses_the_magic_bytes(tmp_path):
    png = tmp_path / "a.bin"
    png.write_bytes(PNG)
    jpg = tmp_path / "b.png"  # the name does not matter
    jpg.write_bytes(JPG_HEAD)
    assert image_part(png) == {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}
    assert image_part(jpg)["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")
    assert text_part("hi") == {"type": "text", "text": "hi"}


def test_image_part_rejects_other_formats(tmp_path):
    bmp = tmp_path / "c.bmp"
    bmp.write_bytes(b"BM" + bytes(60))
    with pytest.raises(SystemExit, match="not a JPEG or PNG"):
        image_part(bmp)


def test_state_from_spec(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(PNG)
    assert state_from_spec("plain") == "plain"
    assert state_from_spec([{"text": "Look: "}, {"image": str(p)}]) == [text_part("Look: "), image_part(p)]
    with pytest.raises(SystemExit):
        state_from_spec([{"text": "a", "image": str(p)}])


def test_decide_command_passes_media_keys(tmp_path, monkeypatch):
    ini = tmp_path / "p.ini"
    ini.write_text("[*]\nctx-size = 4096\n\n[v]\nmodel = m.gguf\nmmproj = mm.gguf\nimage-min-tokens = 1024\n\n[t]\nmodel = t.gguf\n")
    monkeypatch.setenv("ENGINE_PRESETS", str(ini))
    cmd = reference_check.decide_command("m.gguf", "v", 16, dump=False)
    assert cmd[cmd.index("--mmproj") + 1] == "mm.gguf" and cmd[cmd.index("--image-min-tokens") + 1] == "1024"
    assert "--image-max-tokens" not in cmd
    assert "--mmproj" not in reference_check.decide_command("t.gguf", "t", 9, dump=False)


def test_request_body_debias_and_temperature():
    schema = {"instructions": "i", "fields": {"cat": {"type": "bool", "description": "d"}}}
    assert reference_check.request_body("p", schema, ["x"], order_debias=2, temperature=0.5)["options"] == {"order_debias": 2, "temperature": 0.5}
    assert "options" not in reference_check.request_body("p", schema, ["x"])
    assert reference_check.request_body("p", None, ["x"]) == decide_client.build_request("p", ["x"])  # the triage body is today's


def test_build_records_image_tree_variants(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    sha = hashlib.sha256(PNG).hexdigest()
    recs = reference_check.build_records([image_dump(sha)], FIELDS, None, [[{"text": "Look: "}, {"image": str(img)}]])
    assert [r["field"] for r in recs] == ["cat", "animal"]
    r = recs[0]
    assert r["image"] and r["images"] == [{"path": str(img), "sha256": sha}] and r["temperature"] == 0.5
    assert [v["prompt_text"] for v in r["variants"]] == ["<s0>Look: <__media__><t>{\n", "<s1>Look: <__media__><t>{\n"]
    assert r["variants"][1]["nodes"] == [{"node": 0, "pieces_hex": ["2063"], "children": [11, 12], "options": [[0], [1]]}]


def test_build_records_rejects_another_image(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    with pytest.raises(SystemExit, match="differs"):
        reference_check.build_records([image_dump("0" * 64)], FIELDS, None, [[{"image": str(img)}]])


def test_build_records_image_full_path(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    data = image_dump(hashlib.sha256(PNG).hexdigest(), variants=1, temperature=1.0)
    data["tokens"]["states"][0]["branches"] = [
        {"field": "cat", "mode": "full_path", "option": 0, "tokens": [8, 11], "rows": [0, 1], "logp": [], "variant": 0, "phase": 0,
         "pieces_hex": ["20", "74"]},
        {"field": "cat", "mode": "full_path", "option": 1, "tokens": [8, 12], "rows": [1], "logp": [], "variant": 0, "phase": 0,
         "pieces_hex": ["20", "66"]}]
    recs = reference_check.build_records([data], {"cat": FIELDS["cat"]}, None, [[{"image": str(img)}]])
    assert recs[0]["mode"] == "full_path" and recs[0]["prompt_text"] == "<s0><__media__><t>{\n"
    prompt = reference_check.row_prompt(recs[0], recs[0]["entries"][0], 1)
    assert prompt == {"prompt_string": "<s0><__media__><t>{\n t", "multimodal_data": [base64.b64encode(PNG).decode()]}


def test_piece_text_and_with_temperature():
    assert reference_check.piece_text(["2063", "6174"]) == " cat"
    with pytest.raises(SystemExit):
        reference_check.piece_text(["c3"])  # half of a UTF-8 character
    assert reference_check.with_temperature([0.7, 0.3], 1.0) == [0.7, 0.3]
    p = reference_check.with_temperature([0.7, 0.3], 0.5)
    assert abs(p[0] - 0.49 / 0.58) < 1e-12 and abs(sum(p) - 1) < 1e-12
    assert reference_check.with_temperature([1.0, 0.0], 2.0) == [1.0, 0.0]


def test_compare_replays_every_variant_of_an_image_record(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    sha = hashlib.sha256(PNG).hexdigest()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    recs = reference_check.build_records([image_dump(sha)], FIELDS, None, [[{"text": "Look: "}, {"image": str(img)}]])
    (out_dir / "reference-t-p-dump.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["prompt"]["prompt_string"])
        assert body["prompt"]["multimodal_data"] == [base64.b64encode(PNG).decode()]
        variant0 = body["prompt"]["prompt_string"].startswith("<s0>")
        if body["prompt"]["prompt_string"].endswith(" c"):  # the cat node: variants 0.8 and 0.6, mean 0.7, T 0.5 -> 0.49 / 0.58
            p, ids = ((0.8, 0.2) if variant0 else (0.6, 0.4)), (11, 12)
        else:  # the animal node: 0.5 in both variants
            p, ids = (0.5, 0.5), (21, 22)
        top = [{"id": i, "token": "x", "bytes": [120], "logprob": math.log(q)} for i, q in zip(ids, p)]
        return httpx.Response(200, json={"completion_probabilities": [{"id": ids[0], "logprob": math.log(p[0]), "top_logprobs": top}]})

    real = httpx.Client
    monkeypatch.setattr(reference_check.httpx, "Client", lambda **kw: real(base_url="http://test", transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "compare", "--preset", "p", "--tag", "t", "--out-dir", str(out_dir)])
    reference_check.main()
    out = json.loads((out_dir / "reference-t-p.json").read_text())
    assert out["pass"] and out["fields"]["cat"]["max_abs_diff"] < 1e-9 and out["fields"]["animal"]["max_abs_diff"] < 1e-9
    assert sorted(seen) == sorted(f"<s{v}>Look: <__media__><t>{{\n {c}" for v in (0, 1) for c in "ca")
    with gzip.open(out_dir / "reference-t-p-replay.jsonl.gz", "rt") as f:
        replay = f.read()
    assert "sha256:" + sha in replay and base64.b64encode(PNG).decode() not in replay  # the replay keeps hashes, not images
    monkeypatch.setattr(reference_check.httpx, "Client", real)
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "compare", "--preset", "p", "--tag", "t", "--out-dir", str(out_dir), "--offline"])
    reference_check.main()
    assert json.loads((out_dir / "reference-t-p.json").read_text())["fields"] == out["fields"]


def test_dependent_problems_read_every_variant():
    rec = {"ticket": 0, "field": "dog", "ancestors": {"cat": True}, "engine": {"given": {"cat": True}},
           "variants": [{"nodes": [{"given": {"cat": True}, "lines": '  "cat": true,\n'}]},
                        {"nodes": [{"given": {"cat": False}, "lines": '  "cat": true,\n'}]}]}
    problems = reference_check.dependent_problems(rec)
    assert len(problems) == 1 and "entry given" in problems[0]


def test_dump_needs_a_tag_with_states(tmp_path, monkeypatch):
    states = tmp_path / "s.jsonl"
    states.write_text('"x"\n')
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m.gguf", "--preset", "p", "--states", str(states)])
    with pytest.raises(SystemExit, match="--tag is required"):
        reference_check.main()


def test_dump_sends_image_states_and_switches(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    states = tmp_path / "s.jsonl"
    states.write_text(json.dumps([{"image": str(img)}, {"text": " Indoors?"}]) + "\n")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"instructions": "i", "fields": {"cat": {"type": "bool", "description": "d"}}}))
    seen = []

    def fake_run(cmd, input=None, **kwargs):  # stop cmd_dump right after it built the request lines
        seen.append(input)
        return subprocess.CompletedProcess(cmd, 1, "", "stop")

    monkeypatch.setattr(reference_check.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m.gguf", "--preset", "p", "--states", str(states),
                                      "--schema", str(schema), "--tag", "t", "--order-debias", "2", "--temperature", "0.5", "--n", "1"])
    with pytest.raises(SystemExit):
        reference_check.main()
    body = json.loads(seen[0])
    assert body["states"] == [[image_part(img), text_part(" Indoors?")]]
    assert body["options"] == {"order_debias": 2, "temperature": 0.5}


def text_dump(variant=0, temperature=1.0):
    """One --dump-tokens response with one text state (token ids) and its cat branch in `variant`."""
    return {"results": [{"answers": {"cat": {"value": True, "p_true": 0.8}}}], "engine": {"temperature": temperature},
            "tokens": {"prefix": [1], "states": [{"state": [5], "tail": [7], "branches": [
                {"field": "cat", "mode": "tree", "node": 0, "tokens": [8], "children": [11, 12], "options": [[0], [1]],
                 "variant": variant, "phase": 0}]}]}}


@pytest.mark.parametrize("variant, temperature, ok", [(0, 1.0, True), (1, 1.0, False), (0, 0.5, False)])
def test_dump_rejects_text_states_with_variants_or_temperature(tmp_path, monkeypatch, variant, temperature, ok):
    states = tmp_path / "s.jsonl"
    states.write_text('"x"\n')
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"instructions": "i", "fields": {"cat": {"type": "bool", "description": "d"}}}))
    line = json.dumps(text_dump(variant, temperature)) + "\n"
    monkeypatch.setattr(reference_check.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, line, ""))
    monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m.gguf", "--preset", "p", "--states", str(states),
                                      "--schema", str(schema), "--tag", "t", "--out-dir", str(tmp_path)])
    if ok:
        reference_check.main()
        assert json.loads((tmp_path / "reference-t-p-dump.jsonl").read_text())["nodes"][0]["prompt_tokens"] == [1, 5, 7, 8]
    else:
        with pytest.raises(SystemExit, match="text state 0"):
            reference_check.main()


def test_image_prompt_rejects_a_changed_image(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    recs = reference_check.build_records([image_dump(hashlib.sha256(PNG).hexdigest())], FIELDS, None, [[{"image": str(img)}]])
    img.write_bytes(PNG + b"\0")  # the file changed after the dump
    with pytest.raises(SystemExit, match="differs from the record"):
        reference_check.image_prompt(recs[0], "x")


def test_compare_full_path_of_an_image_record_and_its_offline_replay(tmp_path):
    from test_reference_check_fullpath import PIECES, REPLAY, engine_answer, engine_rows, fake_server, fp_record

    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    rows = engine_rows()
    hexes = {100: "41", 101: "42", 10: "6162", 1: "22", 11: "636422"}  # pieces A, B, ab, ", cd"
    branches = [{"field": "q", "mode": "full_path", "option": 0, "tokens": [100, 101, 10, 1], "rows": [1, 2, 3],
                 "logp": [rows[0], rows[1], rows[2]], "variant": 0, "phase": 0},
                {"field": "q", "mode": "full_path", "option": 1, "tokens": [100, 101, 11], "rows": [2], "logp": [rows[3]],
                 "variant": 0, "phase": 0}]
    for b in branches:
        b["pieces_hex"] = [hexes[t] for t in b["tokens"]]
    data = {"results": [{"answers": {"q": engine_answer()}}], "engine": {"temperature": 1.0},
            "tokens": {"prefix": [1], "prefix_texts": ["<s0>"], "tail_text": "<t>",
                       "states": [{"parts": [{"type": "image", "sha256": hashlib.sha256(PNG).hexdigest(), "n_tokens": 10, "n_pos": 3,
                                              "chunks": []}], "tail": [7], "branches": branches}]}}
    rec = reference_check.build_records([data], {"q": {"type": "choice", "description": "q", "options": ["ab", "cd"]}}, None,
                                        [[{"image": str(img)}]])[0]
    by_text = {"AB": (100, 101), "ABab": (100, 101, 10), 'ABab"': (100, 101, 10, 1), 'ABcd"': (100, 101, 11)}
    seen = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": "".join(PIECES[t] for t in body["tokens"])})
        p = body["prompt"]
        assert p["multimodal_data"] == [base64.b64encode(PNG).decode()] and p["prompt_string"].startswith("<s0><__media__><t>")
        seen.append(p["prompt_string"])
        top = REPLAY[by_text[p["prompt_string"][len("<s0><__media__><t>"):]]]
        return httpx.Response(200, json={"completion_probabilities": [{"id": top[0][0], "top_logprobs": [
            {"id": t, "token": s.decode(errors="replace"), "bytes": list(s), "logprob": math.log(q)} for t, s, q in top]}]})

    live = reference_check.RecordingClient(httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)))
    out_live = reference_check.compare_full_path([rec], live, "p")
    assert sorted(seen) == sorted("<s0><__media__><t>" + s for s in by_text)
    # the string replay of the image record gives what the token-id replay of the same rows gives
    out_tokens = reference_check.compare_full_path([fp_record(rows, engine_answer())], fake_server(REPLAY, PIECES), "p")
    assert json.dumps(out_live["detail"], sort_keys=True) == json.dumps(out_tokens["detail"], sort_keys=True)
    path = tmp_path / "replay.jsonl.gz"
    live.write(path)
    with gzip.open(path, "rt") as f:
        assert base64.b64encode(PNG).decode() not in f.read()
    out_offline = reference_check.compare_full_path([rec], reference_check.offline_client(path), "p")
    assert json.dumps(out_offline, sort_keys=True) == json.dumps(out_live, sort_keys=True)
