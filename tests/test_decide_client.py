import hashlib
import json

import pytest

import decide_client
from decide_client import EXPECTED_PREFIX, build_request, chunks, main, parse_args, request_body, run_name, to_pred


def test_build_request_shape():
    r = build_request("qwen3.5-9b", ["a", "b"])
    assert r["model"] == "qwen3.5-9b" and r["states"] == ["a", "b"]
    assert list(r["fields"]) == ["queue", "urgency", "angry"]
    q = r["fields"]["queue"]
    assert q["type"] == "choice" and list(q["options"]) == ["billing", "technical", "sales", "feedback"]
    assert q["options"]["billing"] == "Payments, refunds, invoices, charges"
    assert r["fields"]["urgency"]["type"] == "score" and len(r["fields"]["urgency"]["levels"]) == 4
    assert r["fields"]["angry"]["type"] == "bool"
    assert r["options"] == {"scoring": "tree", "order_debias": 0}


ANSWERS = {
    "queue": {"value": "sales", "probabilities": {"billing": 0.1, "technical": 0.2, "sales": 0.6, "feedback": 0.1}, "confidence": 0.6},
    "urgency": {"value": 2, "expected": 1.9, "probabilities": [0.1, 0.2, 0.5, 0.2], "confidence": 0.5},
    "angry": {"value": False, "p_true": 0.3, "confidence": 0.7},
}


def test_to_pred_maps_answers():
    p = to_pred(ANSWERS, 0.2, 100.0)
    assert p["queue"] == ANSWERS["queue"]["probabilities"]
    assert p["urgency"] == [0.1, 0.2, 0.5, 0.2]
    assert p["angry"] == pytest.approx(0.3)
    assert p["distribution"] == "full" and p["latency_s"] == 0.2 and p["server_ms"] == 100.0 and p["raw"] is ANSWERS


def test_expected_prefix_table():
    assert EXPECTED_PREFIX["qwen3.5-9b"].endswith("<think>\n\n</think>\n\n")
    assert EXPECTED_PREFIX["gemma-4-e4b"].endswith("<|turn>model\n")


def test_chunks():
    assert chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]


def test_prompt_variants():
    from decide_client import QUEUES, variant_suffix
    from prompt_variants import VARIANTS

    assert set(VARIANTS) == {"default", "merged", "framing", "keywords"}
    for name, v in VARIANTS.items():
        assert v["instructions"].strip() and v["description_mode"] in ("full", "short"), name
        assert build_request("qwen3.5-9b", ["a"], name)["instructions"] == v["instructions"]
    # merged uses the one-word schema
    r = build_request("qwen3.5-9b", ["a"], "merged")
    assert r["fields"]["queue"] == {"type": "choice", "description": "Team.", "options": list(QUEUES)}
    assert variant_suffix("default") == "" and variant_suffix("merged") == "-merged"
    # merged folds every full option description into its compact catalogue
    assert all(f"{k} = {d}" in VARIANTS["merged"]["instructions"] for k, d in QUEUES.items())
    # the parity runs' text
    assert VARIANTS["default"]["instructions"] == "Answer each question about this support ticket from its text."


def test_cli_variant_default():
    from decide_client import DEFAULT_VARIANT

    assert DEFAULT_VARIANT == "keywords"  # the held-out check's winner (REPORT-ENGINE.md)
    assert parse_args(["--model", "m"]).prompt_variant == "keywords"
    assert parse_args(["--model", "m", "--prompt-variant", "default"]).prompt_variant == "default"


# --- sub-project 3 switches (spec 2026-09-25-engine-improvements-design.md, section 3.1) ---------------------------

# sha256 of json.dumps(build_request("qwen3.5-9b", ["a", "b"], variant)) from decide_client.py before the switches
# (project 76877fb): a request without switches must stay byte-identical (spec section 1, 12.2).
TODAY = {
    "keywords": "985770f3e31a6da4257e51c445b77a0411b5813e8f63cc1d134cf476b914e7f5",
    "default": "d569ddd75f3e709f701833c2dcb49c5043d7577deb19ae3503137190587fe7b9",
}
BASE_ARGV = ["--model", "qwen3.5-9b", "--batch", "5"]


def sha(body):
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def cli(*extra):
    """(request body for states ["a"], output name) of a decide_client.py command line against a server whose default
    temperature is 1.0 (the effective temperature: --temperature, else 1.0)."""
    args = parse_args(BASE_ARGV + list(extra))
    return request_body(args, ["a"]), run_name(args, temperature=1.0 if args.temperature is None else args.temperature)


def without(body, *keys):
    return {k: v for k, v in body.items() if k not in keys}


def test_request_without_switches_is_byte_identical_to_today():
    for variant, digest in TODAY.items():
        assert sha(build_request("qwen3.5-9b", ["a", "b"], variant)) == digest, variant
        explicit = build_request("qwen3.5-9b", ["a", "b"], variant, scoring="tree", order_debias=0, after={}, temperature=None)
        assert sha(explicit) == digest, variant
    args = parse_args(BASE_ARGV)
    assert sha(request_body(args, ["a", "b"])) == TODAY["keywords"]
    assert run_name(args, temperature=1.0) == "decide-qwen3.5-9b-b5-keywords"  # the Task 0 baseline's file name
    # the switches at their default values are the same request and the same name
    args = parse_args(BASE_ARGV + ["--scoring", "tree", "--order-debias", "0"])
    assert sha(request_body(args, ["a", "b"])) == TODAY["keywords"]
    assert run_name(args, temperature=1.0) == "decide-qwen3.5-9b-b5-keywords"


def test_scoring_flag():
    body, name = cli("--scoring", "full_path")
    assert body["options"] == {"scoring": "full_path", "order_debias": 0}
    assert without(body, "options") == without(cli()[0], "options")
    assert name == "decide-qwen3.5-9b-b5-keywords-full_path"
    with pytest.raises(SystemExit):
        cli("--scoring", "fullpath")


def test_order_debias_flag():
    body, name = cli("--order-debias", "4")
    assert body["options"] == {"scoring": "tree", "order_debias": 4}
    assert without(body, "options") == without(cli()[0], "options")
    assert name == "decide-qwen3.5-9b-b5-keywords-debias4"
    assert cli("--order-debias", "1")[1] == "decide-qwen3.5-9b-b5-keywords-debias1"  # a different request, own name


def test_temperature_flag():
    body, name = cli("--temperature", "2")
    assert body["options"] == {"scoring": "tree", "order_debias": 0, "temperature": 2.0}
    assert without(body, "options") == without(cli()[0], "options")
    assert name == "decide-qwen3.5-9b-b5-keywords-t2.0"
    assert cli("--temperature", "0.5")[1] == "decide-qwen3.5-9b-b5-keywords-t0.5"
    # the name follows the effective temperature: T = 1 given explicitly has the name of a T = 1 run
    body, name = cli("--temperature", "1")
    assert body["options"]["temperature"] == 1.0 and name == "decide-qwen3.5-9b-b5-keywords"


def test_name_follows_the_effective_temperature():
    args = parse_args(BASE_ARGV)
    assert run_name(args, temperature=1.0) == "decide-qwen3.5-9b-b5-keywords"
    # a server default other than 1.0 (the gemma-4-e4b preset's decide-temperature) names a run without --temperature
    assert run_name(args, temperature=1.6947) == "decide-qwen3.5-9b-b5-keywords-t1.6947"
    args = parse_args(BASE_ARGV + ["--temperature", "1", "--order-debias", "4"])
    assert run_name(args, temperature=1.0) == "decide-qwen3.5-9b-b5-keywords-debias4"
    with pytest.raises(TypeError):
        run_name(args)  # the effective temperature is not optional


def test_ensure_t1_flag():
    assert parse_args(BASE_ARGV + ["--ensure-t1"]).ensure_t1 and not parse_args(BASE_ARGV).ensure_t1
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGV + ["--ensure-t1", "--temperature", "1"])


def test_after_flag_config_a_and_b():
    base = cli()[0]
    body, name = cli("--after", "urgency=queue")  # config A
    assert body["fields"]["urgency"] == dict(base["fields"]["urgency"], after=["queue"])
    assert list(body["fields"]["urgency"])[-1] == "after"
    assert "after" not in body["fields"]["queue"] and "after" not in body["fields"]["angry"]
    assert without(body, "fields") == without(base, "fields")
    assert name == "decide-qwen3.5-9b-b5-keywords-after-urgency=queue"
    body_b, name_b = cli("--after", "urgency=queue", "--after", "angry=queue")  # config B
    assert body_b["fields"]["urgency"]["after"] == ["queue"] and body_b["fields"]["angry"]["after"] == ["queue"]
    assert "after" not in body_b["fields"]["queue"]
    assert name_b == "decide-qwen3.5-9b-b5-keywords-after-urgency=queue-angry=queue"
    # the command-line order of --after does not change the request or its name
    assert cli("--after", "angry=queue", "--after", "urgency=queue") == (body_b, name_b)
    body, name = cli("--after", "angry=queue,urgency")
    assert body["fields"]["angry"]["after"] == ["queue", "urgency"]
    assert name == "decide-qwen3.5-9b-b5-keywords-after-angry=queue+urgency"


@pytest.mark.parametrize("bad, message", [
    (["urgency"], "expected FIELD=PARENT"), (["urgency="], "expected FIELD=PARENT"), (["=queue"], "expected FIELD=PARENT"),
    (["urgency=queue,,angry"], "expected FIELD=PARENT"), (["urgency=queue,"], "expected FIELD=PARENT"),
    (["mood=queue"], "unknown field 'mood'"), (["urgency=team"], "unknown field 'team'"),
    (["urgency=queue", "urgency=angry"], "--after urgency given twice"),
])
def test_after_flag_rejects_malformed_unknown_and_repeated(bad, message):
    with pytest.raises(SystemExit) as e:
        cli(*[x for spec in bad for x in ("--after", spec)])
    assert message in str(e.value.code)


def test_all_switches_together_and_with_another_variant():
    body, name = cli("--scoring", "full_path", "--after", "urgency=queue", "--order-debias", "4", "--temperature", "2")
    assert body["options"] == {"scoring": "full_path", "order_debias": 4, "temperature": 2.0}
    assert body["fields"]["urgency"]["after"] == ["queue"]
    assert name == "decide-qwen3.5-9b-b5-keywords-full_path-after-urgency=queue-debias4-t2.0"
    body, name = cli("--prompt-variant", "default", "--scoring", "full_path")
    assert body["instructions"] == build_request("qwen3.5-9b", ["a"], "default")["instructions"]
    assert name == "decide-qwen3.5-9b-b5-full_path"


def test_to_pred_keeps_the_new_answer_keys():
    answers = {
        "queue": dict(ANSWERS["queue"], coverage=0.99, order_spread=0.2),
        "urgency": dict(ANSWERS["urgency"], coverage=0.98, order_spread=0.1, given={"queue": "sales"}),
        "angry": dict(ANSWERS["angry"], coverage=0.97, order_spread=0.05, given={"queue": "sales"}),
    }
    row = json.loads(json.dumps(to_pred(answers, 0.2, 100.0)))  # as written to the preds file
    assert [row["raw"][f]["coverage"] for f in ("queue", "urgency", "angry")] == [0.99, 0.98, 0.97]
    assert [row["raw"][f]["order_spread"] for f in ("queue", "urgency", "angry")] == [0.2, 0.1, 0.05]
    assert row["raw"]["urgency"]["given"] == row["raw"]["angry"]["given"] == {"queue": "sales"}
    assert "given" not in row["raw"]["queue"]


# --- main(): names by the engine's temperature, --ensure-t1, no overwrite without --force (fake server) --------------

class FakeResponse:
    def __init__(self, data):
        self.status_code, self._data, self.text = 200, data, json.dumps(data)

    def json(self):
        return self._data


class FakeServer:
    """httpx.Client stand-in for /v1/decide/info and /v1/decide: the server default temperature is `default_t`, the
    engine applies options.temperature unless `ignores_option`, every angry answer is `p_true`."""

    def __init__(self, default_t=1.0, ignores_option=False, p_true=0.3):
        self.default_t, self.ignores_option, self.p_true, self.bodies = default_t, ignores_option, p_true, []

    def get(self, path, params=None):
        assert path == "/v1/decide/info"
        return FakeResponse({"prefix": {"assistant_prefix": EXPECTED_PREFIX["gemma-4-e4b"]}, "temperature": self.default_t,
                             "prefix_cache": {"capacity": 8, "entries": 0}})

    def post(self, path, json=None):
        assert path == "/v1/decide"
        self.bodies.append(json)
        t = self.default_t if self.ignores_option else json["options"].get("temperature", self.default_t)
        answers = dict(ANSWERS, angry=dict(ANSWERS["angry"], p_true=self.p_true))
        return FakeResponse({"results": [{"answers": answers} for _ in json["states"]],
                             "timings": {"total_ms": 2.0, "rounds": 1}, "usage": {"prompt_tokens": 10},
                             "engine": {"temperature": t}})


def run_main(tmp_path, monkeypatch, server, *extra):
    """decide_client.main for gemma-4-e4b b5 on three tickets against `server`, outputs under tmp_path/out; returns the
    files there."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tickets = tmp_path / "tickets.jsonl"
    tickets.write_text("".join(json.dumps({"text": f"ticket {i}"}) + "\n" for i in range(3)))
    monkeypatch.setattr(decide_client.httpx, "Client", lambda **kw: server)
    monkeypatch.setattr(decide_client, "_vram_mib", lambda: None)
    monkeypatch.setattr(decide_client, "_fork_commit", lambda: None)
    main(["--model", "gemma-4-e4b", "--batch", "5", "--test-set", str(tickets), "--out-dir", str(tmp_path / "out"), *extra])
    return outputs(tmp_path)


def outputs(tmp_path):
    return sorted(str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("*") if p.is_file())


def test_main_names_the_outputs_by_the_engine_temperature(tmp_path, monkeypatch):
    server = FakeServer(default_t=1.6947)  # the gemma-4-e4b preset since the calibration
    assert run_main(tmp_path / "a", monkeypatch, server) == [
        "preds/decide-gemma-4-e4b-b5-keywords-t1.6947.jsonl", "runs/decide-gemma-4-e4b-b5-keywords-t1.6947.json"]
    assert all("temperature" not in b["options"] for b in server.bodies)
    run = json.loads((tmp_path / "a/out/runs/decide-gemma-4-e4b-b5-keywords-t1.6947.json").read_text())
    assert run["engine"]["temperature"] == 1.6947 and run["switches"]["temperature"] is None
    assert run_main(tmp_path / "b", monkeypatch, FakeServer(default_t=1.6947), "--temperature", "1") == [
        "preds/decide-gemma-4-e4b-b5-keywords.jsonl", "runs/decide-gemma-4-e4b-b5-keywords.json"]
    assert run_main(tmp_path / "c", monkeypatch, FakeServer(), "--temperature", "2")[0] == \
        "preds/decide-gemma-4-e4b-b5-keywords-t2.0.jsonl"


def test_main_ensure_t1(tmp_path, monkeypatch):
    # server default 1.0: the requests are those without the flag, byte for byte
    plain, ensured = FakeServer(), FakeServer()
    run_main(tmp_path / "a", monkeypatch, plain)
    assert run_main(tmp_path / "b", monkeypatch, ensured, "--ensure-t1") == outputs(tmp_path / "a")
    assert [json.dumps(b) for b in ensured.bodies] == [json.dumps(b) for b in plain.bodies]
    # server default 1.6947: options.temperature = 1 in every request, the T = 1 name
    server = FakeServer(default_t=1.6947)
    assert run_main(tmp_path / "c", monkeypatch, server, "--ensure-t1") == [
        "preds/decide-gemma-4-e4b-b5-keywords.jsonl", "runs/decide-gemma-4-e4b-b5-keywords.json"]
    assert len(server.bodies) == 2 and all(b["options"]["temperature"] == 1.0 for b in server.bodies)
    # the engine still applies another T: stop after the warm-up, before writing anything
    server = FakeServer(default_t=1.6947, ignores_option=True)
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path / "d", monkeypatch, server, "--ensure-t1")
    assert "1.6947" in str(e.value.code) and len(server.bodies) == 1 and not (tmp_path / "d/out").exists()


def test_main_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    preds, run = tmp_path / "out" / "preds/decide-gemma-4-e4b-b5-keywords.jsonl", tmp_path / "out" / "runs/decide-gemma-4-e4b-b5-keywords.json"
    assert run_main(tmp_path, monkeypatch, FakeServer(p_true=0.3)) == ["preds/" + preds.name, "runs/" + run.name]
    before = preds.read_text(), run.read_text()
    server = FakeServer(p_true=0.6)
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, server)
    assert "--force" in str(e.value.code) and (preds.read_text(), run.read_text()) == before
    assert len(server.bodies) == 1  # refused after the warm-up (which names the run), before the timed requests
    preds.unlink()
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, FakeServer(p_true=0.6))  # the run JSON alone also blocks
    assert not preds.exists()
    run_main(tmp_path, monkeypatch, FakeServer(p_true=0.6), "--force")
    assert [json.loads(l)["angry"] for l in preds.read_text().splitlines()] == [0.6] * 3 and run.read_text() != before[1]
