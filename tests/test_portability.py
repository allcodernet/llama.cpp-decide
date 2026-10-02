"""Settings from the environment (follow-ups Task 3): TRIAGE_DATA, ENGINE_PRESETS, ENGINE_BIN, the placement keys of
reference_check.decide_command, eval/analysis.py's MODELS and directories, the preset step's decision. Without them
every default holds: TRIAGE_DATA is the bundled data/ (relative values from the project directory), the other settings
keep the values the recorded runs used."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import decide_client
import reference_check

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_analysis():
    """analysis.py reads MODELS and the directories at import: load it with them unset, so the defaults are tested
    whatever the caller exported (e.g. the environment of results-engine/3090/run.sh)."""
    spec = importlib.util.spec_from_file_location("eval_analysis", ROOT / "results-engine" / "sp3" / "eval" / "analysis.py")
    mod = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        for name in ("MODELS", "EVAL_DIR", "BASELINE_DIR", "COST_DIR", "TRIAGE_DATA"):
            mp.delenv(name, raising=False)
        spec.loader.exec_module(mod)
    return mod


analysis = load_analysis()


def data_paths(env_value):
    """decide_client.TEST_SET, score.TEST_SET and heldout_stats.TRAIN_SET in a fresh interpreter (they are read at
    import) with TRIAGE_DATA set to env_value, or unset for None; the working directory is not the project's."""
    env = {k: v for k, v in os.environ.items() if k != "TRIAGE_DATA"}
    if env_value is not None:
        env["TRIAGE_DATA"] = env_value
    env["PYTHONPATH"] = str(ROOT)
    code = ("import decide_client, score, heldout_stats; "
            "print(decide_client.TEST_SET); print(score.TEST_SET); print(heldout_stats.TRAIN_SET)")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd="/", capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_triage_data_unset_is_the_bundled_data():
    assert data_paths(None) == [f"{DATA}/triage_test.jsonl"] * 2 + [f"{DATA}/triage_train.jsonl"]
    assert data_paths("") == data_paths(None)  # empty = unset, as ${TRIAGE_DATA:-...} in the run scripts


def test_triage_data_absolute_and_relative(tmp_path):
    assert data_paths(str(tmp_path)) == [f"{tmp_path}/triage_test.jsonl"] * 2 + [f"{tmp_path}/triage_train.jsonl"]
    # relative: from the project directory, whatever the working directory
    assert data_paths("local/data") == [f"{ROOT}/local/data/triage_test.jsonl"] * 2 + [f"{ROOT}/local/data/triage_train.jsonl"]


PRESETS = """[*]
ngl = 99
ctx-size = 8192
decide-seqs = 33

[big]
model = /m/big.gguf
n-cpu-moe = 12
tensor-split = 3,1
override-tensor = exps=CPU
main-gpu = 1
split-mode = row
device = CUDA0,CUDA1
"""


def test_engine_presets_default_is_models_engine_ini(monkeypatch):
    monkeypatch.delenv("ENGINE_PRESETS", raising=False)
    assert decide_client._preset_settings("gemma-4-e4b")["decide-temperature"] == "1.6947"
    assert decide_client._preset_settings("qwen3.5-9b")["model"] == "./models/Qwen3.5-9B-Q4_K_M.gguf"
    assert reference_check._engine_settings("qwen2.5-0.5b")["cache-type-k"] == "f16"
    assert reference_check._engine_settings()["decide-seqs"] == "21"


def test_engine_presets_absolute_and_relative(monkeypatch, tmp_path):
    ini = tmp_path / "presets.ini"
    ini.write_text(PRESETS)
    monkeypatch.setenv("ENGINE_PRESETS", str(ini))
    assert decide_client._preset_settings("big")["decide-seqs"] == "33"
    assert decide_client._preset_settings("big")["model"] == "/m/big.gguf"
    assert reference_check._engine_settings("big")["ctx-size"] == "8192"
    monkeypatch.chdir(tmp_path)  # relative: from the project directory, not the working directory
    monkeypatch.setenv("ENGINE_PRESETS", "models-engine-3090.ini")
    assert decide_client._preset_settings("qwen3.8-27b")["decide-seqs"] == "64"
    assert reference_check._engine_settings("qwen3.8-27b")["ngl"] == "999"
    assert "decide-temperature" not in decide_client._preset_settings("gemma-4-e4b")


# decide_command with models-engine.ini as committed, before this change
TODAY = {
    "qwen2.5-0.5b": ["-m", "m.gguf", "--decide-seqs", "64", "-c", "4096", "-ngl", "99", "-fa", "on", "-ctk", "f16",
                     "-ctv", "f16", "--dump-tokens"],
    "gemma-4-e4b": ["-m", "m.gguf", "--decide-seqs", "64", "-c", "4096", "-ngl", "99", "-fa", "on", "-ctk", "q8_0",
                    "-ctv", "q8_0", "--dump-tokens"],
    "qwen3-30b-a3b": ["-m", "m.gguf", "--decide-seqs", "64", "-c", "4096", "-ngl", "99", "-fa", "on", "-ctk", "q8_0",
                      "-ctv", "q8_0", "--dump-tokens", "--cpu-moe"],
}


@pytest.mark.parametrize("preset", list(TODAY))
def test_decide_command_defaults_unchanged(monkeypatch, preset):
    monkeypatch.delenv("ENGINE_PRESETS", raising=False)
    monkeypatch.delenv("ENGINE_BIN", raising=False)
    cmd = reference_check.decide_command("m.gguf", preset, 64)
    assert cmd == [str(ROOT / "engine" / "build" / "bin" / "llama-decide"), *TODAY[preset]]


def test_decide_command_engine_bin(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGINE_PRESETS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENGINE_BIN", "engine-63ea2c51a/build/bin")
    assert reference_check.decide_command("m.gguf", "gemma-4-e4b", 64)[0] == str(
        ROOT / "engine-63ea2c51a" / "build" / "bin" / "llama-decide")
    monkeypatch.setenv("ENGINE_BIN", "/opt/x/bin")
    assert reference_check.decide_command("m.gguf", "gemma-4-e4b", 64, dump=False)[0] == "/opt/x/bin/llama-decide"


def test_decide_command_passes_placement_keys(monkeypatch, tmp_path):
    ini = tmp_path / "presets.ini"
    ini.write_text(PRESETS)
    monkeypatch.setenv("ENGINE_PRESETS", str(ini))
    monkeypatch.delenv("ENGINE_BIN", raising=False)
    cmd = reference_check.decide_command("m.gguf", "big", 33, dump=False)
    assert cmd[1:] == ["-m", "m.gguf", "--decide-seqs", "33", "-c", "8192", "-ngl", "99", "-fa", "on", "-ctk", "f16",
                       "-ctv", "f16", "--n-cpu-moe", "12", "--tensor-split", "3,1", "--override-tensor", "exps=CPU",
                       "--main-gpu", "1", "--split-mode", "row", "--device", "CUDA0,CUDA1"]
    # only the keys a preset holds
    ini.write_text("[*]\nngl = 99\n\n[one]\nmain-gpu = 0\n")
    assert reference_check.decide_command("m.gguf", "one", 9, dump=False)[-2:] == ["--main-gpu", "0"]
    assert "--tensor-split" not in reference_check.decide_command("m.gguf", "one", 9, dump=False)


def test_analysis_settings_defaults():
    models, eval_dir, baseline, cost = analysis.settings({})
    assert models == ["qwen3.5-9b", "gemma-4-e4b"]
    assert eval_dir == analysis.HERE == ROOT / "results-engine" / "sp3" / "eval"
    assert baseline == ROOT / "results-engine" / "sp3" / "baseline"
    assert cost == ROOT / "results-engine" / "sp3" / "integrity" / "regression"
    assert analysis.settings({"MODELS": "", "EVAL_DIR": "", "BASELINE_DIR": "", "COST_DIR": ""}) == (
        models, eval_dir, baseline, cost)  # empty = unset, as ${X:-...} in run.sh


def test_analysis_settings_from_the_environment():
    env = {"MODELS": "qwen3.8-27b", "EVAL_DIR": "results-engine/3090/eval", "BASELINE_DIR": "/tmp/b"}
    models, eval_dir, baseline, cost = analysis.settings(env)
    assert (models, eval_dir, baseline) == (["qwen3.8-27b"], ROOT / "results-engine/3090/eval", Path("/tmp/b"))
    assert cost == Path("/tmp/b")  # COST_DIR follows BASELINE_DIR when only BASELINE_DIR is set
    assert analysis.settings({**env, "COST_DIR": "c"})[3] == ROOT / "c"
    assert analysis.settings({"MODELS": " a  b "})[0] == ["a", "b"]


def test_cost_source_names_task_8_only_for_the_default():
    assert analysis.cost_source(ROOT / "results-engine" / "sp3" / "integrity" / "regression") == (
        "the regression run of sp3/integrity/, the same requests on this build")
    assert analysis.cost_source(ROOT / "results-engine" / "3090" / "baseline") == (
        "the tree runs in `results-engine/3090/baseline`")
    assert analysis.cost_source(Path("/tmp/x/baseline")) == "the tree runs in `/tmp/x/baseline`"


def calibration(t, apply):
    return {"t": t, "rule": {"apply": apply}}


@pytest.mark.parametrize("cal, settings, want", [
    (calibration(1.6947, True), {"decide-temperature": "1.6947"}, ("check", 1.6947)),  # today's Gemma check
    (calibration(1.6947, True), {}, ("add", 1.6947)),
    (calibration(1.6947, True), {"decide-temperature": "1.5"}, ("add", 1.6947)),
    (calibration(0.9, False), {}, ("skip", 0.9)),
    (calibration(0.9, False), {"decide-temperature": "0.9"}, ("skip", 0.9)),
])
def test_preset_action(cal, settings, want):
    assert analysis.preset_action(cal, settings) == want


def test_preset_step_defaults_match_todays_gemma_check(monkeypatch, capsys):
    """With the committed calibrations and models-engine.ini: Qwen3.5-9B skipped, Gemma checked at its fitted T."""
    monkeypatch.delenv("ENGINE_PRESETS", raising=False)
    assert analysis.main(["preset", "qwen3.5-9b"]) == 0
    assert capsys.readouterr().out == "skip 1.1187\n"
    assert analysis.main(["preset", "gemma-4-e4b"]) == 0
    assert capsys.readouterr().out == "check 1.6947\n"  # run.sh passes it to `name --temperature` as today
