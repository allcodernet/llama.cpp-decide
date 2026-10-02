import json
import subprocess
import sys

import pytest

import reference_check
from reference_check import option_probs


def test_option_probs_multiplies_along_paths():
    # root -> {refund_* (options 0, 1), outage (2)}; refund node -> {_request (0), _status (1)}
    nodes = [{"options": [[0, 1], [2]]}, {"options": [[0], [1]]}]
    probs = option_probs(nodes, 3, [[0.6, 0.4], [0.75, 0.25]])
    assert probs == [0.6 * 0.75, 0.6 * 0.25, 0.4]
    assert abs(sum(probs) - 1.0) < 1e-12


def test_dump_passes_decide_seqs(monkeypatch, tmp_path):
    test_set = tmp_path / "t.jsonl"
    test_set.write_text(json.dumps({"text": "x", "queue": "billing", "urgency": 0, "angry": False}) + "\n")
    seen = []

    def fake_run(cmd, **kwargs):  # stop cmd_dump right after it built the llama-decide command
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "stop")

    monkeypatch.setattr(reference_check.subprocess, "run", fake_run)
    for extra, expected in [([], "9"), (["--decide-seqs", "64", "--tag", "s64"], "64")]:
        monkeypatch.setattr(sys, "argv", ["reference_check.py", "dump", "--model", "m.gguf", "--preset", "qwen2.5-0.5b",
                                          "--n", "1", "--test-set", str(test_set), *extra])
        with pytest.raises(SystemExit):
            reference_check.main()
        assert seen[-1][seen[-1].index("--decide-seqs") + 1] == expected
