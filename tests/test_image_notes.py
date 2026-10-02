import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("image_notes", ROOT / "results-engine" / "image" / "notes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_integrity_row_skipped_failed_and_binding(tmp_path):
    nt = load()
    skipped = tmp_path / "smolvlm-500m"
    skipped.mkdir()
    (skipped / "SKIPPED.txt").write_text("fit check: does not load\nmore\n")
    row = cells(nt.integrity_row(skipped, "smolvlm-500m"))
    assert len(row) == len(nt.HEAD) and row[1] == "skipped: fit check: does not load"

    part = tmp_path / "gemma-4-e4b-vision"   # non-binding: one failed reference, one file missing
    part.mkdir()
    ref = {"pass": False, "fields": {"cat": {"median_abs_diff": 0.02, "disagreements_over_margin": 1},
                                     "dog": {"median_abs_diff": 0.001, "disagreements_over_margin": 1}}}
    (part / "reference-tree-one-gemma-4-e4b-vision.json").write_text(json.dumps(ref))
    (part / "held-new.json").write_text(json.dumps({"pass": True, "applicable": False, "old_held": 148,
                                                    "budget": {"held": 148}}))
    row = cells(nt.integrity_row(part, "gemma-4-e4b-vision"))
    assert len(row) == len(nt.HEAD)
    assert row[1] == "**FAIL** (0.02; 2)" and row[2] == "missing (see failures.txt)"
    assert row[-2] == "missing (see failures.txt)" and row[-1] == "pass, n/a (148 = 148)"

    with pytest.raises(SystemExit):   # the binding model must have every file
        nt.integrity_row(part, nt.BINDING)


def test_held_cell():
    nt = load()
    assert nt.held_cell(None) == "missing (see failures.txt)"
    assert nt.held_cell({"pass": True, "applicable": True, "old_held": 58, "budget": {"held": 1057}}) == "pass (58 → 1057)"
    assert nt.held_cell({"pass": False, "applicable": False, "old_held": 148, "budget": {"held": 148}}) == "**FAIL**, n/a (148 = 148)"
    assert nt.held_cell({"pass": False, "budget": None}) == "**FAIL** (no budget answer; see held-new.json)"


def test_chat_runs_and_unclear():
    nt = load()
    a = [{"image_id": 1, "fields": {"cat": {"answer": "No", "p_yes": 0.1}, "dog": {"answer": "```", "p_yes": 0.4}}}]
    b = [{"image_id": 1, "fields": {"cat": {"answer": "no", "p_yes": 0.2}, "dog": {"answer": "Yes", "p_yes": 0.6}}}]
    assert nt.chat_runs(a, b) == (2, 2, 1, 1)
    u = [{"answer": "```", "p_yes": 0.2, "label": False}, {"answer": "", "p_yes": None, "label": False}]
    assert nt.unclear(u) == "2 (labels: false 2; p_yes on the label's side of 0.5: 1; answers: \"```\" 1, \"\" 1)"


def test_sections_on_the_committed_results():
    nt = load()
    text = "\n".join(line for section in (nt.regression, nt.no_vision, nt.integrity, nt.swa, nt.coco, nt.final_engine) for line in section())
    assert "| qwen3.8-27b-vision | pass" in text and "smolvlm-500m | skipped:" in text
    assert "## Final engine (spec rev 6)" in text and "**FAIL**" not in text.split("## Final engine")[1]


def test_gate_detail_tolerates_missing_files(tmp_path):
    nt = load()
    rec = {"ticket": 1, "field": "dog", "reference": [0.65, 0.35], "engine": [0.1, 0.9]}
    for t in ("tree-multi", "fp"):
        (tmp_path / f"reference-{t}-gemma-3-4b-vision.json").write_text(json.dumps({"detail": [rec]}))
    text = "\n".join(nt.gate_detail(tmp_path))   # gemma-3 is non-binding: no retry.json, no batch.json
    assert "P(dog = true): reference 0.650" in text and "retry missing (see failures.txt)" in text
    assert "batch missing (see failures.txt)" in text
    assert nt.gate_detail(tmp_path / "absent") == []


def test_held_answers_without_pairs(tmp_path):
    nt = load()
    assert nt.held_answers(tmp_path) == ["", "Held cells, answer to the sized state, old build → new build: no held pair recorded."]
