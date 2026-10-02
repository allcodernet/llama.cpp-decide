import json
from pathlib import Path

import pytest

from score import score_file


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_score_file_metrics(tmp_path: Path):
    test = [
        {"text": "a", "queue": "billing", "urgency": 0, "angry": False},
        {"text": "b", "queue": "sales", "urgency": 3, "angry": True},
        {"text": "c", "queue": "technical", "urgency": 1, "angry": False},
        {"text": "d", "queue": "feedback", "urgency": 2, "angry": True},
    ]
    preds = [
        {"queue": {"billing": 0.9, "technical": 0.05, "sales": 0.03, "feedback": 0.02},
         "urgency": [0.7, 0.2, 0.05, 0.05], "angry": 0.1, "latency_s": 0.1, "distribution": "full"},
        {"queue": {"billing": 0.6, "technical": 0.1, "sales": 0.2, "feedback": 0.1},   # wrong queue
         "urgency": [0.0, 0.1, 0.6, 0.3], "angry": 0.8, "latency_s": 0.3, "distribution": "full"},  # urgency off by 1
        {"queue": {"billing": 0.2, "technical": 0.5, "sales": 0.2, "feedback": 0.1},
         "urgency": [0.2, 0.5, 0.2, 0.1], "angry": 0.4, "latency_s": 0.2, "distribution": "full"},
        {"queue": {"billing": 0.1, "technical": 0.1, "sales": 0.1, "feedback": 0.7},
         "urgency": [0.6, 0.2, 0.1, 0.1], "angry": 0.9, "latency_s": 0.4, "distribution": "full"},  # urgency off by 2
    ]
    p = tmp_path / "x.jsonl"
    write(p, preds)
    m = score_file(p, test)
    assert m["queue"] == pytest.approx(0.75)
    assert m["urgency"] == pytest.approx(0.5)
    assert m["urgency_near"] == pytest.approx(0.75)
    assert m["angry"] == pytest.approx(1.0)
    assert m["p50_ms"] == pytest.approx(250.0)  # median of 100, 300, 200, 400 = 250
    assert m["brier_angry"] == pytest.approx((0.01 + 0.04 + 0.16 + 0.01) / 4)
    assert m["ece_queue"] == pytest.approx(0.375)
    assert m["nll_queue"] == pytest.approx(-(sum(map(__import__("math").log, [0.9, 0.2, 0.5, 0.7]))) / 4)


def test_score_file_winner_only_has_no_nll(tmp_path: Path):
    test = [{"text": "a", "queue": "billing", "urgency": 0, "angry": False}]
    p = tmp_path / "w.jsonl"
    write(p, [{"queue": {"billing": 0.8, "technical": 0, "sales": 0, "feedback": 0},
               "urgency": [0.8, 0, 0, 0], "angry": 0.2, "latency_s": 0.1, "distribution": "winner_only"}])
    m = score_file(p, test)
    assert m["queue"] == 1.0 and m["nll_queue"] is None


def test_score_file_length_mismatch_raises(tmp_path: Path):
    p = tmp_path / "short.jsonl"
    write(p, [])
    with pytest.raises(AssertionError):
        score_file(p, [{"text": "a", "queue": "billing", "urgency": 0, "angry": False}])
