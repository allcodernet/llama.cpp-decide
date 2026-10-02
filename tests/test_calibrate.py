import json
from pathlib import Path

import pytest

from calibrate import GRID_POINTS, METRICS, calibrate, fit_temperature, grid, main, temper, temper_preds
from sp3_metrics import QUEUES, nll, score_preds

# Two ticket profiles of 8 tickets: every ticket of a profile gets the same prediction, and the profile's labels
# follow its distribution q exactly (label counts / 8), so a prediction p equal to q minimises the NLL.
PROFILES = [
    (["billing"] * 4 + ["technical"] * 2 + ["sales", "feedback"], [0, 0, 0, 0, 1, 1, 2, 3], [True] * 2 + [False] * 6),
    (["feedback"] * 5 + ["sales", "technical", "billing"], [3, 3, 2, 2, 2, 1, 1, 0], [True] * 6 + [False] * 2),
]


def normalise(w):
    return [x / sum(w) for x in w]


def synthetic(t0, t0_urgency=None, t0_angry=None):
    """(preds, labels) of a model that reports p ∝ q^t0 (per field when t0_urgency / t0_angry are given): scaling a
    field with T = its t0 gives q back, that field's NLL minimum."""
    t0_urgency = t0 if t0_urgency is None else t0_urgency
    t0_angry = t0 if t0_angry is None else t0_angry
    preds, labels = [], []
    for queue, urgency, angry in PROFILES:
        n = len(queue)
        q_queue = [queue.count(k) / n for k in QUEUES]
        q_urgency = [urgency.count(level) / n for level in range(4)]
        q_true = angry.count(True) / n
        row = {"queue": dict(zip(QUEUES, normalise([x ** t0 for x in q_queue]))),
               "urgency": normalise([x ** t0_urgency for x in q_urgency]),
               "angry": normalise([q_true ** t0_angry, (1 - q_true) ** t0_angry])[0],
               "latency_s": 0.1}
        for label_queue, label_urgency, label_angry in zip(queue, urgency, angry):
            preds.append(row)
            labels.append({"text": "", "queue": label_queue, "urgency": label_urgency, "angry": label_angry})
    return preds, labels


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def loss(preds, labels, t):
    return nll(temper_preds(preds, t), labels)["sum"]


def test_temper_matches_the_spec_examples():
    assert temper([0.8, 0.2], 2.0) == pytest.approx([2 / 3, 1 / 3])  # spec 12.1
    assert temper([0.7, 0.2, 0.1], 1.0) == [0.7, 0.2, 0.1]  # T = 1: the engine returns p unchanged
    assert temper([0.6, 0.4, 0.0], 0.5) == pytest.approx([9 / 13, 4 / 13, 0.0])  # p^2 renormalised; zeros stay zero


def test_temper_preds_scales_every_field_and_keeps_the_decisions():
    preds, labels = synthetic(2.0)
    scaled = temper_preds(preds, 2.0)
    assert scaled[0]["queue"] == pytest.approx(dict(zip(QUEUES, [0.5, 0.25, 0.125, 0.125])))  # q of profile 1
    assert scaled[0]["urgency"] == pytest.approx([0.5, 0.25, 0.125, 0.125])
    assert scaled[0]["angry"] == pytest.approx(0.25)
    assert scaled[0]["latency_s"] == 0.1 and preds[0]["angry"] == pytest.approx(0.1)  # the input rows are unchanged
    for t in (0.25, 3.0):  # argmax and p_true >= 0.5 are invariant under temperature
        before, after = score_preds(preds, labels), score_preds(temper_preds(preds, t), labels)
        assert [before[k] for k in ("queue", "urgency", "angry")] == [after[k] for k in ("queue", "urgency", "angry")]


def test_grid_is_64_log_spaced_points_from_a_quarter_to_four():
    g = grid()
    assert len(g) == GRID_POINTS == 64
    assert g[0] == pytest.approx(0.25) and g[-1] == pytest.approx(4.0)
    ratios = [b / a for a, b in zip(g, g[1:])]
    assert min(ratios) == pytest.approx(max(ratios)) == pytest.approx(16 ** (1 / 63))


@pytest.mark.parametrize("t0", [0.3, 0.5, 1.0, 1.37, 2.0, 3.5])
def test_fit_recovers_a_known_temperature(t0):
    preds, labels = synthetic(t0)
    fit = fit_temperature(preds, labels)
    assert fit["t"] == pytest.approx(t0, abs=1e-4)
    lo, hi = fit["grid"]["bracket"]
    assert lo <= fit["grid"]["best"] <= hi and lo <= fit["t_refined"] <= hi
    assert loss(preds, labels, fit["t_refined"]) <= min(loss(preds, labels, t) for t in grid())


def test_fit_minimises_the_sum_of_the_three_fields():
    preds, labels = synthetic(2.0, t0_urgency=1.0, t0_angry=1.0)  # queue alone wants T = 2, urgency and angry T = 1
    t = fit_temperature(preds, labels)["t_refined"]
    assert 1.05 < t < 1.95
    # the summed NLL is convex in 1/T, so a local minimum is the minimum
    assert loss(preds, labels, t) <= min(loss(preds, labels, t - 1e-3), loss(preds, labels, t + 1e-3))


@pytest.mark.parametrize("t0, edge", [(0.1, 0.25), (8.0, 4.0)])
def test_fit_stays_inside_the_grid(t0, edge):
    assert fit_temperature(*synthetic(t0))["t"] == pytest.approx(edge, abs=1e-4)


def test_calibrate_reports_before_and_after_and_applies_the_rule():
    train, test = synthetic(2.0), synthetic(2.0)  # the test set is miscalibrated the same way
    result = calibrate(*train, *test)
    assert result["t"] == pytest.approx(2.0, abs=1e-4)
    assert METRICS == ["nll_queue", "nll_urgency", "nll_angry", "nll_sum", "ece_queue", "ece_urgency", "ece_angry",
                       "brier_angry"]
    for split, (preds, labels) in (("train", train), ("test", test)):
        before, after = score_preds(preds, labels), score_preds(temper_preds(preds, result["t"]), labels)
        assert result[split] == {"before": {k: before[k] for k in METRICS}, "after": {k: after[k] for k in METRICS}}
        assert result[split]["after"]["nll_sum"] < result[split]["before"]["nll_sum"]
    rule = result["rule"]
    assert rule["test_nll_sum_before"] == result["test"]["before"]["nll_sum"]
    assert rule["test_nll_sum_after"] == result["test"]["after"]["nll_sum"]
    assert rule["apply"] is True


def test_rule_keeps_the_preset_when_t_raises_the_test_nll():
    result = calibrate(*synthetic(2.0), *synthetic(1.0))  # fitted on T = 2 miscalibration; the test set is calibrated
    assert result["t"] == pytest.approx(2.0, abs=1e-4)
    assert result["rule"]["test_nll_sum_after"] > result["rule"]["test_nll_sum_before"]
    assert result["rule"]["apply"] is False


def test_main_writes_one_calibration_json_per_model(tmp_path: Path):
    base = tmp_path / "baseline"
    for split in ("train", "test"):
        preds, labels = synthetic(2.0)
        (base / split / "preds").mkdir(parents=True)
        write_jsonl(base / split / "preds" / "decide-m-b5-keywords.jsonl", preds)  # the client's default variant
        write_jsonl(tmp_path / f"{split}.jsonl", labels)
    assert main(["--models", "m", "--baseline-dir", str(base), "--train-set", str(tmp_path / "train.jsonl"),
                 "--test-set", str(tmp_path / "test.jsonl"), "--out-dir", str(tmp_path / "eval")]) == 0
    out = json.loads((tmp_path / "eval" / "calibration-m.json").read_text())
    assert out["model"] == "m" and (out["n_train"], out["n_test"]) == (16, 16)
    assert out["train_preds"] == str(base / "train" / "preds" / "decide-m-b5-keywords.jsonl")
    assert out["test_preds"] == str(base / "test" / "preds" / "decide-m-b5-keywords.jsonl")
    assert (out["grid"]["lo"], out["grid"]["hi"], out["grid"]["points"]) == (0.25, 4.0, 64)
    assert out["t"] == pytest.approx(2.0, abs=1e-4) and out["rule"]["apply"] is True


def test_rule_needs_a_strictly_lower_test_nll():
    result = calibrate(*synthetic(1.0), *synthetic(1.0))  # already calibrated: T = 1 leaves every probability as is
    assert result["t"] == 1.0
    assert result["rule"]["test_nll_sum_after"] == result["rule"]["test_nll_sum_before"]
    assert result["rule"]["apply"] is False
