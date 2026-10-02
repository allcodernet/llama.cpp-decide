import pytest

from heldout_stats import decide, pick_winner, sign_test


def test_sign_test_matches_the_80_ticket_report():
    # keywords vs default on queue in REPORT-ENGINE.md: 3-0, 5-0, 2-0, 4-1 -> 0.25, 0.063, 0.50, 0.38
    for b, c, p in [(3, 0, 0.25), (5, 0, 0.0625), (2, 0, 0.5), (4, 1, 0.375), (0, 0, 1.0), (2, 2, 1.0)]:
        variant = [True] * b + [False] * c + [True, False]
        default = [False] * b + [True] * c + [True, False]
        t = sign_test(variant, default)
        assert (t["variant_only_right"], t["default_only_right"], t["discordant"]) == (b, c, b + c)
        assert t["p"] == pytest.approx(p)


def test_pick_winner_tie_break_and_rule():
    right = {"default": {"queue": 70, "urgency": 60, "angry": 70},
             "a": {"queue": 72, "urgency": 60, "angry": 70},
             "b": {"queue": 72, "urgency": 62, "angry": 69}}
    assert pick_winner(right) == ("b", ["b"])
    right["a"]["urgency"] = 61
    assert pick_winner(right) == (None, ["a", "b"])
    win = {"winner": "b", "winner_vs_default_queue": {"p": 0.01}}
    assert decide({"m1": win, "m2": win})["change_default"] is True
    assert decide({"m1": win, "m2": {**win, "winner_vs_default_queue": {"p": 0.05}}})["change_default"] is False
    assert decide({"m1": win, "m2": {"winner": None, "winner_vs_default_queue": None}})["change_default"] is False
    assert decide({"m1": win, "m2": {**win, "winner": "a"}})["change_default"] is False
