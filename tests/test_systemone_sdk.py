"""Acceptance: the official TypeSafe AI Python SDK against our /v1/systemone (needs a live server).

git clone https://github.com/typesafe-ai/typesafe-sdk-python local/typesafe-sdk-python   # tested: v0.7.0
SYSTEMONE_URL=http://127.0.0.1:8097 uv run --with local/typesafe-sdk-python pytest -q tests/test_systemone_sdk.py
Skipped when the SDK is not installed or SYSTEMONE_URL is unset. SYSTEMONE_MODEL picks the preset (default qwen3.5-9b).
"""

import math
import os

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")

SYSTEMONE_URL = os.environ.get("SYSTEMONE_URL")
pytestmark = pytest.mark.skipif(not SYSTEMONE_URL, reason="SYSTEMONE_URL not set")

TICKET = "I was charged twice for my subscription this month and nobody answered my last two emails. Fix this today."


def test_sdk_system_one_end_to_end():
    from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

    client = TypeSafeClient(api_key="local", base_url=SYSTEMONE_URL, retry=RetryPolicy(max_retries=0), timeout=180)
    queues = {"billing": "Payments, refunds, invoices, charges", "technical": "Bugs, crashes, outages, errors",
              "sales": "Contracts, plan changes, pricing", "feedback": None}
    levels = ["Can wait", "Normal", {"level": "Today", "hint": "user blocked or money wrongly taken"}, "Drop everything"]
    resp = client.system_one(
        TICKET,
        {
            "queue": Choice(instructions="Which team must act first on this support ticket?", criteria=queues),
            "urgency": Score(instructions="How urgent is this support ticket?", criteria=levels),
            "angry": Noul(instructions="Does the text itself show anger?",
                          criteria={"true": "hostile wording, shouting, threats", "false": "neutral or polite"}),
        },
        model=os.environ.get("SYSTEMONE_MODEL", "qwen3.5-9b"),
    )

    assert list(resp.answers) == ["queue", "urgency", "angry"]
    queue = resp.answers["queue"]
    assert isinstance(queue, typesafe_sdk.ChoiceAnswer)
    assert list(queue.probabilities) == list(queues) and queue.choice in queues
    assert math.isclose(sum(queue.probabilities.values()), 1.0, abs_tol=1e-6)
    assert queue.probabilities[queue.choice] == max(queue.probabilities.values())
    assert 0.0 <= queue.confidence <= 1.0

    urgency = resp.answers["urgency"]
    assert isinstance(urgency, typesafe_sdk.ScoreAnswer)
    assert urgency.legend == dict(enumerate(levels))
    assert list(urgency.probabilities) == [0, 1, 2, 3]
    assert math.isclose(sum(urgency.probabilities.values()), 1.0, abs_tol=1e-6)
    assert isinstance(urgency.score, float) and 0.0 <= urgency.score <= 3.0
    assert 0.0 <= urgency.confidence <= 1.0

    angry = resp.answers["angry"]
    assert isinstance(angry, typesafe_sdk.NoulAnswer)
    assert 0.0 <= angry.noul <= 1.0

    assert resp.usage.input_tokens > 0 and resp.usage.output_tokens == 0
    assert resp.request_id  # the SDK raises when the x-typesafe-request-id header is missing
    print(f"\nSDK {typesafe_sdk.__version__}: model={resp.model} queue={queue.choice} "
          f"urgency={urgency.score:.3f} angry={angry.noul:.3f} usage={resp.usage.input_tokens}/{resp.usage.output_tokens} "
          f"request_id={resp.request_id}")


def test_sdk_raises_typed_422():
    """A one-level Score is valid for the SDK but not for our compiler: the SDK must raise its typed 422 error."""
    from typesafe_sdk import RetryPolicy, Score, TypeSafeClient, TypeSafeUnprocessableEntityError

    client = TypeSafeClient(api_key="local", base_url=SYSTEMONE_URL, retry=RetryPolicy(max_retries=0), timeout=180)
    with pytest.raises(TypeSafeUnprocessableEntityError) as exc:
        client.system_one(TICKET, {"one": Score(criteria=["only level"])}, model=os.environ.get("SYSTEMONE_MODEL", "qwen3.5-9b"))
    print(f"\n422: {exc.value}")
