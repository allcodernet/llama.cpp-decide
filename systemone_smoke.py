"""Smoke for the Jev-compatible /v1/systemone route against our own /v1/decide.

Usage: uv run systemone_smoke.py [--url URL] [--model PRESET]

to_decide_body() is the test oracle: spec 3.2 of
docs/specs/2026-09-25-systemone-compat-design.md written again in Python
(same defaults and text helpers as engine/tools/decide/decide-compat.cpp). The smoke posts
each Jev request to /v1/systemone and the oracle body to /v1/decide and asserts identical
probabilities (+-1e-9) for every question. A last case checks the x-typesafe-request-id header
and that a state nested 40 levels deep is a fast 422 after which the server still answers.
"""

import argparse
import json
import sys
import time

import httpx

from decide_client import ANGRY_INSTRUCTIONS, QUEUE_INSTRUCTIONS, QUEUES, URGENCY, URGENCY_INSTRUCTIONS

TOP_INSTRUCTIONS = "Answer each question about the state."
DEFAULTS = {
    "noul": "Answer yes or no.",
    "choice": "Choose the option that applies.",
    "score": "Rate on the given scale.",
}
TOL = 1e-9


def text(x):
    """Spec 3.1 text(): string as is, object/array pretty JSON (indent 2, UTF-8 kept), null ''."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, indent=2, ensure_ascii=False)


def short(x):
    """Spec 3.1 short(): string as is, object/array compact JSON (nlohmann dump()), null ''."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def _field(q):
    kind = q["type"]
    instr = q.get("instructions")
    d = DEFAULTS[kind] if instr is None or instr == "" else text(instr)
    if kind == "noul":
        crit = q.get("criteria") or {}
        parts = [f"{half}: {short(crit[half])}" for half in ("true", "false") if half in crit and short(crit[half])]
        suffix = f" ({'; '.join(parts)})" if parts else ""
        return {"type": "bool", "description": d + suffix}
    if kind == "choice":
        return {"type": "choice", "description": d, "options": {k: short(v) for k, v in q["criteria"].items()}}
    levels = [short(e) or f"level {i}" for i, e in enumerate(q["criteria"])]
    return {"type": "score", "description": d, "levels": levels}


def to_decide_body(jev_request):
    """The /v1/decide body the server builds from a (valid) Jev System One request."""
    return {
        "model": jev_request["model"],
        "instructions": TOP_INSTRUCTIONS,
        "fields": {qid: _field(q) for qid, q in jev_request["questions"].items()},
        "states": [text(jev_request["state"])],
    }


def docs_example(model):
    """The three examples of Jev's public API docs merged; `sales` null and an object level exercise short()."""
    return {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": model,
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                          "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payments, invoicing, refunds",
                                        "technical": "Bugs, outages, integrations", "sales": None}},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", {"level": "Frustrated"}, "Very angry"]},
        },
    }


TICKET_0 = ("Does the annual receipt break out the sales tax separately? My accountant asked, "
            "and I haven't had one yet as I only joined in May. No hurry.")


def triage_ticket0(model):
    """Triage ticket 0 with the decide_client.py questions in Jev form."""
    return {
        "state": TICKET_0,
        "model": model,
        "questions": {
            "queue": {"type": "choice", "instructions": QUEUE_INSTRUCTIONS, "criteria": dict(QUEUES)},
            "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTIONS, "criteria": list(URGENCY)},
            "angry": {"type": "noul", "instructions": ANGRY_INSTRUCTIONS},
        },
    }


def compare(jev_request, sys_resp, dec_resp):
    """Max abs probability difference between the two endpoints; raises on a shape mismatch."""
    dec = dec_resp["results"][0]["answers"]
    assert list(sys_resp["answers"]) == list(jev_request["questions"]), "answer order differs from question order"
    worst = 0.0
    for qid, q in jev_request["questions"].items():
        a, b = sys_resp["answers"][qid], dec[qid]
        assert a["type"] == q["type"], f"{qid}: type {a['type']}"
        if q["type"] == "noul":
            pairs = [(a["noul"], b["p_true"])]
        elif q["type"] == "choice":
            assert list(a["probabilities"]) == list(b["probabilities"]) and a["choice"] == b["value"], qid
            pairs = [(a["probabilities"][k], b["probabilities"][k]) for k in b["probabilities"]]
        else:
            assert list(a["probabilities"]) == [str(i) for i in range(len(b["probabilities"]))], qid
            assert list(a["legend"].values()) == list(q["criteria"]), f"{qid}: legend"
            pairs = [(a["probabilities"][str(i)], p) for i, p in enumerate(b["probabilities"])]
            pairs.append((a["score"], b["expected"]))
        worst = max([worst] + [abs(x - y) for x, y in pairs])
    return worst


def run_case(client, name, jev_request):
    rs = client.post("/v1/systemone", json=jev_request)
    rd = client.post("/v1/decide", json=to_decide_body(jev_request))
    if rs.status_code != 200 or rd.status_code != 200:
        sys.exit(f"{name}: /v1/systemone {rs.status_code} {rs.text[:300]} | /v1/decide {rd.status_code} {rd.text[:300]}")
    sys_resp, dec_resp = rs.json(), rd.json()
    diff = compare(jev_request, sys_resp, dec_resp)
    print(f"== {name}")
    print(json.dumps(sys_resp, indent=2, ensure_ascii=False))
    print(f"max abs diff vs /v1/decide oracle body: {diff:.3g}")
    return diff


def nested(depth):
    """A value nested `depth` levels deep: a scalar is depth 0, [1] depth 1."""
    value = 1
    for _ in range(depth):
        value = [value]
    return value


def limits_case(client, model):
    """x-typesafe-request-id on every response; a depth-40 state is a 422 in < 1 s and the server keeps serving."""
    t0 = time.perf_counter()
    deep = client.post("/v1/systemone", json={"state": nested(40), "model": model, "questions": {"q": {"type": "noul"}}})
    dt = time.perf_counter() - t0
    after = client.post("/v1/systemone", json=docs_example(model))
    ids = [deep.headers.get("x-typesafe-request-id"), after.headers.get("x-typesafe-request-id")]
    print("== limits")
    print(f"depth 40: {deep.status_code} in {dt * 1000:.1f} ms: {deep.text}")
    print(f"follow-up docs example: {after.status_code}; x-typesafe-request-id: {ids}")
    if not all(ids) or ids[0] == ids[1]:
        sys.exit(f"FAIL: x-typesafe-request-id missing or repeated: {ids}")
    detail = deep.json().get("detail") if deep.status_code == 422 else None
    if not detail or detail[0]["loc"] != ["body", "state"] or detail[0]["msg"] != "nesting deeper than 32 levels" or dt >= 1.0:
        sys.exit(f"FAIL: depth 40 gave {deep.status_code} in {dt:.3f} s: {deep.text[:300]}")
    if after.status_code != 200:
        sys.exit(f"FAIL: follow-up request gave {after.status_code}: {after.text[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--model", default="qwen3.5-9b")
    args = ap.parse_args()
    client = httpx.Client(base_url=args.url, timeout=600)
    diffs = [run_case(client, "docs example", docs_example(args.model)),
             run_case(client, "triage ticket 0", triage_ticket0(args.model))]
    worst = max(diffs)
    if worst > TOL:
        sys.exit(f"FAIL: probabilities differ by {worst:.3g} (> {TOL})")
    limits_case(client, args.model)
    print(f"OK: identical probabilities through both endpoints (max abs diff {worst:.3g}); "
          "request id header and depth-40 422 checked")


if __name__ == "__main__":
    main()
