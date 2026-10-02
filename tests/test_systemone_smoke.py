"""The Python oracle of spec 3.2 (systemone_smoke.to_decide_body), mirroring the test-decide-compat ctest cases."""

import json

from systemone_smoke import docs_example, to_decide_body, triage_ticket0


def test_docs_noul_example():  # ctest case 1
    body = to_decide_body({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}}},
    })
    assert json.dumps(body) == json.dumps({
        "model": "jev-latest",
        "instructions": "Answer each question about the state.",
        "fields": {"is_urgent": {"type": "bool",
                                 "description": "Does this convey urgency? (true: Explicitly time-sensitive; false: No urgency expressed)"}},
        "states": ["Help! My payouts have been failing for 3 days."],
    })


def test_absent_null_empty_instructions_defaults():  # ctest case 3
    fields = to_decide_body({
        "state": "s", "model": "m",
        "questions": {
            "n": {"type": "noul"},
            "c": {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            "s": {"type": "score", "instructions": None, "criteria": ["lo", "hi"]},
            "e": {"type": "noul", "instructions": "", "criteria": {"true": "x"}},
        },
    })["fields"]
    assert fields["n"]["description"] == "Answer yes or no."
    assert fields["c"]["description"] == "Choose the option that applies."
    assert fields["s"]["description"] == "Rate on the given scale."
    assert fields["e"]["description"] == "Answer yes or no. (true: x)"


def test_object_values_and_noul_halves():  # ctest cases 4 and 5
    req = {
        "state": {"ticket": "hé", "tags": [1, 2]}, "model": "m",
        "questions": {
            "c": {"type": "choice", "instructions": {"question": "Pick", "data": {"k": "v"}},
                  "criteria": {"a": {"rubric": "first", "n": 1}, "b": ["x", "y"], "z": None}},
            "s": {"type": "score", "criteria": ["low", "", "high"]},
            "f": {"type": "noul", "instructions": "Q?", "criteria": {"false": "No"}},
            "t0": {"type": "noul", "instructions": "Q?", "criteria": {"true": None}},
            "o": {"type": "noul", "instructions": "Q?", "criteria": {"true": {"a": 1}}},
        },
    }
    body = to_decide_body(req)
    assert body["states"] == ['{\n  "ticket": "hé",\n  "tags": [\n    1,\n    2\n  ]\n}']
    f = body["fields"]
    assert list(f) == ["c", "s", "f", "t0", "o"]
    assert f["c"]["description"] == '{\n  "question": "Pick",\n  "data": {\n    "k": "v"\n  }\n}'
    assert f["c"]["options"] == {"a": '{"rubric":"first","n":1}', "b": '["x","y"]', "z": ""}
    assert f["s"]["levels"] == ["low", "level 1", "high"]
    assert f["f"]["description"] == "Q? (false: No)"
    assert f["t0"]["description"] == "Q?"
    assert f["o"]["description"] == 'Q? (true: {"a":1})'


def test_smoke_requests_translate():
    docs = to_decide_body(docs_example("qwen3.5-9b"))
    assert docs["fields"]["department"]["options"]["sales"] == ""
    assert docs["fields"]["frustration"]["levels"] == ["Calm", '{"level":"Frustrated"}', "Very angry"]
    triage = to_decide_body(triage_ticket0("qwen3.5-9b"))
    assert [v["type"] for v in triage["fields"].values()] == ["choice", "score", "bool"]
    assert len(triage["fields"]["queue"]["options"]) == 4 and len(triage["fields"]["urgency"]["levels"]) == 4
