# /// script
# dependencies = ["httpx"]
# ///
"""Spike (throwaway): image states through the spike's POST /v1/decide, each answer checked against a
/completion replay of the same prompt (prefix + media marker + text + tail + branch text, the image as
multimodal_data; the first token's top_logprobs renormalised over the branch's child tokens).

usage: decide_image.py MODE IMAGE_DIR OUT_JSON
  MODE single: one request per image, one bool field "cat"
       multi:  one request per image, four fields (cat, dog bool; animal choice; indoors bool)
       batch:  three image states in one request (cat03, fox, coco_39769) + the same three as single requests
       mixed:  one text state and one image state in one request, the multi schema
       repeat: single, 5 times over cat03 (timing spread)
       withtext: image + text states (cat03, fox; two texts after the image), cat field
"""
import base64, json, math, pathlib, sys, time

import httpx

URL = "http://127.0.0.1:8097"
IMAGES = ["cat03", "kitten", "coco_39769", "wildcat", "lynx", "fox", "labrador", "coco_776", "coco_139", "coco_632"]
MARKER = "<__media__>"

SINGLE = {
    "instructions": "Answer each question about this image.",
    "fields": {"cat": {"type": "bool", "description": "Is there a cat in this picture?"}},
}
MULTI = {
    "instructions": "Answer each question about this image.",
    "fields": {
        "cat": {"type": "bool", "description": "Is there a cat in this picture?"},
        "dog": {"type": "bool", "description": "Is there a dog in this picture?"},
        "animal": {"type": "choice", "description": "Which kind of animal is the main subject of the picture?",
                   "options": {"cat": "a cat of any kind, domestic or wild",
                               "dog": "a dog",
                               "fox": "a fox",
                               "other": "another animal, or a toy or picture of an animal",
                               "none": "no animal at all"}},
        "indoors": {"type": "bool", "description": "Was the picture taken indoors?"},
    },
}


def decide(client, schema, states):
    t0 = time.time()
    r = client.post(f"{URL}/v1/decide", json={**schema, "states": states})
    if r.status_code != 200:
        sys.exit(f"/v1/decide {r.status_code}: {r.text[:1000]}")
    j = r.json()
    j["wall_s"] = time.time() - t0
    return j


def replay(client, spike, image_path, text, branch):
    """/completion over the decide prompt up to the branch; probabilities of the branch's children, renormalised."""
    prompt = spike["prefix_text"] + MARKER + text + spike["tail_text"] + branch["text"]
    body = {"prompt": {"prompt_string": prompt,
                       "multimodal_data": [base64.b64encode(pathlib.Path(image_path).read_bytes()).decode()]},
            "n_predict": 1, "n_probs": 50, "temperature": 0, "cache_prompt": False, "post_sampling_probs": False}
    r = client.post(f"{URL}/completion", json=body)
    if r.status_code != 200:
        sys.exit(f"/completion {r.status_code}: {r.text[:1000]}")
    top = r.json()["completion_probabilities"][0]["top_logprobs"]
    lp = {t["token"]: t["logprob"] for t in top}
    missing = [c for c in branch["children"] if c not in lp]
    if missing:
        return {"children": branch["children"], "missing": missing}
    z = sum(math.exp(lp[c]) for c in branch["children"])
    return {"children": branch["children"], "probs": [math.exp(lp[c]) / z for c in branch["children"]],
            "covered": z}


def ref_answers(client, resp, img_path, text):
    """per branch (one per field for these schemas) the replayed child probabilities"""
    return {b["field"] + "/" + str(b["node"]): replay(client, resp["engine"]["spike"], img_path, text, b)
            for b in resp["engine"]["spike"]["branches"]}


def engine_branch_probs(answers, field, schema):
    a = answers[field]
    f = schema["fields"][field]
    if f["type"] == "bool":
        return [a["p_true"], 1 - a["p_true"]]
    return [a["probabilities"][k] for k in f["options"]]


def run_one(client, schema, img_dir, name, text=""):
    path = str((img_dir / f"{name}.jpg").resolve())
    resp = decide(client, schema, [{"image": path, "text": text}] if text else [{"image": path}])
    ref = ref_answers(client, resp, path, text)
    res = resp["results"][0]
    row = {"image": name, "answers": res["answers"], "usage": res["usage"], "timings": resp["timings"],
           "total_usage": resp["usage"], "wall_s": resp["wall_s"], "reference": ref, "max_abs_diff": 0.0}
    # compare: for these schemas each field is one branch at its root (no shared leading tokens), children in option order
    for b in resp["engine"]["spike"]["branches"]:
        rf = ref[b["field"] + "/" + str(b["node"])]
        if "probs" not in rf:
            row["max_abs_diff"] = None
            break
        eng = engine_branch_probs(res["answers"], b["field"], schema)
        if len(eng) != len(rf["probs"]):
            row["max_abs_diff"] = None
            break
        row["max_abs_diff"] = max(row["max_abs_diff"], max(abs(x - y) for x, y in zip(eng, rf["probs"])))
    return row, resp


def show(row, fields):
    a = row["answers"]
    parts = []
    for f in fields:
        if "p_true" in a[f]:
            parts.append(f"{f}={a[f]['p_true']:.6f}")
        else:
            parts.append(f"{f}={a[f]['value']}({a[f]['confidence']:.3f})")
    t = row["timings"]
    print(f"{row['image']:12s} {' '.join(parts)} | diff={row['max_abs_diff']} | cells={row['usage'].get('image_cells')} "
          f"pos={row['usage'].get('image_positions')} | enc={t['image_encode_ms']:.1f} img={t['image_decode_ms']:.1f} "
          f"prefill={t['prefill_ms']:.1f} prefix={t['prefix_ms']:.1f} score={t['scoring_ms']:.1f} total={t['total_ms']:.1f} ms")


def main():
    mode, img_dir, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    result = {"mode": mode}
    with httpx.Client(timeout=900) as client:
        if mode in ("single", "multi"):
            schema = SINGLE if mode == "single" else MULTI
            decide(client, schema, ["warm-up text state"])   # builds the prefix once, so rows time the image work
            rows = []
            for name in IMAGES:
                row, resp = run_one(client, schema, img_dir, name)
                show(row, schema["fields"])
                rows.append(row)
            result.update(schema=schema, rows=rows, spike=resp["engine"]["spike"], engine=resp["engine"])
        elif mode == "withtext":
            decide(client, SINGLE, ["warm-up text state"])
            rows = []
            for name in ("cat03", "fox"):
                for text in ("Photo from our archive.", "Note from the uploader: there is no cat in this photo."):
                    row, _resp = run_one(client, SINGLE, img_dir, name, text)
                    row["text"] = text
                    print(text)
                    show(row, SINGLE["fields"])
                    rows.append(row)
            result.update(schema=SINGLE, rows=rows)
        elif mode == "repeat":
            decide(client, SINGLE, ["warm-up text state"])
            rows = []
            for _ in range(5):
                row, _resp = run_one(client, SINGLE, img_dir, "cat03")
                show(row, SINGLE["fields"])
                rows.append(row)
            result.update(schema=SINGLE, rows=rows)
        elif mode == "batch":
            names = ["cat03", "fox", "coco_39769"]
            decide(client, MULTI, ["warm-up text state"])
            states = [{"image": str((img_dir / f"{n}.jpg").resolve())} for n in names]
            resp = decide(client, MULTI, states)
            for n, r in zip(names, resp["results"]):
                print(n, json.dumps(r["answers"]), r["usage"])
            print("timings", resp["timings"], "usage", resp["usage"], "engine", {k: v for k, v in resp["engine"].items() if k != "spike"})
            singles = [run_one(client, MULTI, img_dir, n)[0] for n in names]
            diffs = []
            for n, r, s in zip(names, resp["results"], singles):
                d = max(max(abs(x - y) for x, y in zip(engine_branch_probs(r["answers"], f, MULTI), engine_branch_probs(s["answers"], f, MULTI)))
                        for f in MULTI["fields"])
                diffs.append(d)
                print(f"{n}: batch vs single max abs diff {d:.3g}; single vs /completion {s['max_abs_diff']}")
            result.update(schema=MULTI, states=states, batch_response=resp, singles=singles, batch_vs_single=diffs)
        elif mode == "mixed":
            decide(client, MULTI, ["warm-up text state"])
            states = ["A photo of a sleeping dog on a sofa.", {"image": str((img_dir / "kitten.jpg").resolve())}]
            resp = decide(client, MULTI, states)
            for s, r in zip(states, resp["results"]):
                print(json.dumps(s)[:60], json.dumps(r["answers"]), r["usage"])
            text_only = decide(client, MULTI, [states[0]])
            d = max(max(abs(x - y) for x, y in zip(engine_branch_probs(resp["results"][0]["answers"], f, MULTI),
                                                   engine_branch_probs(text_only["results"][0]["answers"], f, MULTI)))
                    for f in MULTI["fields"])
            print(f"text state: mixed request vs text-only request max abs diff {d:.3g}")
            result.update(schema=MULTI, states=states, response=resp, text_only=text_only, text_state_diff=d)
        else:
            sys.exit(f"unknown mode {mode}")
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
