"""Sub-project 4 smoke checks (the server smoke of Task 2 and the CLI smoke of Task 3) on the test machine; each prints its checks, writes them with the raw
responses to --out and exits 1 when one fails.

  uv run results-engine/image/smoke.py server --url URL --model PRESET --images DIR --out FILE
      a vision server: info, image states (one image, two images, a PNG), a text part holding the media marker, and a string
      state against its one-part form
  uv run results-engine/image/smoke.py cli-text --old-bin DIR --model GGUF --preset P --out FILE
      llama-decide of the old build (DIR) and of engine/build/bin on text requests (tree, full_path, order_debias 2) with
      --dump-tokens: responses and dumps must be identical apart from timings (spec 6)
  uv run results-engine/image/smoke.py cli-image --model GGUF --preset P --images DIR --out FILE
      llama-decide with the preset's projector: --info, the dump of an image state (spec 6), and the same line without
      --mmproj (400 vision_unavailable)
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import reference_check as rc  # noqa: E402
from decide_client import TEST_SET, _preset_settings, build_request, chunks  # noqa: E402

ONE = {"instructions": "Answer each question about this image.",
       "fields": {"cat": {"type": "bool", "description": "Is there a cat in this picture?"}}}
TEXT_TIMINGS = {"prefill_ms", "prefix_ms", "scoring_ms", "total_ms", "rounds", "decodes", "retries"}


def data_uri(path):
    data = Path(path).read_bytes()
    kind = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64," + base64.b64encode(data).decode()}}


def body(model, states, schema=ONE):
    return {"model": model, **schema, "states": states}


def check(results, name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail[:300]}")


def finish(results, raw, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"checks": results, **raw}, indent=1) + "\n")
    sys.exit(0 if all(r["pass"] for r in results) else 1)


def without_timings(resp):
    return {k: v for k, v in resp.items() if k != "timings"}


def cmd_server(args):
    c = httpx.Client(base_url=args.url, timeout=900)
    d = Path(args.images)
    res, raw = [], {}
    raw["info"] = c.get("/v1/decide/info", params={"model": args.model}).json()
    v = raw["info"].get("vision", {})
    check(res, "info: vision available", v.get("available") is True and v.get("max_images_per_state") == 4, json.dumps(v))

    r = c.post("/v1/decide", json=body(args.model, [[data_uri(d / "coco-000000039769.jpg")]]))
    raw["cat"] = r.json()
    check(res, "one image: 200", r.status_code == 200, str(r.status_code))
    if r.status_code == 200:
        j = r.json()
        u, t = j["results"][0]["usage"], j["timings"]
        check(res, "one image: two cats -> p_true > 0.5", j["results"][0]["answers"]["cat"]["p_true"] > 0.5, json.dumps(j["results"][0]["answers"]))
        check(res, "one image: usage images, cells, positions", u.get("images") == 1 and u.get("image_cells", 0) > 0 and u.get("image_positions", 0) > 0, json.dumps(u))
        check(res, "one image: prompt_tokens counts the image cells", j["usage"]["prompt_tokens"] >= u.get("image_cells", 0), json.dumps(j["usage"]))
        check(res, "one image: image timings inside prefill",
              t.get("image_encode_ms", 0) > 0 and t.get("image_decode_ms", 0) > 0 and t["prefill_ms"] >= t["image_encode_ms"] + t["image_decode_ms"], json.dumps(t))

    r = c.post("/v1/decide", json=body(args.model, [[data_uri(d / "coco-000000000776.jpg"), {"type": "text", "text": " and "},
                                                     data_uri(d / "coco-000000000139.jpg")]]))
    raw["two"] = r.json()
    check(res, "two images in one state", r.status_code == 200 and r.json()["results"][0]["usage"].get("images") == 2, str(r.status_code))

    r = c.post("/v1/decide", json=body(args.model, [[data_uri(d / "coco-000000039769.png")]]))
    raw["png"] = r.json()
    check(res, "a PNG image", r.status_code == 200 and r.json()["results"][0]["usage"].get("images") == 1, str(r.status_code))

    r = c.post("/v1/decide", json=body(args.model, [[{"type": "text", "text": "<__media__> is only text here."}]]))
    raw["marker"] = r.json()
    ok = r.status_code == 200 and "images" not in r.json()["results"][0]["usage"] and set(r.json()["timings"]) == TEXT_TIMINGS
    check(res, "media marker in a text part stays text", ok, json.dumps(r.json())[:300])

    s1 = c.post("/v1/decide", json=body(args.model, ["A cat sleeps on the sofa."])).json()
    s2 = c.post("/v1/decide", json=body(args.model, [[{"type": "text", "text": "A cat sleeps on the sofa."}]])).json()
    raw["string"], raw["one_part"] = s1, s2
    check(res, "a string and its one-part form give the same response", without_timings(s1) == without_timings(s2))
    check(res, "a text response has today's keys", set(s1["results"][0]["usage"]) == {"state_tokens", "scored_tokens"} and set(s1["timings"]) == TEXT_TIMINGS,
          json.dumps({"usage": s1["results"][0]["usage"], "timings": sorted(s1["timings"])}))
    finish(res, raw, args.out)


MEDIA_FLAGS = ("--mmproj", "--image-min-tokens", "--image-max-tokens")


def without_media(cmd):
    """cmd without the projector flags and their values (reference_check.decide_command adds them from the preset)."""
    out, skip = [], False
    for x in cmd:
        if skip:
            skip = False
        elif x in MEDIA_FLAGS:
            skip = True
        else:
            out.append(x)
    return out


def with_media(cmd, settings):
    cmd = without_media(cmd) + ["--mmproj", settings["mmproj"]]
    for key in ("image-min-tokens", "image-max-tokens"):
        if key in settings:
            cmd += [f"--{key}", settings[key]]
    return cmd


def run_cli(cmd, lines):
    """llama-decide on request lines: (exit code, stdout responses, the last JSON line of stderr or None)."""
    proc = subprocess.run(cmd, input="".join(json.dumps(x) + "\n" for x in lines), capture_output=True, text=True)
    out = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    err = next((json.loads(x) for x in reversed(proc.stderr.splitlines()) if x.startswith("{")), None)
    return proc.returncode, out, err


def cmd_cli_text(args):
    texts = [json.loads(x)["text"] for x in Path(TEST_SET).read_text().splitlines() if x.strip()][:10]
    bodies = [build_request(args.preset, b) for b in chunks(texts, 5)]
    bodies += [build_request(args.preset, b, scoring="full_path") for b in chunks(texts, 5)]
    bodies += [build_request(args.preset, b, order_debias=2) for b in chunks(texts, 5)]
    res, raw = [], {}
    for name, binary in (("old", ROOT / args.old_bin / "llama-decide"), ("new", ROOT / "engine/build/bin/llama-decide")):
        cmd = rc.decide_command(args.model, args.preset, 64)   # with --dump-tokens
        cmd[0] = str(binary)
        code, out, err = run_cli(cmd, bodies)
        raw[name] = {"returncode": code, "responses": out, "error": err}
        check(res, f"{name}: every line answered", code == 0 and len(out) == len(bodies), json.dumps(err))
    same = [without_timings(a) == without_timings(b) for a, b in zip(raw["old"]["responses"], raw["new"]["responses"])]
    check(res, "old == new: responses and dumps apart from timings", len(same) == len(bodies) and all(same), str(same))
    finish(res, raw, args.out)


def cmd_cli_image(args):
    s = _preset_settings(args.preset)
    seqs = int(s["decide-seqs"])
    image = Path(args.images) / "coco-000000039769.jpg"
    state = [{"type": "text", "text": "Photo from our archive. "}, data_uri(image), {"type": "text", "text": " Is it indoors?"}]
    line = body(args.preset, [state, "A cat sleeps on the sofa."])
    res, raw = [], {}
    code, out, err = run_cli(with_media(rc.decide_command(args.model, args.preset, seqs, dump=False), s) + ["--info"], [line])
    raw["info"] = out
    check(res, "--info with --mmproj: vision available", code == 0 and bool(out) and out[0].get("vision", {}).get("available") is True,
          json.dumps(out[0].get("vision") if out else err))
    code, out, err = run_cli(with_media(rc.decide_command(args.model, args.preset, seqs), s), [line])
    raw["dump"] = out
    check(res, "dump: the line is answered", code == 0 and len(out) == 1, json.dumps(err))
    if code == 0 and len(out) == 1:
        t = out[0]["tokens"]
        img_state, text_state = t["states"]
        check(res, "dump: prefix_texts (one variant) and tail_text",
              isinstance(t.get("prefix_texts"), list) and len(t["prefix_texts"]) == 1 and t.get("tail_text", "").endswith("{\n"))
        types = [p["type"] for p in img_state.get("parts", [])]
        check(res, "image state: parts text, image, text; no state key", types == ["text", "image", "text"] and "state" not in img_state, str(types))
        ip = img_state["parts"][1] if types == ["text", "image", "text"] else {}
        check(res, "image part: SHA-256 of the file", ip.get("sha256") == hashlib.sha256(image.read_bytes()).hexdigest())
        chunk_cells = sum(c["n_tokens"] if c["type"] == "image" else len(c["tokens"]) for c in ip.get("chunks", []))
        check(res, "image part: cells >= positions > 0, chunks add up",
              ip.get("n_tokens", 0) >= ip.get("n_pos", 0) > 0 and chunk_cells == ip.get("n_tokens"), json.dumps({k: ip.get(k) for k in ("n_tokens", "n_pos")}))
        check(res, "image state: pieces_hex for every branch token",
              len(img_state["branches"]) > 0 and all(len(b.get("pieces_hex", [])) == len(b["tokens"]) for b in img_state["branches"]))
        check(res, "text state: today's entry", "state" in text_state and "parts" not in text_state and all("pieces_hex" not in b for b in text_state["branches"]))
        u = out[0]["results"][0]["usage"]
        check(res, "usage of the image state", u.get("images") == 1 and u.get("image_cells") == ip.get("n_tokens") and u.get("image_positions") == ip.get("n_pos"), json.dumps(u))
    code, out, err = run_cli(without_media(rc.decide_command(args.model, args.preset, seqs, dump=False)), [line])
    raw["no_mmproj"] = {"returncode": code, "error": err}
    e = (err or {}).get("error", {})
    check(res, "without --mmproj: exit 1, 400 vision_unavailable at states[0]", code == 1 and e.get("code") == "vision_unavailable" and e.get("field") == "states[0]", json.dumps(err))
    finish(res, raw, args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("server")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_server)
    p = sub.add_parser("cli-text")
    p.add_argument("--old-bin", required=True, help="the old build's bin directory, relative to the project directory")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cli_text)
    p = sub.add_parser("cli-image")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cli_image)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
