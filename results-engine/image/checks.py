"""Sub-project 4 live checks (spec 9.2-9.5), called by results-engine/image/run.sh on the test machine. Every command writes
JSON with a `pass` key to --out and exits 1 when it is false.

  text-run --url U --model P --out F.jsonl [--debias K] [--limit N] [--test-set T] [--server-bin BIN]
      spec 9.2: the text request sequence on the test tickets (TRIAGE_DATA) in batches of 5: tree, full_path, after (urgency
      and angry after queue), order_debias K (default 2), temperature 0.5, then 8 requests alternating the keywords and default
      prompt variants (prefix cache); one line per request: label, status, response. Exit 1 when a request is not 200.
      The server's /v1/decide/info goes to F.info.json (build and projector evidence); with --server-bin (the llama-server
      the router was started with) it gains server_build: that path, its --version lines and its engine checkout's commit.
  text-compare A.jsonl B.jsonl --out F.json
      spec 9.2: request by request the same status and the same response apart from timings; pass = no difference
      (max abs diff 0)
  no-vision --url U --model P --image IMG --out F.json [--log ROUTER_LOG] [--skip-rejected]
      spec 9.3 on a server without a projector: info, an image request, a mixed request, info again, then text-run's sequence
      on the first 10 tickets; --skip-rejected sends only info and the text sequence (the run without the rejected requests)
  no-vision-compare WITH.json WITHOUT.json --out F.json
      spec 9.3: the text sequence after the rejected requests equals the run without them (probabilities, cache counts)
  cli-no-vision --model GGUF --preset P --image IMG --out F.json
      spec 9.3: llama-decide without --mmproj answers an image state with 400 vision_unavailable (exit 1, nothing on stdout)
  malformed --url U --model P --image JPG --out F.json [--skip-cases]
      spec 9.4 and 3.2 on a fresh server: a fixed valid request, then the failing cases of spec 9.1 (malformed_cases, each
      400 invalid_states at its state; two of them fail in pass 3 on the main loop) and an image state over the budget (400
      budget naming the image cells, also on the main loop), info unchanged, the fixed request again. --skip-cases sends
      only the fixed request, the two info calls and the fixed request again (the run of a fresh server without the cases)
      The three aspect-ratio cases (spec 3.1 rev 6) must also name the 200:1 limit in their message.
  limits --url U --model P --image JPG --out F.json
      spec 3.1, 3.4, 5.1 rev 6 on a fresh server: info's vision.max_image_aspect_ratio is 200; decodable PNGs of 200 x 1 and
      1 x 200 answer 200, of 201 x 1 and 1 x 201 400 invalid_states naming the limit; the early budget check: a state of a
      sized text, then k copies of JPG (k = 1, 2), an undecodable PNG and JPG again, whose cells cross the limit at image k,
      is 400 budget naming the cells of k images (the undecodable PNG is never reached; a control within the budget shows
      it gives 400 invalid_states when it is); info unchanged by these rejected requests (read after the last 200)
  double-fault --url U --model P --image JPG --out F.json
      the order of the two budget checks on a fresh server: a schema with as many bool fields as decide sequences (too few
      sequences) and a short text state is 400 budget naming the sequences; one field and a state of a sized text and JPG
      whose cells cross the limit at the image is 400 budget naming the image cells; both faults at once answer the
      sequence-budget 400 (it runs before the prefixes are prepared and before any image is loaded; the engine before that
      change answered the image cells); info unchanged by these rejected requests
  vision-reason --model GGUF --preset P
      prints vision.reason of llama-decide --info with the preset's projector, "available" when vision is available
  malformed-compare WITH.json WITHOUT.json --out F.json
      spec 3.2 "a 400 leaves the engine as it was": both fixed responses equal those of the run without the cases, apart
      from timings (prefix cache counts included)
  batch --url U --model P --states S.jsonl --schema F.json --out F.json
      spec 9.4: the states alone, together in one request and mixed with as many text states (tickets); the statistic
      (sp3_metrics.compare) of together and of mixed against alone
  retry --model GGUF --preset P --states S.jsonl --schema F.json --out F.json
      spec 9.4: llama-decide with LLAMA_DECIDE_FAULT=rc1 against a run without it, under the statistic; retries >= 1 vs 0
  held --url U --model P --image JPG --out F.json [--old OLD.json]
      spec 9.4: a chat request with the image stays in the server's slot; a decide probe's 400 budget text gives held, prefix,
      tail, branches and the limit; a text state sized to fit exactly under that held count is sent. Without --old (the old
      build) the answer is recorded; with --old (the new build, same sequence, the old run's size) it must be 400 budget when
      the new held count is larger, else the check is not applicable (cells equal positions)
  ubatch --model GGUF --preset P --image JPG --out F.json
      spec 5.3: llama-decide -ub 128 answers an image state of a non-causal projector with 400 budget naming n_ubatch
  swa --url U --model P --out F.json [--debias K] [--chat] [--round] [--expect-swa-cells N]
      spec 5.3 rev 5: optionally a text chat resident in the slot; info's memory.swa_cells; a probe's 400 budget gives the cell
      counts and the limit min(n_ctx, swa_cells) - 16 (its text names the SWA cache exactly when that is the bound); a sweep of
      one-state requests around the largest state the budget admits: up to it 200 (state_tokens as sized), over it 400 budget
      with the same limit, never 500. --round: two requests of three states each whose first two fill the limit, all three
      over it but within n_ctx - 16 (the sizing before rev 5 put them in one round): three copies of one text, and three
      distinct texts; each must answer 200 in two rounds without a retry, its first two results must equal the first two
      states sent as one request and its third result that state sent alone (max abs diff 0); the comparison with every
      text alone is recorded, not a gate
  swa-info --model GGUF --preset P --out F.json [--expect-swa-cells N] [--prefix-words N]
      spec 5.3 rev 5: llama-decide --info -v (--swa-full when the preset has it): memory.swa_cells equals the size of the SWA
      KV cache in llama.cpp's log. --prefix-words: llama-decide --info for instructions of N words: with the SWA cache below
      n_ctx a prefix over its limit (and within n_ctx - 16) is 400 budget naming the SWA cache, else info answers
  gate --dir D --preset P --out F.json
      spec 9.5: the gate passes when every reference-<tag>-<preset>.json of REF_TAGS and batch.json in D pass and D has no
      SKIPPED.txt
  gate-off --url U --model P --image JPG --out F.json
      spec 9.5 after a failed gate: info reports non_causal_projector and an image request gets 400 vision_unavailable
"""
import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import reference_check as rc  # noqa: E402
from decide_client import TEST_SET, build_request, chunks, image_part, state_from_spec, text_part  # noqa: E402
from sp3_metrics import compare  # noqa: E402
from decide_client import _preset_settings  # noqa: E402

ONE_FIELD = {"instructions": "Answer each question about this image.",
             "fields": {"cat": {"type": "bool", "description": "Is there a cat in this picture?"}}}
ERROR_LINE = re.compile(r"error|exception|abort|fail", re.IGNORECASE)
REF_TAGS = ("tree-one", "tree-multi", "fp", "after", "debias2", "t0.5")  # run.sh ref_args; spec 9.4
SWA_SCHEMA = {"instructions": "Answer the question about this text.",
              "fields": {"animal": {"type": "choice", "description": "Which animal does the text name?", "options": ["cat", "dog", "bird"]}}}
ROUND_WORDS = (("dog", "cat", "bird", "bird"), ("cat", "bird"), ("bird", "dog", "dog", "cat"))   # swa --round: three texts
PREFIX_BUDGET = re.compile(r"prefix needs (\d+) cells with (\d+) held by slots, (?:n_ctx - 16 is (\d+)|the sliding-window cache "
                           r"holds (\d+) cells \(limit (\d+)\); start with --swa-full to use n_ctx)")
SWEEP_OFFSETS = (-64, -16, -2, -1, 0, 1, 2, 16)   # swa: state sizes around the largest admitted one
TINY_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAIAAAA2iEnWAAAAGElEQVR4nAXBAQEAAAjDIG7/zhOEoHJiPUfdBv3wKgpGAAAAAElFTkSuQmCC")
# decide-debias.cpp state_cells_message: K_eff only for K > 1; the limit's text names n_ctx or the SWA cache (spec 5.3 rev 5)
BUDGET = re.compile(r"needs (\d+) cells \(held (\d+), prefix (\d+), (?:K_eff (\d+) \u00d7 \()?state (\d+)(?: \(image cells \d+\))?, "
                    r"tail (\d+), branches (\d+)\)\)?, (?:n_ctx - 16 is (\d+)|the sliding-window cache holds (\d+) cells \(limit (\d+)\))")


def body_for(model, schema, states):
    return {"model": model, "instructions": schema.get("instructions", ""), "fields": schema["fields"], "states": states}


def post(client, path, body):
    r = client.post(path, json=body)
    return {"status": r.status_code, "response": r.json()}


def info(client, model):
    r = client.get("/v1/decide/info", params={"model": model})
    if r.status_code != 200:
        sys.exit(f"/v1/decide/info: {r.status_code} {r.text[:300]}")
    return r.json()


def last_json(text):
    """The last line of `text` that starts with '{', parsed (llama-decide writes its error JSON last on stderr)."""
    for line in reversed(text.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


def write(out, result):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("checks", "max_abs_diff", "differences", "pass") if k in result}, indent=1)[:4000])
    sys.exit(0 if result["pass"] else 1)


def json_diff(a, b, path="", skip=("timings",)):
    """(max abs diff over the numbers, [every place where a and b differ]); keys in `skip` are left out at any depth."""
    if isinstance(a, dict) and isinstance(b, dict):
        worst, diffs = 0.0, []
        if set(a) != set(b):
            diffs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            if k in skip:
                continue
            w, d = json_diff(a[k], b[k], f"{path}/{k}", skip)
            worst, diffs = max(worst, w), diffs + d
        return worst, diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, [f"{path}: length {len(a)} vs {len(b)}"]
        worst, diffs = 0.0, []
        for i, (x, y) in enumerate(zip(a, b)):
            w, d = json_diff(x, y, f"{path}/{i}", skip)
            worst, diffs = max(worst, w), diffs + d
        return worst, diffs
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0, ([] if type(a) is type(b) and a == b else [f"{path}: {a!r} vs {b!r}"])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b), ([] if a == b else [f"{path}: {a!r} vs {b!r}"])
    return 0.0, ([] if a == b else [f"{path}: {a!r} vs {b!r}"])


def text_requests(model, test_set, debias, limit=None):
    """spec 9.2: (label, body) of the text sequence: each of the five switches of sub-project 3 on the tickets in batches of 5."""
    texts = [json.loads(l)["text"] for l in Path(test_set).read_text().splitlines() if l.strip()]
    batches = chunks(texts[:limit] if limit else texts, 5)
    seq = [("tree", build_request(model, b, "keywords")) for b in batches]
    seq += [("full_path", build_request(model, b, "keywords", scoring="full_path")) for b in batches]
    seq += [("after", build_request(model, b, "keywords", after={"urgency": ["queue"], "angry": ["queue"]})) for b in batches]
    seq += [(f"debias{debias}", build_request(model, b, "keywords", order_debias=debias)) for b in batches]
    seq += [("t0.5", build_request(model, b, "keywords", temperature=0.5)) for b in batches]
    seq += [("cache", build_request(model, b, "keywords" if i % 2 == 0 else "default")) for i, b in enumerate(batches[:8])]
    return seq


def load_lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def cmd_text_run(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    lines = [{"label": label, **post(client, "/v1/decide", body)} for label, body in text_requests(args.model, args.test_set, args.debias, args.limit)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(l) + "\n" for l in lines))
    # the server's info next to the output: it shows the build and the projector (the old build has no `vision` key)
    sidecar = info(client, args.model)
    if args.server_bin:
        sidecar["server_build"] = server_build(args.server_bin)
    Path(args.out).with_suffix(".info.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    bad = [(l["label"], l["status"]) for l in lines if l["status"] != 200]
    print(f"{len(lines)} requests, not 200: {bad}")
    sys.exit(1 if bad else 0)


def server_build(binary):
    """what identifies a server binary: its path, the version lines of its --version and the commit of the engine checkout it
    lives in (<engine>/build/bin/llama-server; None when that is no git checkout)"""
    p = Path(binary)
    v = subprocess.run([str(p), "--version"], capture_output=True, text=True)
    g = subprocess.run(["git", "-C", str(p.resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"path": str(binary),
            "version": [l for l in (v.stdout + v.stderr).splitlines() if l.startswith(("version:", "built with"))],
            "engine_commit": g.stdout.strip() if g.returncode == 0 else None}


def cmd_text_compare(args):
    a, b = load_lines(args.a), load_lines(args.b)
    worst, differences = 0.0, ([] if len(a) == len(b) else [f"{len(a)} vs {len(b)} requests"])
    for i, (x, y) in enumerate(zip(a, b)):
        w, d = json_diff(x, y, f"#{i} {x['label']}")
        worst, differences = max(worst, w), differences + d
    write(args.out, {"a": str(args.a), "b": str(args.b), "requests": len(a), "max_abs_diff": worst, "differences": differences[:50],
                     "pass": not differences})


def log_window_checks(lines):
    """(error lines, checks) of the router log lines written after the rejected requests: the window holds their router
    lines (at least two `proxying request` lines, so a wrong offset cannot pass) and no error line."""
    errors = [l for l in lines if ERROR_LINE.search(l)]
    return errors, {"the log window holds the rejected requests (2+ proxying request lines)":
                    sum("proxying request" in l for l in lines) >= 2,
                    "no error line from the rejected requests": not errors}


def cmd_no_vision(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    image_state = [image_part(args.image)]
    out = {"model": args.model, "info_before": info(client, args.model)}
    checks = {"info_reports_no_projector": out["info_before"].get("vision") == {"available": False, "reason": "no_projector"}}
    if not args.skip_rejected:
        offset = Path(args.log).stat().st_size
        out["image_request"] = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [image_state]))
        out["mixed_request"] = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, ["A cat sleeps on the sofa.", image_state]))
        time.sleep(2)  # the router writes its log asynchronously
        out["info_after"] = info(client, args.model)
        log = Path(args.log).read_bytes()
        out["log_lines"] = log[offset:].decode("utf-8", "replace").splitlines()
        out["log_error_lines"], window = log_window_checks(out["log_lines"])
        for name, field in (("image_request", "states[0]"), ("mixed_request", "states[1]")):
            r, e = out[name], out[name]["response"].get("error", {})
            checks[f"{name}: 400 vision_unavailable at {field}"] = (r["status"] == 400 and e.get("code") == "vision_unavailable"
                                                                    and e.get("field") == field and "--mmproj" in e.get("message", ""))
        checks["info unchanged by the rejected requests"] = out["info_after"] == out["info_before"]
        checks["the log holds the engine's own lines"] = b"decide: " in log
        checks.update(window)
    out["text"] = [{"label": l, **post(client, "/v1/decide", b)} for l, b in text_requests(args.model, args.test_set, 2, 10)]
    checks["the server keeps serving (text requests 200)"] = all(t["status"] == 200 for t in out["text"])
    out["checks"] = checks
    out["pass"] = all(checks.values())
    write(args.out, out)


def cmd_no_vision_compare(args):
    w, wo = json.loads(Path(args.with_rejected).read_text()), json.loads(Path(args.without).read_text())
    worst, differences = 0.0, ([] if len(w["text"]) == len(wo["text"]) else ["request counts differ"])
    for i, (x, y) in enumerate(zip(w["text"], wo["text"])):
        d_w, d = json_diff(x, y, f"#{i} {x['label']}")
        worst, differences = max(worst, d_w), differences + d
    write(args.out, {"model": w["model"], "with_rejected_pass": w["pass"], "without_pass": wo["pass"], "requests": len(w["text"]),
                     "max_abs_diff": worst, "differences": differences[:50], "pass": w["pass"] and wo["pass"] and not differences})


def cmd_cli_no_vision(args):
    cmd = rc.decide_command(args.model, args.preset, 9, dump=False)
    body = body_for(args.preset, ONE_FIELD, [[image_part(args.image)]])
    proc = subprocess.run(cmd, input=json.dumps(body) + "\n", capture_output=True, text=True)
    err = last_json(proc.stderr) or {}
    e = err.get("error", {})
    write(args.out, {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "error": err,
                     "pass": proc.returncode == 1 and e.get("code") == "vision_unavailable" and e.get("field") == "states[0]"
                     and not proc.stdout.strip()})


def png_header(w, h):
    """A PNG signature and IHDR chunk (8-bit RGB) without pixel data: the header check passes, stb_image cannot decode it."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def jpeg_sof(data):
    """Offset of the first SOF0/SOF1/SOF2 marker, walking the JPEG's segments from SOI."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError(f"no JPEG marker at offset {i}")
        if data[i + 1] in (0xC0, 0xC1, 0xC2):
            return i
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError("no SOF marker")


def data_url(data, subtype="png"):
    return f"data:image/{subtype};base64," + base64.b64encode(data).decode()


def img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def malformed_cases(jpg):
    """spec 9.4 Malformed input: (name, states, field) of the failing cases of spec 9.1, each a 400 invalid_states. `jpg` is a
    real JPEG; its 12-bit copy passes the header check and fails in stb_image (pass 3, in run_request)."""
    ok, b64 = img(data_url(TINY_PNG)), base64.b64encode(TINY_PNG).decode()
    big = png_header(1, 1)
    big += bytes(10485761 - len(big))
    twelve = bytearray(jpg)
    twelve[jpeg_sof(jpg) + 4] = 12
    return [
        ("empty part list", ["A text state.", []], "states[1]"),
        ("adjacent text parts", [[text_part("a"), text_part("b")]], "states[0]"),
        ("unknown part type", [[{"type": "input_image", "image_url": {"url": data_url(TINY_PNG)}}]], "states[0]"),
        ("an extra key (detail)", [[{"type": "image_url", "image_url": {"url": data_url(TINY_PNG), "detail": "low"}}]], "states[0]"),
        ("an empty text part", [[text_part("")]], "states[0]"),
        ("a part that is not an object", [["text"]], "states[0]"),
        ("5 images in a state", [[ok] * 5], "states[0]"),
        ("33 images in a request", [[ok] * 4] * 9, "states[8]"),
        ("10 MiB + 1 byte", [[img(data_url(big))]], "states[0]"),
        ("8000 x 4001 pixels", [[img(data_url(png_header(8000, 4001)))]], "states[0]"),
        ("an http URL", [[img("http://images.cocodataset.org/val2017/000000039769.jpg")]], "states[0]"),
        ("a file: URL", [[img("file:///etc/hostname")]], "states[0]"),
        ("a server path", [[img("/home/marduk/llama.cpp-decision/local/image/integrity/coco-000000039769.jpg")]], "states[0]"),
        ("empty data", [[img("data:image/png;base64,")]], "states[0]"),
        ("IMAGE/ in upper case", [[img("data:IMAGE/png;base64," + b64)]], "states[0]"),
        ("the subtype PNG in upper case", [[img("data:image/PNG;base64," + b64)]], "states[0]"),
        (";base64 without the comma", [[img("data:image/png;base64" + b64)]], "states[0]"),
        ("bare base64", [[img(b64)]], "states[0]"),
        ("DATA: in upper case", [[img("DATA:image/png;base64," + b64)]], "states[0]"),
        ("a URL parameter", [[img("data:image/png;charset=utf-8;base64," + b64)]], "states[0]"),
        ("URL-safe base64", [[img("data:image/png;base64," + b64.replace("+", "-").replace("/", "_"))]], "states[0]"),
        ("unpadded base64", [[img("data:image/png;base64,iVBORw")]], "states[0]"),
        ("a newline in the base64", [[img("data:image/png;base64," + b64[:40] + "\n" + b64[40:])]], "states[0]"),
        ("BMP bytes", [[img(data_url(b"BM" + bytes(60), "bmp"))]], "states[0]"),
        ("12-bit JPEG (pixel decoding, pass 3)", [[ok], [img(data_url(bytes(twelve), "jpeg"))]], "states[1]"),
        ("a PNG header without pixel data", [[img(data_url(png_header(2, 3)))]], "states[0]"),
        ("201 x 1 pixels (aspect ratio)", [[img(data_url(png_header(201, 1)))]], "states[0]"),
        ("a 16000 x 2 strip (aspect ratio)", [[img(data_url(png_header(16000, 2)))]], "states[0]"),
        ("a 16000000 x 2 strip (aspect ratio)", [[img(data_url(png_header(16000000, 2)))]], "states[0]"),
    ]


RATIO_TEXT = "more than 200 times its shorter side"   # decide-image.cpp parse_image, spec 3.1 rev 6
SEQ_TEXT = "sequences per state"   # decide-debias.cpp sequence_budget_message


def solid_png(w, h):
    """A decodable 8-bit gray PNG of w x h pixels, every pixel 128."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + bytes([128]) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


def cmd_malformed(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    fixed = body_for(args.model, ONE_FIELD, [[image_part(args.image)], "A cat sleeps on the sofa."])
    out = {"model": args.model, "skip_cases": args.skip_cases, "fixed": [post(client, "/v1/decide", fixed)]}
    before = info(client, args.model)
    rows, checks = [], {}
    if not args.skip_cases:
        for name, states, field in malformed_cases(Path(args.image).read_bytes()):
            r = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, states))
            e = r["response"].get("error", {})
            checks[name] = r["status"] == 400 and e.get("code") == "invalid_states" and e.get("field") == field
            if "(aspect ratio)" in name:
                checks[name] = checks[name] and RATIO_TEXT in e.get("message", "")
            rows.append({"case": name, "field": field, "status": r["status"], "error": e})
        # spec 3.2, 5.3: an image state over the budget, rejected on the main loop after its image was tokenised
        r = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[image_part(args.image), text_part("cat" + " cat" * 19999)]]))
        e = r["response"].get("error", {})
        checks["image state over the budget: 400 budget naming the image cells"] = (
            r["status"] == 400 and e.get("code") == "budget" and "(image cells " in e.get("message", ""))
        rows.append({"case": "image state over the budget", "status": r["status"], "error": e})
    checks["info unchanged"] = info(client, args.model) == before
    out["fixed"].append(post(client, "/v1/decide", fixed))
    checks["the fixed request: 200 before and after"] = all(f["status"] == 200 for f in out["fixed"])
    write(args.out, {**out, "cases": rows, "checks": checks, "pass": all(checks.values())})


def cmd_malformed_compare(args):
    w, wo = json.loads(Path(args.with_cases).read_text()), json.loads(Path(args.without).read_text())
    worst, differences = 0.0, []
    for i, (x, y) in enumerate(zip(w["fixed"], wo["fixed"])):
        d_w, d = json_diff(x, y, f"fixed #{i}")
        worst, differences = max(worst, d_w), differences + d
    write(args.out, {"model": w["model"], "with_cases_pass": w["pass"], "without_pass": wo["pass"], "max_abs_diff": worst,
                     "differences": differences[:50],
                     "pass": w["pass"] and wo["pass"] and len(w["fixed"]) == len(wo["fixed"]) == 2 and not differences})


def cmd_limits(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    v = info(client, args.model).get("vision") or {}
    out, checks = {"model": args.model, "vision": v}, {}
    checks["info: vision.max_image_aspect_ratio 200"] = v.get("max_image_aspect_ratio") == 200
    # spec 3.1 rev 6: decodable PNGs at 200:1 are answered, at 201:1 rejected in pass 1, in both orientations
    for w, h in ((200, 1), (1, 200), (201, 1), (1, 201)):
        r = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[img(data_url(solid_png(w, h)))]]))
        e, name = r["response"].get("error", {}), f"{w} x {h}"
        out[name] = r
        if max(w, h) <= 200 * min(w, h):
            checks[f"{name}: 200 with one image"] = r["status"] == 200 and r["response"]["results"][0]["usage"].get("images") == 1
        else:
            checks[f"{name}: 400 invalid_states naming the aspect-ratio limit"] = (
                r["status"] == 400 and e.get("code") == "invalid_states" and e.get("field") == "states[0]" and RATIO_TEXT in e.get("message", ""))
    # spec 5.1 rev 6: the cells so far are checked right after each image; an image after the one that crosses the limit is never
    # loaded, so the undecodable PNG behind it (pass 3: 400 invalid_states when it is reached) does not decide the answer
    image, broken = image_part(args.image), img(data_url(png_header(2, 3)))
    alone = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[image]]))
    probe = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, ["cat" + " cat" * 19999]))
    b = budget_parts(probe["response"].get("error", {}).get("message"))
    out["budget"] = b
    checks["probes: the image alone 200, a long text 400 budget with the cell counts"] = alone["status"] == 200 and b is not None
    if not all(checks.values()):
        write(args.out, {**out, "alone": alone, "probe": probe, "checks": checks, "pass": False})
    cells, room = alone["response"]["results"][0]["usage"]["image_cells"], largest_state(b)
    out["image_cells"], out["largest_state"] = cells, room
    before = info(client, args.model)   # after the last 200: only rejected requests follow
    control = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[text_part("A short text."), image, broken, image]]))
    out["control"] = control
    checks["control within the budget: the undecodable PNG gives 400 invalid_states (pass 3)"] = (
        control["status"] == 400 and control["response"].get("error", {}).get("code") == "invalid_states")
    for k in (1, 2):
        n = room - k * cells + cells // 2   # fits with k - 1 images, over the limit with k
        r = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[text_part(sized_text(client, args.model, n))] + [image] * k + [broken, image]]))
        e = r["response"].get("error", {})
        s, m = budget_parts(e.get("message")) or {}, re.search(r"\(image cells (\d+)\)", e.get("message", ""))
        out[f"over after image {k}"] = {"text_tokens": n, "status": r["status"], "error": e}
        checks[f"over the limit after image {k}: 400 budget with the cells of {k} image(s), before the undecodable PNG"] = (
            r["status"] == 400 and e.get("code") == "budget" and s.get("state") == n + k * cells and m is not None
            and int(m.group(1)) == k * cells)
    after = info(client, args.model)
    checks["info unchanged by the rejected requests"] = after == before
    if after != before:
        out["info_before"], out["info_after"] = before, after
    write(args.out, {**out, "checks": checks, "pass": all(checks.values())})


def cmd_double_fault(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    seqs = info(client, args.model)["decide_seqs"]
    # one branch sequence per bool field: K (1 + seqs) > seqs - K, so no state fits a round (the sequence budget)
    many = {"instructions": ONE_FIELD["instructions"],
            "fields": {f"q{i}": {"type": "bool", "description": f"Is statement {i} about this picture true?"} for i in range(seqs)}}
    image = image_part(args.image)
    alone = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[image]]))
    probe = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, ["cat" + " cat" * 19999]))
    b = budget_parts(probe["response"].get("error", {}).get("message"))
    out, checks = {"model": args.model, "decide_seqs": seqs, "budget": b}, {}
    checks["probes: the image alone 200, a long text 400 budget with the cell counts"] = alone["status"] == 200 and b is not None
    if not all(checks.values()):
        write(args.out, {**out, "alone": alone, "probe": probe, "checks": checks, "pass": False})
    cells = alone["response"]["results"][0]["usage"]["image_cells"]
    n = largest_state(b) - cells // 2   # the text alone fits, with the image it is over the limit (more fields only add cells)
    over = [[text_part(sized_text(client, args.model, n)), image]]
    before = info(client, args.model)   # after the last 200: only rejected requests follow
    cases = {"sequences only": (many, ["A short text."]), "cells only": (ONE_FIELD, over), "both": (many, over)}
    for name, (schema, states) in cases.items():
        r = post(client, "/v1/decide", body_for(args.model, schema, states))
        out[name] = {"status": r["status"], "error": r["response"].get("error", {})}
    msg = {name: out[name]["error"].get("message", "") for name in cases}
    budget = {name: out[name]["status"] == 400 and out[name]["error"].get("code") == "budget" for name in cases}
    checks["sequences only: 400 budget naming the sequences"] = budget["sequences only"] and SEQ_TEXT in msg["sequences only"]
    checks["cells only: 400 budget naming the image cells (the early check after the image)"] = (
        budget["cells only"] and "(image cells " in msg["cells only"])
    checks["both: the sequence budget answers first"] = budget["both"] and msg["both"] == msg["sequences only"]
    after = info(client, args.model)
    checks["info unchanged by the rejected requests"] = after == before
    if after != before:
        out["info_before"], out["info_after"] = before, after
    write(args.out, {**out, "checks": checks, "pass": all(checks.values())})


def cmd_vision_reason(args):
    """prints vision.reason of llama-decide --info with the preset's projector ("available" when vision is available)"""
    cmd = rc.decide_command(args.model, args.preset, int(_preset_settings(args.preset)["decide-seqs"]), dump=False) + ["--info"]
    proc = subprocess.run(cmd, input=json.dumps(body_for(args.preset, ONE_FIELD, ["cat"])) + "\n", capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        sys.exit(f"llama-decide --info exit {proc.returncode}: {proc.stderr[-1500:]}")
    v = json.loads(lines[-1]).get("vision") or {}
    print("available" if v.get("available") else v.get("reason", "unknown"))


def answer_entries(results, fields):
    """sp3_metrics.compare entries (state, field, {option: probability}) of a response's results."""
    out = []
    for i, r in enumerate(results):
        for f in fields:
            a = r["answers"][f]
            if "p_true" in a:
                probs = {"true": a["p_true"], "false": 1.0 - a["p_true"]}
            elif isinstance(a["probabilities"], dict):
                probs = dict(a["probabilities"])
            else:
                probs = {str(k): p for k, p in enumerate(a["probabilities"])}
            out.append((i, f, probs))
    return out


def answers_match(ref, results, fields):
    """sp3_metrics.compare (the statistic) of `results` against `ref`, state by state; ref is the reference for the margin"""
    return compare(answer_entries(ref, fields), answer_entries(results, fields))


def load_states(path):
    return [state_from_spec(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def cmd_batch(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    schema = json.loads(Path(args.schema).read_text())
    fields = list(schema["fields"])
    states = load_states(args.states)
    texts = [json.loads(l)["text"] for l in Path(args.test_set).read_text().splitlines() if l.strip()][: len(states)]

    def results(batch):
        r = post(client, "/v1/decide", body_for(args.model, schema, batch))
        if r["status"] != 200:
            sys.exit(f"/v1/decide: {r['status']} {json.dumps(r['response'])[:500]}")
        return r["response"]["results"]

    alone = [results([s])[0] for s in states]
    together = results(states)
    mixed = results([x for pair in zip(states, texts) for x in pair])[0::2]   # image state, ticket, image state, …
    ref = answer_entries(alone, fields)
    out = {"model": args.model, "states": len(states), "together": compare(ref, answer_entries(together, fields)),
           "mixed": compare(ref, answer_entries(mixed, fields))}
    out["pass"] = out["together"]["pass"] and out["mixed"]["pass"]
    write(args.out, out)


def cli_response(cmd, body, fault=None):
    env = {k: v for k, v in os.environ.items() if k != "LLAMA_DECIDE_FAULT"}
    if fault:
        env["LLAMA_DECIDE_FAULT"] = fault
    proc = subprocess.run(cmd, input=json.dumps(body) + "\n", capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit(f"llama-decide (fault {fault}) exit {proc.returncode}: {proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cmd_retry(args):
    schema = json.loads(Path(args.schema).read_text())
    cmd = rc.decide_command(args.model, args.preset, int(_preset_settings(args.preset)["decide-seqs"]), dump=False)
    body = body_for(args.preset, schema, load_states(args.states))
    clean, fault = cli_response(cmd, body), cli_response(cmd, body, "rc1")
    stat = compare(answer_entries(clean["results"], list(schema["fields"])), answer_entries(fault["results"], list(schema["fields"])))
    checks = {"run without the fault: no retry": clean["timings"]["retries"] == 0,
              "rc1: at least one retry": fault["timings"]["retries"] >= 1,
              "rc1: the statistic against the run without it": stat["pass"],
              "rc1: same image usage": [r["usage"] for r in clean["results"]] == [r["usage"] for r in fault["results"]]}
    write(args.out, {"command": cmd, "timings": {"clean": clean["timings"], "rc1": fault["timings"]}, "statistic": stat,
                     "checks": checks, "pass": all(checks.values())})


def budget_parts(message):
    """the numbers of a 400 budget text: K 1 without K_eff; swa_cells None when n_ctx is the bound"""
    m = BUDGET.search(message or "")
    if not m:
        return None
    needs, held, prefix, k, state, tail, branches, ctx_limit, swa_cells, swa_limit = m.groups()
    return {"needs": int(needs), "held": int(held), "prefix": int(prefix), "K": int(k or 1), "state": int(state), "tail": int(tail),
            "branches": int(branches), "limit": int(ctx_limit or swa_limit), "swa_cells": int(swa_cells) if swa_cells else None}


def largest_state(b):
    """the largest state the budget b admits: held + prefix + K (state + tail + branches) <= limit"""
    return (b["limit"] - b["held"] - b["prefix"]) // b["K"] - b["tail"] - b["branches"]


def held_budget_matches(b, old):
    """held --old: the same prefix, tail and branches as the old build, and its limit or, on an iSWA model whose SWA cache is
    the bound (spec 5.3 rev 5; the old build ignored it), the SWA cache's"""
    return all(b[k] == old[k] for k in ("prefix", "tail", "branches")) and (
        b["limit"] == old["limit"] or (b["swa_cells"] is not None and b["limit"] == b["swa_cells"] - 16 < old["limit"]))


def sized_text(client, model, n, cycle=("cat",)):
    """A text of exactly n tokens ("cat cat …", or the words of `cycle` in turn), checked with the server's /tokenize."""
    words = n
    for _ in range(20):
        text = " ".join(cycle[i % len(cycle)] for i in range(words))
        r = client.post("/tokenize", json={"model": model, "content": text})
        r.raise_for_status()
        got = len(r.json()["tokens"])
        if got == n:
            return text
        words += n - got
    sys.exit(f"could not size a text to {n} tokens")


def sized_answer_ok(sized):
    """spec 3.2, 5.3: a state within the budget answers 200; one over it 400 budget; a 500 is never the answer to cells"""
    return sized["status"] == 200 or (sized["status"] == 400 and sized["response"].get("error", {}).get("code") == "budget")


def cmd_held(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    out = {"model": args.model, "build": "new" if args.old else "old"}
    chat = client.post("/v1/chat/completions", json={
        "model": args.model, "max_tokens": 1, "temperature": 0, "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [text_part("Describe this picture in one word."), image_part(args.image)]}]})
    out["chat"] = {"status": chat.status_code, "usage": chat.json().get("usage")}
    out["probe"] = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, ["cat" + " cat" * 19999]))
    b = budget_parts(out["probe"]["response"].get("error", {}).get("message"))
    out["budget"] = b
    checks = {"chat request: 200": chat.status_code == 200, "probe: 400 budget with the cell counts": b is not None}
    if not all(checks.values()):
        write(args.out, {**out, "checks": checks, "pass": False})
    if args.old:
        old = json.loads(Path(args.old).read_text())
        out["old_held"], size = old["budget"]["held"], old["size"]
        checks["same prefix, tail, branches and limit as the old build (or the SWA cache's limit)"] = held_budget_matches(b, old["budget"])
    else:
        size = b["limit"] - b["held"] - b["prefix"] - b["tail"] - b["branches"]   # fits exactly under this held count
    out["size"] = size
    out["sized"] = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [sized_text(client, args.model, size)]))
    if args.old:
        out["applicable"] = b["held"] > out["old_held"]
        if out["applicable"]:
            e = out["sized"]["response"].get("error", {})
            s = budget_parts(e.get("message")) or {}
            checks["new build: 400 budget for the state that fits under the old count"] = (
                out["sized"]["status"] == 400 and e.get("code") == "budget" and s.get("state") == size and s.get("held") == b["held"])
        else:
            out["note"] = "not applicable: the slot's cells equal its positions, so both counts agree"
            checks["not applicable: the sized state answers 200 or 400 budget, never 500"] = sized_answer_ok(out["sized"])
    write(args.out, {**out, "checks": checks, "pass": all(checks.values())})


def swa_body(model, states, debias):
    body = body_for(model, SWA_SCHEMA, states)
    if debias > 1:
        body["options"] = {"order_debias": debias}
    return body


def round_split_gate(multi, pair, last_alone, fields):
    """swa --round: each round of the split answers as that composition does on its own: the multi-round request's first two
    results equal the same two states sent as one request, its third result equals that state sent alone (max abs diff 0)"""
    a = answers_match(pair, multi[:2], fields)["max_abs_diff"]
    c = answers_match(last_alone, multi[2:], fields)["max_abs_diff"]
    return {"first_two_vs_pair_max_abs_diff": a, "third_vs_alone_max_abs_diff": c, "pass": a == 0 and c == 0}


def round_cases(client, model, debias, b, n_ctx, checks):
    """swa --round: per case the sizes, the cells of the first two and of all three states, the three-state request, the first
    two states as one request and each text alone (full results); adds the case's checks to `checks`"""
    fields = list(SWA_SCHEMA["fields"])
    two = (b["limit"] - b["held"] - b["prefix"]) // b["K"] - 2 * (b["tail"] + b["branches"])   # text tokens of two states at the limit

    def need(sizes):
        return b["held"] + b["prefix"] + b["K"] * sum(n + b["tail"] + b["branches"] for n in sizes)

    def ask(texts):
        r = post(client, "/v1/decide", swa_body(model, texts, debias))
        t = r["response"].get("timings") or {}
        return {"status": r["status"], "error": r["response"].get("error"), "results": r["response"].get("results"),
                "rounds": t.get("rounds"), "retries": t.get("retries")}

    first = two // 2 + 100
    cases = {"same": ([two // 2] * 3, [ROUND_WORDS[0]] * 3), "distinct": ([first, two - first, 3000], list(ROUND_WORDS))}
    out = {}
    for name, (sizes, cycles) in cases.items():
        texts = [sized_text(client, model, n, c) for n, c in zip(sizes, cycles)]
        multi = ask(texts)
        pair = ask(texts[:2])
        alone = {t: ask([t]) for t in dict.fromkeys(texts)}
        case = {"state_tokens": sizes, "need_2": need(sizes[:2]), "need_3": need(sizes), "limit": b["limit"], "n_ctx_limit": n_ctx - 16,
                "multi": multi, "pair": pair, "alone": [alone[t] for t in dict.fromkeys(texts)]}
        checks[f"round {name}: the first two states fill the limit, all three are over it but within n_ctx - 16"] = (
            b["limit"] - 2 * b["K"] < case["need_2"] <= b["limit"] < case["need_3"] <= n_ctx - 16)
        checks[f"round {name}: 200 in two rounds without a retry"] = multi["status"] == 200 and multi["rounds"] == 2 and multi["retries"] == 0
        checks[f"round {name}: the pair and each text alone answer 200"] = (
            pair["status"] == 200 and all(a["status"] == 200 for a in alone.values()))
        if multi["status"] == 200 and pair["status"] == 200 and all(a["status"] == 200 for a in alone.values()):
            case["gate"] = round_split_gate(multi["results"], pair["results"], [alone[texts[2]]["results"][0]], fields)
            checks[f"round {name}: round 1 equals the pair request and round 2 the third state alone (max abs diff 0)"] = case["gate"]["pass"]
            # recorded data, not a gate: a state in a two-state batch answers differently from the same state alone on this
            # model, also with --swa-full and on the old build (swa/control/), so the comparison documents that gap
            case["vs_alone"] = answers_match([alone[t]["results"][0] for t in texts], multi["results"], fields)
            case["pair_vs_alone"] = answers_match([alone[t]["results"][0] for t in texts[:2]], pair["results"], fields)
        out[name] = case
    return out


def cmd_swa(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    out = {"model": args.model, "debias": args.debias, "chat": args.chat}
    checks = {}
    if args.chat:
        chat = client.post("/v1/chat/completions", json={
            "model": args.model, "max_tokens": 1, "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "Describe a cat in one word."}]})
        out["chat_answer"] = {"status": chat.status_code, "usage": chat.json().get("usage")}
        checks["chat request: 200"] = chat.status_code == 200
    i = info(client, args.model)
    n_ctx, swa = i["n_ctx"], i["memory"].get("swa_cells")
    out["info"] = {k: i.get(k) for k in ("n_ctx", "n_batch", "decide_seqs", "n_parallel", "memory")}
    swa_bound = swa is not None and swa < n_ctx
    if args.expect_swa_cells:
        checks[f"info: memory.swa_cells {args.expect_swa_cells}"] = swa == args.expect_swa_cells
    probe = post(client, "/v1/decide", swa_body(args.model, ["cat" + " cat" * 19999], args.debias))
    b = budget_parts((probe["response"].get("error") or {}).get("message"))
    out["probe"] = {"status": probe["status"], "error": probe["response"].get("error"), "budget": b}
    checks["probe: 400 budget with the cell counts"] = probe["status"] == 400 and b is not None
    if b is None:
        write(args.out, {**out, "checks": checks, "pass": False})
    checks[f"probe: K_eff {args.debias}"] = b["K"] == args.debias
    checks["probe: limit min(n_ctx, swa_cells) - 16"] = b["limit"] == min(n_ctx, swa or n_ctx) - 16
    checks["probe: names the SWA cache (and --swa-full) exactly when it is the bound"] = (
        b["swa_cells"] == swa and "--swa-full" in probe["response"]["error"]["message"] if swa_bound else b["swa_cells"] is None)
    if args.chat:
        checks["probe: held > 0 (the chat's cells)"] = b["held"] > 0
    m = largest_state(b)
    out["largest_admitted"] = m
    sweep = []
    for d in SWEEP_OFFSETS:
        n = m + d
        r = post(client, "/v1/decide", swa_body(args.model, [sized_text(client, args.model, n)], args.debias))
        e = r["response"].get("error") or {}
        sweep.append({"state_tokens": n, "status": r["status"], "error": e or None, "budget": budget_parts(e.get("message")),
                      "usage": r["response"].get("usage")})
    out["sweep"] = sweep
    checks["sweep: every state up to the largest admitted answers 200 with exactly its tokens"] = all(
        s["status"] == 200 and s["usage"]["state_tokens"] == s["state_tokens"] for s in sweep if s["state_tokens"] <= m)
    checks["sweep: every state over it answers 400 budget with the probe's held, limit and bound"] = all(
        s["status"] == 400 and s["error"].get("code") == "budget" and s["budget"] is not None and s["budget"]["state"] == s["state_tokens"]
        and all(s["budget"][k] == b[k] for k in ("held", "limit", "swa_cells")) for s in sweep if s["state_tokens"] > m)
    checks["sweep: no 500"] = all(s["status"] != 500 for s in sweep)
    if args.round:
        out["round"] = round_cases(client, args.model, args.debias, b, n_ctx, checks)
    write(args.out, {**out, "checks": checks, "pass": all(checks.values())})


def cmd_swa_info(args):
    settings = _preset_settings(args.preset)
    cmd = rc.decide_command(args.model, args.preset, int(settings["decide-seqs"]), dump=False) + ["--info", "-v"]
    if settings.get("swa-full", "0").lower() in ("1", "true", "on", "enabled"):
        cmd.append("--swa-full")
    proc = subprocess.run(cmd, input=json.dumps(body_for(args.preset, SWA_SCHEMA, ["cat"])) + "\n", capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    i = json.loads(lines[-1]) if lines else {}
    logged = [int(x) for x in re.findall(r"creating\s+SWA KV cache, size = (\d+) cells", proc.stderr)]
    swa = (i.get("memory") or {}).get("swa_cells")
    checks = {"llama-decide --info: exit 0": proc.returncode == 0,
              "info: memory.swa_cells equals the SWA KV cache size in the log": swa is not None and logged[-1:] == [swa]}
    if args.expect_swa_cells:
        checks[f"info: memory.swa_cells {args.expect_swa_cells}"] = swa == args.expect_swa_cells
    out = {"command": cmd, "returncode": proc.returncode, "n_ctx": i.get("n_ctx"), "memory": i.get("memory"), "logged_swa_cells": logged,
           "kv_lines": [l for l in proc.stderr.splitlines() if "KV cache" in l or "n_seq_max" in l or "n_ubatch" in l or "n_swa " in l][:20]}
    if args.prefix_words:
        schema = dict(SWA_SCHEMA, instructions=" ".join(["cat"] * args.prefix_words))
        p = subprocess.run([a for a in cmd if a != "-v"], input=json.dumps(body_for(args.preset, schema, ["cat"])) + "\n",
                           capture_output=True, text=True)
        e = (last_json(p.stderr) or {}).get("error") or {}
        m = PREFIX_BUDGET.search(e.get("message", ""))
        n_ctx = i.get("n_ctx") or 0
        out["prefix"] = {"words": args.prefix_words, "returncode": p.returncode, "error": e or None,
                         "info_prefix_tokens": ((last_json(p.stdout) or {}).get("prefix") or {}).get("tokens")}
        if swa is not None and swa < n_ctx:
            checks["prefix over the SWA limit, within n_ctx - 16: 400 budget naming the SWA cache"] = (
                p.returncode == 1 and e.get("code") == "budget" and m is not None and m.group(4) is not None
                and int(m.group(4)) == swa and int(m.group(5)) == swa - 16 < int(m.group(1)) <= n_ctx - 16)
        else:
            checks["prefix: info answers (n_ctx is the bound)"] = p.returncode == 0 and out["prefix"]["info_prefix_tokens"] is not None
    write(args.out, {**out, "checks": checks, "pass": all(checks.values())})


def cmd_ubatch(args):
    cmd = rc.decide_command(args.model, args.preset, int(_preset_settings(args.preset)["decide-seqs"]), dump=False) + ["-ub", "128"]
    proc = subprocess.run(cmd, input=json.dumps(body_for(args.preset, ONE_FIELD, [[image_part(args.image)]])) + "\n",
                          capture_output=True, text=True)
    e = (last_json(proc.stderr) or {}).get("error", {})
    write(args.out, {"command": cmd, "returncode": proc.returncode, "error": e,
                     "pass": proc.returncode == 1 and e.get("code") == "budget" and "n_ubatch is 128" in e.get("message", "")
                     and "--ubatch-size" in e.get("message", "")})


def cmd_gate(args):
    d = Path(args.dir)
    files = {t: d / f"reference-{t}-{args.preset}.json" for t in REF_TAGS}
    files["batch"] = d / "batch.json"
    checks = {k: f.exists() and json.loads(f.read_text())["pass"] is True for k, f in files.items()}
    skipped = d / "SKIPPED.txt"
    checks["the model ran (no SKIPPED.txt)"] = not skipped.exists()
    write(args.out, {"preset": args.preset, "dir": str(d), "skipped": skipped.read_text() if skipped.exists() else None,
                     "checks": checks, "pass": all(checks.values())})


def cmd_gate_off(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    v = info(client, args.model).get("vision")
    r = post(client, "/v1/decide", body_for(args.model, ONE_FIELD, [[image_part(args.image)]]))
    e = r["response"].get("error", {})
    checks = {"info: available false, reason non_causal_projector": v == {"available": False, "reason": "non_causal_projector"},
              "image request: 400 vision_unavailable at states[0], naming the reason": r["status"] == 400
              and e.get("code") == "vision_unavailable" and e.get("field") == "states[0]" and "non_causal_projector" in e.get("message", "")}
    write(args.out, {"model": args.model, "vision": v, "response": r, "checks": checks, "pass": all(checks.values())})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("text-run")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-set", type=Path, default=TEST_SET)
    p.add_argument("--debias", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--server-bin", default=None, help="the llama-server the router was started with (recorded in the sidecar)")
    p.set_defaults(func=cmd_text_run)
    p = sub.add_parser("text-compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_text_compare)
    p = sub.add_parser("no-vision")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None, help="the router's log (required without --skip-rejected)")
    p.add_argument("--skip-rejected", action="store_true")
    p.add_argument("--test-set", type=Path, default=TEST_SET)
    p.set_defaults(func=cmd_no_vision)
    p = sub.add_parser("no-vision-compare")
    p.add_argument("with_rejected", type=Path)
    p.add_argument("without", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_no_vision_compare)
    p = sub.add_parser("cli-no-vision")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cli_no_vision)
    for name, func, url in (("malformed", cmd_malformed, True), ("gate-off", cmd_gate_off, True), ("limits", cmd_limits, True),
                            ("double-fault", cmd_double_fault, True), ("ubatch", cmd_ubatch, False)):
        p = sub.add_parser(name)
        if url:
            p.add_argument("--url", required=True)
            p.add_argument("--model", required=True)
        else:
            p.add_argument("--model", required=True, help="the GGUF")
            p.add_argument("--preset", required=True)
        p.add_argument("--image", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "malformed":
            p.add_argument("--skip-cases", action="store_true")
        p.set_defaults(func=func)
    p = sub.add_parser("malformed-compare")
    p.add_argument("with_cases", type=Path)
    p.add_argument("without", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_malformed_compare)
    p = sub.add_parser("batch")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--test-set", type=Path, default=TEST_SET)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_batch)
    p = sub.add_parser("retry")
    p.add_argument("--model", required=True, help="the GGUF")
    p.add_argument("--preset", required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_retry)
    p = sub.add_parser("held")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--old", type=Path, default=None, help="the old build's held-old.json (makes this the new build's run)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_held)
    p = sub.add_parser("swa")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--debias", type=int, default=1)
    p.add_argument("--chat", action="store_true", help="a text chat resident in the slot first")
    p.add_argument("--round", action="store_true", help="also the three-state round")
    p.add_argument("--expect-swa-cells", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_swa)
    p = sub.add_parser("swa-info")
    p.add_argument("--model", required=True, help="the GGUF")
    p.add_argument("--preset", required=True)
    p.add_argument("--expect-swa-cells", type=int, default=None)
    p.add_argument("--prefix-words", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_swa_info)
    p = sub.add_parser("gate")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_gate)
    p = sub.add_parser("vision-reason")
    p.add_argument("--model", required=True, help="the GGUF")
    p.add_argument("--preset", required=True)
    p.set_defaults(func=cmd_vision_reason)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
