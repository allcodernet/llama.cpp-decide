"""Answer the triage questions for the test tickets through our engine's /v1/decide.

Usage: uv run decide_client.py --model <preset> --batch <N> [--url URL] [--test-set PATH] [--limit K] [--accept-prefix]
                               [--prompt-variant NAME] [--out-dir DIR] [--force]
                               [--scoring tree|full_path] [--order-debias K] [--after FIELD=PARENT[,PARENT]]...
                               [--temperature T | --ensure-t1]

Writes <out-dir>/preds/decide-<preset>-b<N>.jsonl (score.py format; `raw` keeps the whole answers object, including
coverage, order_spread and given when the engine returns them) and
<out-dir>/runs/decide-<preset>-b<N>.json (setup and latency summary); --out-dir defaults to
results-engine, a relative path is taken from the project directory. If either file exists, the run stops after the
warm-up without writing anything; --force overwrites them.
--prompt-variant NAME picks instructions + description mode from prompt_variants.py; the variants except `default`
get the suffix -<NAME>. Without the flag the variant is DEFAULT_VARIANT (`keywords`, chosen by the held-out check in
REPORT-ENGINE.md); --prompt-variant default is the request of all runs before that check.
Sub-project 3 switches (spec 2026-09-25-engine-improvements-design.md, section 3.1): --scoring sets options.scoring,
--order-debias options.order_debias, --temperature options.temperature (absent: the server default,
--decide-temperature), --after FIELD=P the field's `after` list (repeatable, one per field; names are queue, urgency,
angry). Without them the request is today's, byte for byte. Suffixes after the variant's, in this order: -full_path,
-after-<field>=<parent>[+<parent>]... (fields in declaration order), -debias<K> for the switches that change the request,
then -t<T> when the temperature the engine applied (engine.temperature of the warm-up answer: --temperature, else the
server default) is not 1.0. So a run on a preset with decide-temperature is never named like a T = 1 run, and
--temperature 1 gets the T = 1 name. --ensure-t1 runs at T = 1 whatever the server default: it sends
options.temperature = 1 only when /v1/decide/info reports another default (otherwise the request is today's, byte for
byte) and stops after the warm-up, without writing anything, if the engine still applied another temperature.
--test-set defaults to TRIAGE_DATA's triage_test.jsonl (score.py). The run's preset_settings, model_file and decide_seqs
are read from the presets file: ENGINE_PRESETS, default models-engine.ini (relative: from the project directory).
States may be strings or part lists (sub-project 4, spec 3.1): image_part(path) turns a JPEG or PNG file into a data-URI part,
text_part(text) makes a text part, and state_from_spec(spec) builds a state from its file form (a string, or a list of
{"text": T} / {"image": PATH} items; PATH relative to the project directory).
"""

import argparse
import base64
import configparser
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from prompt_variants import VARIANTS
from score import TRIAGE_DATA

ROOT = Path(__file__).parent
TEST_SET = TRIAGE_DATA / "triage_test.jsonl"  # TRIAGE_DATA: score.py
DEFAULT_VARIANT = "keywords"  # REPORT-ENGINE.md, "Held-out check (320 train tickets)"

# Origin: the earlier triage experiment's task definition (data/README.md).
QUEUES = {
    "billing": "Payments, refunds, invoices, charges",
    "technical": "Bugs, crashes, outages, errors, performance, login problems",
    "sales": "Contracts, plan changes, pricing questions, cancellation threats, procurement",
    "feedback": "Praise, feature requests, general non-problem questions",
}
URGENCY = [
    "Can wait: praise, feature requests, idle questions, no problem",
    "Normal: a real problem for one user without time pressure",
    "Should be handled today: user is blocked, money wrongly taken, a deadline within days, or repeated contact",
    "Drop everything: outage affecting many users, security incident, data loss, or enterprise customer about to churn",
]
QUEUE_INSTRUCTIONS = "Which team must act first on this support ticket?"
URGENCY_INSTRUCTIONS = "How urgent is this support ticket?"
ANGRY_INSTRUCTIONS = "The text itself shows anger (hostile wording, shouting, threats, sarcasm aimed at the company)."
FIELDS = ("queue", "urgency", "angry")  # build_request's field names, in declaration order

# Tail of the rendered chat prompt (thinking disabled) that must precede the opening brace.
# The engine's assistant_prefix also carries the end-of-user-turn text before this tail
# (e.g. "<|im_end|>\n..."), so the client compares with endswith, not equality.
EXPECTED_PREFIX = {
    "qwen3.5-9b": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma-4-e4b": "<|turn>model\n",
    "qwen2.5-0.5b": "<|im_start|>assistant\n",
    "qwen3.5-4b": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    # the 26B-A4B template closes an empty thought channel when thinking is off
    "gemma-4-26b-a4b": "<|turn>model\n<|channel>thought\n<channel|>",
    "qwen3.5-35b-a3b": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "qwen3-30b-a3b": "<|im_start|>assistant\n",  # Instruct-2507: no thinking block (observed)
    "qwen3.8-27b": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
}


def build_request(preset, contexts, variant="default", *, scoring="tree", order_debias=0, after=None, temperature=None):
    fields = {
        "queue": {"type": "choice", "description": QUEUE_INSTRUCTIONS, "options": dict(QUEUES)},
        "urgency": {"type": "score", "description": URGENCY_INSTRUCTIONS, "levels": list(URGENCY)},
        "angry": {"type": "bool", "description": ANGRY_INSTRUCTIONS},
    }
    if VARIANTS[variant]["description_mode"] == "short":
        fields = {
            "queue": {"type": "choice", "description": "Team.", "options": list(QUEUES)},
            "urgency": {"type": "score", "description": "Urgency.", "levels": ["Wait", "Normal", "Today", "Immediately"]},
            "angry": {"type": "bool", "description": "Anger."},
        }
    for name, parents in (after or {}).items():
        fields[name]["after"] = list(parents)
    options = {"scoring": scoring, "order_debias": order_debias}
    if temperature is not None:
        options["temperature"] = temperature
    return {
        "model": preset,
        "instructions": VARIANTS[variant]["instructions"],
        "fields": fields,
        "states": list(contexts),
        "options": options,
    }


IMAGE_MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"))


def image_part(path):
    """A state part for an image file (spec 3.1): data:image/<jpeg|png>;base64,<data>, the subtype from the magic bytes; exits
    on a file that is neither JPEG nor PNG."""
    data = Path(path).read_bytes()
    for magic, subtype in IMAGE_MAGIC:
        if data.startswith(magic):
            return {"type": "image_url", "image_url": {"url": f"data:image/{subtype};base64," + base64.b64encode(data).decode()}}
    sys.exit(f"{path}: not a JPEG or PNG file")


def text_part(text):
    """A text part of a state (spec 3.1)."""
    return {"type": "text", "text": text}


def state_from_spec(spec):
    """A state from its file form: a string stays a string; a list of {"text": T} and {"image": PATH} items becomes the part
    list (PATH relative: from the project directory)."""
    if isinstance(spec, str):
        return spec
    parts = []
    for item in spec:
        if isinstance(item, dict) and set(item) == {"text"}:
            parts.append(text_part(item["text"]))
        elif isinstance(item, dict) and set(item) == {"image"}:
            parts.append(image_part(ROOT / item["image"]))
        else:
            sys.exit(f"state item {item!r}: expected {{'text': TEXT}} or {{'image': PATH}}")
    return parts


def variant_suffix(variant):
    """Output-name suffix: none for default, else -<variant>."""
    return "" if variant == "default" else f"-{variant}"


def parse_after(specs, fields=FIELDS):
    """--after values FIELD=PARENT[,PARENT] -> {FIELD: [PARENT, ...]}. Exits on a malformed value, a name that is not
    one of `fields` (default: this client's FIELDS) or a FIELD given twice; the other `after` rules (self, cycles) are
    the engine's 400."""
    after = {}
    for spec in specs:
        field, _, rest = spec.partition("=")
        parents = rest.split(",")
        if not field or "" in parents:
            sys.exit(f"--after {spec!r}: expected FIELD=PARENT[,PARENT]")
        unknown = [n for n in [field, *parents] if n not in fields]
        if unknown:
            sys.exit(f"--after {spec!r}: unknown field {unknown[0]!r} (fields: {', '.join(fields)})")
        if field in after:
            sys.exit(f"--after {field} given twice")
        after[field] = parents
    return after


def switch_suffix(scoring="tree", order_debias=0, after=None, temperature=1.0):
    """Output-name suffix of the sub-project 3 switches: empty for today's request at T = 1, else one part per switch
    that changes it and -t<T> for an effective temperature other than 1.0 (see the module docstring)."""
    parts = []
    if scoring != "tree":
        parts.append(f"-{scoring}")
    if after:
        parts.append("-after" + "".join(f"-{f}={'+'.join(after[f])}" for f in sorted(after, key=FIELDS.index)))
    if order_debias != 0:
        parts.append(f"-debias{order_debias}")
    if temperature != 1.0:
        parts.append(f"-t{float(temperature)!r}")
    return "".join(parts)


def to_pred(answers, latency_s, server_ms):
    return {
        "queue": answers["queue"]["probabilities"],
        "urgency": list(answers["urgency"]["probabilities"]),
        "angry": float(answers["angry"]["p_true"]),
        "latency_s": latency_s,
        "server_ms": server_ms,
        "distribution": "full",
        "raw": answers,
    }


def check_prefix(assistant_prefix, preset):
    tail = assistant_prefix[-120:]
    expected = EXPECTED_PREFIX.get(preset)
    return (expected is not None and assistant_prefix.endswith(expected)), tail


def chunks(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def presets_file():
    """The server's presets file: ENGINE_PRESETS, default models-engine.ini (relative: from the project directory)."""
    return ROOT / (os.environ.get("ENGINE_PRESETS") or "models-engine.ini")


def _preset_settings(preset):
    """The [*] defaults overlaid with the preset's own section of the presets file (presets_file)."""
    ini = configparser.ConfigParser()
    ini.read(presets_file())
    settings = {}
    for section in ("*", preset):
        if ini.has_section(section):
            settings.update(ini.items(section))
    return settings


def _vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def _fork_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT / "engine"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _decide_info(client, model):
    r = client.get("/v1/decide/info", params={"model": model})
    if r.status_code != 200:
        sys.exit(f"/v1/decide/info failed: {r.status_code} {r.text[:500]}")
    return r.json()


def _percentiles(values):
    s = sorted(values)
    return {"p50": s[len(s) // 2], "p90": statistics.quantiles(values, n=10)[8]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--accept-prefix", action="store_true")
    ap.add_argument("--prompt-variant", choices=list(VARIANTS), default=None,
                    help=f"default {DEFAULT_VARIANT}; `default` is the request of the runs before the held-out check")
    ap.add_argument("--out-dir", type=Path, default=Path("results-engine"),
                    help="preds/ and runs/ are written under this directory (relative: from the project directory)")
    ap.add_argument("--scoring", choices=["tree", "full_path"], default="tree", help="options.scoring")
    ap.add_argument("--order-debias", type=int, default=0, metavar="K", help="options.order_debias, 0-8 (0 and 1: off)")
    ap.add_argument("--after", action="append", default=[], metavar="FIELD=PARENT[,PARENT]",
                    help="the field's `after` list (repeatable, one per field)")
    ap.add_argument("--force", action="store_true", help="overwrite existing preds/runs files of the same name")
    t = ap.add_mutually_exclusive_group()
    t.add_argument("--temperature", type=float, default=None, metavar="T",
                   help="options.temperature, 0.05-20 (absent: the server default, --decide-temperature)")
    t.add_argument("--ensure-t1", action="store_true",
                   help="run at T = 1: options.temperature = 1 only if the server default is not 1.0; stop if the "
                        "engine applies another temperature")
    args = ap.parse_args(argv)
    args.prompt_variant = args.prompt_variant or DEFAULT_VARIANT
    args.after = parse_after(args.after)
    return args


def request_body(args, contexts):
    """The /v1/decide body for these states with the run's prompt variant and switches."""
    return build_request(args.model, contexts, args.prompt_variant, scoring=args.scoring,
                         order_debias=args.order_debias, after=args.after, temperature=args.temperature)


def run_name(args, *, temperature):
    """The preds/runs file name (without extension): decide-<preset>-b<N>, then the variant's and the switches' suffixes;
    `temperature` is the one the engine applied (engine.temperature), -t<T> unless it is 1.0."""
    return (f"decide-{args.model}-b{args.batch}" + variant_suffix(args.prompt_variant)
            + switch_suffix(args.scoring, args.order_debias, args.after, temperature))


def main(argv=None):
    args = parse_args(argv)

    tickets = [json.loads(line) for line in args.test_set.read_text().splitlines() if line.strip()]
    if args.limit:
        tickets = tickets[: args.limit]
    client = httpx.Client(base_url=args.url, timeout=600)

    # 1. Prefix check via /v1/decide/info (works before any decision: the engine renders
    # the template-only assistant prefix as soon as the model is loaded).
    info = _decide_info(client, args.model)
    ok, tail = check_prefix(info["prefix"]["assistant_prefix"], args.model)
    print(f"assistant_prefix tail: {tail!r}")
    if not ok and not args.accept_prefix:
        sys.exit(f"prefix mismatch for {args.model}; expected tail {EXPECTED_PREFIX.get(args.model)!r} (use --accept-prefix to proceed)")
    if args.ensure_t1 and info["temperature"] != 1.0:
        args.temperature = 1.0  # only here: with a server default of 1.0 the request stays today's, byte for byte
        print(f"server default temperature {info['temperature']}: sending options.temperature 1", file=sys.stderr)

    # 2. Warmup. The temperature the engine applied (options.temperature, else the server default) names the outputs.
    warm = client.post("/v1/decide", json=request_body(args, [tickets[0]["text"]]))
    if warm.status_code != 200:
        sys.exit(f"warmup failed: {warm.status_code} {warm.text[:500]}")
    temperature = warm.json()["engine"]["temperature"]
    if args.ensure_t1 and temperature != 1.0:
        sys.exit(f"--ensure-t1: the engine applied temperature {temperature}, not 1; nothing written")
    name = run_name(args, temperature=temperature)
    preds_path = ROOT / args.out_dir / "preds" / f"{name}.jsonl"
    run_path = ROOT / args.out_dir / "runs" / f"{name}.json"
    existing = [str(p) for p in (preds_path, run_path) if p.exists()]
    if existing and not args.force:
        sys.exit(f"refusing to overwrite {' and '.join(existing)} (--force overwrites); nothing written")
    # Re-fetch after the warm-up: the cached prefix in the info is now this run's, not the previous invocation's.
    info = _decide_info(client, args.model)

    # 3. Timed runs.
    preds, requests_meta, engine_block = [], [], None
    for batch in chunks(tickets, args.batch):
        body = request_body(args, [t["text"] for t in batch])
        t0 = time.perf_counter()
        resp = client.post("/v1/decide", json=body)
        wall = time.perf_counter() - t0
        if resp.status_code != 200:
            sys.exit(f"decide failed: {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        if len(data["results"]) != len(batch):
            sys.exit(f"expected {len(batch)} results, got {len(data['results'])}")
        per_ticket_wall = wall / len(batch)
        per_ticket_server = data["timings"]["total_ms"] / len(batch)
        for res in data["results"]:
            try:
                pred = to_pred(res["answers"], per_ticket_wall, per_ticket_server)
            except KeyError as e:
                sys.exit(f"missing field {e} in response: {json.dumps(res)[:800]}")
            preds.append(pred)
        engine_block = data["engine"]
        requests_meta.append({"n": len(batch), "wall_s": wall, "timings": data["timings"],
                              "usage": data["usage"], "engine": data["engine"]})
        print(f"{len(preds):3d}/{len(tickets)}  {wall * 1000:7.1f} ms  rounds={data['timings'].get('rounds')}", file=sys.stderr)

    # 4. Outputs.
    settings = _preset_settings(args.model)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.write_text("".join(json.dumps(p) + "\n" for p in preds))
    usage_sum = {}
    for m in requests_meta:
        for k, v in m["usage"].items():
            usage_sum[k] = usage_sum.get(k, 0) + v
    run = {
        "preset": args.model,
        "batch": args.batch,
        "prompt_variant": args.prompt_variant,
        "switches": {"scoring": args.scoring, "order_debias": args.order_debias, "after": args.after,
                     "temperature": args.temperature},
        "n_tickets": len(preds),
        "fork_commit": _fork_commit(),
        "preset_settings": settings,
        "model_file": settings.get("model"),
        "decide_seqs": int(settings["decide-seqs"]) if "decide-seqs" in settings else None,
        "vram_mib_after_run": _vram_mib(),
        "prefix_tail": tail,
        "prefix_ok": ok,
        "info": info,
        "engine": engine_block,
        "distribution": sorted({p["distribution"] for p in preds}),
        "latency_s_per_ticket": _percentiles([p["latency_s"] for p in preds]),
        "server_ms_per_ticket": _percentiles([p["server_ms"] for p in preds]),
        "rounds": sorted({m["timings"].get("rounds") for m in requests_meta}),
        "usage_sum": usage_sum,
        "requests": requests_meta,
    }
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(run, indent=2))
    print(f"wrote {preds_path} and {run_path}")


if __name__ == "__main__":
    main()
