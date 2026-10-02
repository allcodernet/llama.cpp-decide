"""Independent reference check for the engine's tree-scored and full-path distributions.

Two phases, because the 8 GB GPU cannot hold the model twice:

  uv run reference_check.py dump --model <gguf> --preset <name> --n 20 [--schema PATH --tag TAG]
                                 [--decide-seqs S] [--scoring tree|full_path] [--after FIELD=PARENT[,PARENT]]...
                                 [--batch B] [--out-dir DIR]
      (server STOPPED) runs `llama-decide --dump-tokens` on the first N test tickets
      (B states per request line, default 1; --decide-seqs S (default 9), same -c/-fa/cache-type
      as the server) and writes DIR/reference-[TAG-]<preset>-dump.jsonl (DIR default results-engine):
      one record per (ticket, field) with the engine's own answer. Tree records hold, per branch
      node of the field, the prompt token ids (prefix + state + tail + branch tokens), the
      children token ids and the option indices through each child. Without --schema
      the triage schema of decide_client.py is used; --schema takes the fields and
      instructions of a request body, e.g. tests/data/decide-multibranch.json (a choice
      field with three branch nodes on Qwen). --scoring full_path sets options.scoring (spec 4);
      its records hold prompt_prefix (prefix + state + tail) and the field's entries as dumped
      (option; tokens = stem + path; rows = the indices into tokens whose logits were read; logp per
      row: children log-probs keyed by token id, `end` on a leaf row, `merged` per leaf child; null =
      -inf). --after F=P[,P] (repeatable) sets field F's `after` (spec 5); every record then carries
      `ancestors` = {ancestor: its returned value} (transitive, declaration order; empty for level-0
      fields), and dependent nodes/entries keep their dumped `lines` and `given`. --tag is required
      with --scoring full_path, --after, --batch > 1, --decide-seqs other than 9 or --schema: without
      it the dump would overwrite the committed tree dump reference-<preset>-dump.jsonl.

  uv run reference_check.py compare --preset <name> --url http://127.0.0.1:8097 [--n N] [--tag TAG] [--out-dir DIR] [--offline]
      (server RUNNING with that preset) replays the dumped prompts through POST /completion
      (n_predict 1, n_probs 512, post_sampling_probs off, temperature 0, cache_prompt off) and
      writes DIR/reference-[TAG-]<preset>.json. Every request with its answer (the generated token,
      the listed ids and log-probs, the listed pieces; /detokenize bodies) is kept in
      DIR/reference-[TAG-]<preset>-replay.jsonl.gz; --offline answers from that file instead of a
      server, so a comparison can be recomputed without the model.
      Tree records: reads the children's log-probabilities from
      completion_probabilities[0].top_logprobs, renormalises over that node's children,
      multiplies the child probabilities along each option's path, and compares against the
      engine's distribution. Writes, per field: median absolute difference, max absolute
      difference, argmax agreement rate, the count of disagreements where the reference top-2
      margin exceeds 0.15, and argmax_abs_diff_median (abs diff at the reference's argmax
      option only -- arity-independent, reported but not gated). Pass (spec 8.3):
      median <= 0.01 and zero such disagreements.
      Full-path records (spec 12.3): every row of every option is replayed (prompt_prefix + the
      entry's tokens up to the row). The reference takes each path token's log-probability from
      the top-512 list and sums END (tokens whose piece satisfies ends(), spec 4.1) and MERGED
      (pieces = the leaf token's piece + r with ends(r)) over the listed tokens only (lower
      bounds), then computes log P(o), the distribution and the coverage as spec 4.1 does. A path
      token absent from the list is an unverifiable edge: the reference uses the engine's value
      for it (the edge is left out of the node comparison), its engine probability must not
      exceed twice the list's smallest (512th) probability, and at most 5 % of all edges may be
      unverifiable. Per field: the statistic on the distribution (sp3_metrics.compare, the
      reference as A), coverage median abs diff <= 0.02, and the node comparison (every dumped
      children/end/merged value against the replay, as probabilities; pairs that are -inf on
      both sides are counted, not compared) median abs diff <= 0.01. Also written: the coverage
      sanity gate (median engine coverage per field >= 0.8; spec 12.3 applies it to Qwen3.5-9B
      and Gemma 4 E4B, so it is not part of `pass`) and per record the values behind it all.
      Both modes, records with `ancestors` (--after): every dependent answer and node/entry has
      `given` = the ancestors' returned values (same order) and every dependent node/entry's
      `lines` = stem text + JSON value + ",\n" over them (spec 5.2); level-0 fields have
      neither. Part of `pass`.
      Image states (sub-project 4, spec 7): dump --states FILE takes JSONL state specs instead of the test tickets (a string, or
      a list of {"text": T} / {"image": PATH} items) and --order-debias K / --temperature T set those options (each needs --tag).
      The CLI then runs with the preset's mmproj / image-min-tokens / image-max-tokens. An image state's records keep the image
      files (checked against the dump's SHA-256), the temperature and, per catalogue variant, the prompt text (prefix text +
      the state's text parts with MARKER per image + tail text) and each node's or entry's pieces. compare replays them through
      /completion with {"prompt_string": text + branch pieces, "multimodal_data": [base64 images]} (reference servers run with
      LLAMA_MEDIA_MARKER=<__media__>), tree mode: the mean over the variants, then the temperature, against the engine; full
      path: as for text (K = 1, T = 1). The recorded replay keeps sha256:<hex> in place of each image, so --offline needs the
      image files.

  uv run reference_check.py split --model <gguf> --preset <name> --decide-seqs S --tag TAG [--n 20] [--batch 5]
                                  [--out-dir DIR]
      (server STOPPED) row-cap split check (spec 12.3): the first N tickets as B-state and as
      1-state request lines, full-path and tree (four llama-decide runs; responses written to
      DIR/split-TAG-<preset>-<scoring>-b<B|1>.jsonl). Per mode the B-state answers against the
      1-state ones (the reference) under the statistic; per full-path B-state line the branch
      decodes = timings.decodes - rounds - 1 when the line built its prefix. Pass: the full-path
      statistic, every full-path B-state line in one round with >= 2 branch decodes, and every
      full-path 1-state line with 1 (the unsplit reference). The tree comparison is the
      batch-composition reference; `side_by_side` lists both per field.
      Writes DIR/split-TAG-<preset>.json.

The replay uses the engine's own token ids, so it checks the KV choreography and the
logit read-out, not where the engine splits text into tokens.

Settings: ENGINE_PRESETS (the presets file, default models-engine.ini) and ENGINE_BIN (the directory of llama-decide,
default engine/build/bin, e.g. engine-63ea2c51a/build/bin); relative values from the project directory.
"""

import argparse
import base64
import configparser
import copy
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import httpx

from decide_client import TEST_SET, build_request, chunks, parse_after, presets_file, state_from_spec

ROOT = Path(__file__).parent
# preset keys that place the model on devices, passed to llama-decide as --KEY VALUE when the preset holds them
PLACEMENT_KEYS = ("n-cpu-moe", "tensor-split", "override-tensor", "main-gpu", "split-mode", "device")
# preset keys of a projector, passed to llama-decide as --KEY VALUE when the preset holds them (sub-project 4)
MEDIA_KEYS = ("mmproj", "image-min-tokens", "image-max-tokens")
MARKER = "<__media__>"  # LLAMA_MEDIA_MARKER of the reference servers (spec 7)

N_PROBS = 512              # replay list length (spec 12.3)
NODE_BOUND = 0.01          # full-path node comparison: per field median abs diff in probability
COVERAGE_BOUND = 0.02      # full-path coverage: per field median abs diff
UNVERIFIABLE_FACTOR = 2.0  # an unverifiable edge's engine probability <= this x the list's smallest probability
UNVERIFIABLE_MAX = 0.05    # at most this fraction of a model's edges unverifiable
COVERAGE_GATE = 0.8        # sanity gate: median engine coverage per field
JSON_SPACE = b" \t\n\r"


def _engine_settings(preset=None):
    """The [*] defaults overlaid with the preset's own section, e.g. a per-preset cache-type override."""
    ini = configparser.ConfigParser()
    ini.read(presets_file())
    settings = dict(ini.items("*"))
    if preset and ini.has_section(preset):
        settings.update(ini.items(preset))
    return settings


def _stem(preset, tag):
    return f"reference-{tag + '-' if tag else ''}{preset}"


def _out_dir(path):
    """--out-dir: a relative path is taken from the project directory."""
    return path if path.is_absolute() else ROOT / path


def request_body(preset, schema, texts, scoring="tree", after=None, order_debias=0, temperature=None):
    """The request for `texts` (states: strings or part lists): the triage schema, or the fields/instructions of `schema`;
    options.scoring set for full-path, `after` set on the fields named by --after, options.order_debias / temperature when
    given."""
    if schema is None:
        body = build_request(preset, list(texts))
    else:
        body = {"instructions": schema.get("instructions", ""), "fields": copy.deepcopy(schema["fields"]), "states": list(texts)}
    if scoring != "tree":
        body.setdefault("options", {})["scoring"] = scoring
    if order_debias:
        body.setdefault("options", {})["order_debias"] = order_debias
    if temperature is not None:
        body.setdefault("options", {})["temperature"] = temperature
    for name, parents in (after or {}).items():
        if name not in body["fields"]:
            sys.exit(f"--after {name}={','.join(parents)}: the request has no field {name!r}")
        body["fields"][name]["after"] = list(parents)
    return body


def ancestors_of(fields):
    """Field name -> its ancestors (transitive closure of `after`) in declaration order (spec 5.2)."""
    names = list(fields)
    out = {}
    for name in names:
        seen, todo = set(), list(fields[name].get("after", []))
        while todo:
            a = todo.pop()
            if a not in seen:
                seen.add(a)
                todo.extend(fields[a].get("after", []))
        out[name] = [n for n in names if n in seen]
    return out


def field_keys(field_def):
    """Option/level/bool keys in the declaration order the engine assigns to children."""
    if field_def["type"] == "choice":
        return list(field_def["options"])
    if field_def["type"] == "score":
        return [str(i) for i in range(len(field_def["levels"]))]
    if field_def["type"] == "bool":
        return ["true", "false"]
    raise ValueError(f"unknown field type {field_def['type']!r}")


def engine_probs(record):
    """The engine's own distribution, in the same order as record['keys']."""
    ans = record["engine"]
    if record["type"] == "bool":
        p = ans["p_true"]
        return [p, 1.0 - p]
    if record["type"] == "score":
        return list(ans["probabilities"])
    if record["type"] == "choice":
        return [ans["probabilities"][k] for k in record["keys"]]
    raise ValueError(record["type"])


def option_probs(nodes, n_options, node_probs):
    """Per option: product over the branch nodes on its path of the renormalised child probability."""
    probs = [1.0] * n_options
    for node, child_p in zip(nodes, node_probs):
        for child_options, p in zip(node["options"], child_p):
            for i in child_options:
                probs[i] *= p
    return probs


def llama_decide():
    """ENGINE_BIN (default engine/build/bin; relative: from the project directory) / llama-decide."""
    return ROOT / (os.environ.get("ENGINE_BIN") or "engine/build/bin") / "llama-decide"


def decide_command(model, preset, decide_seqs, dump=True):
    """llama-decide with the server's -c/-ngl/-fa/cache types of the preset (and --cpu-moe for MoE presets, the
    placement keys when the preset holds them, the projector keys when the preset holds them)."""
    settings = _engine_settings(preset)
    cmd = [
        str(llama_decide()), "-m", model,
        "--decide-seqs", str(decide_seqs),
        "-c", settings.get("ctx-size", "4096"),
        "-ngl", settings.get("ngl", "99"),
        "-fa", settings.get("fa", "on"),
        "-ctk", settings.get("cache-type-k", "f16"),
        "-ctv", settings.get("cache-type-v", "f16"),
    ]
    if dump:
        cmd.append("--dump-tokens")
    if settings.get("cpu-moe", "0").lower() in ("1", "true", "on", "yes"):
        cmd.append("--cpu-moe")  # MoE presets keep the expert weights on the CPU, as the server does
    for key in PLACEMENT_KEYS:
        if key in settings:
            cmd += [f"--{key}", settings[key]]
    for key in MEDIA_KEYS:
        if key in settings:
            cmd += [f"--{key}", settings[key]]
    return cmd


def run_decide(cmd, bodies):
    """One response per request body (llama-decide reads one body per stdin line); exits on any failure."""
    proc = subprocess.run(cmd, input="".join(json.dumps(b) + "\n" for b in bodies), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"llama-decide failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if len(out_lines) != len(bodies):
        sys.exit(f"expected {len(bodies)} response lines, got {len(out_lines)}:\n{proc.stderr[-2000:]}")
    return [json.loads(l) for l in out_lines]


def _with_dependent(branch, item):
    """item plus the branch's `lines` and `given` (dumped for dependent fields only, spec 10)."""
    for key in ("lines", "given"):
        if key in branch:
            item[key] = branch[key]
    return item


def build_records(responses, fields, ancestors=None, specs=None):
    """One record per (state, field) from --dump-tokens responses, states numbered over all lines in order; with
    `ancestors` (field -> its ancestors) every record gets the ancestors' returned values for its state. `specs`: the state
    specs of a --states dump in the same order (image states take their image files from them)."""
    records, ticket = [], 0
    for data in responses:
        prefix = data["tokens"]["prefix"]
        for res, state in zip(data["results"], data["tokens"]["states"]):
            if "parts" in state:  # an image state (spec 6)
                records += image_records(data, res, state, fields, ancestors, specs[ticket], ticket)
                ticket += 1
                continue
            answers = res["answers"]
            branches_by_field = {}
            for b in state["branches"]:
                branches_by_field.setdefault(b["field"], []).append(b)
            for name, fdef in fields.items():
                branches = branches_by_field[name]
                rec = {"ticket": ticket, "field": name, "type": fdef["type"], "keys": field_keys(fdef), "engine": answers[name]}
                if branches[0].get("mode") == "full_path":
                    rec["mode"] = "full_path"
                    rec["prompt_prefix"] = prefix + state["state"] + state["tail"]
                    rec["entries"] = [_with_dependent(b, {"option": b["option"], "tokens": b["tokens"], "rows": b["rows"],
                                                          "logp": b["logp"]}) for b in branches]
                else:
                    rec["nodes"] = [_with_dependent(b, {"node": b["node"],
                                                        "prompt_tokens": prefix + state["state"] + state["tail"] + b["tokens"],
                                                        "children": b["children"], "options": b["options"]})
                                    for b in branches]
                if ancestors is not None:
                    rec["ancestors"] = {g: answers[g]["value"] for g in ancestors[name]}
                records.append(rec)
            ticket += 1
    return records


def check_text_states(responses):
    """Exits when a text state was dumped with order_debias > 1 (a branch of variant > 0) or a temperature other than 1:
    text records replay token ids on variant 0's prefix and compare without a temperature, so their reference would be
    wrong without an error. Image states are recorded per variant with the temperature. build_records itself does not check
    this, because results-engine/sp3/integrity/checks.py reads debiased text dumps through it for answers and ancestors only."""
    ticket = 0
    for data in responses:
        T = data.get("engine", {}).get("temperature", 1.0)  # engines before sub-project 3 write none: T = 1
        for state in data["tokens"]["states"]:
            if "parts" not in state and (any(b.get("variant", 0) > 0 for b in state["branches"]) or T != 1.0):
                sys.exit(f"text state {ticket}: dumped with order_debias > 1 or temperature {T}; "
                         "a text state's reference replays variant 0 at T = 1 only (dump text states without "
                         "--order-debias / --temperature)")
            ticket += 1


def image_files(spec, state):
    """The image files of a state spec in part order, checked against the dump's SHA-256 of each image part."""
    paths = [item["image"] for item in spec if "image" in item]
    shas = [p["sha256"] for p in state["parts"] if p["type"] == "image"]
    if len(paths) != len(shas):
        sys.exit(f"state {spec!r}: {len(paths)} image files, {len(shas)} image parts in the dump")
    files = []
    for path, sha in zip(paths, shas):
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != sha:
            sys.exit(f"{path}: SHA-256 {actual} differs from the dump's {sha}")
        files.append({"path": path, "sha256": sha})
    return files


def state_text(spec):
    """A state as replay text (spec 7): its text parts, MARKER for each image."""
    return "".join(MARKER if "image" in item else item["text"] for item in spec)


def image_records(data, res, state, fields, ancestors, spec, ticket):
    """The records of one image state: per field the tree nodes of every catalogue variant with that variant's prompt text,
    or the full-path entries (one variant, T = 1)."""
    tokens = data["tokens"]
    files, text = image_files(spec, state), state_text(spec)
    by_field = {}
    for b in state["branches"]:
        by_field.setdefault(b["field"], {}).setdefault(b.get("variant", 0), []).append(b)
    records = []
    for name, fdef in fields.items():
        variants = by_field[name]
        rec = {"ticket": ticket, "field": name, "type": fdef["type"], "keys": field_keys(fdef), "engine": res["answers"][name],
               "image": True, "state": spec, "images": files, "temperature": data["engine"]["temperature"]}
        if variants[0][0].get("mode") == "full_path":
            if len(variants) != 1 or rec["temperature"] != 1.0:
                sys.exit("full-path image records need order_debias < 2 and T = 1")
            rec["mode"] = "full_path"
            rec["prompt_text"] = tokens["prefix_texts"][0] + text + tokens["tail_text"]
            rec["entries"] = [_with_dependent(b, {"option": b["option"], "tokens": b["tokens"], "rows": b["rows"], "logp": b["logp"],
                                                  "pieces_hex": b["pieces_hex"]}) for b in variants[0]]
        else:
            rec["variants"] = [{"variant": v, "prompt_text": tokens["prefix_texts"][v] + text + tokens["tail_text"],
                                "nodes": [_with_dependent(b, {"node": b["node"], "pieces_hex": b["pieces_hex"],
                                                              "children": b["children"], "options": b["options"]}) for b in bs]}
                               for v, bs in sorted(variants.items())]
        if ancestors is not None:
            rec["ancestors"] = {g: res["answers"][g]["value"] for g in ancestors[name]}
        records.append(rec)
    return records


def piece_text(pieces_hex):
    """The text of dumped token pieces (hex bytes); exits when they are not UTF-8 text. A branch node or full-path row that
    ends inside a multi-byte UTF-8 character therefore stops the run: string replays need whole characters."""
    try:
        return bytes.fromhex("".join(pieces_hex)).decode("utf-8")
    except ValueError as e:  # UnicodeDecodeError is a ValueError
        sys.exit(f"pieces {pieces_hex!r} are not UTF-8 text: {e}")


def with_temperature(p, T):
    """The engine's temperature on a distribution (decide::apply_temperature): softmax(log p / T); T = 1 returns p."""
    if T == 1.0:
        return list(p)
    z = [-math.inf if x == 0 else math.log(x) / T for x in p]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


_IMAGE_DATA = {}


def image_prompt(rec, text):
    """The /completion prompt of an image record (spec 7): the text and the record's images as multimodal_data; exits when
    an image file's SHA-256 is not the record's (the file changed since the dump)."""
    data = []
    for f in rec["images"]:
        if f["path"] not in _IMAGE_DATA:
            raw = (ROOT / f["path"]).read_bytes()
            _IMAGE_DATA[f["path"]] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode())
        sha, b64 = _IMAGE_DATA[f["path"]]
        if sha != f["sha256"]:
            sys.exit(f"{f['path']}: SHA-256 {sha} differs from the record's {f['sha256']} (the image changed since the dump)")
        data.append(b64)
    return {"prompt_string": text, "multimodal_data": data}


def row_prompt(rec, entry, idx):
    """The replay prompt of a full-path row: token ids, or for an image record the prompt text + the entry's pieces up to it
    (a row inside a multi-byte UTF-8 character stops the run in piece_text: string replays need whole characters)."""
    if rec.get("image"):
        return image_prompt(rec, rec["prompt_text"] + piece_text(entry["pieces_hex"][: idx + 1]))
    return rec["prompt_prefix"] + entry["tokens"][: idx + 1]


def cmd_dump(args):
    if (args.scoring != "tree" or args.after or args.batch != 1 or args.decide_seqs != 9 or args.schema or args.states
            or args.order_debias or args.temperature is not None) and not args.tag:
        sys.exit("dump: --tag is required with --scoring full_path, --after, --batch > 1, --decide-seqs other than 9, --schema, "
                 "--states, --order-debias or --temperature (without it the dump overwrites the committed tree dump "
                 "reference-<preset>-dump.jsonl)")
    if args.states:
        specs = [json.loads(l) for l in args.states.read_text().splitlines() if l.strip()][: args.n]
        states = [state_from_spec(s) for s in specs]
    else:
        specs = None
        states = [json.loads(l)["text"] for l in args.test_set.read_text().splitlines() if l.strip()][: args.n]
    schema = json.loads(args.schema.read_text()) if args.schema else None
    after = parse_after(args.after, list(request_body(args.preset, schema, ["x"])["fields"]))  # decide_client's rules
    fields = request_body(args.preset, schema, ["x"], args.scoring, after)["fields"]
    bodies = [request_body(args.preset, schema, batch, args.scoring, after, args.order_debias, args.temperature)
              for batch in chunks(states, args.batch)]
    responses = run_decide(decide_command(args.model, args.preset, args.decide_seqs), bodies)
    check_text_states(responses)
    records = build_records(responses, fields, ancestors_of(fields) if after else None, specs)

    out_path = _out_dir(args.out_dir) / f"{_stem(args.preset, args.tag)}-dump.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    if args.scoring == "tree":
        n_nodes = sum(len(_parts(r)) for r in records)
        print(f"wrote {out_path} ({len(records)} records, {n_nodes} branch nodes, {len(states)} tickets)")
    else:
        n_rows = sum(len(e["rows"]) for r in records for e in r["entries"])
        print(f"wrote {out_path} ({len(records)} records, {n_rows} rows, {len(states)} tickets)")


def child_probs(client, preset, rec, node, prompt=None):
    """Reference probabilities of one branch node's children, renormalised over the children; `prompt` replaces the node's
    prompt_tokens (image records)."""
    resp = client.post("/completion", json={
        "model": preset,
        "prompt": node["prompt_tokens"] if prompt is None else prompt,
        "n_predict": 1,
        "n_probs": 512,
        "temperature": 0,
        "post_sampling_probs": False,
        "cache_prompt": False,
    })
    if resp.status_code != 200:
        sys.exit(f"/completion failed: {resp.status_code} {resp.text[:500]}")
    probs = resp.json()["completion_probabilities"][0]
    if "top_logprobs" in probs:
        by_id = {p["id"]: p["logprob"] for p in probs["top_logprobs"]}
        to_prob = math.exp
    elif "top_probs" in probs:
        by_id = {p["id"]: p["prob"] for p in probs["top_probs"]}
        to_prob = lambda x: x
    else:
        sys.exit(f"unexpected completion_probabilities shape: {json.dumps(probs)[:500]}")
    missing = [c for c in node["children"] if c not in by_id]
    if missing:
        sys.exit(f"ticket {rec['ticket']} field {rec['field']}: children {missing} not in top_logprobs "
                 f"(n_probs may need to be larger)")
    raw = [to_prob(by_id[c]) for c in node["children"]]
    total = sum(raw)
    return [x / total for x in raw]


# ---- raw replay data (every comparison can be recomputed offline) ----

def _recordable(body):
    """A request as recorded and as looked up offline: multimodal_data replaced by sha256:<hex> of each image, which keeps the
    replay files small; other requests unchanged."""
    p = body.get("prompt")
    if not isinstance(p, dict) or "multimodal_data" not in p:
        return body
    shas = ["sha256:" + hashlib.sha256(base64.b64decode(d)).hexdigest() for d in p["multimodal_data"]]
    return {**body, "prompt": {**p, "multimodal_data": shas}}


class RecordingClient:
    """Passes every request to `client` and keeps it with its answer: /completion as the generated token and the listed
    (id, log-prob) pairs, /detokenize as its body; the listed pieces once per token id."""

    def __init__(self, client):
        self.client, self.calls, self.pieces = client, [], {}

    def post(self, path, json):
        resp = self.client.post(path, json=json)
        if resp.status_code == 200:
            call = {"path": path, "request": _recordable(json)}
            if path == "/completion":
                probs = resp.json()["completion_probabilities"][0]
                listed = probs.get("top_logprobs", [])
                call.update(id=probs.get("id"), logprob=probs.get("logprob"), top=[[p["id"], p["logprob"]] for p in listed])
                self.pieces.update((p["id"], bytes(p["bytes"]).hex()) for p in listed)
            else:
                call["response"] = resp.json()
            self.calls.append(call)
        return resp

    def write(self, path):
        """gzip JSONL: one line per request, then {"pieces": {id: hex bytes}}."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            for call in self.calls:
                f.write(json.dumps(call) + "\n")
            f.write(json.dumps({"pieces": self.pieces}) + "\n")


def _request_key(body):
    return json.dumps(body, sort_keys=True)


def offline_client(path):
    """An httpx client answering from a RecordingClient file; a request it has no answer for gets a 404."""
    answers, pieces = {}, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if "pieces" in rec:
                pieces = {int(k): bytes.fromhex(v) for k, v in rec["pieces"].items()}
            else:
                answers[rec["path"], _request_key(rec["request"])] = rec

    def handler(request):
        rec = answers.get((request.url.path, _request_key(_recordable(json.loads(request.content)))))
        if rec is None:
            return httpx.Response(404, json={"error": "not in the recorded replay"})
        if rec["path"] != "/completion":
            return httpx.Response(200, json=rec["response"])
        listed = [{"id": t, "token": pieces[t].decode("utf-8", "replace"), "bytes": list(pieces[t]), "logprob": lp}
                  for t, lp in rec["top"]]
        return httpx.Response(200, json={"completion_probabilities": [
            {"id": rec["id"], "logprob": rec["logprob"], "top_logprobs": listed}]})

    return httpx.Client(base_url="http://offline", transport=httpx.MockTransport(handler))


# ---- full-path replay (spec 4.1, 12.3) ----

def ends_value(w):
    """Spec 4.1 ends(w) on a piece's bytes, exactly the engine's decide::ends_value: w is not empty and only JSON
    whitespace, or its first non-whitespace byte is ',' or '}' (so ',&' ends a value)."""
    for c in w:
        if c not in JSON_SPACE:
            return c in b",}"
    return len(w) > 0


def in_merged(piece, base):
    """Spec 4.1: a token with this piece is in MERGED[base], i.e. piece = base + r with base not empty and ends(r)
    (which also rules out an empty r)."""
    return len(base) > 0 and piece.startswith(base) and ends_value(piece[len(base):])


def _log_sum(logps):
    total = sum(math.exp(lp) for lp in logps)
    return math.log(total) if total > 0 else -math.inf


def _log_add(a, b):
    hi, lo = max(a, b), min(a, b)
    return hi if lo == -math.inf else hi + math.log1p(math.exp(lo - hi))


def _num(x):
    return -math.inf if x is None else x  # the engine writes -inf as null


def top_logprobs(client, preset, prompt, pieces):
    """The replay's top-512 at the prompt's last position as (token id, piece bytes, log-prob); records the pieces."""
    resp = client.post("/completion", json={
        "model": preset,
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": N_PROBS,
        "temperature": 0,
        "post_sampling_probs": False,
        "cache_prompt": False,
    })
    if resp.status_code != 200:
        sys.exit(f"/completion failed: {resp.status_code} {resp.text[:500]}")
    probs = resp.json()["completion_probabilities"][0]
    if "top_logprobs" not in probs:
        sys.exit(f"unexpected completion_probabilities shape: {json.dumps(probs)[:500]}")
    top = [(p["id"], bytes(p["bytes"]), p["logprob"]) for p in probs["top_logprobs"]]
    pieces.update((tok, piece) for tok, piece, _ in top)
    return top


def token_piece(client, preset, tok, pieces):
    """A token's piece: from a replayed list, else from /detokenize (exact for pieces that are valid UTF-8)."""
    if tok not in pieces:
        resp = client.post("/detokenize", json={"model": preset, "tokens": [tok]})
        if resp.status_code != 200:
            sys.exit(f"/detokenize failed: {resp.status_code} {resp.text[:500]}")
        pieces[tok] = resp.json()["content"].encode()
    return pieces[tok]


def replay_row(row, top, piece_of):
    """The replay's values for one dumped row, in log space: its children (None = not in the list), the END sum on a
    leaf row and per leaf child the MERGED sum, both over the listed tokens only (lower bounds); p_min = the smallest
    probability of the list (its 512th)."""
    by_id = {tok: lp for tok, _, lp in top}
    out = {"node": row["node"], "p_min": min(math.exp(lp) for _, _, lp in top)}
    if "children" in row:
        out["children"] = {t: by_id.get(int(t)) for t in row["children"]}
    if "merged" in row:
        out["merged"] = {}
        for t in row["merged"]:
            base = piece_of(int(t))
            out["merged"][t] = _log_sum(lp for _, piece, lp in top if in_merged(piece, base))
    if "end" in row:
        out["end"] = _log_sum(lp for _, piece, lp in top if ends_value(piece))
    return out


def field_nodes(entries):
    """One field's full-path entries: the stem length, the dumped row per node, the node per token prefix (a node's row
    is read in the first option through it, so every node has exactly one)."""
    stem = next(e for e in entries if e["option"] == 0)["rows"][0] + 1  # option 0 reads the root at the stem's last token
    rows, node_at = {}, {}
    for e in entries:
        for idx, row in zip(e["rows"], e["logp"]):
            if row["node"] in rows:
                raise ValueError(f"node {row['node']} dumped twice")
            rows[row["node"]] = row
            node_at[tuple(e["tokens"][: idx + 1])] = row["node"]
    return stem, rows, node_at


def option_log_probs(entries, value):
    """Spec 4.1 log P(o) per option in option order; value(kind, node, token) gives the log value of kind 'children'
    or 'merged' (token = the child) or 'end' (token None) at a node."""
    stem, _, node_at = field_nodes(entries)
    out = []
    for e in sorted(entries, key=lambda x: x["option"]):
        toks = e["tokens"]
        path = toks[stem:]
        nodes = [node_at[tuple(toks[: stem + i])] for i in range(len(path) + 1)]  # r_0 .. r_k
        lp = sum(value("children", nodes[i], t) for i, t in enumerate(path[:-1]))
        last = path[-1]
        lp += _log_add(value("children", nodes[-2], last) + value("end", nodes[-1], None), value("merged", nodes[-2], last))
        out.append(lp)
    return out


def _normalise(logps, what):
    """(distribution, coverage) from per-option log P (log-sum-exp)."""
    m = max(logps)
    if m == -math.inf:
        sys.exit(f"{what}: every option has probability 0")
    lse = m + math.log(sum(math.exp(lp - m) for lp in logps))
    return [math.exp(lp - lse) for lp in logps], math.exp(lse)


def _node_pair(node, kind, tok, lp_engine, lp_replay):
    engine, replay = _num(lp_engine), lp_replay
    return {"node": node, "kind": kind, "token": None if tok is None else int(tok), "p_engine": math.exp(engine),
            "p_replay": math.exp(replay), "abs_diff": abs(math.exp(engine) - math.exp(replay)),
            "both_zero": engine == -math.inf and replay == -math.inf}


def full_path_record_check(rec, replays):
    """One full-path record against its replayed rows (node -> replay_row): the reference distribution and coverage
    (spec 4.1 on the replay's values, the engine's value on an unverifiable edge), the node comparison, the edges."""
    _, rows, _ = field_nodes(rec["entries"])

    def eng(kind, node, tok):
        row = rows[node]
        return _num(row["end"] if kind == "end" else row[kind][str(tok)])

    def ref(kind, node, tok):
        r = replays[node]
        if kind == "end":
            return r["end"]
        v = r[kind][str(tok)]
        return eng(kind, node, tok) if v is None else v  # a child absent from the list: the engine's value

    what = f"ticket {rec['ticket']} field {rec['field']}"
    reference, ref_cov = _normalise(option_log_probs(rec["entries"], ref), what + " (reference)")
    recomputed, rec_cov = _normalise(option_log_probs(rec["entries"], eng), what + " (engine rows)")
    engine = engine_probs(rec)
    nodes, unverifiable, edges = [], [], 0
    for node, row in sorted(rows.items()):
        r = replays[node]
        for tok, lp in row.get("children", {}).items():
            edges += 1
            if r["children"][tok] is None:
                p = math.exp(lp)
                unverifiable.append({"node": node, "token": int(tok), "p_engine": p, "p_min": r["p_min"],
                                     "ratio": p / r["p_min"] if r["p_min"] > 0 else math.inf,
                                     "within_bound": p <= UNVERIFIABLE_FACTOR * r["p_min"]})
            else:
                nodes.append(_node_pair(node, "children", tok, lp, r["children"][tok]))
        for tok, lp in row.get("merged", {}).items():
            nodes.append(_node_pair(node, "merged", tok, lp, r["merged"][tok]))
        if "end" in row:
            nodes.append(_node_pair(node, "end", None, row["end"], r["end"]))
    ref_argmax = max(range(len(reference)), key=lambda i: reference[i])
    top2 = sorted(reference, reverse=True)
    return {
        "ticket": rec["ticket"], "field": rec["field"], "reference": reference, "engine": engine,
        "coverage_reference": ref_cov, "coverage_engine": rec["engine"]["coverage"],
        "argmax_agree": ref_argmax == max(range(len(engine)), key=lambda i: engine[i]),
        "ref_top2_margin": top2[0] - top2[1] if len(top2) > 1 else 1.0,
        "engine_recompute_max_abs_diff": max([abs(a - b) for a, b in zip(recomputed, engine)]
                                             + [abs(rec_cov - rec["engine"]["coverage"])]),
        "edges": edges, "unverifiable": unverifiable, "nodes": nodes,
    }


def unverifiable_summary(items, n_edges):
    """Spec 12.3 over a model's edges: every unverifiable edge within the bound, at most 5 % of the edges."""
    fraction = len(items) / n_edges if n_edges else 0.0
    within = all(u["within_bound"] for u in items)
    return {"edges": n_edges, "unverifiable": len(items), "fraction": fraction, "max_fraction": UNVERIFIABLE_MAX,
            "bound_factor": UNVERIFIABLE_FACTOR, "max_ratio": max((u["ratio"] for u in items), default=None),
            "within_bound": within, "pass": within and fraction <= UNVERIFIABLE_MAX, "detail": items}


def json_text(value):
    """A value's JSON text as the engine writes it (nlohmann dump: UTF-8, no ASCII escaping)."""
    return json.dumps(value, ensure_ascii=False)


def expected_lines(ancestors):
    """Spec 5.2 lines(F) from the ancestors' values in declaration order: stem text + encoded value + ",\\n" each."""
    return "".join(f"  {json_text(name)}: {json_text(value)},\n" for name, value in ancestors.items())


def _parts(rec):
    """A record's nodes or entries; an image tree record's nodes over all its catalogue variants."""
    if "variants" in rec:
        return [n for v in rec["variants"] for n in v["nodes"]]
    return rec["entries"] if rec.get("mode") == "full_path" else rec["nodes"]


def dependent_problems(rec):
    """A record from an --after dump: a dependent field's answer and every node/entry have `given` = the ancestors'
    returned values (same order) and every node/entry `lines` = expected_lines; a level-0 field has neither."""
    where = f"ticket {rec['ticket']} {rec['field']}"
    parts = _parts(rec)
    anc, problems = rec["ancestors"], []
    if not anc:
        if "given" in rec["engine"]:
            problems.append(f"{where}: level-0 answer has given {rec['engine']['given']}")
        problems += [f"{where}: level-0 entry has given/lines" for p in parts if "given" in p or "lines" in p]
        return problems
    want, lines = list(anc.items()), expected_lines(anc)
    given = rec["engine"].get("given")
    if given is None or list(given.items()) != want:
        problems.append(f"{where}: answer given {given}, the ancestors returned {anc}")
    for p in parts:
        if p.get("given") is None or list(p["given"].items()) != want:
            problems.append(f"{where}: entry given {p.get('given')}, the ancestors returned {anc}")
        if p.get("lines") != lines:
            problems.append(f"{where}: entry lines {p.get('lines')!r}, built from the ancestors {lines!r}")
    return problems


def dependent_check(records):
    """The `after` check over an --after dump's records (None for a dump without --after)."""
    if not any("ancestors" in r for r in records):
        return None
    dependent = [r for r in records if r["ancestors"]]
    problems = [p for r in records for p in dependent_problems(r)]
    return {"dependent_records": len(dependent),
            "dependent_entries": sum(len(_parts(r)) for r in dependent),
            "problems": problems, "pass": not problems}


def summarize_full_path(checks, records):
    """Per field and per model: the statistic, coverage, node comparison, unverifiable edges, gate, `after` check."""
    from sp3_metrics import compare as statistic  # sp3_metrics imports this module, so not at the top

    def entries(key):
        return [(c["ticket"], c["field"], dict(zip(r["keys"], c[key]))) for r, c in zip(records, checks)]

    stat = statistic(entries("reference"), entries("engine"))
    fields = {}
    for name in dict.fromkeys(c["field"] for c in checks):
        cs = [c for c in checks if c["field"] == name]
        cov = [abs(c["coverage_reference"] - c["coverage_engine"]) for c in cs]
        pairs = [p for c in cs for p in c["nodes"]]
        compared = [p["abs_diff"] for p in pairs if not p["both_zero"]]
        excess = [p["p_replay"] - p["p_engine"] for p in pairs if p["kind"] != "children"]
        s = stat["fields"][name]
        f = {k: s[k] for k in ("median_abs_diff", "max_abs_diff", "argmax_agreement", "argmax_disagreements",
                               "disagreements_over_margin")}
        f.update({
            "statistic_pass": s["pass"],
            "coverage_median_abs_diff": statistics.median(cov), "coverage_max_abs_diff": max(cov),
            "coverage_engine_median": statistics.median([c["coverage_engine"] for c in cs]),
            "coverage_reference_median": statistics.median([c["coverage_reference"] for c in cs]),
            "coverage_pass": statistics.median(cov) <= COVERAGE_BOUND,
            "node_values": len(compared), "node_both_zero": len(pairs) - len(compared),
            "node_median_abs_diff": statistics.median(compared), "node_max_abs_diff": max(compared),
            "node_pass": statistics.median(compared) <= NODE_BOUND,
            "end_merged_replay_excess_max": max(excess, default=None),  # > 0: the replay's lower bound above the engine
            "edges": sum(c["edges"] for c in cs), "unverifiable_edges": sum(len(c["unverifiable"]) for c in cs),
            "engine_recompute_max_abs_diff": max(c["engine_recompute_max_abs_diff"] for c in cs),
        })
        f["pass"] = f["statistic_pass"] and f["coverage_pass"] and f["node_pass"]
        fields[name] = f
    unverifiable = unverifiable_summary([dict(u, ticket=c["ticket"], field=c["field"]) for c in checks for u in c["unverifiable"]],
                                        sum(c["edges"] for c in checks))
    medians = {name: f["coverage_engine_median"] for name, f in fields.items()}
    out = {"mode": "full_path", "n_records": len(checks), "fields": fields, "unverifiable": unverifiable,
           "coverage_gate": {"threshold": COVERAGE_GATE, "median_engine_coverage": medians,
                             "pass": all(v >= COVERAGE_GATE for v in medians.values())}}
    passed = all(f["pass"] for f in fields.values()) and unverifiable["pass"]
    dep = dependent_check(records)
    if dep is not None:
        out["dependent"] = dep
        passed = passed and dep["pass"]
    out["pass"] = passed
    out["detail"] = checks
    return out


def compare_full_path(records, client, preset):
    """Replay every row of every full-path record through /completion and compare (spec 12.3)."""
    pieces = {}

    def piece_of(tok):
        return token_piece(client, preset, tok, pieces)

    checks = []
    for rec in records:
        replays = {}
        for e in rec["entries"]:
            for idx, row in zip(e["rows"], e["logp"]):
                top = top_logprobs(client, preset, row_prompt(rec, e, idx), pieces)
                replays[row["node"]] = replay_row(row, top, piece_of)
        checks.append(full_path_record_check(rec, replays))
    return summarize_full_path(checks, records)


def cmd_compare(args):
    dump_path = _out_dir(args.out_dir) / f"{_stem(args.preset, args.tag)}-dump.jsonl"
    records = [json.loads(l) for l in dump_path.read_text().splitlines() if l.strip()]
    if args.n:
        keep = sorted({r["ticket"] for r in records})[: args.n]
        records = [r for r in records if r["ticket"] in keep]
    out_path = _out_dir(args.out_dir) / f"{_stem(args.preset, args.tag)}.json"
    replay_path = _out_dir(args.out_dir) / f"{_stem(args.preset, args.tag)}-replay.jsonl.gz"

    if args.offline:
        client = offline_client(replay_path)
    else:
        client = RecordingClient(httpx.Client(base_url=args.url, timeout=600))
    if records and records[0].get("mode") == "full_path":
        out = {"preset": args.preset, "tag": args.tag, **compare_full_path(records, client, args.preset)}
        if not args.offline:
            client.write(replay_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        summary = {"fields": out["fields"], "unverifiable": {k: v for k, v in out["unverifiable"].items() if k != "detail"},
                   "coverage_gate": out["coverage_gate"]}
        if "dependent" in out:
            summary["dependent"] = {**out["dependent"], "problems": out["dependent"]["problems"][:5]}
        print(json.dumps(summary, indent=2))
        print(f"pass: {out['pass']}")
        print(f"wrote {out_path}")
        return

    abs_diffs, argmax_agree, over_margin, argmax_diffs = {}, {}, {}, {}
    detail = []
    for rec in records:
        if rec.get("image"):
            # spec 7: per catalogue variant the replay of its prompt text, the mean over the variants, then the temperature
            per_variant, nodes = [], []
            for var in rec["variants"]:
                probs = [child_probs(client, args.preset, rec, n, image_prompt(rec, var["prompt_text"] + piece_text(n["pieces_hex"])))
                         for n in var["nodes"]]
                per_variant.append(option_probs(var["nodes"], len(rec["keys"]), probs))
                nodes += var["nodes"]
            mean = [sum(p[i] for p in per_variant) / len(per_variant) for i in range(len(rec["keys"]))]
            ref = with_temperature(mean, rec["temperature"])
        else:
            # dumps written before multi-branch support hold one node with one option per child
            nodes = rec.get("nodes") or [{"prompt_tokens": rec["prompt_tokens"], "children": rec["children"],
                                          "options": [[k] for k in range(len(rec["children"]))]}]
            node_probs = [child_probs(client, args.preset, rec, n) for n in nodes]
            ref = option_probs(nodes, len(rec["keys"]), node_probs)
        eng = engine_probs(rec)

        diffs = [abs(a - b) for a, b in zip(ref, eng)]
        abs_diffs.setdefault(rec["field"], []).extend(diffs)
        ref_argmax = max(range(len(ref)), key=lambda i: ref[i])
        eng_argmax = max(range(len(eng)), key=lambda i: eng[i])
        agree = ref_argmax == eng_argmax
        argmax_agree.setdefault(rec["field"], []).append(agree)
        argmax_diffs.setdefault(rec["field"], []).append(abs(ref[ref_argmax] - eng[ref_argmax]))
        top2 = sorted(ref, reverse=True)
        margin = top2[0] - top2[1] if len(top2) > 1 else 1.0
        if not agree and margin > 0.15:
            over_margin[rec["field"]] = over_margin.get(rec["field"], 0) + 1
        detail.append({"ticket": rec["ticket"], "field": rec["field"], "branch_nodes": len(nodes),
                       "reference": ref, "engine": eng, "argmax_agree": agree, "ref_top2_margin": margin})

    fields = {}
    for name, diffs in abs_diffs.items():
        fields[name] = {
            "median_abs_diff": statistics.median(diffs),
            "max_abs_diff": max(diffs),
            "argmax_agreement": statistics.mean(argmax_agree[name]),
            "disagreements_over_margin": over_margin.get(name, 0),
            "argmax_abs_diff_median": statistics.median(argmax_diffs[name]),
        }
    passed = all(v["median_abs_diff"] <= 0.01 and v["disagreements_over_margin"] == 0 for v in fields.values())
    dep = dependent_check(records)  # --after dumps only; None otherwise, and the output is as before
    if dep is not None:
        passed = passed and dep["pass"]

    out = {"preset": args.preset, "tag": args.tag, "n_records": len(records), "fields": fields, "pass": passed,
           "detail": detail}
    if dep is not None:
        out["dependent"] = dep
    if not args.offline:
        client.write(replay_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(fields, indent=2))
    print(f"pass: {passed}")
    print(f"wrote {out_path}")


# ---- row-cap split (spec 12.3) ----

def branch_decodes(resp):
    """Branch-phase decodes of one response: timings.decodes minus one trunk decode per round and the prefix decode of a
    request that built its prefix (usage.cached_tokens 0; prefix and trunks stay under n_batch tokens here)."""
    t = resp["timings"]
    return t["decodes"] - t["rounds"] - (1 if resp["usage"]["cached_tokens"] == 0 else 0)


def answer_entries(responses):
    """sp3_metrics entries (state, field, {key: probability}) over every result of every response, in order."""
    out, i = [], 0
    for resp in responses:
        for res in resp["results"]:
            for name, a in res["answers"].items():
                if "p_true" in a:
                    out.append((i, name, {"true": a["p_true"], "false": 1.0 - a["p_true"]}))
                elif isinstance(a["probabilities"], dict):
                    out.append((i, name, dict(a["probabilities"])))
                else:
                    out.append((i, name, {str(k): p for k, p in enumerate(a["probabilities"])}))
            i += 1
    return out


def split_summary(fp_batched, fp_single, tree_batched, tree_single):
    """Per mode the batched answers against the one-state answers (the reference) under the statistic; the full-path
    batched lines' branch decodes; both statistics side by side per field."""
    from sp3_metrics import compare as statistic  # sp3_metrics imports this module, so not at the top

    fp = statistic(answer_entries(fp_single), answer_entries(fp_batched))
    tree = statistic(answer_entries(tree_single), answer_entries(tree_batched))
    lines = [{"states": len(r["results"]), "rounds": r["timings"]["rounds"], "decodes": r["timings"]["decodes"],
              "prefix_built": r["usage"]["cached_tokens"] == 0, "branch_decodes": branch_decodes(r),
              "rows_read": r["usage"]["scored_tokens"], "states_per_round": r["engine"]["states_per_round"]}
             for r in fp_batched]
    split = all(l["rounds"] == 1 and l["branch_decodes"] >= 2 for l in lines)
    single = [branch_decodes(r) for r in fp_single]
    single_unsplit = all(d == 1 for d in single)  # the reference lines decode their branch phase in one chunk
    keys = ("median_abs_diff", "max_abs_diff", "argmax_disagreements", "disagreements_over_margin", "pass")
    side = {name: {"full_path": {k: fp["fields"][name][k] for k in keys}, "tree": {k: tree["fields"][name][k] for k in keys}}
            for name in fp["fields"]}
    return {"full_path": {"vs_b1": fp, "lines": lines, "split": split, "single_branch_decodes": single,
                          "single_unsplit": single_unsplit},
            "tree": {"vs_b1": tree, "branch_decodes": [branch_decodes(r) for r in tree_batched]},
            "side_by_side": side, "pass": fp["pass"] and split and single_unsplit}


def cmd_split(args):
    texts = [json.loads(l)["text"] for l in args.test_set.read_text().splitlines() if l.strip()][: args.n]
    out_dir = _out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for scoring in ("full_path", "tree"):
        for batch in (args.batch, 1):
            bodies = [request_body(args.preset, None, b, scoring) for b in chunks(texts, batch)]
            responses = run_decide(decide_command(args.model, args.preset, args.decide_seqs, dump=False), bodies)
            (out_dir / f"split-{args.tag}-{args.preset}-{scoring}-b{batch}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in responses))
            runs[scoring, batch] = responses
    out = {"preset": args.preset, "model": args.model, "decide_seqs": args.decide_seqs,
           "row_cap": args.decide_seqs + 1,  # llama_n_seq_max in llama-decide: the slot sequence + --decide-seqs
           "n_states": len(texts), "batch": args.batch,
           **split_summary(runs["full_path", args.batch], runs["full_path", 1], runs["tree", args.batch], runs["tree", 1])}
    out_path = out_dir / f"split-{args.tag}-{args.preset}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"side_by_side": out["side_by_side"],
                      "branch_decodes": [l["branch_decodes"] for l in out["full_path"]["lines"]],
                      "split": out["full_path"]["split"], "pass": out["pass"]}, indent=2))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--preset", required=True)
    d.add_argument("--n", type=int, default=20)
    d.add_argument("--test-set", type=Path, default=TEST_SET)
    d.add_argument("--schema", type=Path, default=None, help="request body whose fields/instructions replace the triage schema")
    d.add_argument("--tag", default="", help="output name part: reference-<tag>-<preset>")
    d.add_argument("--decide-seqs", type=int, default=9, help="llama-decide --decide-seqs (default 9)")
    d.add_argument("--scoring", choices=["tree", "full_path"], default="tree", help="options.scoring (default tree)")
    d.add_argument("--after", action="append", default=[], metavar="FIELD=PARENT[,PARENT]",
                   help="set a field's `after` (repeatable)")
    d.add_argument("--states", type=Path, default=None,
                   help="JSONL of state specs (a string, or a list of {\"text\": T} / {\"image\": PATH} items) instead of the tickets")
    d.add_argument("--order-debias", type=int, default=0, help="options.order_debias (default 0: not sent)")
    d.add_argument("--temperature", type=float, default=None, help="options.temperature (default: not sent)")
    d.add_argument("--batch", type=int, default=1, help="states per request line (default 1)")
    d.add_argument("--out-dir", type=Path, default=Path("results-engine"),
                   help="output directory (default results-engine; relative: from the project directory)")
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("compare")
    c.add_argument("--preset", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8097")
    c.add_argument("--n", type=int, default=None)
    c.add_argument("--tag", default="", help="output name part: reference-<tag>-<preset>")
    c.add_argument("--out-dir", type=Path, default=Path("results-engine"),
                   help="directory of the dump and the result (default results-engine)")
    c.add_argument("--offline", action="store_true",
                   help="answer every request from the recorded replay reference-[TAG-]<preset>-replay.jsonl.gz, no server")
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("split")
    s.add_argument("--model", required=True)
    s.add_argument("--preset", required=True)
    s.add_argument("--decide-seqs", type=int, required=True)
    s.add_argument("--tag", required=True, help="output name part: split-<tag>-<preset>")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--batch", type=int, default=5, help="states per batched request line (default 5)")
    s.add_argument("--test-set", type=Path, default=TEST_SET)
    s.add_argument("--out-dir", type=Path, default=Path("results-engine"),
                   help="output directory (default results-engine; relative: from the project directory)")
    s.set_defaults(func=cmd_split)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
