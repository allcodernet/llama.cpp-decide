# llama.cpp-decide

A one-pass decision endpoint for llama.cpp: `POST /v1/decide` lets an ordinary open-weights chat model (GGUF) answer
many typed questions about a text or an image (choice, score, yes/no) with a full probability distribution per
question, from one batched scoring pass and without generating text. The answer is chosen among the allowed values by
reading next-token probabilities, so the output always matches the schema. It ships as a patch series on a pinned
upstream llama.cpp commit, with evaluation scripts, a synthetic triage dataset and the recorded results.

It started as a local reproduction of the "System One" decision model idea of TypeSafe AI's **Jev**, and also serves
the System One request shape (`POST /v1/systemone`). This project is not affiliated with TypeSafe AI / Jev or with the
llama.cpp authors.

## How it works

```
system: instructions + field catalogue      <- rendered once, KV cache kept (prefix P)
user:   <state text or image>               <- per state: fork P -> trunk, prefill the state
assistant: {                                <- trunk ends with "{\n"
  "queue": "billing" | "technical" | ...    <- per field: fork trunk -> branch, place the
  "urgency": 0 | 1 | 2 | 3                     field name, read the logits of the option
  "angry": true | false                        tokens, softmax over the options only
}
```

Per request the engine decodes the prefix when it is not cached, then per round one batched trunk decode (state +
assistant tail for every state of the round) and one batched branch decode (every branch node of every state). Fields
do not see each other unless a field declares `after` (it is then scored after its parent fields and sees their
answers). llama.cpp's batch splitter groups sequences by length; nothing is padded by hand.

Checked on Qwen2.5, Qwen3, Qwen3.5 (dense and hybrid recurrent), Qwen3.6, Qwen3.8 and Gemma 4 (sliding window), and
on MoE models with the experts on the CPU; other families are untested. SentencePiece vocabularies with
`add_space_prefix` (Llama 2, Mistral 7B v0.x and similar) are rejected at startup: the engine tokenises prefix, state,
tail and branch separately and checks that the pieces join the way the whole prompt would tokenise.

## Requirements

- Linux, git, CMake, a C++17 compiler; for the GPU build the CUDA toolkit (the author used CUDA 13.3 with `g++-15` on
  an RTX 3090 and an RTX 3070 Ti laptop GPU). `BUILD_CUDA=0` builds for the CPU.
- [uv](https://docs.astral.sh/uv/) for the Python scripts (Python 3.12+, dependency `httpx`; tests use `pytest`).
- Models in GGUF format. Nothing is downloaded automatically; the quick start names the files.
- VRAM: every decide sequence holds KV cells, and hybrid models (Qwen3.5, Qwen3.8) also hold one recurrent state per
  sequence (about 156 MiB each on Qwen3.8-27B), so size `decide-seqs` to the GPU.

## Quick start

```bash
git clone https://github.com/allcodernet/llama.cpp-decide && cd llama.cpp-decide
./build-engine.sh          # clones upstream llama.cpp 60b06ab9a into engine/, applies patches/decide/*.patch with git am,
                           # builds with CUDA (see "Build settings"); BUILD_CUDA=0 for a CPU-only build
git -C engine rev-parse HEAD^{tree}    # must print c3daa53ab47099be8a69e6d81749f52072768fa7 (the patched tree)
./test-engine.sh           # the 11 decide ctests, no model needed
uv run pytest -q           # the Python tests, no model or server needed (the SDK test is skipped)
```

`git am` gives new commit ids on every machine, so compare the tree id, not the commit id. The commit ids quoted in the
reports name the development builds the runs were made on (`REPORT-ENGINE.md`, Setup, "Commit ids in this report").

### A text request

Any chat GGUF works; the small Qwen2.5-0.5B is enough to see it run. The presets files `models-engine.ini` and
`models-engine-3090.ini` are the author's machine presets: their absolute model paths (`/home/<user>/.lmstudio/...`)
must be edited to where your files are, and relative paths resolve from the project directory. The quick-start model
is named there under `./models/`, so it needs no edit:

```bash
mkdir -p models
curl -L -o models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf
ENGINE_PRESETS=models-engine-3090.ini ./serve-engine.sh       # separate terminal: llama-server on 127.0.0.1:8097
curl -s http://127.0.0.1:8097/v1/decide -H 'Content-Type: application/json' -d '{
  "model": "qwen2.5-0.5b",
  "instructions": "Answer each question about this support ticket.",
  "fields": {
    "queue": {"type": "choice", "description": "Which team must act first?",
              "options": {"billing": "Payments, refunds, invoices", "technical": "Bugs, crashes, outages"}},
    "angry": {"type": "bool", "description": "The text itself shows anger."}},
  "states": ["I was charged twice this month. Fix it NOW."]}'
```

The router loads a preset on the first request that names it. The answer carries `value`, `probabilities` (or
`p_true`) and `confidence` per field. A 0.5B model only shows the mechanics: on the bundled test set it gets 33.8% of
the queues right (below); the recorded results use Qwen3.5-9B, Gemma 4 E4B and Qwen3.8-27B.

### An image request

Image states need a vision model and its projector (`--mmproj`, preset key `mmproj`). The recorded image results use
Qwen3.8-27B (IQ3_S, about 13 GB, plus the BF16 projector) on a 24 GB GPU: download
`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` and `mmproj-Qwen3.8-27B-BF16.gguf` from
https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF and put their paths into the `model` and `mmproj` lines of
the `[qwen3.8-27b-vision]` preset in `models-engine-3090.ini` (the paths there are the author's). With the server from
above still running:

```bash
curl -L -o cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg     # COCO val2017 39769, CC BY-SA 2.0 (Flickr)
uv run python - <<'EOF'
import httpx
from decide_client import image_part
body = {"model": "qwen3.8-27b-vision", "instructions": "Answer each question about this image.",
        "fields": {"cat": {"type": "bool", "description": "Is there a cat in this picture?"}},
        "states": [[{"type": "text", "text": "A customer sent this photo: "}, image_part("cats.jpg")]]}
print(httpx.post("http://127.0.0.1:8097/v1/decide", json=body, timeout=900).json()["results"][0])
EOF
```

### The bundled data and the evaluation scripts

`data/` holds the synthetic support-ticket set every triage number comes from (80 test, 320 train tickets;
provenance and labels in `data/README.md`). With the server running:

```bash
ENGINE_PRESETS=models-engine-3090.ini uv run decide_client.py --model qwen2.5-0.5b --batch 5 --out-dir /tmp/qs
uv run score.py --preds-dir /tmp/qs/preds       # accuracy, urg±1, ECE, Brier, NLL, p50 latency
```

`decide_client.py` checks the model's thinking-off assistant prefix against `EXPECTED_PREFIX` (a new model needs an
entry there) and refuses to overwrite existing outputs (`--force`). The other scripts: `reference_check.py` (replays
the engine's token sequences through `/completion` and compares), `batch_invariance.py`, `calibrate.py` (fits a
temperature on a train run), `heldout_stats.py`, `sp3_metrics.py`, `systemone_smoke.py`; each has its usage in its
docstring, and `REPORT-ENGINE.md`, "How to reproduce", lists the commands of every recorded result.

## API

### `POST /v1/decide`

Enabled when the server runs with `--decide-seqs N` (preset key `decide-seqs`): N extra sequences, above the normal
slots, for trunks and branches.

```json
{
  "model": "qwen3.5-9b",
  "instructions": "Answer each question about this support ticket.",
  "fields": {
    "queue":   {"type": "choice", "description": "Which team must act first?",
                "options": {"billing": "Payments, refunds, invoices, charges", "technical": "Bugs, crashes, outages"}},
    "urgency": {"type": "score", "description": "How urgent is this support ticket?",
                "levels": ["Can wait", "Normal", "Today", "Drop everything"]},
    "angry":   {"type": "bool", "description": "The text itself shows anger."}
  },
  "states": ["ticket text 1", "ticket text 2"],
  "options": {"scoring": "tree", "order_debias": 0, "temperature": 1.0}
}
```

| type | keys | answer |
|---|---|---|
| `choice` | `options`: object key → description, or array of keys (2–255) | `value` (argmax key), `probabilities` by key, `confidence` |
| `score` | `levels`: array of level descriptions (2–10) | `value` (argmax index), `expected` (Σ i·p_i), `probabilities` in level order, `confidence` |
| `bool` | — | `value` (`p_true >= 0.5`), `p_true`, `confidence` |

- `description` is required; `after` (array of field names) makes a field sequential (below). `states`: 1–256
  entries, each a non-empty string or an array of text and image parts (below). `model` is required in router mode.
- Response: `results[i]` answers `states[i]` (`answers`, `usage`); the request's `usage` (`prompt_tokens`,
  `cached_tokens`, `state_tokens`, `scored_tokens`), `timings` (`prefill_ms`, `scoring_ms`, `total_ms`, `rounds`,
  `decodes`, …) and `engine` (`scoring`, `seqs_per_state`, `states_per_round`).
- Errors: 400 `{"error": {"message", "code"}}` with `invalid_schema`, `invalid_options`, `invalid_states`, `budget` (a
  state, the catalogue or the number of decide sequences does not fit; the message names the numbers),
  `vision_unavailable`; 500 `engine`; 404 `disabled` without `--decide-seqs`. A request is answered whole or not at all,
  and a 400 leaves the engine as it was.
- `GET /v1/decide/info` reports `decide_seqs`, `n_ctx`, memory type (recurrent, sliding window and the SWA cache
  size), the resident prefixes, the prefix cache, the temperature, the scoring modes and `vision`.
- `llama-decide` (`engine/build/bin/llama-decide`) is the same engine without HTTP: one request per line on stdin
  (the body without `model`), one response per line; `--info`, `--dump-tokens` (the token sequences for
  `reference_check.py`), `--mmproj`:

```bash
engine/build/bin/llama-decide -m ./models/Qwen2.5-0.5B-Instruct-Q8_0.gguf --decide-seqs 9 -c 4096 -ngl 99 -fa on \
    < tests/data/decide-triage.json
```

### Engine switches

Per-request switches of `POST /v1/decide` and server flags (spec
`docs/specs/2026-09-25-engine-improvements-design.md`). A request without them gets the same answers as
before the switches existed when its prefix is resident or built into the same cells (a prefix restored from the
cache, or rebuilt after other cells were allocated, can land in other KV cells, which moves placement-sensitive models
such as Qwen2.5-0.5B), except on the `gemma-4-e4b` preset of `models-engine.ini`, which applies T = 1.6947 (same
argmax, tempered probabilities; `"options": {"temperature": 1}` returns the untempered ones). Measured on the triage
tickets (test 80 + train 320, batch 5, `--decide-seqs 21`, RTX 3070 Ti laptop, Qwen3.5-9B / Gemma 4 E4B); details in
`REPORT-ENGINE.md`, section "Engine improvements (sub-project 3)".

| switch | how | default | effect on the triage tickets | throughput vs tree | recommendation |
|---|---|---|---|---|---|
| full-path scoring | `"options": {"scoring": "full_path"}` | `"tree"` | scores each option's whole token path including the end of the value, so option keys may share token prefixes (`sale`/`sales`, a 400 in tree mode); adds `coverage`. No significant accuracy difference from tree (sign tests p = 1; 7 / 2 of 1200 field decisions change), coverage median ≥ 0.9969 | 0.34x / 0.20x | off; use for option keys that share prefixes |
| sequential fields | field `"after": ["queue"]` | none | the field is scored after its ancestors and sees their answers; adds `given`. Urgency after queue: Qwen3.5-9B +17 of 400 urgency answers right (unadjusted p = 0.0046, Holm 0.10: not significant over the 24 sign tests; 78.5% → 82.8% over the 400 tickets, test 78.8% → 85.0%, train 78.4% → 82.2%), Gemma +5 (p = 0.30); urgency NLL over the 400 tickets: 9B 0.548 → 0.542, Gemma 0.534 → 0.570 | 0.66x / 0.58x (one dependent field), 0.52x / 0.47x (two) | where a gain is observed on labelled tickets |
| prefix cache | `--decide-prefix-cache N` (preset key `decide-prefix-cache`) | 8 | a prefix of another schema, prompt or debias variant is restored instead of decoded again; per request with two alternating prompts (`keywords` / `default`): Qwen3.5-9B 1252 → 692 / 1001 → 690 ms, Gemma 326 → 150 / 252 → 149 ms | host RAM per snapshot: 60.5 MB / 13.4 MB | on (default) |
| option-order debiasing | `"options": {"order_debias": K}` | 0 (off) | averages K catalogues with rotated choice options; adds `order_spread`. No significant accuracy difference (p ≥ 0.5; 8 / 7 of 1200 field decisions change); test summed NLL −0.030 / −0.009 at K = 4 | 0.19x / 0.13x (K = 4) | off; a diagnostic of option-order sensitivity |
| temperature | `"options": {"temperature": T}`; `--decide-temperature T` (preset key `decide-temperature`) | 1.0 | p ∝ p^(1/T), every argmax unchanged. Preset `gemma-4-e4b`: T = 1.6947 (fitted on the 320 train tickets): test summed NLL 0.909 → 0.780, angry ECE 0.031 → 0.041 | no cost | per preset, by the calibration rule (Qwen3.5-9B: none) |

Full-path rows got faster later: the full-vocabulary log-sum-exp went from 0.753 ms to 0.163 ms (AVX2) per
262144-token row, and Gemma's scoring went from 131.7-136.5 ms to 85.2-87.2 ms per 5-state request at
`--decide-seqs 64` on the RTX 3090 (`results-engine/followups/task-1/notes.md`). The 0.34x / 0.20x above were measured
before that change and were not re-measured on the laptop.

`decide_client.py` sets them with `--scoring full_path`, `--after FIELD=PARENT[,PARENT]` (repeatable), `--order-debias K`
and `--temperature T` (output names get `-full_path`, `-after-<field>=<parent>`, `-debias<K>`, and `-t<T>` when the
temperature the engine applied is not 1.0, e.g. `-t1.6947` for a Gemma run without `--temperature`); `--ensure-t1` runs
at T = 1 whatever the preset, `--force` overwrites existing outputs. `calibrate.py` fits a model's temperature on a
stored train run (`results-engine/sp3/eval/calibration-<preset>.json`).

### Image input

A `/v1/decide` state can be an ordered list of text and image parts (spec
`docs/specs/2026-10-02-image-input-design.md`). The server needs the model's projector: `--mmproj` (preset
key `mmproj`, optionally `image-min-tokens` / `image-max-tokens`); `llama-decide` takes the same flags.

```json
{"model": "qwen3.8-27b-vision",
 "instructions": "Answer each question about this image.",
 "fields": {"cat": {"type": "bool", "description": "Is there a cat in this picture?"}},
 "states": [
   [{"type": "text", "text": "A customer sent this photo: "},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}}],
   "A plain text state, answered as before."]}
```

- Parts: `{"type": "text", "text": "<non-empty>"}` and `{"type": "image_url", "image_url": {"url": "data:image/<subtype>;base64,<data>"}}`,
  no other keys (OpenAI's `detail` is rejected with 400 `invalid_states`), in any order; two text parts must not be
  adjacent. A string `S` and `[{"type": "text", "text": S}]` are the same state. Images belong to states only (not to
  the instructions or field descriptions).
- URL: `data:image/` is matched case-sensitively (`DATA:` or `image/JPEG` is a 400), `<subtype>` uses only `a-z 0-9 . + -`,
  and no further parameters are allowed. Images: JPEG or PNG (recognised by their bytes, not by `<subtype>`), standard
  padded base64 without whitespace; no `http(s)` or `file:` URLs or paths. Limits: 4 images per state, 32 images per
  request, 10 MiB and 32,000,000 pixels per image, and an image's longer side at most 200 times its shorter side (a
  projector pads the short side of a narrow strip up to its patch size, so a 16,000,000 x 2 PNG within the pixel limit
  would have become gigabytes of preprocessed pixels).
- Errors: 400 `invalid_states` (`field` names the state; also an image over 200:1), 400 `vision_unavailable` when the
  server has no usable projector, 400 `budget` (the message names the image cells; a state's cells are checked again
  right after each of its images is tokenised, so a state that is already over the limit loads no further image; too
  few decide sequences for the schema is checked before any image is loaded), 500 `engine` when libmtmd cannot
  tokenise a decoded image, the projector fails to encode it, or an image decode fails with a code other than 1. A
  state that is neither a string nor an array gets 400 `invalid_states` "states[i]: states entries must be strings or
  arrays of parts" with `field` `states[i]`.
- `GET /v1/decide/info` (and `llama-decide --info`) reports `vision`: `{"available": true, "mrope": …, "non_causal": …,
  "max_images_per_state": 4, "max_images_per_request": 32, "max_image_bytes": 10485760, "max_image_pixels": 32000000,
  "max_image_aspect_ratio": 200}` or `{"available": false, "reason": "no_projector" | "projector_without_vision" | "non_causal_projector"}`.
- Response: an image state's `usage` adds `images`, `image_cells` and `image_positions`; `timings` adds
  `image_encode_ms` and `image_decode_ms`. A request without images gets the same response keys and values as a text
  request before image support, apart from two error cases: the message for a state that is neither a string nor an
  array (above) and the SWA budget (below).
- Every engine switch works with image states. With `order_debias` K the image is decoded K times; each variant and
  each retry encodes it again (there is no image cache).
- Projectors with non-causal image attention (Gemma 3): not available (`non_causal_projector`); gemma-3-4b failed the
  check that the engine's answers for an image do not depend on the round's layout.
- Models with sliding-window attention layers (Gemma 3, Gemma 4): every cell limit is `min(n_ctx, SWA cells) − 16`,
  where the SWA cache holds `min(n_ctx, n_swa × sequences + n_ubatch)` cells padded to a multiple of 256, or `n_ctx`
  with `--swa-full` (preset key `swa-full`); "sequences" is `n_seq_max`, the decide sequences plus the server slots.
  Where that cache is smaller than `n_ctx` (Gemma 4 E4B at ctx 16384: n_swa 512 × 17 sequences (16 decide sequences +
  1 server slot) + n_ubatch 512 = 9216 cells), a state over it, text or image, gets 400 `budget` naming the SWA cache
  and `--swa-full`; before image support it passed the budget and ended in 500 `engine`. Where the cache has `n_ctx`
  cells (Gemma 3 at its preset, Gemma 4 E4B at ctx 4096) nothing changes. info reports `memory.swa_cells` for such
  models.
- Without `--mmproj` the server and `llama-decide` start and run as before; image requests get `vision_unavailable`.
  The environment variables `LLAMA_ARG_MMPROJ`, `LLAMA_ARG_IMAGE_MIN_TOKENS` and `LLAMA_ARG_IMAGE_MAX_TOKENS` set the
  same options for `llama-decide` (and are accepted, unused, by `llama-completion`): unset them for a text-only CLI run.
  `llama-completion --help` lists `--mmproj`, `--image-min-tokens` and `--image-max-tokens`, which it ignores.
- Vision presets in `models-engine-3090.ini`: `qwen3.8-27b-vision`, `smolvlm-500m`, `gemma-3-4b-vision`,
  `qwen3.6-35b-a3b-vision`, `gemma-4-e4b-vision` (the substitute for `smolvlm-500m`, which the join check rejects), and
  the text presets `gemma-4-e4b-16k` and `gemma-4-e4b-16k-swa-full` (Gemma 4 E4B at ctx 16384 without and with
  `swa-full`, for the SWA budget checks).
- Checked live: Qwen3.8-27B (binding; every integrity check passes, `results-engine/image/final/`), Gemma 4 E4B and
  Qwen3.6-35B-A3B (non-binding; one kept failure each). Gemma 3 is not supported (above). No model that ran splits an
  image into several image chunks (SmolVLM, the model for that case, could not run), so that path has no live check.
- Client side: `decide_client.image_part(path)` builds the part; `reference_check.py dump --states FILE` and `compare`
  replay image states through `/completion` (servers started with `LLAMA_MEDIA_MARKER='<__media__>'`).
- Results and limits: `REPORT-ENGINE.md`, "Image input (sub-project 4)".

### Jev compatibility: `POST /v1/systemone`

The engine server also answers TypeSafe AI's System One shape
(`{"state", "model", "questions": {id: noul | choice | score}}` → `{"model", "answers", "usage"}`),
so unmodified Jev clients work against it, within the limits listed below. The official Python SDK only
needs `base_url` and a preset name as `model` (any API key, unless the server runs with `--api-key`):

```python
from typesafe_sdk import TypeSafeClient, Choice, RetryPolicy
TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8097", timeout=180,
               retry=RetryPolicy(max_retries=0)).system_one(
    "My password reset email never arrives.", {"team": Choice(criteria={"billing": "Payments", "technical": None})}, model="qwen3.5-9b")
```

`retry=RetryPolicy(max_retries=0)` matters: the SDK's default policy retries 5xx responses and
client timeouts twice, which re-sends the whole request while a cold model is still loading.

The request is translated to a `/v1/decide` body (spec
`docs/specs/2026-09-25-systemone-compat-design.md`, section 3); `systemone_smoke.py`
rebuilds that body in Python and checks that both endpoints give identical probabilities.
SDK acceptance test (not a project dependency; skipped without `SYSTEMONE_URL`):

```bash
git clone https://github.com/typesafe-ai/typesafe-sdk-python local/typesafe-sdk-python    # tested at v0.7.0 (2ce5c65)
SYSTEMONE_URL=http://127.0.0.1:8097 uv run --with local/typesafe-sdk-python \
    pytest -q tests/test_systemone_sdk.py      # SYSTEMONE_MODEL=<preset>, default qwen3.5-9b
```

Known differences from Jev (documented, not bugs):
- probabilities are not rounded or clipped (Jev rounds to 2 decimals and clips `noul` to [0.01, 0.99]);
- `usage.output_tokens` is always 0; `usage.input_tokens` is `prompt_tokens + cached_tokens`
  of the decide run, which also counts the branch tokens decoded for scoring;
- `model` in the response is the request's value (Jev resolves aliases such as `jev-latest` to `jev-1.13.0`);
  pass a preset name (`model="qwen3.5-9b"` or `TYPESAFE_DEFAULT_MODEL`), since the SDK default `jev-latest` is not a preset;
- `confidence` is the docs demo's approximation `clip((n·peak − 1)/(n − 1), 0, 1)`, not Jev's undisclosed definition;
  `noul` has no confidence;
- limits: 2–10 score levels, 2–255 choice options, at most 64 questions per request (Jev/SDK allow 1-level scores and
  no question limit); violations are 422 `detail` errors; the shipped presets cap the question count lower (next item);
- question cap with the shipped presets: every question needs at least one branch sequence, so with `decide-seqs = 21`
  a request holds at most 19 single-branch questions (fewer when a choice's option keys share leading tokens); more is
  a 400 `budget` (SDK `TypeSafeBadRequestError`); raise `decide-seqs` in the presets (VRAM cost on hybrid
  models such as Qwen3.5: one recurrent state per sequence);
- context: the question catalogue, the state and the answers share `ctx-size` (4096 in most presets; Jev documents
  64k), so a long state or a long question catalogue is a 400 `budget`; raise `ctx-size` in the presets;
- body limits: a raw body above 1 MiB is a 413 with a `detail` body (SDK `TypeSafeAPIError`); a `state`,
  `instructions` or criteria value nested deeper than 32 levels, and more than 2 MiB of text serialised from the
  request's values, are 422 `detail` errors;
- choice option keys whose tokens are string prefixes of each other (e.g. `sale`/`sales`) are a 422; rename one of them
  (full-path scoring lifts this for `/v1/decide`);
- question ids and the whole question catalogue are visible to the model (Jev keeps keys away from the model and
  evaluates questions in isolation); absent, null or empty `instructions` fall back to a default per type;
- only `state` is tokenised without special-token parsing; `instructions`, criteria and question ids go into the
  catalogue, where text such as `<|im_start|>` becomes a control token, so put untrusted text in `state` only;
- auth: without `--api-key` (as in `serve-engine.sh`) any key works and the `Authorization` header is ignored; with
  `--api-key` the SDK's key must match it, otherwise 401 (`TypeSafeAuthenticationError`); no 429/529;
- SDK: `response.request_id` works (every `/v1/systemone` response from our handler carries `x-typesafe-request-id`);
  `client.models.list()` fails validation against llama-server's OpenAI-style `GET /v1/models`, so take the model
  names from the presets;
- `loc` omits the question-type segment that Jev includes (ours `["body","questions",id,"criteria"]`, Jev's
  `["body","questions",id,"score","criteria"]`) and gives array indices as strings;
- router mode (`--models-preset`, as in `serve-engine.sh`): a missing or unknown `model` gets the router's own error body
  (missing: 400 `{"error":{..., "type":"invalid_request_error"}}`) and a body that is not valid JSON gets a 500
  `server_error` from the router, both before our handler runs (so without `x-typesafe-request-id`); the FastAPI-style
  422 shapes for these two cases (`loc ["body","model"]`, `json_invalid`) only hold on a single-model server;
- `budget` (state, question catalogue or question count over the context / `--decide-seqs` budget) is a 400 with a
  `detail` body; `engine` 500 and `disabled` 404 keep the `{"error":{...}}` body of `/v1/decide` (the SDK parses both
  shapes).

## Build settings, presets and environment

- `build-engine.sh`: `BUILD_CUDA=0` (CPU-only), `TARGETS="a b"` / `EXTRA_TARGETS`, `LOCAL_SEED=path` (clone upstream
  from a local llama.cpp clone that holds commit `60b06ab9a` instead of GitHub), `CUDA_BIN` (default `/opt/cuda/bin`),
  `CUDA_ARCH` (default `86`, the RTX 30 series; set yours, e.g. `89`), `CUDA_HOST_COMPILER` (default
  `/usr/bin/g++-15`); an empty value drops that setting. An existing `engine/` is left as it is and the patches are not
  applied again.
- Presets: `serve-engine.sh` starts `llama-server --models-preset $ENGINE_PRESETS --models-max 1 --port 8097`.
  `models-engine.ini` (default) holds the presets of the 8 GB laptop runs and `models-engine-3090.ini` those of the
  24 GB runs. They are the author's examples: relative `./models/` paths resolve from the project directory, and the
  absolute paths (`/home/<user>/.lmstudio/models/...`) must be edited to where your files are; llama-server does not
  expand `~` or variables there. `decide_client.py` and `reference_check.py` read the same file.
- Settings from the environment (relative paths from the project directory):

| variable | default | used by |
|---|---|---|
| `TRIAGE_DATA` | `data` (the bundled set) | the test and train sets of every script (`DATA` in the run scripts still wins) |
| `ENGINE_PRESETS` | `models-engine.ini` | `serve-engine.sh`, `decide_client.py`, `reference_check.py` |
| `ENGINE_BIN` | `engine/build/bin` | `serve-engine.sh`, `reference_check.py` |
| `MODELS`, `EVAL_DIR`, `BASELINE_DIR`, `COST_DIR` | the sub-project 3 presets and directories | `results-engine/sp3/eval/run.sh` (also its `baseline` step) and `analysis.py` |
| `GPU_VRAM_MAX`, `GPU_WAIT_MIN` | 200 MiB, 0 min | `gpu_ready` of the run scripts (they wait for a GPU without a `llama-server`) |
| `CUDA_BIN`, `CUDA_ARCH`, `CUDA_HOST_COMPILER` | `/opt/cuda/bin`, `86`, `/usr/bin/g++-15` | `build-engine.sh` (empty: setting dropped) |

- The run scripts under `results-engine/` (`sp3/eval/run.sh`, `sp3/integrity/run.sh`, `3090/run.sh`, `image/run.sh`)
  start and stop their own server on 127.0.0.1:8097 and write into the committed result directories unless their
  header says how to redirect them; read the header before running one. The image checks also need the four COCO
  integrity images under `local/image/integrity/` (`REPORT-ENGINE.md`, "How to reproduce").
- `remote-sync.sh` keeps a copy of the repository and of `engine/` on a second (GPU) machine without a git remote:
  `REMOTE=user@gpu-host REMOTE_DIR=/abs/path ./remote-sync.sh push|pull|run …` (both must be exported); its header
  describes the safety checks.

## What is in the repository

- `patches/decide/` — the engine: 45 patches on ggml-org/llama.cpp `60b06ab9a` (new code under `tools/decide/`, the
  `/v1/decide`, `/v1/decide/info` and `/v1/systemone` routes in `tools/server/`, `--decide-seqs` and friends in
  `common/`, ctests under `tests/`). `build-engine.sh`, `test-engine.sh`, `serve-engine.sh`.
- Phase 1: the baseline fork, a third-party llama.cpp fork with a `POST /v1/decision` endpoint, measured in an earlier
  phase of this work; neither the fork nor the scripts that ran it are part of this repository. That phase ran the 80
  synthetic test tickets, three questions each, through the fork; the numbers survive as the baseline rows of
  `REPORT-ENGINE.md` and the files under `results/`. The engine here was built separately afterwards.
- Phase 2 and its sub-projects: the own engine (parity with the baseline, integrity checks against `/completion`), Jev
  compatibility, the engine switches, the follow-ups (faster full-path rows, fault modes for the 500 paths, a test
  machine with an RTX 3090), Qwen3.8-27B, and image input. All results: `REPORT-ENGINE.md` (sections per sub-project,
  "How to reproduce" at the end) and `results-engine/`.
- `docs/specs/` — the five design documents. The code's "spec N" and "spec N.M" comments refer to their
  sections; "Task N" in comments, notes and result directories (`task-N/`) names the step of the work in which a
  result was produced.
- `data/` — the synthetic triage set (`data/README.md`). `tests/` — the Python tests (no model needed).
- Recorded result files (JSON, JSONL, logs under `results/` and `results-engine/`) are kept as recorded, apart from
  mechanical renames of a few labels and keys with every value unchanged (listed in `REPORT-ENGINE.md`); they contain
  absolute paths of the author's machines (home directory, model files) and the names "laptop" (RTX 3070 Ti, 8 GB) and
  "test machine" (RTX 3090, 24 GB).

## Results in short

Phase 1, the baseline fork, 80 synthetic support tickets, 3 questions each, RTX 3070 Ti laptop GPU:

| Model / method | Queue acc. | Urgency acc. | Angry acc. | p50 per ticket |
|---|---|---|---|---|
| Qwen3.5-9B, one-pass /v1/decision, batch 1 | 92.5% | 77.5% | 97.5% | 99 ms |
| Qwen3.5-9B, one-pass, batch 5 | 95.0% | 78.8% | 97.5% | 89 ms |
| Gemma 4 E4B (4.6 GB), one-pass, batch 1 | 90.0% | 81.2% | 95.0% | 60 ms |
| Gemma 4 E4B, one-pass, batch 5 | 92.5% | 81.2% | 95.0% | 32 ms |

- The own engine ran the same tickets at 0.99-1.21x the baseline fork's p50 latency with its own prompt format (accuracy
  differs from the baseline's; with the first prompt, Qwen3.5-9B batch 5 queue was 88.8% against 95.0%), and passes an
  independent reference check: its token sequences replayed through llama.cpp's `/completion` give the same
  distributions under a pre-set statistic (`REPORT-ENGINE.md`).
- A held-out check on the 320 train tickets picked the `keywords` prompt variant (queue 94.7 / 94.4% vs 89.7 / 89.4%
  for `default` on Qwen3.5-9B / Gemma 4 E4B, sign test p = 3.1e-05 / 1.4e-04), now the client default.
- Qwen3.8-27B (IQ3_S) on an RTX 3090: test set queue 98.8%, urgency 88.8%, angry 100% (tree, batch 5), p50 101.3 ms
  per ticket at batch 5, every integrity check passes, `/v1/systemone` smoke and SDK test pass; its preset applies
  the calibrated `decide-temperature = 0.8992` (`results-engine/3090/notes.md`).
- Images, Qwen3.8-27B: on 200 COCO val2017 images with five yes/no questions each, the test half had 2 of 487 decide
  answers wrong against 15-17 (strict) or 5-6 (by `p_yes`) for a chat baseline asking one question per call. The
  strict gap is mostly a format effect of the 1-token chat baseline (answers that are neither yes nor no), and the
  counts are too small to call decide more accurate; read by the model's own yes/no preference the two are close. The
  claim is speed at comparable accuracy on this model: one five-field decide request takes 1.32 s per image (median)
  against 6.66-6.81 s for the five chat calls (`REPORT-ENGINE.md`, "Image input (sub-project 4)").
- No training anywhere: the models answer as they come. The data is synthetic and small (80 test tickets: 2.5 points
  are 2 tickets); small differences are not significant.

Known limitations (details in `REPORT-ENGINE.md`, "Limits and open points" and the sections per sub-project): fields
are independent unless `after` is used; option keys that share leading tokens need full-path scoring; Gemma-3-style
projectors are not supported; there is no image cache and no body-size cap for `/v1/decide` beyond the HTTP library's
default; untested model families may fail the tokenisation join check at startup.

## Related work

Checked on 2026-10-02. One-pass typed decisions on open models are an active area; this repository is not the first
or the only implementation, and the upstream work below may supersede parts of it.

- llama.cpp [#29752](https://github.com/ggml-org/llama.cpp/pull/29752) (open): `POST /v1/decision` in llama-server,
  many contexts per request, one branch per field with `llama_memory_seq_cp`, trie scoring.
- llama.cpp [#29818](https://github.com/ggml-org/llama.cpp/pull/29818) (merged 2026-10-02): `/v1/systemone`
  for trained decision models (Laya, Julia-1, Lev, OpenJev with vision, Kev) through a decision head on the graph.
- llama.cpp [#29832](https://github.com/ggml-org/llama.cpp/pull/29832) (draft; its description says it is a
  demonstration with no intent to merge yet): a model-agnostic `/v1/systemone` for ordinary chat models, with image, audio and video input.
- llama.cpp [#29321](https://github.com/ggml-org/llama.cpp/pull/29321) (draft): `llama-system-one`, a typed
  decision readout library and CLI.
- [aakash-chaddha/llama.cpp](https://github.com/aakash-chaddha/llama.cpp): a llama.cpp fork with
  `/v1/decision`, image and audio contexts and a media encoder cache.
- [SGLang](https://github.com/sgl-project/sglang): `/v1/decisions` turns a chat model into a decision model, and
  `POST /v1/systemone` serves the same decisions in the System One request shape
  ([docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/supported-models/decision_models.mdx)).
- [Ollama](https://github.com/ollama/ollama): `/v1/systemone` with decision models such as Nimble
  ([docs](https://github.com/ollama/ollama/blob/main/docs/capabilities/decision.mdx)).
- [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev): a curated list of projects built around Jev and
  System One (also mirrored as the fork vicfei/awesome-jev-2).

Image input exists elsewhere too (#29818 for OpenJev, #29832, the aakash-chaddha fork), and so does a `/v1/systemone`
route (#29818, #29832, SGLang, Ollama); SGLang also documents a per-request temperature on the label logits and a
replay of the scored ids through its `/v1/score`. Not found elsewhere as of 2026-10-02 (a search of the projects above,
not an exhaustive one): fields conditioned on earlier fields (`after`) inside the batched shared-KV engine; option-order
debiasing in the server; a published integrity and calibration evaluation that goes with a `/completion` replay check
of the engine's token sequences; and that combination running on hybrid, sliding-window and MoE models. It is a
research implementation measured on a small synthetic set, not a product.

## Licence

MIT (`LICENSE`). The patches modify llama.cpp (MIT, © The ggml authors); third-party material, the
COCO images and the System One API are described in `THIRD-PARTY.md`.
