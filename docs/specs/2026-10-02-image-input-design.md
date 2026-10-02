# Sub-project 4: image input for `/v1/decide` — design

Status: rev 7 (rev 3 approved on 2026-10-02; rev 4 added three clarifications: no separate part
limit, unknown keys in a part, a failed image tokenisation; rev 5 adds the SWA cache to the cell
budget, a defect that predates images and was found in §9.4; rev 6 adds the aspect-ratio limit and
the early budget check of §5.1, found in a code review of the final engine; rev 7
checks the sequence budget before the prefixes are prepared and before any image is loaded, §3.2;
rev 7 wording fix: §3.2 and §5.1 state that order once).
Parent: `2026-09-25-own-decide-engine-design.md` (sub-project 1). Siblings:
`2026-09-25-systemone-compat-design.md` (sub-project 2), `2026-09-25-engine-improvements-design.md`
(sub-project 3). Prototype: `results-engine/spikes/image/` (throwaway; numbers quoted below come
from its `notes.md`). Code references are to the engine fork `engine/` at `49785a95d` (tree
`32d63de3ad1b00c086fb4abd2b9da53631d8d043`): E = `tools/decide/decide-engine.cpp`, SC =
`decide-schema.cpp`, API = `decide-api.cpp`, CLI = `decide-cli.cpp`, SRV =
`tools/server/server-context.cpp`, M = `tools/mtmd/mtmd.h`, MH = `tools/mtmd/mtmd-helper.h`.

## 1. Goal

A state of a `/v1/decide` request may contain images. The engine answers the schema's fields about
it in one pass, with the guarantees it gives for text states.

Success criteria:

1. **Image states work.** A state is a string (as today) or an ordered list of text and image
   parts (§3). Results for image states pass the reference check and the batch-invariance check
   under the parent spec's statistic (§9.4).
2. **Every existing switch works with image states**: full-path scoring, `after`, order
   debiasing, temperature, the prefix cache (§5.5).
3. **Models without vision are unaffected** (§4). Nothing new is required to start them, text
   requests give the results of the current build, and a request with an image gets one defined
   400 before the engine touches any state. No path ends in a 500 or a crash because vision is
   missing.
4. **Text requests are unaffected when a projector is loaded.** A request without image parts
   makes no call into libmtmd and returns the response of the current build (§9.2).

Non-goals (§10): images in the instructions, remote URLs and server paths, audio and video, an
image snapshot cache, batched image decoding across states, images in `/v1/systemone`.

## 2. Facts the design relies on

Engine and server (verified by reading the code at `49785a95d`):

- **States today.** `states_of` (SC:200-214) requires 1 to 256 non-empty strings. It has three call
  sites: the HTTP thread of `/v1/decide` before the task is queued (SRV:5423), the HTTP thread of
  `/v1/systemone` (SRV:5486), and the main loop (API:24). `systemone_to_decide` produces string
  states.
- **Errors.** `error{code, http, field}`; codes `invalid_schema`, `invalid_options`,
  `invalid_states`, `budget` (400), `engine` (500), `disabled` (404).
- **Body size.** `/v1/decide` has no cap of its own; cpp-httplib's default of 100 MB applies.
- **Threads.** A decide request runs as one task on the server's main loop, between slot updates;
  it never overlaps a slot's multimodal work. Slots encode through their own mtmd batch. Sharing
  the server's `mtmd_context` with the engine is therefore safe.
- **Trunks.** The only place a state's tokens are decoded is `run_round` (E:577-591), once per
  state and debias variant. The `rc = 1` retry runs the round again (E:764-789). Full-path
  scoring and `after` phases only copy trunks.
- **Positions and cells.** `trunk_len` is a position. Four places take a sequence's
  `pos_max + 1` as its cell count: `slot_live` (E:128-130), the restore check in `fill_slot`
  (E:368), `prefixes[].tokens` in info (E:984-997) and `held_cells` (E:267-273). The first three
  concern prefix slots only.
- **Server slots.** `slot.prompt.n_tokens()` is the slot's true cell count, media included
  (`server-task.h:594`). llama.cpp has no per-sequence cell count.
- **CLI.** `llama-decide` parses its arguments as `LLAMA_EXAMPLE_COMPLETION`; `--mmproj` and
  `--image-min-tokens`/`--image-max-tokens` are not registered for it (`common/arg.cpp`).

libmtmd:

- **Capabilities.** `mtmd_support_vision`, `mtmd_support_audio`, `mtmd_decode_use_mrope`,
  `mtmd_decode_use_non_causal` (M:141-150).
- **Loading.** `mtmd_helper_bitmap_init_from_buf` (MH:77-81) decodes an image from memory, but
  it also accepts audio buffers, and when its image decoder (stb_image) fails it tries ffmpeg
  (webp, then video), which starts a subprocess. The engine therefore does not call it.
  `mtmd_bitmap_init(nx, ny, rgb)` builds a bitmap from decoded RGB pixels.
- **Tokenising.** `mtmd_tokenize` with the marker alone as text, `add_special = false` and
  `parse_special = true`, yields the image's chunks: the
  model's own marker tokens as text chunks (for Qwen `<|vision_start|>` and `<|vision_end|>`) and
  one or more image chunks. With `add_special = true` it would wrap BOS/EOS around every image.
  An image's token object holds its preprocessed pixels (about 10-25 MB for a 1k-cell image).
- **Decoding.** `mtmd_encode_chunk` writes embeddings into a buffer the context owns, valid until
  the next encode. `mtmd_helper_decode_image_chunk` decodes one image chunk into one sequence. It
  sets 2D positions for M-RoPE models and returns the next position; its return value is
  `llama_decode`'s. For non-causal projectors it
  switches the whole context to non-causal attention for the duration of that decode, and the
  image must fit in one `n_ubatch`.
- **Marker.** The server uses a random marker per process unless `LLAMA_MEDIA_MARKER` is set.
- **Server chat path.** Its base64 decoder is lenient (stops at the first invalid character
  without an error) and its media handler also fetches URLs. Both are `static`; neither is reused.

Measured in the spike (Qwen3.8-27B IQ3_S, RTX 3090, `--image-min-tokens 1024`):

- An image takes 1038-1842 KV cells but advances the position by only 39-48 (M-RoPE).
- Encoding costs about 0.3 ms per cell and decoding about 0.8 ms per cell: 0.4-2.3 s per image.
  A further field costs about 15 ms.
- The engine matched a `/completion` replay under the statistic (largest abs diff 0.021).

## 3. API

### 3.1 States

`states[i]` is one of:

- a non-empty string (unchanged), or
- a non-empty array of parts. A part is
  - `{"type": "text", "text": "<non-empty string>"}`, or
  - `{"type": "image_url", "image_url": {"url": "data:image/<subtype>;base64,<data>"}}`.

Rules:

- Parts are used in the order given. Images may come before, between or after text.
- A string `S` and the array `[{"type": "text", "text": S}]` are the same state: same tokens,
  same response.
- Two text parts must not be adjacent. (They would tokenise differently from their concatenation,
  which the reference replay uses.)
- Limits: at most 4 images per state and 32 images per request. A decoded image is at most
  10 MiB and at most 32,000,000 pixels (width × height), and its longer side is at most 200
  times its shorter side (rev 6: a projector pads the short side of an extreme strip up to its
  patch size, so the pixel limit alone does not bound the memory of the preprocessed image). The
  count of states stays 1 to 256.
  (With at most 4 images and no adjacent text parts a state has at most 9 parts, so there is no
  separate part limit.)
- A part has exactly the keys shown above. Any other key, for example OpenAI's `detail`, is an
  error.
- The URL is `data:image/<subtype>;base64,<data>`, matched case-sensitively, with `<subtype>` of
  the characters `a-z 0-9 . + -` and no further parameters. `http(s)` URLs, `file:` URLs, paths
  and bare base64 are rejected.
- `<data>` is standard base64 (RFC 4648 §4): the alphabet `A-Z a-z 0-9 + /`, `=` padding, a
  length that is a multiple of 4, no whitespace. Anything else is an error; the URL-safe alphabet
  and unpadded data are errors.
- The decoded bytes must be a JPEG or a PNG, recognised by their magic bytes, not by `<subtype>`.
  The engine reads width and height from the header and applies the pixel limit before it decodes
  any pixels.
- The engine decodes the pixels itself with the vendored stb_image and hands RGB pixels to
  libmtmd (`mtmd_bitmap_init`). A file stb_image cannot decode is an error. libmtmd's own loader,
  with its audio, webp and video paths, is never called. libmtmd already compiles stb_image with
  external linkage, so the decide library compiles its own copy in one file with
  `STB_IMAGE_STATIC`, `STBI_ONLY_JPEG` and `STBI_ONLY_PNG`, and takes `vendor::stb` as a private
  dependency.
- Text parts are untrusted text, tokenised as today (`tokenize_plain`, no special tokens). A text
  part that contains the media marker string is plain text; it cannot insert an image.

Images belong to states only. Instructions, field descriptions and options stay text.

### 3.2 Errors

| condition | http | code | field |
|---|---|---|---|
| a part is malformed, a limit of §3.1 is exceeded, the URL or base64 is invalid, the bytes are not a JPEG or PNG, the pixels cannot be decoded | 400 | `invalid_states` | `states[i]` |
| a state has an image part and vision is not available (§4) | 400 | `vision_unavailable` | `states[i]` |
| the state's cells (text + image) exceed the budget | 400 | `budget` | as today; the message names the image cells |
| libmtmd fails to tokenise a decoded image, the projector fails to encode an image, or an image decode fails with a code other than 1 | 500 | `engine` | none |

- The checks run in three passes, each over the states in order; the first error ends the
  request and `field` names its state:
  1. the checks that need no projector: parts, limits, URL, base64, format, header pixels. The
     limit of 32 images names the state that holds the 33rd image;
  2. vision availability: `vision_unavailable` names the first state with an image;
  3. in `run_request`: first the sequence budget (too few decide sequences for the schema, 400
     `budget`; it needs only the schema), then the preparation of the prefixes, then pixel
     decoding (`invalid_states`) state by state and part by part; right after each image is
     tokenised the state's cells so far are checked against the cell limit (400 `budget`,
     §5.1), and once every part is tokenised the per-state cell check of §5.3 follows.
  Passes 1 and 2 run in `states_of`, so on the HTTP thread too.
- A request is answered whole or not at all: one bad state rejects the request.
- A request that ends in one of the 400 answers above leaves the engine as it was: it decodes
  nothing and changes no prefix slot, cache entry or server slot (§9.3 tests this).

### 3.3 Response

For a state with at least one image, its `usage` gains `images` (count), `image_cells` and
`image_positions`. `state_tokens` keeps counting text tokens only. The request's
`usage.prompt_tokens` counts every decoded cell, so it includes image cells, once per decode.
When the request has at least
one image, `timings` gains `image_encode_ms` and `image_decode_ms`; both are parts of the prefill
time.

A response to a request without images has exactly today's keys and values.

### 3.4 `GET /v1/decide/info`

A new top-level key `vision`:

- Vision available: `{"available": true, "mrope": <bool>, "non_causal": <bool>,
  "max_images_per_state": 4, "max_images_per_request": 32, "max_image_bytes": 10485760,
  "max_image_pixels": 32000000, "max_image_aspect_ratio": 200}`.
- Not available: `{"available": false, "reason": "<code>"}`, with `reason` one of §4.1.

The CLI's `--info` prints the same key.

### 3.5 `/v1/systemone`

Unchanged. Its inputs are strings and its validation stays as it is.

## 4. Models without vision

### 4.1 When vision is available

Vision is available when all of these hold; otherwise `reason` names the first that fails:

| reason | condition that fails |
|---|---|
| `no_projector` | a projector is loaded (`--mmproj`) |
| `projector_without_vision` | `mtmd_support_vision` is true (an audio-only projector fails here) |
| `non_causal_projector` | the projector decodes images causally, or the non-causal gate of §9.5 passed |

`engine_config` gains `mtmd_context * mctx` (null without a projector). One function computes
the status from it. The engine calls it when it is created. The server calls it once after the
model and projector are loaded and keeps the result in an atomic field of its own, which the
HTTP thread reads; the HTTP thread never dereferences the engine, which is reset on sleep and
rebuilt on resume. A resume loads the same model and projector, so the value does not change.

### 4.2 Behaviour

- **Start.** The server and `llama-decide` start without a projector exactly as today: no new
  required flag, no warning, no error. The decide library links libmtmd (the server already
  does), so the build has no new option.
- **Text requests.** A request without image parts runs today's code path. The engine calls
  libmtmd only for states that have image parts.
- **Image request.** After the pass-1 checks of §3.2 it is rejected with `vision_unavailable` on
  the HTTP thread, before the task is queued. Nothing is decoded and no slot or cache entry
  changes. The message says which case of §4.1 applies and, for `no_projector`, that the server
  needs `--mmproj`.
- **Mixed request.** Text states next to an image state are not answered either (§3.2).
- **Clients** can read `vision.available` from info before sending images.
- **No 500.** Missing vision never reaches the engine's decode path. The engine still checks it
  when it parses the states on the main loop (the second `states_of` call) and answers the same
  400, so a caller that skips the HTTP-thread check (the CLI) gets the same result.

## 5. Engine

### 5.1 Parsed states

`states_of(body, vision_available)` returns, per state, its ordered parts: text (a string) or
image (the decoded bytes). A string state is one text part. All three call sites pass the
server's vision status; `/v1/systemone` states are always strings, so the value has no effect
there.

In `run_request`, after the sequence-budget check and the preparation of the prefixes (§3.2 pass
3) and before any model decode, each image part is
decoded to RGB pixels (stb_image), wrapped in a bitmap (`mtmd_bitmap_init`) and tokenised
(`mtmd_tokenize` on the marker alone, `add_special = false`, `parse_special = true`). That gives
the image's chunks, its cell count `n_tokens` and its position count `n_pos`. The pixels and the
bitmap are freed at once. Text parts are tokenised as today. The chunks live until the request
ends. Their memory grows with the projector's image token cap (`--image-max-tokens`); the limits
of §3.1 bound it and the time the main loop is held. Rev 6: right after an image is tokenised,
the state's cells so far are checked against the cell limit; a state that is already over it
ends the request with 400 `budget` at once, before further images are loaded.

### 5.2 Building a trunk

For each state and debias variant, after the prefix is copied into the trunk sequence:

1. The parts up to and including the last image are decoded into the trunk in order, state by
   state: text chunks with `decode_chunked`, image chunks with `mtmd_encode_chunk` followed by
   `mtmd_helper_decode_image_chunk`. The position advances by the token count of a text chunk and
   by the helper's returned position for an image chunk.
2. The text after the last image (if any) and the tail join the round's batched trunk decode, as
   all state text does today. A state without images takes only this step, so text states are
   batched exactly as before.

`trunk_len` is the resulting position, so branch phases, `after` and full-path scoring work on
the trunk unchanged.

Return codes: a 1 from an image or text decode of step 1 is the round's `rc = 1` and takes the
existing retry path. Any other non-zero code, and a failed `mtmd_encode_chunk`, end the request
with 500 `engine` through the engine's existing error cleanup.

An image is encoded again each time it is decoded: once per debias variant and once more per
`rc = 1` retry. On the 27B that adds about 0.3 ms per cell to the 0.8 ms per cell of the decode.
Reusing embeddings would need a host copy of every image in the round and is left out (§10).

### 5.3 Budget

- **Cells.** A state's cells are its text tokens plus the `n_tokens` of its images. Both budget
  checks (E:720, E:749) use this sum.
- **Positions.** Only `trunk_len` uses positions.
- **Prefix slots.** Prefixes never contain images, so `slot_live`, the `fill_slot` restore check
  and `prefixes[].tokens` stay correct.
- **Server slots.** `engine_config` gains `std::function<int64_t()> held_cells`. The server sets
  it to the sum of `slot.prompt.n_tokens()` over its slots. When it is empty (the CLI),
  `engine::held_cells()` keeps today's count. This fixes the under-count when a chat slot holds
  an image.
- **SWA cache (rev 5).** A model with sliding-window attention layers has a second, smaller KV
  cache (`n_swa × sequences + n_ubatch` cells, padded; `n_ctx` with `--swa-full`). Every cell
  limit of the engine is `min(n_ctx, SWA cells) − 16`. A state over it gets 400 `budget` that
  names the SWA cache and `--swa-full`; before rev 5 it passed the budget and ended in a 500. This
  applies to text and image states alike. info reports `memory.swa_cells` for such models.
- **Non-causal projectors.** If an image chunk has more cells than `n_ubatch`, the request gets
  400 `budget` with a message that names `n_ubatch`.

### 5.4 Projector families

| family | positions | attention | handled by |
|---|---|---|---|
| Qwen-style (M-RoPE) | 2D, position jump | causal | the helper's positions; cells and positions counted apart (§5.3) |
| sequential (SmolVLM, idefics3) | sequential; one image can be several image chunks | causal | the chunk loop of §5.2 |
| Gemma-3-style | sequential | non-causal during the image decode | the helper's context-wide switch; safe because an image chunk is decoded on its own, with no other sequence in the batch |

### 5.5 Existing switches

| switch | with image states |
|---|---|
| full-path scoring, `after`, temperature | unchanged: they act on trunk copies and logits |
| order debiasing (K variants) | the image is decoded K times, once per variant prefix |
| prefix cache | unchanged: keyed by the prefix, which has no images |
| `rc = 1` retry and the fault hook | the round is rebuilt, images included |

## 6. CLI and dump

- `llama-decide` accepts `--mmproj`, `--image-min-tokens` and `--image-max-tokens`. These
  arguments are registered for `LLAMA_EXAMPLE_COMPLETION` one by one; the example is not added to
  `mmproj_examples`, which would also switch on projector auto-download for `llama-completion`.
- With `--mmproj` the CLI creates an `mtmd_context` as the server does. Without it the CLI
  behaves as today and answers an image state with `vision_unavailable`.
- `--dump-tokens`: an image state's entry lists its parts in order. A text part has its tokens.
  An image part has the SHA-256 of its bytes, `n_tokens`, `n_pos` and its chunk layout. The dump
  also carries what a string replay needs: the prefix text, the tail text and each branch's text.
  The dump of a request without images is unchanged.

## 7. Project tools

- `decide_client.py`: states may be strings or part lists; a helper turns an image file into a
  data-URI part.
- `reference_check.py`: an image state is replayed through `/completion` with
  `{"prompt": {"prompt_string": ..., "multimodal_data": [...]}}`: prefix text, a media marker per
  image, the text parts, the tail and the branch text. Reference servers run with
  `LLAMA_MEDIA_MARKER=<__media__>` so the marker is known. The statistic is the text one.
- Presets: vision presets are added next to the existing ones in `models-engine-3090.ini` (for
  example `qwen3.8-27b-vision` with the projector and its own `decide-seqs`). Existing presets do
  not change.
- The replay parses special tokens in the whole string and the engine does not parse them in text
  parts. Test texts therefore contain no special-token strings, as in the text reference check.

## 8. Models and data

Models (all on the test machine, inside `<repo>/models` when downloaded):

| model | role | binding for the statistic |
|---|---|---|
| Qwen3.8-27B IQ3_S + its BF16 projector (present) | main: M-RoPE, hybrid | yes |
| SmolVLM-500M-Instruct Q8_0 + projector (`ggml-org/SmolVLM-500M-Instruct-GGUF`, about 546 MB, Apache-2.0) | sequential positions, several chunks per image | no (very small model; reported). If it cannot run (for example its template fails the join check), it is recorded as skipped with the reason |
| Gemma 4 E4B Q4_0 (present) + its BF16 projector (`ggml-org/gemma-4-E4B-it-GGUF`), added during execution as the substitute for SmolVLM, which the engine's join check rejects | sequential positions, causal attention, one chunk per image | no (reported). With SmolVLM skipped, no live model has several image chunks per image |
| gemma-3-4b-it + projector (`ggml-org/gemma-3-4b-it-GGUF`, about 3.3 GB) | non-causal gate (§9.5) | decides the gate |
| Qwen3.6-35B-A3B + projector (present) | MoE with a projector | no; run if it loads on the 24 GB card with the placement keys, otherwise recorded as skipped with the reason |
| Qwen2.5-0.5B (present, no projector) | the no-vision checks of §9.3 | n/a |

Data: a labelled subset of COCO val2017.

- 200 images, chosen with a fixed seed from `instances_val2017.json` for the five object
  categories person, car, bicycle, cat and dog (20 positives and 20 negatives each, every image
  used once). A category is positive when the image has an instance with `iscrowd = 0` and an
  area of at least max(32², 1 % of the image), negative when it has no instance of the category
  at all, and unlabelled otherwise.
- Every image is asked all five yes/no fields in one request; a field is scored only where its
  label is positive or negative.
- The repository gets the list (image id, COCO URL, licence id, labels) and a notice crediting
  the COCO Consortium (annotations: CC BY 4.0). The images are downloaded to the git-ignored
  `local/` directory and are not committed.

## 9. Verification

### 9.1 Unit tests (ctest, no model)

- `states_of`: every rule of §3.1 with a passing and a failing case, including strict base64,
  every limit, the format and pixel checks, adjacent text parts, unknown part types, `field`, and
  the equality of a string and its one-part form.
- `vision_available = false` with an image part gives `vision_unavailable`; with only text parts
  it gives the same parse as today.
- Budget functions with image cells; the `n_ubatch` rule of §5.3.
- The systemone tests pass unchanged.

### 9.2 Text regression (live)

Old build (`49785a95d`) against new build on the test machine, same model and placement:

- The triage test set with each of the five switches of sub-project 3. The same request
  sequence is sent to a freshly started server of each build; the probabilities are identical
  (max abs diff 0).
- Two runs: both builds without a projector, and both builds with the same `--mmproj` (27B).
- The project's pytest suite and the existing integrity scripts pass on the new build.

### 9.3 No-vision behaviour (live)

On Qwen2.5-0.5B (no projector) and on the 27B started without `--mmproj`:

- info reports `vision.available = false`, reason `no_projector`.
- An image request and a mixed request each get 400 `vision_unavailable`.
- info taken before and after those requests is identical, and the text requests sent after
  them give the probabilities and the cache `hits`/`misses` of the same text requests in a
  server run without the rejected requests.
- The server log has no error line from these requests and the server keeps serving.
- The CLI without `--mmproj` gives the same 400 for an image state.

### 9.4 Integrity for image states (live)

On every vision model of §8 that runs:

- **Reference check**: tree mode with one and with several fields, full-path, `after`, debiasing
  (K = 2), temperature, image-only states, text before and after the image, two images in one
  state.
- **Batch invariance**: the same image states alone, together and mixed with text states.
- **Retry**: the `rc1` fault mode with an image state gives the result of the run without it,
  under the statistic.
- **Held cells**: a multimodal chat request is resident in a server slot. A decide request is
  sized to fit the budget under the old count (the slot's `pos_max + 1`) but not under the slot's
  true cells. The new build answers 400 `budget`. The old build's answer to the same sequence is
  recorded next to it.
- **Malformed input**: the failing cases of §9.1 against the live server return their 400 and
  leave info unchanged.

### 9.5 Non-causal gate

gemma-3-4b runs the reference check and batch invariance of §9.4. If it passes the statistic,
non-causal projectors stay enabled. If it fails, or the model cannot be run, the engine reports
`vision.available = false` with `non_causal_projector` for such projectors, their image requests
get `vision_unavailable`, and the report says so. Either outcome completes this sub-project.

### 9.6 Evaluation (live, Qwen3.8-27B)

On the COCO subset, split 100/100 into calibration and test with the fixed seed:

- accuracy and NLL per field at T = 1, and the fitted temperature on the calibration half
  (reported only; the text preset's temperature is not changed);
- a chat baseline that asks one question per call: accuracy, and time per image against one
  decide request with five fields.

### 9.7 Report

A section in `REPORT-ENGINE.md`, the API in `README.md`, the regenerated patch
series with its tree id, and the results under `results-engine/image/`.

## 10. Out of scope

- Images in the instructions or the prefix; an image shared by several states.
- Remote URLs, `file:` URLs and server paths.
- Audio and video input.
- A cache of image trunks or embeddings; reuse of embeddings across variants and retries.
- Decoding the images of several states in one batch.
- Images in `/v1/systemone`.
- A calibrated temperature preset for image requests.
