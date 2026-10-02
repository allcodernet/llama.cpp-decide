# Image-input feasibility spike (2026-10-02)

> **Throwaway.** `spike.patch` is a prototype, not a proposal for the `decide` series. It is `git diff a32ee701e` of
> the worktree `<repo>/engine-spike-image` (branch `spike-image`, uncommitted changes) on the
> test machine. The worktree and its build (`engine-spike-image/build`, CUDA, arch 86, g++-15, Release, tests off)
> stay there so the runs can be repeated. While the spike ran, `engine/` on the test machine moved on to `49785a95d`
> (the follow-ups' last fixes, 0038); the patch applies to `a32ee701e`.

## Answer

**Works, with limits.** Qwen3.8-27B (IQ3_S) with its BF16 projector answers "is there a cat in this picture?"
through `/v1/decide` in one pass. All 10 test images are on the right side of 0.5. The engine matches a `/completion`
replay of the same prompt under the reference statistic used for text states. With four questions about one image
in one request, each extra question costs about 15 ms; the image itself costs 0.4-2.3 s.

Limits:
- The prototype decodes each image state on its own, so three image states take as long as three requests.
- Image states support only the default tree mode with K = 1. Full-path scoring, `after` fields and debiasing were not
  tested.
- `held_cells()` under-counts cells held by multimodal server slots with M-RoPE (found by reading the code; see Risks).
- The image arrives as a path on the server. A real API needs base64.

## Setup

- Server: `serve-spike.sh` (the spike build's `llama-server -m <Qwen3.8-27B IQ3_S> --mmproj <mmproj BF16>`):
  - `--image-min-tokens 1024` (the setting of the author's existing server for this model), or `IMG_MIN=0` for the projector's default;
  - `-fa on -ctk q8_0 -ctv q8_0 --jinja --reasoning off -np 1 -c 16384 --decide-seqs 16`;
  - `127.0.0.1:8097`;
  - `LLAMA_MEDIA_MARKER=<__media__>`. The server otherwise picks a random marker per process, which the `/completion`
    replay needs to know.
- Every run used T = 1, the server default; the preset's 0.8992 was not applied. The GPU was checked before each run
  (no foreign server, 612 MiB in use) and left idle afterwards.
- Images: in `local/spike-image/` on the test machine (git-ignored). They were not committed. The ground truth is my
  own reading of each picture:

| name | truth | URL | sha256 |
|---|---|---|---|
| cat03 | cat | https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg?width=1024 | 6abda6611dab9d7d7754259c00575baba1d6c3e1b9cf7d61ef79a0fc67729ebe |
| kitten | cat | https://commons.wikimedia.org/wiki/Special:FilePath/Kittyply_edit1.jpg?width=1024 | 47015ba67a3aab52c3d677d2e12f427b6fb5dc4f1103b696a9ba27f804f49b47 |
| coco_39769 | cat (two, on a sofa) | http://images.cocodataset.org/val2017/000000039769.jpg | dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e |
| wildcat | cat (European wildcat) | https://commons.wikimedia.org/wiki/Special:FilePath/Felis_silvestris_silvestris.jpg?width=1024 | aed6caa694a20c9f225f039193b529b471cb92f68d87f4ba1cb201508ef46a46 |
| lynx | cat (hard: wild cat) | https://commons.wikimedia.org/wiki/Special:FilePath/Lynx_lynx_poing.jpg?width=1024 | 0da57a19518d4b3b2661df8f50c618f0f3946148c59d860dc3839a398a84976c |
| fox | no cat (hard) | https://commons.wikimedia.org/wiki/Special:FilePath/Vulpes_vulpes_ssp_fulvus.jpg?width=1024 | 3b37bef747c937d2c42a39c4830bb5bc09ce1e20b5a1da8d0c2ecaafa6661d79 |
| labrador | no cat (dog) | https://commons.wikimedia.org/wiki/Special:FilePath/YellowLabradorLooking_new.jpg?width=1024 | d4cb12ea8a768582562e1f1c0c930ea6fc4757f5b66551b986b41498efd75ca1 |
| coco_776 | no cat (hard: teddy bears) | http://images.cocodataset.org/val2017/000000000776.jpg | 1dd31e9059c491992be2f562624eb4093e17aee08b4f7baf5ff9ea24543b0a33 |
| coco_139 | no cat (living room) | http://images.cocodataset.org/val2017/000000000139.jpg | ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d |
| coco_632 | no cat (bedroom) | http://images.cocodataset.org/val2017/000000000632.jpg | a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550 |

Licences: Wikimedia Commons files follow the licence on each file page. COCO images are Flickr images under the
CC licences listed in the COCO annotations.

## How the prompts differ

- **Chat reference** (`reference_chat.py`, `/v1/chat/completions`):
  - no system message;
  - user content `[image, "Is there a cat in this picture? Answer with yes or no."]`;
  - thinking off (`<think>\n\n</think>\n\n`);
  - one token generated. `p_yes` = P(yes) / (P(yes) + P(no)), summed over case and space variants among the top 20
    first tokens.
- **Decide** (`decide_image.py`):
  - system = the engine's catalogue (`Answer each question about this image.` + `- "cat": Is there a cat in this
    picture? true or false.`);
  - user = `<|vision_start|>` + image + `<|vision_end|>`, plus the optional text after it;
  - assistant `<think>\n\n</think>\n\n{\n` + branch `  "cat":`, options ` true` / ` false`.
- **`/completion` replay**: the decide prompt as one string (prefix + media marker + text + tail + branch text). The
  image goes in as `multimodal_data`, so the server's own multimodal path builds slot 0. `n_probs` 50; the probabilities
  are renormalised over the branch's children. This is the reference check for image states.

## Numbers

p(cat) per image (image-min-tokens 1024 unless noted). Cells are the image's KV cells including `<|vision_start|>` and
`<|vision_end|>`; positions are what the image advances the M-RoPE position by:

| image | truth | chat p_yes | decide p_true (cat only) | replay | decide p_true (4 fields) | cells / positions | chat, default tokens | decide, default tokens | cells / positions, default |
|---|---|---|---|---|---|---|---|---|---|
| cat03 | cat | 0.9990 | 0.9999 | 0.9999 | 1.0000 | 1602 / 42 | 0.9991 | 0.9999 | 1602 / 42 |
| kitten | cat | 0.9842 | 0.9998 | 0.9998 | 1.0000 | 1082 / 42 | 0.9867 | 0.9998 | 1082 / 42 |
| coco_39769 | cat | 0.9986 | 0.9996 | 0.9996 | 1.0000 | 1038 / 39 | 1.0000 | 0.9999 | 302 / 22 |
| wildcat | cat | 0.9985 | 0.9999 | 0.9999 | 1.0000 | 1842 / 48 | 0.9978 | 0.9999 | 1842 / 48 |
| lynx | cat | 0.9922 | 0.8383 | 0.8226 | 0.6576 | 1082 / 42 | 0.9782 | 0.9505 | 427 / 27 |
| fox | no cat | 0.0185 | 0.0012 | 0.0011 | 0.0000 | 1202 / 42 | 0.0126 | 0.0008 | 1202 / 42 |
| labrador | no cat | 0.0038 | 0.0001 | 0.0001 | 0.0000 | 1082 / 38 | 0.0349 | 0.0001 | 398 / 24 |
| coco_776 | no cat | 0.0004 | 0.0004 | 0.0003 | 0.0000 | 1082 / 42 | 0.0277 | 0.0001 | 262 / 22 |
| coco_139 | no cat | 0.0762 | 0.0716 | 0.0617 | 0.0287 | 1082 / 42 | 0.3368 | 0.0312 | 262 / 22 |
| coco_632 | no cat | 0.0185 | 0.0046 | 0.0057 | 0.0009 | 1038 / 39 | 0.0818 | 0.0048 | 302 / 22 |

- **The model can do it.** Every image is on the correct side of 0.5 in every column. The only uncertain case is
  the lynx. Its "cat" probability depends on the wording: 0.99 in chat, 0.84 with the bare catalogue, 0.66 next to
  the `animal` field, whose own answer is `cat` at 0.836. That is a prompt-definition question ("does a wild cat
  count?"), not an engine problem.
- **Chat reference.** The bare chat question is less clean on the small COCO images at default tokens. For coco_139
  the top token is "```" and p_yes is 0.34 there.

Engine vs `/completion` replay (statistic of the text reference check: per field median abs diff <= 0.01 and no
argmax disagreement where the reference's top-2 margin > 0.15):

| run | field | median abs diff | max abs diff | argmax agreement | pass |
|---|---|---|---|---|---|
| cat only, 1024 | cat | 9.4e-05 | 0.0157 | 1.00 | PASS |
| cat only, default tokens | cat | 3.6e-05 | 0.0177 | 1.00 | PASS |
| 4 fields, 1024 | cat | 2.1e-05 | 0.0205 | 1.00 | PASS |
| 4 fields, 1024 | dog | 0.00067 | 0.0090 | 1.00 | PASS |
| 4 fields, 1024 | animal (choice, 5 options) | 6.2e-05 | 0.0208 | 1.00 | PASS |
| 4 fields, 1024 | indoors | 0.0013 | 0.0056 | 1.00 | PASS |

- **Same range as text states.** The 27B's text reference check had max abs diffs of 0.011-0.022. The largest image
  diffs are on the uncertain answers (lynx, coco_139).
- **Image + text states.** These also match their replays (`runs/decide-withtext.json`, max abs diff 0.0074). A text
  after the image moves the answer: cat03 with "there is no cat in this photo" gives p_true 0.887 instead of 0.9997.

Time per request with one image state (ms, medians over the 10 images, prefix resident):

| run | image cells | projector encode | image decode (LLM) | trunk prefill incl. both | branch scoring | total |
|---|---|---|---|---|---|---|
| cat only, 1024 | 1082 | 324 | 853 | 1231 | 36.3 | 1286 |
| cat only, default tokens | 412 | 88 | 361 | 504 | 36.2 | 550 |
| 4 fields, 1024 | 1082 | 326 | 866 | 1248 | 82.8 | 1351 |

- **Cost per image.** The image dominates: about 0.8 ms per image cell to decode (about 1250 cells/s) and 0.3 ms per
  cell to encode. cat03 (1602 cells) takes 2.0 s and wildcat (1842 cells) 2.3 s. The text around it and the tail
  cost a few ms.
- **Chat reference for comparison.** Its wall time per image (one generated token, base64 upload) has a median of
  1339 ms at 1024 tokens and 614 ms at default tokens: the same image cost.
- **One pass, many fields.** Four fields cost 1351 ms against 1286 ms for one field. Asking the same four questions
  through chat without a prompt cache would cost about 4 x 1.3 s.
- **Several image states in one request.** Three image states with four fields (cat03, fox, coco_39769): 4708 ms in
  one request against 4870 ms for three single requests.
  - The images are decoded state by state, so they gain nothing from batching. Scoring is batched: 146 ms for 3 states
    against 83 ms for 1.
  - Batch vs single max abs diff: 0.0073 / 0.0002 / 0.0010.
- **Mixed request.** One text state and one image state work in one request. The text state's answers move by at most
  0.0071 against the same text alone. That is a placement effect: the round's batch differs.
- **Repeats.** cat03 five times gave p_true 0.999862 the first time, then 0.999894 four times (placement again); the
  total time was 1998-2011 ms.
- **Text-only requests.** These still work in the spike build: `tests/data/decide-triage.json`, as it was then (its first
  example state has been replaced since), gives billing/2/false and feedback/0/false (`runs/decide-text-triage.json`).
  The only change on the text path is an image term of 0 in the budget.

VRAM (`vram_sweep.sh`, image-min-tokens 1024; idle after load, peak during the 3-image request; includes the
desktop's 612 MiB):

| decide-seqs | ctx | idle MiB | peak MiB | 3-image request ms |
|---|---|---|---|---|
| 8 | 16384 | 14960 | 15254 | 4764 |
| 16 | 16384 | 16192 | 16488 | 4656 |
| 16 | 32768 | 16782 | 17078 | 4664 |
| 32 | 16384 | 18686 | 18982 | 4661 |
| 48 | 16384 | 21182 | 21476 | 4663 |
| 64 | 16384 | 23672 | 23966 | 4664 |

- **Memory per sequence and projector.** Each decide sequence costs about 156 MiB (the recurrent state), as without
  the projector. The projector file is 0.93 GB.
- **Request overhead.** The image request adds only about 300 MiB above idle.
- **Even `--decide-seqs 64` fits** at 16k context, with about 600 MiB headroom on the 24 GB card. The recommendation is
  16-32 sequences with 16k-32k context.
- **Context, not sequences, limits images per round.** An image takes 0.26k-1.8k cells, so 16k context holds about
  8-14 image states at 1k-1.8k cells each (budget check by cells).

## M-RoPE accounting (cells vs positions)

- **Cells and positions differ.** An image takes `n_tokens` KV cells (1038-1842 at min 1024) but only `n_pos`
  positions: max(nx, ny) of the merged patch grid plus the two marker tokens, 39-48 here.
  - The text, the tail and every branch token after it start at `pos + n_pos`. The engine's `trunk_len` is a position,
    so the branch phase works unchanged.
  - The batch allocator allows the position jump for M-RoPE models ("X < Y").
  - The budget must count cells (`n_tokens`), never positions. The prototype adds `n_tokens` to the state's cells in
    both budget checks.
- **The engine assumes position = cell count elsewhere:**
  - `slot_live` (prefix only, still correct while images stay out of prefixes);
  - `info` `prefixes[].tokens` (pos_max + 1);
  - `held_cells()`, which takes `seq_pos_max + 1` of every server slot as its cell count.
  - With a multimodal chat request resident in a server slot, `held_cells()` under-counts by about the image's cells
    (for example 1100 cells counted as about 150). A decide request could then pass the budget check and hit
    `rc = 1` / 500. This was found by reading the code, not reproduced live; the spike's 16k context never came near
    the limit.

## What the spike skipped

- Image source and parsing:
  - only server paths: no base64 or URL;
  - no validation of the state object;
  - the in-band encoding `"\x1d<path>\x1d<text>"` (a text state starting with `\x1d` is taken as an image);
  - one image per state, always before the text.
- Engine paths:
  - images in the prefix or instructions;
  - batching image decodes across states, and the embedding reuse across K variants (the code reuses them at
    `v == 0`, but only K = 1 ran);
  - full-path scoring, `after`, debiasing, temperature presets for images;
  - prefix-cache or image-cache snapshots of image trunks;
  - the `rc = 1` retry with images (code path present, never triggered).
- Tests, tooling and polish:
  - the `llama-decide` CLI (no `--mmproj`), the token dump, `reference_check.py`;
  - ctests;
  - `/v1/systemone` (Jev's API has no images);
  - error polish;
  - non-M-RoPE and non-causal projector models (Gemma-style vision);
  - a held-out labelled image set and calibration;
  - the `held_cells()` fix.

## Recommendation for a real sub-project

**API.**
- A state is either a string (as today) or an ordered array of parts that mirrors the OpenAI content parts the server
  already parses: `[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}, {"type": "text", "text": "..."}]`.
- Image and text parts appear in the order given, and several images per state are allowed.
- **Base64 (data URI) is the only default.** Remote URLs and server paths stay off (security: a path reads any file
  the server can read; a URL means fetches and latency). A server path could come back behind an explicit flag for
  local batch jobs.
- `/v1/decide/info` reports whether a projector is loaded. `usage` reports image cells and positions per state.

**Engine pieces.**
1. **Trunk construction with mtmd.**
   - Tokenise each image part as its own marker through `mtmd_tokenize` (mtmd adds `<|vision_start|>`/`<|vision_end|>`
     for Qwen); tokenise the text parts with `tokenize_plain` (untrusted, no special tokens).
   - Encode once per state and decode the embeddings into each variant's trunk, then the text and the tail.
   - Optional: batch the image embeddings of several states into one embedding decode. This is only possible when no
     model needs non-causal image attention, which mtmd toggles context-wide.
2. **Cell and position budgeting.**
   - Count cells (`n_tokens`) for `n_ctx`, positions (`n_pos`) for `trunk_len`.
   - Fix `held_cells()` and `info` so they do not equate positions with cells. This needs the cell count per server
     slot from the server (`server_tokens` size) or a memory API.
3. **Prefix cache and hybrid state.**
   - Images in states do not touch the prefix cache.
   - An image cache keyed by the image hash (snapshot of the trunk after the image: KV cells + recurrent state: about 36 MB
     of q8 KV per 1k cells, from the 16k -> 32k VRAM step, plus 156 MiB recurrent on the 27B) would remove the main cost for repeated images.
     It is a separate, optional step.
4. **Join checks.** The markers are special tokens, so the joins prefix|image and image|text are safe. Keep the
   existing probes for text|tail, and add a check that the template renders an image part the way mtmd expects (the
   server's chat path uses the same marker replacement).
5. **Reference check for image states.**
   - Extend `reference_check.py`: dump the prompt pieces plus the image hash, replay through `/completion` with
     `multimodal_data`, as `decide_image.py` does.
   - The CLI needs `--mmproj` for the dump, or the dump moves to the server.
   - Batch invariance with image states.
6. **Tests and evaluation.**
   - The schema and state-parsing ctests do not need a model.
   - A small vision model is needed for CI-style live tests.
   - A labelled image set for accuracy and calibration: the 27B's text temperature does not carry over, and image
     answers are very confident (0.9999).

**Risks.**
- **Prompt definitions matter.** The lynx moves between 0.66 and 0.99 with the wording.
- **Image prefill dominates latency** (about 0.8 ms per cell on the 3090). The engine's gain is in the number of
  questions, not in the image cost.
- **Context use.** At 1k-1.8k cells per image, a long instruction catalogue and several images compete for `n_ctx`.
- **Other model families:**
  - Gemma (non-causal image attention, no M-RoPE);
  - models whose projector puts newline or separator tokens inside the image;
  - iSWA plus large images.
- **Placement noise.** It is in the same range as for text (up to 0.02 on uncertain answers).
- **The `held_cells()` under-count** with multimodal slots.

**Size: medium.**
- Smaller than sub-project 3 (five switches, each with its own integrity and evaluation work). About the size of
  sub-project 2.
- The core is small: the prototype is 142 added lines and worked on its first real run.
- The work is in the API (parts, base64, validation, limits), the budget fixes, the reference and batch-invariance
  tooling for image states, the tests with a vision model, a labelled image set for evaluation and calibration, and
  the interaction with K > 1, `after` and full-path.
- The optional image snapshot cache would make it large.

## Files

- `spike.patch`: the throwaway engine diff (5 files, 142 added lines, 5 changed).
- Scripts:
  - `serve-spike.sh`: start the spike server; `SEQS`, `CTX`, `IMG_MIN`;
  - `reference_chat.py`: step 1;
  - `decide_image.py`: modes `single`, `multi`, `batch`, `mixed`, `repeat`, `withtext`;
  - `vram_sweep.sh`;
  - `analyze.py`: the tables above, also in `runs/analysis.md`.
- Request examples: `runs/request-single.json`, `runs/request-multi.json`.
- Responses and replays: `runs/*.json`; the VRAM sweep: `runs/vram-sweep.tsv`.
- Run on the test machine in `<repo>/local/spike-image` (images and server logs are there).
  Example: `./serve-spike.sh /tmp/log`, then `uv run decide_image.py single . out.json`, then kill the printed PID.
