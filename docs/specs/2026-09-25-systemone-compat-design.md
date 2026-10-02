# Sub-project 2: `/v1/systemone` compatibility layer — design

Status: rev 5. Parent: `2026-09-25-own-decide-engine-design.md` (sub-project 1,
merged). Sources of the Jev API facts: the author's snapshot of TypeSafe's public docs taken on 2026-09-23
(`{api,primitives,confidence}.md`) and the official Python SDK 0.7.0 checked out at `path/to/typesafe-sdk-python`
(`src/typesafe_sdk/_schemas/models.py` = the OpenAPI models; `_core/config.py` = `base_url`
override; `_core/constants.py` = `DEFAULT_MODEL "jev-latest"`). The docs snapshot and these paths are the author's
local copies (not part of this repository); the SDK is public at https://github.com/typesafe-ai/typesafe-sdk-python
(v0.7.0, commit `2ce5c65`).

## 1. Goal

Serve TypeSafe AI's System One request/response shape (`POST /v1/systemone`) from our
`llama-server` so that unmodified Jev clients (including the official Python SDK with
`base_url` pointed at us) get answers from our engine. The layer is a pure translation on
top of `/v1/decide`: no new engine behaviour, no new scoring.

Known differences, documented in README (not bugs):
- probabilities are not rounded or clipped (Jev rounds to 2 decimals, clips noul to [0.01, 0.99]);
- `usage.output_tokens` is always 0 (Jev bills synthetic output tokens);
- `model` in the response is the request's value (Jev resolves aliases such as `jev-latest`
  to `jev-1.13.0`);
- `confidence` uses the approximation from the docs demo (section 3.2), not Jev's
  undisclosed definition;
- limits: ours needs 2–10 score levels, 2–255 choice options and at most 64 questions per
  request (Jev/SDK allow 1-level scores and put no limit on questions); violations are 422;
- question cap with the shipped presets: every question needs at least one branch sequence,
  so with `decide-seqs = 21` a request holds at most 19 single-branch questions (fewer when
  a choice's option keys share leading tokens); more is a 400 `budget` (SDK
  `TypeSafeBadRequestError`); raise `decide-seqs` in `models-engine.ini` (VRAM cost on
  hybrid models: one recurrent state per sequence);
- context: the catalogue, the state and the answers share `ctx-size` (4096 in the presets;
  Jev documents 64k), so a long state or a long catalogue is a 400 `budget`; raise
  `ctx-size`;
- body limits: a raw body above 1 MiB is a 413; a `state`, `instructions` or criteria value
  nested deeper than 32 levels and more than 2 MiB of text serialised from the request's
  values are 422 (sections 3.2 and 4);
- choice option keys whose tokens are string prefixes of each other (e.g. `sale`/`sales`)
  are a 422; rename one (sub-project 3's full-path scoring lifts this for `/v1/decide`);
- the question ids and the whole catalogue are visible to the model (Jev says keys are not
  sent to the model and questions are evaluated in isolation);
- only `state` is tokenised without special-token parsing; `instructions`, criteria and ids
  go into the catalogue, which is tokenised with special-token parsing, so untrusted text
  belongs in `state`;
- auth: without `--api-key` any key works (the `Authorization` header is ignored); with it
  the SDK's key must match, otherwise 401; no 429/529;
- SDK: `response.request_id` works (every response of our handler carries
  `x-typesafe-request-id`); `client.models.list()` does not validate against
  llama-server's OpenAI-style `GET /v1/models`, so clients use the preset names;
- `loc` omits the question-type segment Jev includes (ours
  `["body","questions",id,"criteria"]`, Jev's `[..., id, "score", "criteria"]`) and gives
  array indices as strings;
- `budget` (state, question catalogue or question count over the context /
  `--decide-seqs` budget) is a 400 with a `detail` body;
- in router mode (`--models-preset`) a missing or unknown `model` and a body that is not
  valid JSON are answered by llama-server's router (`proxy_post`) with its own error body
  before our handler runs; the 422 shapes below for those two cases hold on a single-model
  server.

## 2. Jev shape (facts)

Request: `{"state": string|object|array, "model": string (required), "questions": {id: Question}}`.
`Question = {"type": "noul"|"choice"|"score", "instructions"?: string|object|array, "criteria"?: ...}`.
`instructions` is optional (SDK default `None`, omitted from the wire). Criteria:
noul optional `{"true"?: X, "false"?: X}`; choice: map option → `X | null`, max 255;
score: array of level entries `X`, where `X` = string | object | array.

Response: `{"model": string, "answers": {id: Answer}, "usage": {"input_tokens": int, "output_tokens": int}}`.
choice `{"type":"choice","choice":key,"probabilities":{key:p},"confidence":c}`;
score `{"type":"score","score":expected,"legend":{"0":X,...},"probabilities":{"0":p,...},"confidence":c}`
(legend holds the original criteria entries, objects stay objects);
noul `{"type":"noul","noul":p_true}` (no confidence). The SDK models are `strict=True`:
`usage` values must be JSON integers.

Errors: Jev is FastAPI, so validation failures are 422 with
`{"detail": [{"loc": ["body", "questions", id, ...], "msg": string, "type": string}]}`.

## 3. Translation

### 3.1 Text helper

`text(X)`: string → as is; object/array → `json.dump(X, indent 2)` (our choice; Jev's own
serialisation is undocumented, its cookbook uses `json.dumps(..., indent=2)`); null → `""`.
`short(X)`: string → as is (a string with newlines stays multi-line); object/array →
compact `json.dump(X)` (single line); null → `""`. Any other JSON type (number, bool) in a
place that expects `X` is a 422 shape error.

### 3.2 Request → `/v1/decide` body

| Jev | ours |
|---|---|
| `model` | `model` (unchanged; the router proxies by it). Missing → 422 `loc ["body","model"]` |
| `state` | `states: [text(state)]` |
| `questions` (ordered map) | `fields` in the same order (`common_json` keeps insertion order; duplicate keys in the wire JSON resolve last-wins in the parser, not addressed) |
| `instructions` present and not `""` | `D = text(instructions)` |
| `instructions` absent, null or `""` | `D` = default per type: noul `"Answer yes or no."`, choice `"Choose the option that applies."`, score `"Rate on the given scale."` |
| noul | `{"type":"bool","description": D + suffix}` with suffix `" (true: " + short(criteria.true) + "; false: " + short(criteria.false) + ")"`; a half that is missing or null is omitted (`" (true: …)"`), no criteria or both halves empty → no suffix |
| choice | `{"type":"choice","description": D,"options": {key: short(value) or "" for null}}` |
| score | `{"type":"score","description": D,"levels": [short(entry) or "level <i>" when the result is empty]}` |
| — | top-level `instructions` = `"Answer each question about the state."` |

Validation (422, FastAPI body, `loc` naming the field) happens in the HTTP thread before
any task is posted, in two layers:
1. shape checks in the translator: `model` missing/not a string; `state` missing or not
   string/object/array; `questions` missing/not an object/empty; a question not an object;
   `type` missing or unknown; `instructions` present but not string/object/array/null;
   choice `criteria` missing/not an object, or a value not string/object/array/null; score
   `criteria` missing/not an array, or an entry not string/object/array; noul `criteria`
   present but not an object, or a half not string/object/array/null; bounds: a `state`,
   `instructions` or criteria value nested deeper than 32 levels (at the value's `loc`; a
   scalar is depth 0; checked before that value is serialised), `state` empty after
   serialisation (`["body","state"]`), more than 64 questions (`["body","questions"]`,
   checked after the state and before any question is translated), more than 2 MiB of
   `text()`/`short()` output in total (`["body"]`, checked after each value is serialised,
   so the largest transient string is bounded by the 1 MiB body limit and the depth cap);
2. our compiler (`compile_request` on the translated body): every 400 it raises
   (`invalid_schema`, `invalid_options`, `invalid_states`, option/level counts, empty
   ids) is re-emitted as 422 with `loc ["body","questions",<field>]` when the error
   carries a field name, else `["body"]`, and `msg` = our message. For this,
   `decide::error` gains an optional `field` member that the compiler and the trie builder
   fill; no message parsing. `error_result` in `decide-api.cpp` adds `"field"` to the
  `{"error":{...}}` body when it is non-empty, so `/v1/decide` error bodies gain an
  optional key and the route can build the `loc` for post-task errors.
The sibling-prefix check needs the vocabulary and runs in the task thread
(`build_tries`), so it returns from the task as a 400 `invalid_schema`; the route maps any
post-task 400 (`invalid_schema`, `budget`) to the same 422/400 `detail` shape:
`invalid_schema` → 422 with `loc ["body","questions",field]`, `budget` → 400 with
`{"detail":[{"loc":["body"],"msg":...,"type":"budget"}]}`. `engine` 500 and `disabled`
404 keep our usual `{"error":{...}}` body (the SDK parses both shapes).

### 3.3 `/v1/decide` response → Jev response

From `results[0].answers`, keyed by question id in request order:

| ours | Jev |
|---|---|
| choice `value`, `probabilities` | `choice`, `probabilities` (same map), `confidence` |
| score `expected`, `probabilities` list | `score` = `expected`, `probabilities` = `{"0": p0, ...}`, `legend` = `{"0": criteria[0], ...}` (original entries), `confidence` |
| bool `p_true` | `noul` = `p_true` |
| `usage.prompt_tokens + usage.cached_tokens` | `usage.input_tokens` (Jev counts the whole input; ours excludes the cached prefix from `prompt_tokens`; the sum also includes the branch tokens decoded for scoring, so it is slightly above the request's text — documented) |
| — | `usage.output_tokens` = 0 |
| request `model` | `model` |

`confidence = clip((n·peak − 1)/(n − 1), 0, 1)`, `n` = options/levels, `peak` = max
probability. This is the approximation the docs' interactive demo uses (it reproduces the
docs examples to ±0.01); Jev's real definition is not published. Noul has no confidence.

## 4. Components

- `engine/tools/decide/decide-compat.{h,cpp}` (library `llama-decide`), pure functions:
  - `common_json systemone_to_decide(const common_json & body)` — throws
    `compat_error{loc (vector<string>), msg, type}`;
  - `common_json decide_to_systemone(const common_json & decide_response, const common_json & questions, const std::string & model)`;
  - `double jev_confidence(const std::vector<double> & probs)`;
  - `common_json validation_body(const std::vector<compat_error> & errs)` (FastAPI shape) and
    `compat_error from_decide_error(const decide::error &)` (uses `error.field` for the `loc`);
  - `std::pair<int, common_json> map_decide_error(int status, const common_json & decide_error_body)`
    for the post-task 400s (3.2): `{400, detail}` for `budget`, `{422, detail}` otherwise;
    other statuses come back unchanged.
- `decide-schema.h`: `decide::error` gains `std::string field;` (empty when not about a
  field); `compile_field`/`build_tries` set it where they already name the field in the
  message. `decide-api`'s only change is the optional `"field"` key in `error_result`
  (the CLI keeps using `handle_decide`/`handle_info`; there is a single translation
  path, in the server route).
- Server: `POST /v1/systemone` (`post_systemone`): every response carries
  `x-typesafe-request-id` (`"llamacpp-"` + a random id); 404 `disabled` before parsing; a
  body above 1 MiB → 413 with a `detail` body (`loc ["body"]`); parse;
  `systemone_to_decide` + `compile_request` in the HTTP thread (422 on failure, no task);
  then post the existing `SERVER_TASK_TYPE_DECIDE` task with the translated body (no new
  task fields). The existing `decide_run` lambda writes straight into the response, so it
  is split: a `decide_task(req, res, body, info) -> std::optional<std::pair<int, std::string>>`
  helper used by `post_decide`, `get_decide_info` and `post_systemone`. It posts the task
  and waits; it returns `nullopt` after handling the two non-result outcomes itself
  (connection closed → return with nothing written, as today; generic task error →
  `res.error(result->to_json())`), otherwise the decide result's `{status, data}`.
  `post_decide`/`get_decide_info` write the pair straight into `res`; `post_systemone`
  applies `map_decide_error` on a 400 and translates the body on 200 with the request's `questions` and `model`. Router mode: `proxy_post`
  (the SDK always sends `model`; with our presets a client must pass `model="qwen3.5-9b"`
  or set `TYPESAFE_DEFAULT_MODEL`).
- `tests/test-decide-compat.cpp` (ctest, no model): request translation golden for the
  docs example (noul with criteria, choice with a null description, score), absent
  `instructions` defaults, object state / object instructions / object criteria
  serialisation, null criteria halves, 422 cases with their `loc` (missing model, missing
  state, empty questions, bad type, choice criteria not an object, a numeric criteria
  value), response translation from a synthetic decide body (choice/score/noul; legend
  keeps an object entry; usage sum; integers), `map_decide_error` for an
  `invalid_schema` body with a field (422) and for `budget` (400), the bounds of 3.2,
  confidence values (certain → 1.0, uniform → 0.0, `[0.5,0.3,0.2]` → 0.25).
- Project: `systemone_smoke.py` — builds the translated `/v1/decide` body in Python by the
  same rules (the test oracle), posts the docs example and triage ticket 0 to both
  endpoints on `:8097` and asserts identical probabilities; `tests/test_systemone_sdk.py`
  — `pytest.importorskip("typesafe_sdk")` and skipped unless `SYSTEMONE_URL` is set; runs
  the official SDK client (`base_url=SYSTEMONE_URL`, any api key, `model=<preset>`,
  `timeout` ≥ 120 s for a cold model) with one choice, one score and one noul question on a
  support ticket and asserts typed answers with probabilities summing to 1 (±1e-6). The
  SDK is not added to `pyproject.toml` (a path dependency would make `uv.lock`
  machine-specific); the README documents
  `uv run --with path/to/typesafe-sdk-python pytest tests/test_systemone_sdk.py`.
  Note: the SDK's default `RetryPolicy` is `max_retries=2` on {408, 429, 5xx} and on
  connection/timeout errors, so an `engine` 500 runs three times and a client timeout on
  a cold model re-posts the request; the test passes `RetryPolicy(max_retries=0)` and a
  timeout ≥ 120 s.
- Docs: README endpoint list + known differences; REPORT-ENGINE.md "Jev compatibility"
  section with the smoke and SDK output; `patches/decide/` regenerated.

## 5. Verification

1. ctest `test-decide-compat` green, no model.
2. Live on `qwen3.5-9b`: `/v1/systemone` with the docs example returns the Jev shape; the
   smoke's oracle body through `/v1/decide` gives identical probabilities.
3. SDK acceptance: the official SDK, pointed at our server, completes the call end to end.
4. 422 path: a request with a bad `type` returns 422 with `loc ["body","questions",id,"type"]`;
   a missing `model` returns 422 with `loc ["body","model"]`.
5. Router mode: works through `--models-preset` (model in the body).
