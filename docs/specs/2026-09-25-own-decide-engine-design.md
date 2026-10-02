# Own parallel-decision engine for llama.cpp — design (sub-project 1 of 3)

Date: 2026-09-25. Status: revision 2; implemented as Phase 2 (`REPORT-ENGINE.md`).

Note for readers of the published repository: this is the design as written before the work; `FINDINGS.md` and "the earlier experiment" are the author's notes and directory of that work and are not part of this repository. The triage set is bundled in `data/` (`data/README.md`).
The baseline fork, a third-party llama.cpp fork with a `POST /v1/decision` endpoint, measured in an earlier phase of
this work (neither the fork nor the scripts that ran it are part of this repository), stays as the baseline this work
is measured against.

## 1. Goal and decomposition

Build our own implementation of the one-pass "decision" mechanism on top of upstream
`ggml-org/llama.cpp`, in our own fork, with a schema designed for the features we want
later. The baseline fork is read as a reference for lessons; no code is copied from it.

Three sub-projects, each with its own spec and implementation:

1. **Engine and parity (this spec).** Core API (`/v1/decide`), engine with tree scoring,
   server hook, CLI, tests, and a regression against the baseline results on the
   80-ticket triage set.
2. **Jev compatibility.** `POST /v1/systemone` translating TypeSafe's `state` +
   `questions` (choice/score/noul) to and from the core API.
3. **Engine improvements**, each independently switchable: full-path scoring,
   sequential fields (`after`), prefix LRU, option-order debiasing.

This spec fixes the core API and the engine interfaces so that 2 and 3 extend without
breaking changes; it implements only what section 9 lists.

## 2. Facts the design rests on

Verified in the llama.cpp tree at `<repo>/llama.cpp` (the baseline
fork, upstream base `60b06ab9a`; line numbers below are from that checkout
and will be re-verified against our pinned upstream commit during implementation).

- **Sequence copy is cheap in a unified KV cache.** `llama_memory_seq_cp` within one
  stream only adds the destination seq id to the cells (`src/llama-kv-cache.cpp:463-491`).
  `kv_unified` in `llama_context_params` selects one stream (`include/llama.h:407-409`).
  Sibling branches need distinct seq ids only; overlapping positions are fine, and each
  branch continues at its trunk's `pos_max + 1` (`src/llama-batch.cpp:288-318`).
- **Recurrent state copies are lazy.** For recurrent/hybrid memory `seq_cp` aliases the
  source row and the duplication happens in the next decode's graph
  (`src/llama-memory-recurrent.cpp:249-284`, `src/llama-graph.cpp:1105-1131`). Every
  sequence owns a state row allocated at context creation, so `n_seq_max` costs memory
  up front on such models.
- **No manual padding.** Hybrid memory splits with `split_equal(n_ubatch, !unified, …)`
  (`src/llama-memory-hybrid.cpp:89`), recurrent with `sequential = true`
  (`src/llama-memory-recurrent.cpp:445`), iSWA tries `split_simple` first
  (`src/llama-kv-cache-iswa.cpp:209`). In every case sequences of different lengths are
  grouped into ubatches inside one `llama_decode` call; the engine never pads.
- **Batch and output limits.** `llama_decode` asserts `n_tokens ≤ n_batch`
  (`src/llama-context.cpp:1734`); the caller chunks. Logit rows are `n_vocab` floats per
  token with `batch.logits[i] != 0`, packed in batch order (`include/llama.h:1038-1049`),
  bounded by `n_outputs_max`; the context asserts `n_seq_max ≤ n_outputs_max` at
  creation (`src/llama-context.cpp:374, 2212`), and the server derives that number as
  `n_parallel · (1 + n_draft)` (`tools/server/server-context.cpp:42-57, 1019-1021`).
  **Consequence: adding decision sequences requires raising the server's output
  reservation by the same amount**, which the baseline fork's hook does at
  `server-context.cpp:52`.
- **Whole-sequence removal never fails**; partial ranges may (`include/llama.h:756`).
- **Prefix snapshots move between sequences.** `llama_state_seq_get_data` /
  `set_data(_ext)` restore into any seq id; the server's slot cache does this
  (`tools/server/server-task.cpp:1832`). Sub-project 3 LRU hook.
- **Server threading.** HTTP handlers post tasks to `queue_tasks`
  (`tools/server/server-queue.h:30`); `llama_decode` runs on the update-loop thread.
  The baseline fork's hook follows this pattern (`SERVER_TASK_TYPE_DECISION`, executed in
  `process_single_task`, `server-context.cpp:2580-2595, 5242-5260`); the only upstream
  precedent for a synchronous task using the context is `SLOT_SAVE/RESTORE`, and the
  generic `is_yielding` decline at `server-context.cpp:2455-2457` covers new task types.
  We follow the same pattern.
- **Unified KV and slots.** With `kv_unified`, every slot's `n_ctx_seq` equals `n_ctx`
  and all slots plus the decision sequences share one pool of `n_ctx` cells
  (`tools/server/server-context.cpp:1218-1235`). Cells held by slot prompt caches are
  not available to the engine.
- **Thinking-off rendering.** `common_chat_templates_apply` with
  `add_generation_prompt = true`, `enable_thinking = false` (`common/chat.h:249-266`);
  `--reasoning off` also sets the template kwarg (`common/arg.cpp:3687-3703`).
- **Tokenisation flags.** `common_tokenize(vocab, text, add_special, parse_special)`
  (`common/common.h:1049-1059`). Rendered templates (Gemma) already contain `<bos>`; the
  baseline fork's engine removes a duplicate BOS (`decision-engine.cpp:185-192`).
- **Routes and router mode.** Routes are registered in `tools/server/server.cpp`; in
  `--models-preset` mode `proxy_post`/`proxy_get` forward by the `model` field or query
  (`tools/server/server-models.cpp:1965`).
- **Build conventions.** `tests/CMakeLists.txt:83-112` `llama_build_and_test`; tools are
  `add_subdirectory` entries linking `llama-common llama`.
- **Measured numerics.** On this laptop the baseline fork's probabilities shift with
  batch shape: median per-ticket queue shift 0.003 (Qwen) / 0.0001 (Gemma), maximum
  0.10, seven argmax flips out of 320 predictions, all near ties (earlier phase). Any
  invariance test must use tolerances of that order, not 1e-3.

Baseline (the baseline fork, this laptop, `results/score-table.txt` in this repository): Qwen3.5-9B
Q4_K_M 92.5/77.5/97.5 % at 99 ms per ticket (batch 1), 95.0/78.8/97.5 % at 89 ms
(batch 5); Gemma 4 E4B Q4_0 90.0/81.2/95.0 % at 60 ms and 92.5/81.2/95.0 % at 32 ms.

Target models are universal: dense (Qwen3.8-27B, Mistral Small), hybrid-recurrent
(Qwen3.5, Nemotron-H), MoE (Qwen3.5-35B-A3B), sliding-window (Gemma 4). The engine never
assumes an architecture; what differs is the memory cost per sequence, which is
reported, not guessed.

## 3. Core API

### 3.1 `POST /v1/decide`

```json
{
  "model": "qwen3.5-9b",
  "instructions": "Answer each question about this support ticket.",
  "fields": {
    "queue":   {"type": "choice", "description": "Which team must act first?",
                "options": {"billing": "Payments, refunds, invoices, charges",
                            "technical": "Bugs, crashes, outages, errors, performance, login problems",
                            "sales": "Contracts, plan changes, pricing questions, cancellation threats, procurement",
                            "feedback": "Praise, feature requests, general non-problem questions"}},
    "urgency": {"type": "score", "description": "How urgent is this support ticket?",
                "levels": ["Can wait: ...", "Normal: ...", "Should be handled today: ...", "Drop everything: ..."]},
    "angry":   {"type": "bool", "description": "The text itself shows anger (hostile wording, shouting, threats, sarcasm aimed at the company)."}
  },
  "states": ["ticket text 1", "ticket text 2"],
  "options": {"scoring": "tree", "order_debias": 0}
}
```

| type | keys | constraints |
|---|---|---|
| `choice` | `options`: object key → description, or array of keys | 2–255 options; keys non-empty, unique |
| `score` | `levels`: array of level descriptions | 2–10 levels; value = index (single digit by construction) |
| `bool` | — | — |

Common: `description` required and non-empty. `after` (array of field names) is
reserved for sub-project 3: 400 `"after" is not supported yet`. `instructions`
optional. `states`: 1–256 non-empty strings. `options.scoring`: `tree` (default) only;
`options.order_debias`: 0 only. Any other value or unknown key: 400 naming it and the
accepted values. `model` required in router mode, ignored otherwise.

`tree` scores every trie node where option paths diverge (exact distribution over the
options given the branching tokens). Sub-project 3's `full_path` additionally scores the
non-branching continuation tokens of every option, i.e. the exact probability of each
option's full string; it is defined there, and it also lifts the sibling-prefix
restriction of 5.2 by scoring to a terminator.

### 3.2 Response

```json
{
  "object": "decide",
  "results": [{
    "answers": {
      "queue":   {"value": "sales", "probabilities": {"billing": 0.1, "technical": 0.2, "sales": 0.6, "feedback": 0.1}, "confidence": 0.6},
      "urgency": {"value": 2, "expected": 1.9, "probabilities": [0.1, 0.2, 0.5, 0.2], "confidence": 0.5},
      "angry":   {"value": false, "p_true": 0.3, "confidence": 0.7}
    },
    "usage": {"state_tokens": 41, "scored_tokens": 3}
  }],
  "usage": {"prompt_tokens": 320, "cached_tokens": 270, "state_tokens": 82, "scored_tokens": 6},
  "timings": {"prefill_ms": 40.1, "scoring_ms": 12.3, "total_ms": 52.4, "rounds": 1, "decodes": 2},
  "engine": {"scoring": "tree", "seqs_per_state": 4, "states_per_round": 5}
}
```

- `results[i]` ↔ `states[i]`.
- `choice`: `value` = argmax key; `probabilities` keyed by option key, all present;
  `confidence` = max probability.
- `score`: `value` = argmax index; `expected` = Σ i·p_i; `probabilities` list in level
  order; `confidence` = max probability.
- `bool`: `value` = `p_true >= 0.5`; `p_true`; `confidence` = max(p_true, 1 − p_true).
- Ties: first option in declaration order wins.
- `usage.prompt_tokens` = prefix + state + branch tokens decoded in this request;
  `cached_tokens` = prefix tokens reused (0 when rebuilt); `scored_tokens` = logit rows
  read. `timings.total_ms` covers work on the update-loop thread, after
  `llama_synchronize`; `decodes` = number of `llama_decode` calls.

Errors: 400 `{"error": {"message": ..., "code": "invalid_schema" | "invalid_options" | "invalid_states" | "budget"}}`;
500 `{"error": {"message": ..., "code": "engine"}}` (5.4, 5.7).

### 3.3 `GET /v1/decide/info`

```json
{
  "decide_seqs": 21, "n_ctx": 4096, "n_batch": 2048, "kv_unified": true, "n_parallel": 1,
  "memory": {"recurrent": false, "swa": false},
  "prefix": {"hash": "…", "tokens": 270, "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "valid": true},
  "scoring_modes": ["tree"]
}
```

`memory.recurrent` / `swa` come from the model's memory type; the per-sequence memory
cost is what llama.cpp logs at context creation (buffer sizes), which the server prints
at startup; the engine adds one log line with `decide_seqs` and the memory type so the
operator can size the flag. `assistant_prefix` is the text the chat template emitted
after the user turn with thinking disabled; it depends only on the template, so the
endpoint renders it even before any schema has been cached (`prefix.valid` is then
false and `tokens` 0). `decide_client.py` compares it with its `EXPECTED_PREFIX` table
(the `/apply-template` probe is retired).

## 4. Prompt rendering (one place, all models)

`decide-schema` renders the catalogue text; `decide-engine` passes system + user
messages through the model's chat template (`add_generation_prompt = true`,
`enable_thinking = false`), with the user content set to a sentinel string containing
`\x1f` that cannot occur in normal text, then splits the rendered string at the
sentinel. If the sentinel is not found once (a template that drops or duplicates the
user content) the engine fails at first use with 500 `engine` naming the template. A
model without an explicit chat template (`common_chat_templates_was_explicit` false;
llama.cpp would silently fall back to ChatML) fails at startup: "no chat template; pass
`--chat-template`".

Catalogue (system message; `instructions` first when non-empty):

```
Answer with one JSON object. Read the input, then fill every field below.

Fields:
- "queue": Which team must act first?
  Allowed values:
    "billing": Payments, refunds, invoices, charges
    "technical": Bugs, crashes, ...
- "urgency": How urgent is this support ticket? Integer level:
    0: Can wait: ...
    3: Drop everything: ...
- "angry": The text itself shows anger (...). true or false.
```

Templates that fold the system message into the first user turn (Gemma) are fine: the
sentinel still marks where the state goes, and everything before it is the prefix.

Trunk tail: the template's generation prompt followed by `{\n`. Branch text per field:
`  "<name>": ` + the option's JSON encoding (`"billing"`, `2`, `true`). Qwen tokenizers
have a single token for `{\n`, and Gemma splits on newlines before merging, so the join
tokenises as `tokenize("{\n") + tokenize(stem)` on both families (unit-tested).

Tokenisation:
- rendered prefix and trunk tail: `parse_special = true, add_special = false`; if the
  vocab wants BOS and the first prefix token is not BOS, prepend it; if the template
  produced two leading BOS tokens, drop one;
- state text: `parse_special = false, add_special = false` (state text cannot inject
  control tokens);
- branch text: `parse_special = false`.

Boundary rule: for each option, stem + value is tokenised as one string; the stem tokens
are the longest common token prefix across the field's options; what follows is the
option's path. Then a sibling check at every trie node: no child token's text may be a
prefix of a sibling child's text (SentencePiece can yield `▁"` and `▁"s` as siblings,
which would let one option absorb another's mass). For `choice` fields a conflict first
triggers the quote-anchored fallback: the stem becomes stem text + `"` tokenised jointly,
and each key + `"` is tokenised on its own after the quote; the response lists such
fields in `engine.quote_split_fields` and the report discusses the tokenisation-boundary
cost. A conflict that survives the fallback (or any conflict in `score`/`bool`) is 400
`invalid_schema` naming the field and the two option keys, with the hint to rename. Paths that
are strict prefixes of one another are impossible for quoted strings and single-digit
levels; the engine still asserts it.

## 5. Engine

### 5.1 Sequence layout

The server creates the context with `kv_unified = true`,
`n_seq_max = n_parallel + decide_seqs`, and its output reservation raised by
`decide_seqs`. Slots keep `0..n_parallel-1`; the engine owns
`[n_parallel, n_parallel + decide_seqs)`:

| role | count | content |
|---|---|---|
| prefix `P` | 1, persistent | rendered prefix tokens; rebuilt when the prefix hash changes or `P` is found invalid |
| trunk `T_c` | one per state in the round | `seq_cp(P → T_c)` + state tokens + trunk tail |
| branch `B_c,f,k` | ≥1 per field | `seq_cp(T_c → B)` + branch tokens up to trie node `k`; logits at the last token |

### 5.2 Token trie and branches

Per field, option paths (section 4) form a trie under the stem. A branch is created for
every trie node with two or more children, carrying the tokens from the stem to that
node; its logit row gives the distribution over the children. A field whose options
diverge at the first token has one branch. `bool` uses `true`/`false`; `score` uses
`0..n-1`.

### 5.3 Rounds, budget, batching

- `seqs_per_state = 1 + Σ_f branches(f)`.
- `states_per_round = (decide_seqs − 1) // seqs_per_state`; 0 → 400 `budget`
  ("schema needs K sequences per state, budget is S").
- Cell check before each round: `held + prefix_tokens + Σ_c (state_c + tail + Σ_b
  branch_b) ≤ n_ctx − 16`, where `held = Σ over slot seq ids of (pos_max + 1)` (cells
  slot prompt caches currently occupy). If it fails, `states_per_round` is reduced; if
  one state does not fit → 400 `budget` with the numbers. If `llama_decode` still
  returns 1 (no cell space), the engine removes its trunks/branches, halves
  `states_per_round`, retries once, and otherwise fails with 500 `engine`.
- `rounds = ceil(N / states_per_round)`. Each round: trunk phase (all trunks), branch
  phase (all branches), then `seq_rm` of every trunk and branch.
- Every phase is submitted in chunks of at most `n_batch` tokens, whole sequences never
  reordered, and the logit rows of a chunk are read right after its `llama_decode`
  (the logits buffer belongs to one call). Trunk chunks request no logits.

The context created for a decide-enabled server should run with `--parallel 1` unless
the operator wants chat traffic on the same instance; the info endpoint reports
`n_parallel` and the engine logs a warning at startup when `n_parallel > 1` that slots
and decisions share `n_ctx` cells.

### 5.4 Scoring

For every branch tail, take the child tokens' logits from the `n_vocab` row,
log-softmax over exactly those, and add each child's log-probability to the paths
through it. An option's log-probability is the sum along its path; the field's
distribution is the softmax of the option log-probabilities. Float32 log space.

Degenerate cases: a child logit of `-inf` gives probability 0 and the field
renormalises; all options `-inf` → 500 `engine` naming the field. The distribution
sum is asserted within 1e-4 of 1 in debug builds and in the unit tests.

### 5.5 Prefix management and hash

Prefix hash = FNV-1a 64-bit of the **rendered prefix and tail text** (llama.cpp ships no SHA-1) (so a template that inserts the
date changes the hash, and the same schema with the same rendering hits). Before each
request the engine checks `P` is valid: `llama_memory_seq_pos_max(P) == prefix_tokens −
1`; otherwise it rebuilds. The engine object is created after the context and reset
whenever the server recreates the context (sleep/resume path,
`server-context.cpp:959-975`). Sub-project 3 replaces "clear and rebuild" with a
`get_data`/`set_data` LRU keyed by the same hash.

### 5.6 Server and common integration

- `common_params` gains `int32_t decide_seqs = 0`; `--decide-seqs N` (env
  `LLAMA_ARG_DECIDE_SEQS`, preset key `decide-seqs`) in `common/arg.cpp`; minimum 3 when
  non-zero.
- `common_context_params_to_llama`: when `decide_seqs > 0`, `kv_unified = true`,
  `n_seq_max += decide_seqs`. The server's output reservation adds `decide_seqs`. The
  speculative/draft params copy (`common_base_params_to_speculative`,
  `common/speculative.cpp:2467-2501`) sets `decide_seqs = 0`, so the draft context gets
  neither the extra sequences nor the raised reservation; a test with `--model-draft`
  plus `--decide-seqs` covers this.
- New task type `SERVER_TASK_TYPE_DECIDE` carrying the compiled request; the HTTP
  handler validates (400s without touching the model), posts the task, waits for the
  result. The update loop executes the engine synchronously when it dequeues the task;
  slots are not advanced during that call. Mixed batching with chat slots is out of
  scope.
- Routes `POST /v1/decide`, `GET /v1/decide/info` (router mode: proxied by `model` body
  field / `?model=` query). `--decide-seqs 0`: the routes answer 404.
- The engine lives in `server_context`, constructed after model load and after every
  context (re)creation, destroyed before context free.

### 5.7 CLI `llama-decide`

`llama-decide -m model.gguf --decide-seqs N [-c N] [-b N] [-ngl N] [-fa on]`: creates
its own context through the same `common` path as the server with `n_parallel = 1`
(`kv_unified = true`, `n_seq_max = 1 + decide_seqs`, output reservation raised to
match; seq 0 unused, engine base 1); reads one JSON request per line on stdin (the `/v1/decide`
body without `model`), writes one response per line, exits non-zero on the first error
with the error JSON on stderr. `--info` prints the info JSON for a request's schema and
exits. `--dump-tokens` makes every response line carry a `tokens` object: `prefix`
(ids), and per state `state` (ids), `tail` (ids) and per branch `branch` (ids) with the
`children` token ids scored at its tail, so `reference_check.py` can replay the exact
token sequence through `/completion`. It is the test surface: no HTTP, no slots, same
engine object.

## 6. Components and files (our llama.cpp fork, branch `decide`)

```
tools/decide/
  CMakeLists.txt          static lib llama-decide (schema, trie, engine, api) + llama-decide CLI
  decide-schema.{h,cpp}   JSON → compiled_schema (fields, option encodings); catalogue rendering; JSON-level 400s
  decide-trie.{h,cpp}     tokenisation with the boundary rule, sibling check, trie construction, branch enumeration (needs llama_vocab)
  decide-engine.{h,cpp}   prefix, budget, rounds, chunked batches, decode, logit read-out, aggregation
  decide-api.{h,cpp}      request → compiled request (validation), response JSON, info JSON
  decide-cli.cpp          llama-decide
common/                   decide_seqs param, arg, context-param mapping (one commit)
tools/server/             DECIDE task, routes, proxy wiring, output reservation (one commit)
tests/
  test-decide-schema.cpp  JSON validation 400s, catalogue golden text (no model)
  test-decide-math.cpp    aggregation from hand-written logits, expected, ties, -inf (no model)
  test-decide-trie.cpp    boundary rule, sibling check, branch counts on two tokenizer families (small GGUFs; skipped when absent)
```

Interfaces sub-projects 2 and 3 rely on:

- `compiled_schema decide::compile(const json & fields, const std::string & instructions, const decide_options &)` — JSON-level validation, throws `decide_error{code, message}`.
- `std::string decide::render_catalogue(const compiled_schema &, int variant)` — variant 0 now; debiasing adds permutations.
- `trie_set decide::build_tries(const compiled_schema &, const llama_vocab *)` — throws for sibling-prefix conflicts.
- `decide_result engine::decide(const compiled_schema &, const trie_set &, const std::vector<std::string> & states)` — per-state answers, usage, timings.
- `json engine::info(const compiled_schema *)`.

Project repo (this worktree): `build-engine.sh` (clone `ggml-org/llama.cpp` at a
pinned commit into `engine/`, check out our branch `decide` — local until a remote is
provided — build `llama-server`, `llama-decide` and the tests with the CUDA flags of the
baseline build), `models-engine.ini` (`decide-seqs = 21`, `parallel = 1`),
`decide_client.py` (`/v1/decide` client, same output format as the baseline runs, prefix check via
`/v1/decide/info`), `reference_check.py` (8.3), `results-engine/`, `REPORT-ENGINE.md`.

## 7. Error handling summary

| condition | response |
|---|---|
| JSON/schema/option/state validation | 400, before any model work |
| `after`, `scoring ≠ tree`, `order_debias ≠ 0`, unknown keys | 400 `invalid_options` / `invalid_schema` |
| sibling-prefix trie conflict | 400 `invalid_schema`, field and two keys named |
| budget too small for one state, or one state exceeds free cells | 400 `budget` with numbers |
| sentinel not found in rendered template; all options `-inf` | 500 `engine` |
| `llama_decode` failure after the one retry | 500 `engine` with the return code |
| endpoint disabled | 404 |
| router mode (`--models-preset`), body that is not valid JSON | 500 from the router itself (upstream `proxy_post` parses the body for `model` before any child handler runs); a single-model server returns our 400 |

Sequence hygiene: every exit path removes the request's trunks and branches. The prefix
`P` survives a failed request unless the failure happened while building `P`; the next
request re-validates `P` (5.5) and rebuilds if needed.

## 8. Verification

1. **Unit tests, no model (ctest):** every JSON-level 400 of section 7; catalogue golden
   text for the triage schema; math: 4-way distribution from given logits, `expected`,
   first-option tie, one `-inf` option, all `-inf`; sum within 1e-4.
2. **Unit tests with small GGUFs (`Qwen2.5-0.5B-Instruct-Q8_0` and a Gemma family
   tokenizer, both present under `~/.lmstudio/models` or `models/`; skipped when
   absent):** the `angry` field yields paths for `true` and `false` that differ and pass
   the sibling check; the triage fields each produce exactly one branch; a schema
   engineered to violate the sibling check is rejected with both keys named; the
   trunk-tail/branch join tokenises without a token spanning the join.
3. **Engine integrity (CLI + `reference_check.py`, real models):**
   - **independent reference:** for 20 states × 3 fields, the engine's `tree`
     distribution is compared with the distribution read from `llama-server
     /completion` given the **same token ids** (the client tokenises nothing; the engine's
     `--dump-tokens` emits the prefix + state + branch token ids), `n_predict 1`,
     `n_probs 512`, `post_sampling_probs` off, renormalised over the child tokens, every
     child required to be present. Pass: median absolute difference ≤ 0.01, no argmax
     disagreement where the reference top-2 margin exceeds 0.15;
   - **batch invariance:** each state alone vs inside a batch of 5, same statistic and
     thresholds, on the 0.5B model and on `Qwen3.5-4B` (hybrid);
   - **rounds and budget:** a request needing more sequences than the budget returns
     400; 12 states with `states_per_round 5` reports `rounds 3` and its answers match
     three separate requests of 5/5/2 states under the same statistic and thresholds
     (cell layout differs between the two runs, so bit-identity is not expected).
4. **Parity regression (80 tickets, Qwen3.5-9B Q4_K_M and Gemma 4 E4B Q4_0, batch 1
   and 5).** Gate: p50 per ticket ≤ 1.5× the baseline number for the same configuration,
   and all four runs complete with `rounds 1` for batch 5. Accuracy is **reported, not
   gated**: the prompt text differs from the baseline's catalogue (which packs option
   definitions into a field description), and the baseline showed prompt changes move
   results by 8–10 tickets while batch shape moves them by 2. The report lists per-ticket
   disagreements between the two engines and the two rendered prompts side by side. A
   metric more than 5 points below the baseline row triggers a one-off ablation: the
   baseline's catalogue text passed as `instructions` with one-word descriptions, to
   separate prompt from engine.
5. `REPORT-ENGINE.md`: setup, the four rows next to the baseline rows, latency
   breakdown, the integrity results, the prompt diff, the ablation if it ran (the published
   report does not include the prompt diff and the ablation).

## 9. Scope of this sub-project

In: sections 3–8 with `tree` scoring, one cached prefix, no `after`, no debiasing; two
models on this laptop; CLI; tests; regression; report.

Out (later sub-projects): `/v1/systemone`; full-path scoring; sequential fields; prefix
LRU; order debiasing; mixed batching with chat slots; multi-GPU and MoE-offload
performance work (the engine must run there; speed is not a goal here).

## 10. Risks

- Upstream `tools/server` and `common` change often; the hooks are two small commits on
  a pinned upstream commit so rebases stay tractable.
- Hybrid models: `split_equal` may turn one branch chunk into several ubatches;
  correctness unaffected, latency measured.
- Gemma's sliding-window cache allocates per sequence; `decide-seqs 21` may not fit on
  8 GB with larger contexts. The budget error and the info endpoint make this visible.
- The sibling check may reject option sets that the baseline fork accepts (it scores whatever
  token diverges first); this is deliberate, the error names the fix.
- Batch-shape numerics: probabilities near ties flip between shapes; every comparison
  in section 8 uses statistical thresholds derived from the measured baseline.
