# Sub-project 3: engine improvements — design

Status: rev 5. Parent: `2026-09-25-own-decide-engine-design.md`
(sub-project 1, §1 decomposition: "full-path scoring, sequential fields (`after`), prefix
LRU, option-order debiasing", each independently switchable). Sibling:
`2026-09-25-systemone-compat-design.md` (sub-project 2). Code references are to the
engine fork `engine/` at `f042104fa` (E = `tools/decide/decide-engine.cpp`, T =
`decide-trie.cpp`, S = `decide-score.cpp`, SC = `decide-schema.cpp`).

## 1. Goal and switches

Add five improvements to the decide engine. When a request uses none of the per-request
switches, it gets the same probabilities as the current build (max abs diff ≤ 1e-6)
whenever its prefix is resident or built into the same cells; a prefix restored from the
cache (feature 3), or rebuilt after other cells were allocated in between, can land in
other KV cell positions. Some models are sensitive to cell placement: in the
moved-cells checks of `results-engine/sp3/checks/task-4/` on Qwen2.5-0.5B, probabilities moved by up to 0.039 after a restore
(the statistic passed) and by up to 0.074 after a plain rebuild (the statistic failed on
`angry`, median 0.0139). Such results are therefore compared under the parent spec's
batch-invariance statistic; an A/B/A sequence without interleaved cells gives identical
results (§12.3).

| # | feature | switch | default | cost when on |
|---|---|---|---|---|
| 1 | full-path scoring | request `options.scoring: "full_path"` | `"tree"` | one branch sequence per option instead of per branch node; more logit rows |
| 2 | sequential fields | field `"after": [names]` | none | one extra decode phase per dependency level |
| 3 | prefix cache | server/CLI `--decide-prefix-cache N` | `8` | host RAM for up to N prefix snapshots (≤ 2 GiB total) |
| 4 | option-order debiasing | request `options.order_debias: K` | `0` (off) | K prefixes (K−1 of them restored from the cache or rebuilt on every request, §6.1), K× trunks and branches |
| 5 | temperature | request `options.temperature: T`; server/CLI `--decide-temperature T` | `1.0` | none |

Plus hardening (§9): backend-sampler detachment for decide sequences, a fault-injection
hook that exercises the `rc == 1` retry path live, and the `--decide-seqs` bound checked
where `n_parallel` is final.

Out of scope: non-canonical tokenisations of an option (§4.8), mixed batching with chat
slots, per-field temperatures, exposing features 1, 2 and 4 through `/v1/systemone`
(Jev's schema has no place for them; features 3 and 5 reach it through the server
defaults).

## 2. Facts the design relies on

- **Output rows per decode are capped.** One `llama_decode` with more output rows than
  `cparams.n_outputs_max` aborts on a `GGML_ASSERT` (`src/llama-context.cpp:1805,2059,2212`).
  `n_outputs_max ≥ n_seq_max` holds in the server and the CLI (`llama-context.cpp:375`,
  `common/common.cpp:1750-1757`), so `llama_n_seq_max(ctx)` is a safe per-decode row cap.
  Tree mode never reaches it; full-path mode can.
- **Rows per sequence are capped when a backend sampler is attached.** With
  `--backend-sampling`, `common_init_from_params` attaches samplers to every sequence
  (`common.cpp:1385-1395`); `llama_decode` then fails when one sequence has more than
  `n_outputs_max_per_seq` rows (`llama-context.cpp:1687-1711`). Full-path puts several rows
  on one branch sequence. `llama_set_sampler(ctx, seq, nullptr)` (`include/llama.h:1339`)
  detaches a sequence.
- **Sequence ids.** Today: prefix P = `seq_base`; trunk c = `seq_base+1+c·sps`; its
  branches follow (E:136,204,212,246); `sps = 1 + branches_per_state`,
  `spr = (decide_seqs − 1) / sps` (E:307-313). `LLAMA_MAX_SEQ` = 256.
- **Splitting a sequence's tokens across decodes** (continuing at the next position in the
  next `llama_decode`) is valid for attention, iSWA, recurrent and hybrid memory.
- **Snapshots.** `llama_state_seq_get_size/get_data/set_data` (`include/llama.h:877-896`)
  serialise one sequence (attention KV, iSWA KV without masked cells, recurrent state,
  hybrid both) into host memory and restore it into any sequence id. `set_data` returns 0
  on failure; hybrid and iSWA memory restore their parts one after the other
  (`llama-memory-hybrid.cpp:197-202`, `llama-kv-cache-iswa.cpp:267-273`), so a failure in a
  later part can leave cells of an earlier part behind. Restored cells are placed by
  `find_slot(cont = false)` (`llama-kv-cache.cpp:2419`), not necessarily where they were.
  Restore followed by `seq_cp` works for recurrent memory. Flags 0 (host copy) are used.
- **Recurrent memory scales with sequences.** Hybrid models keep one recurrent state per
  sequence id (about 63 MiB per sequence on Qwen3.5-35B-A3B, REPORT-ENGINE.md MoE
  section), so features that need more sequences (1, 4) must fail with a clear `budget`
  error rather than assume a large pool.
- **Retries.** A round that gets `rc == 1` is cleaned up and retried once with
  `spr = max(1, spr / 2)`; a second failure of the same round is a 500 (E:366-385). The
  `retried` flag is per round.
- **Catalogue order.** `render_catalogue(cs, variant)` ignores `variant` today (SC:145);
  the branch text (`stem_text + encoded`) does not depend on the catalogue, so one trie set
  serves every catalogue variant.

## 3. API changes

### 3.1 Request

- `options.scoring`: `"tree"` (default) or `"full_path"`.
- `options.order_debias`: integer 0–8; 0 and 1 mean off (one catalogue).
- `options.temperature`: number in [0.05, 20]; absent → the server default
  (`--decide-temperature`, default 1.0).
- Field `after`: array of 1–8 distinct names of other fields in the same request.
  Validation (400 `invalid_schema`, `error.field` = the dependent field; for a cycle, the
  first field in declaration order on the cycle): not an array, empty, a non-string
  entry, a duplicate, an unknown name, the field itself, a cycle (message lists it, e.g.
  `after: cycle queue -> urgency -> queue`), or a dependency level above 8 (§5.1).
- Other option values or unknown option keys: 400 `invalid_options` as today, messages
  `options.scoring must be "tree" or "full_path"`, `options.order_debias must be an
  integer from 0 to 8`, `options.temperature must be a number from 0.05 to 20`.

### 3.2 Response

Per-field answer keys added only when the feature is on (existing keys unchanged):
- `coverage` (full-path): the probability, given the field's trie stem, that the model
  continues with one of the options and ends the value (§4.1). The trie stem is the
  common token prefix of all options' tokenisations, so for choice fields it can already
  contain the opening quote and shared leading pieces of the values; coverage values are
  therefore not comparable across fields. With debiasing, the mean over variants.
- `order_spread` (K ≥ 2): max over options of (max over variants − min over variants) of
  that option's probability (before temperature).
- `given` (after): object `{ancestor name: its answer value}` for every field in the
  dependent field's ancestor set (§5.2), in declaration order.

`engine` block: `scoring`, `order_debias` (effective K, §7.1), `temperature`, `phases`
(1 + deepest `after` level), `seqs_per_state` (all sequences one state uses, all variants
included: `K·(1 + B)`), `states_per_round`, `prefix_cache` = `{capacity, entries, hits,
misses}` (hits/misses of this request), `resplit_options` (full-path, §4.8; omitted when
empty). `timings`: `prefill_ms` keeps its meaning (prefix and trunk work), adds
`prefix_ms` (the part of `prefill_ms` spent building, restoring or snapshotting prefixes)
and `retries` (number of rounds retried). `usage.cached_tokens` counts prefix tokens that
were resident or restored; `usage.prompt_tokens` counts tokens actually decoded;
`usage.scored_tokens` and `results[].usage.scored_tokens` count logit rows read.

### 3.3 `GET /v1/decide/info` and flags

Info adds `scoring_modes: ["tree", "full_path"]`, `temperature` (server default),
`prefix_cache` = `{capacity, entries, bytes}`, and `prefixes` = one object per resident
prefix slot `{slot, hash, tokens, valid}`; the existing `prefix` object stays (slot 0).
A slot is listed when it is valid or its sequence holds cells (`seq_pos_max ≥ 0` or
`seq_pos_min ≥ 0`, so cells in only one part of hybrid memory show up; a slot listed only
through `seq_pos_min` reports `tokens` 0, since it has cells in one memory part only); iSWA reports only
its SWA part, so cells held only in its base part stay invisible here. Trunk and branch ids
are not listed.
With a schema body, only slot 0 (variant 0) is built or validated, and `schema` reports
`seqs_per_state` and `states_per_round` for the body's `options` (scoring, K) and `after`.

New `common` parameters, registered exactly like `--decide-seqs` (server + completion
examples, env vars `LLAMA_ARG_DECIDE_PREFIX_CACHE`, `LLAMA_ARG_DECIDE_TEMPERATURE`),
copied into `engine_config` by the server and the CLI, defaulted in the speculative copy:
`--decide-prefix-cache N` (0–16, default 8) and `--decide-temperature T` (0.05–20,
default 1.0).

## 4. Full-path scoring

### 4.1 Definitions

- **Value-end predicate** `ends(w)` for a string `w` that directly follows a JSON value
  inside an object: `w` is non-empty and either consists only of whitespace
  (` `, `\t`, `\n`, `\r`), or its first non-whitespace character is `,` or `}`.
- **Vocabulary indexes**, built once at engine construction over every non-control token
  `u` with piece `s = common_token_to_piece(vocab, u, false)`: `END = {u : ends(s)}`;
  `MERGED[b] = {u : s = b + r, b non-empty, ends(r)}` for every split of `s` (a hash map
  from base string `b` to token list).
- For an option `o` with trie path `t_1..t_k` (the tokens after the field's trie stem,
  exactly as the trie builds them today) and logit rows `r_0` (after the stem's last token)
  … `r_k` (after `t_k`), with `p(t | r)` the full-vocabulary softmax of row `r`:

  `log P(o) = Σ_{i=1..k−1} log p(t_i | r_{i−1}) + log( p(t_k | r_{k−1}) · Σ_{v∈END} p(v | r_k) + Σ_{u∈MERGED[piece(t_k)]} p(u | r_{k−1}) )`

  The last factor is the probability that the model writes `t_k` and then ends the value,
  or writes one token that completes `t_k` and ends the value inside it (a closing quote
  merged with `,` or a newline, e.g. `",` or `",\n`). The two terms are disjoint events,
  and for valid JSON encodings no `MERGED` token is another option's child at the same
  node.
- **Distribution** `P̂(o) = P(o) / Σ_o' P(o')`; **coverage** `= Σ_o P(o)`. All in log
  space (log-sum-exp). All `P(o) = 0` → 500 `engine` naming the field (as tree mode).
  The log-probabilities are stored as floats, so the sum can land just above 1: a coverage
  in (1, 1 + 1e-6] is reported as exactly 1, in a single answer and in the debias mean
  (§7.2); larger values are left alone.

### 4.2 Why it lifts the sibling-prefix restriction

Tree mode rejects sibling children whose pieces are string prefixes of each other
(`sale`/`sales`, T:146-170) because the child probability of `sale` also contains the
model's intent to continue with `s`. Full-path scoring multiplies by the value-end
probability after the option's last token, so "sale then continue" no longer counts as
"sale". In full-path mode the sibling check and the quote-split fallback are skipped; the
distinct-path check (T:117-129) stays (it guards against tokenisers that map distinct
keys to the same path).

### 4.3 Trie and sequences

`build_tries(cs, vocab, scoring)` gains the scoring mode. For full-path fields the trie
records, per option, its leaf and the nodes on its path. **One branch sequence per
option**: `seq_cp(trunk)` + `branch_tokens(t, leaf)` (stem + all path tokens), with logit
rows requested at the stem's last token (row `r_0`) and at every path token. A node's row
is requested only in the first option (declaration order) whose path contains it, so
every node is read exactly once. Per state and variant, `B = Σ_f #options_f` in
full-path mode (tree: `B = Σ_f #branch_nodes_f`, unchanged).

### 4.4 Rows, chunking and scoring

- `decode_chunked` flushes a chunk when its tokens reach `n_batch` (today) **or its
  requested rows reach `llama_n_seq_max(ctx)`**. A chunk boundary may split one branch
  sequence; its later tokens continue at the next position in the next chunk.
- The row callback computes the full-vocabulary log-sum-exp of the row once and stores,
  per node: the log-probabilities of its children's tokens; for a leaf (row `r_k`)
  `log Σ_{v∈END} p(v)`; for a leaf's parent, per leaf child `t_k`,
  `log Σ_{u∈MERGED[piece(t_k)]} p(u)` (−inf when empty). The row pointer is not kept past
  the callback.
- `score_field_full(field, trie, node_logp) → field_answer` implements §4.1; it validates
  that every node on every path has its row (500 `engine` otherwise), rejects NaN/+inf
  (500), and returns the distribution and coverage.

### 4.5 Budget

The per-state cell check and `need_for` use the full-path token counts (Σ over options of
stem + path tokens). If `spr = 0`, the unified sequence-budget message of §7.3.

### 4.6 Backend sampling

At construction the engine calls `llama_set_sampler(ctx, s, nullptr)` for every decide
sequence `s ∈ [seq_base, seq_base + decide_seqs)`, so a server or CLI started with
`--backend-sampling` does not cap its rows per sequence (the call detaches an attached
sampler and is a no-op otherwise).

### 4.7 Dump format

`--dump-tokens` output is described once for all features in §10.

### 4.8 Known approximations

- Only the canonical tokenisation of each option (the joint tokenisation of stem + value)
  is scored; other token splits of the same value are not added.
- In context the value is followed by a terminator, and the tokeniser can re-split more
  than the last token (merges that `MERGED` does not cover because they reach back past
  `t_k`). At trie build the
  engine tokenises `stem + encoded + ",\n"` jointly; when its tokens up to `t_{k−1}` differ
  from the path, the option is listed in `engine.resplit_options` (and in the dump) and
  scored with the end-of-string path as usual.

## 5. Sequential fields (`after`)

### 5.1 Levels

The `after` graph is a DAG (validated, §3.1). Level of a field = 0 without `after`,
otherwise 1 + max level of its `after` fields; level > 8 is rejected. Fields are scored
level by level; each level ≥ 1 is one more decode phase per round (`engine.phases`
= 1 + max level).

### 5.2 Context lines

Ancestor set `anc(F)` = transitive closure of `F.after`. For a state, the dependent
field's branch text is `lines(F) + stem_text(F) + encoded(o)` where `lines(F)` is the
concatenation over `G ∈ anc(F)` in declaration order of
`stem_text(G) + encoded(v_G) + ",\n"`, and `v_G` is G's answer value for the same state
(after G's own conditioning and debias averaging; temperature does not change it). The
dependent field therefore sees exactly its declared ancestors, in declaration order,
between the opening `{` and its own key — not the other fields and not the catalogue
order.

### 5.3 Tries per context

For each (field F, lines text) the engine builds F's context trie with the usual rule
applied to the joint tokenisations of `lines + stem_text(F) + encoded(o)` for every option:
the stem is their longest common token prefix and the paths are the remainders (the lines
join the stem, and the stem keeps absorbing whatever value tokens all options share, e.g.
` "` or a leading space, exactly as in the lines-free trie). The context trie must have
the same paths as F's lines-free trie (built once per request with the usual rules,
including the sibling check and the quote-split decision); nodes, branch nodes and the
full-path plan are then reused. A quote-split field keeps its rule (stem part
`lines + stem_text(F) + "\""` tokenised jointly, key parts as today), so its paths are
equal by construction. Any path difference → 500 `engine` naming the field (`answer lines
change the tokenisation of field "urgency"; remove its "after"`). The tail|new-stem join is
checked like today's tail|stem join. Context tries are cached per request by (field,
lines text). Branch sequence ids stay the ones pre-assigned per field (E:246); level-0
branches are kept until the round's cleanup, so the ids never collide.

### 5.4 Budget

Before a round, the per-state cell check and `need_for` use a worst case for each
dependent field: Σ over ancestors of (max over the ancestor's options of the token count
of the jointly tokenised line `stem_text(G) + encoded(o) + ",\n"`) + 2 tokens slack per
ancestor. After phase L, the exact token counts of phase L+1 are known; if they exceed
the reserved cells, the round fails with 500 `engine` (`cell estimate exceeded`) before
decoding — a defect signal, not an expected path.

### 5.5 Phases in a round

Prefill trunks → phase 0: branches of level-0 fields → read rows, score level 0 →
phase 1: build lines per state, branches of level-1 fields → read, score → … Every exit
path still removes all trunks and branches (the existing `catch(...)` + `cleanup`). With
debiasing, each variant runs its own branches of a dependent field with the same lines
(from the averaged ancestor answers).

## 6. Prefix slots and prefix cache

### 6.1 Slots

The engine has up to 8 prefix slots: slot v lives in sequence `seq_base + v`. A request
with K catalogue variants (K = 1 without debiasing) uses slots 0..K−1. State c of a round
owns the block of `K·(1 + B)` sequence ids starting at `seq_base + K + c·K·(1 + B)`;
inside the block, variant v's trunk is at offset `v·(1 + B)` and its B branches follow.
With K = 1 this is exactly today's layout. `render_catalogue(cs, variant, variants)` and
`prepare_prefix(cs, variant, variants)` take the variant; `(0, 1)` is today's rendering.
Slot hygiene:
- at request start (after the budget checks), resident slots ≥ K are evicted, and every id
  in `[seq_base + K, seq_base + decide_seqs)` is cleared with `llama_memory_seq_rm`
  (whole-sequence removal never fails; ids without cells cost a scan) so no trunk can
  inherit stale cells;
- at the end of every request, on the success path and on every error path (a
  request-wide guard — RAII or a try/catch spanning the whole request, since some 500s
  are thrown outside the per-round `catch(...)`, e.g. the second `rc == 1` at E:380),
  slots ≥ 1 are evicted (§6.2), so between
  requests only slot 0 holds cells, as today. A debias request therefore restores (cache
  on) or rebuilds (cache off) its K_eff − 1 extra prefixes each time; the default cache
  size of 8 holds the variants of two K = 4 schemas.

### 6.2 Cache

`--decide-prefix-cache N`: an LRU map `hash → {snapshot bytes, prefix_state}` with at most
N entries and at most 2 GiB of snapshot bytes (0 = off: today's clear-and-rebuild).
- **Order in a request**: tokenise the wanted prefixes, run the budget checks (held cells
  + the wanted prefixes' tokens + the round), and only then evict and fill slots; a
  request that fails a budget check leaves every resident slot untouched.
- **Evict slot v**: if N > 0, slot v is valid and its hash is not cached, snapshot it
  (`get_size`, `get_data`, flags 0) into the cache, evicting least recently used entries
  while the entry count or byte total is over the limit (an entry larger than 2 GiB is not
  stored); then `llama_memory_seq_rm(slot v)`. The trim never evicts an entry this request
  wants (the hashes of all slots it fills, collected before the first eviction); when only
  wanted entries would remain over the limit, the new snapshot is not stored. The decision
  comes before the copy: from the entries' LRU order, wanted flags and sizes and the new
  snapshot's size (`get_size`) the engine decides whether the snapshot is stored and which
  entries go. If it is not stored, nothing is erased and nothing is copied off the GPU;
  otherwise the victims are erased first and the snapshot is copied after, so host memory
  stays within the limits. If that copy fails, the entry is not stored and the erased
  victims stay erased.
- **Fill slot v with prefix `ps`**: if the hash is cached, `set_data` into slot v and check
  `ret != 0 && seq_pos_max == tokens − 1`; on failure call `llama_memory_seq_rm(slot v)`
  (hybrid/iSWA may have restored a part), drop the entry and build. Otherwise build
  (decode) as today. A cached entry becomes most recently used on every hit.
- `prefix_ms` covers snapshot, restore and build time; `cached_tokens` counts restored and
  resident prefix tokens.

### 6.3 Hash

Unchanged (FNV-1a of prefix text + tail text). Each catalogue variant has its own hash.

## 7. Option-order debiasing

### 7.1 Variants

`order_debias: K ≥ 2` renders `K_eff` catalogues, `K_eff = min(K, n_max)` where `n_max` is
the largest option count among the request's choice fields; without choice fields
`K_eff = 1` (reported as `engine.order_debias: 1`). Variant v (0 ≤ v < K_eff) lists each
choice field's options rotated left by `floor(v·n / K_eff)` positions (n = that field's
option count); score levels and booleans keep their order. Every option appears in every
position equally often when n divides `K_eff`; otherwise as evenly as that formula allows
(documented; K = n is the recommended setting). Variant 0 equals today's catalogue.

### 7.2 Execution

Slots 0..K_eff−1 hold the variant prefixes (built, resident or restored). Each state gets
`K_eff` trunks (`seq_cp(P_v)` + state + tail) and, per trunk, the branch sequences of the
request's scoring mode, decoded in the same phases. The trie set is shared. Per field,
the `K_eff` distributions (declaration order) are averaged arithmetically; `value`,
`expected`, `confidence` come from the average (then temperature); `order_spread` from the
`K_eff` distributions; full-path coverage is the mean.

### 7.3 Budget

Per state `K_eff·(1 + B)` sequences, `spr = (decide_seqs − K_eff) / (K_eff·(1 + B))`;
cells: held + Σ_v |P_v| + per state `K_eff·(state + tail + branch tokens)`. `spr = 0` → 400
`budget` with one message for every mode (it replaces today's text at E:310):
`schema needs S sequences per state (K_eff K × (1 trunk + B branch sequences, <mode>)); --decide-seqs D leaves A after K prefix slots`
with S = `K_eff·(1 + B)`, A = `max(0, D − K_eff)`, `<mode>` = `tree` or `full_path`; e.g. the §12.3
all-switch request on Qwen3.5-4B: `schema needs 44 sequences per state (K_eff 4 × (1 trunk +
10 branch sequences, full_path)); --decide-seqs 32 leaves 28 after 4 prefix slots`.

## 8. Temperature

Per field, after debias averaging: `p_T(o) ∝ exp(log p(o) / T)`; `value` (argmax) is
unchanged; `probabilities`, `confidence`, `expected`, `p_true` use `p_T`; `coverage` and
`order_spread` are reported before temperature. T comes from `options.temperature`, else
`--decide-temperature`. `/v1/systemone` gets the server default. A T fitted on tree-mode
probabilities also becomes the default for the other modes; requests can override it.

`calibrate.py` (project): fits one T per model by minimising the summed NLL of the three
triage fields on the 320-ticket train set, from the stored probabilities of the
tree-mode b5 train run with the client's default prompt variant (post-hoc scaling equals
the engine's computation); grid over [0.25, 4] (64 log-spaced points) then golden-section
refinement; reports train and test NLL, ECE (10 bins) and Brier before and after. Rule
(fixed before looking at the test set): the fitted T goes into that model's preset in
`models-engine.ini` as `decide-temperature` only if it lowers the test-set summed NLL.
Calibration runs last in the evaluation (§12.4), so every other run uses T = 1.

## 9. Hardening

- **Backend samplers**: §4.6.
- **Fault hook.** Environment variable `LLAMA_DECIDE_FAULT` (read at engine construction;
  documented in the CLI help): `rc1` makes the first branch-phase decode of the first
  attempt of each round return 1 without decoding (the retry of that round runs
  normally); `rc1-twice` fails the first branch-phase decode of both attempts of the first
  round of the first request after construction only (that request ends in the 500
  `no KV cell space`, exercising the error-path cleanup; later requests run normally); `rc1-phase1` does the same as `rc1` for the first phase-1 decode (`after`); `restore`
  hands the first cache restore after construction a snapshot one byte short, so
  `set_data` fails and the restore-failure path of §6.2 runs (later restores run
  normally); `cell-estimate` makes the first exact post-phase cell check (§5.4) behave as
  exceeded, `lines-join` makes the first context join check (§5.3) fail at its call site,
  and `tail-mismatch` makes the first tail comparison of a catalogue variant v ≥ 1 fail;
  each of these three ends its request in the 500 `engine` of that path (`cell estimate
  exceeded`, `answer lines change the tokenisation …`, `catalogue variant v renders another
  chat template tail`). A one-shot mode (`rc1-twice`, `restore`, `cell-estimate`,
  `lines-join`, `tail-mismatch`) is armed at construction and disarmed only in its own code
  path. This drives the existing cleanup + `spr = max(1, spr/2)` + retry path;
  `timings.retries` counts retried rounds. Unset → nothing changes.
- **`--decide-seqs` bound where `n_parallel` is final.** The server resolves `-np -1` at
  `server.cpp:152`, before `load_model`; the check
  `n_parallel + decide_seqs ≤ llama_max_parallel_sequences()` is repeated there and fails
  the start with a message naming both numbers. The argument parser check stays.

## 10. Dump format (`--dump-tokens`)

- `tokens.prefix` stays (variant 0); `tokens.prefixes` lists the token arrays of all
  `K_eff` variants.
- Per state: `state`, `tail` (unchanged) and `branches`, one entry per branch sequence
  actually decoded for that state, each with `field`, `variant`, `phase`, `mode`
  (`"tree"` / `"full_path"`), `tokens` (the sequence's tokens after the trunk: lines when
  dependent, stem, path) and:
  - tree: `node`, `children`, `options` (as today);
  - full-path: `option` (index), `rows` (indices into `tokens` whose logits were read),
    and `logp`: per row, the values the engine stored (child log-probs keyed by token id,
    and for the leaf row `end`, for a leaf's parent `merged`);
  - dependent fields: `given` (ancestor values) and `lines` (text).
- `engine.resplit_options` is repeated in the dump.

## 11. Errors (additions to the parent spec §7)

| condition | response |
|---|---|
| bad `after` (not array, empty, non-string, duplicate, unknown, self, cycle, level > 8) | 400 `invalid_schema`, `field` as §3.1 |
| bad `options.scoring` / `order_debias` / `temperature` | 400 `invalid_options` |
| full-path or debias needs more sequences than `--decide-seqs` | 400 `budget` with the numbers |
| a full-path node without its row; NaN/+inf row values; answer lines change a field's tokenisation; cell estimate exceeded | 500 `engine` |
| snapshot restore fails | not an error: slot cleaned, entry dropped, prefix built |

## 12. Verification

### 12.1 Unit tests (ctest, no model)

- schema: every §3.1 validation case with its message and `field`; option parsing;
  `render_catalogue` goldens: variant 0 equal to today's text; `K_eff` = 4 rotations of a
  4-option field; a 3-option field under `K_eff` = 4; `K_eff` computation (no choice field
  → 1; K > n_max → n_max).
- score: full-path log-probability from hand-built node log-probs (a two-token option and
  a one-token option, with and without merged tokens), coverage, all-zero → 500;
  temperature (T = 1 identity; T = 2 on `[0.8, 0.2]` → `[2/3, 1/3]`); debias averaging and
  `order_spread`.
- trie (tokenizers via `DECIDE_TEST_MODEL`: Qwen2.5-0.5B and Gemma 4): full-path accepts
  `sale`/`sales`; one branch sequence per option; the row plan reads every node exactly
  once; `END` contains the tokens for `,` `}` `\n`; `MERGED["\""]` members are printed and
  asserted non-empty when the vocabulary has a `",` token; a dependent field's context
  trie keeps the lines-free paths and its stem ends with the lines + stem tokens;
  the re-split comparison helper unit-tested on synthetic token vectors (a re-split case
  and a no-re-split case), plus an informational printout of which keys from a candidate
  list (keys ending in `)`, `.`, `!`, `%`, a digit, `N/A`) re-split on each tokenizer.

### 12.2 Baseline and regression (live, required)

Before any engine change of this sub-project, on the engine HEAD after sub-project 2's
follow-up fixes: tree-mode b5 runs of Qwen3.5-9B and Gemma 4 E4B on the test (80) and train (320)
sets with the client's default prompt variant (`keywords`), saved as the baseline. After
the sub-project, the same runs must match the baseline preds with max abs diff ≤ 1e-6;
each run uses a freshly started server (or `--decide-prefix-cache 0`), so its prefix is
built or resident, never restored.

### 12.3 Integrity (live)

All checks use the batch-invariance statistic of the parent spec (per field: median abs
diff over options ≤ 0.01, no argmax disagreement where the reference top-2 margin > 0.15)
unless stated.
- **Full-path reference** (`reference_check.py`, `--decide-seqs` a parameter): replays
  every node of every option (prefix + state + tail + stem + path prefix → `/completion`
  `top_logprobs`, 512). Per field: the statistic on `P̂`; median abs diff of coverage ≤ 0.02;
  per-node comparison of the dumped `logp` values with the replay, median abs diff in
  probability ≤ 0.01 per field. A path token absent from the replay's top-512 is an
  unverifiable edge: the reference uses the engine's value for that edge (so the edge is
  excluded from the comparison), the engine's probability for it must not exceed twice the
  replay's 512th probability, and at most 5 % of a model's edges may be unverifiable. The
  replay computes the `END` and `MERGED` sums over the returned top-512 tokens (a lower
  bound). Models: Qwen2.5-0.5B
  (f16 KV preset, `--decide-seqs 64`), Qwen3.5-4B (hybrid, `--decide-seqs 32`: its
  recurrent state is ~50 MiB per sequence), Gemma 4
  E4B (iSWA, SPM tokenizer, `--decide-seqs 64`), Qwen3.5-9B (`--decide-seqs 21`);
  20 states × 3 fields each. Sanity gate on Qwen3.5-9B and Gemma 4 E4B: median coverage
  per field ≥ 0.8 (a lower value means missing `END`/`MERGED` mass or a model that prefers
  another format; either is investigated and reported).
- **Row-cap splits**: on Qwen2.5-0.5B and Gemma 4 E4B with `--decide-seqs 64`, a full-path
  b5 request splits its branch phase into ≥ 2 chunks (asserted from `timings.decodes`);
  its answers match the same 5 states sent as b1 requests (no split) under the statistic.
- **After reference**: the same replay for dependent branches (their tokens include the
  lines), same criteria, plus a check that every dependent entry's `given` equals the
  ancestors' returned values and its `lines` text is built from them.
- **Prefix cache**: on Qwen2.5-0.5B (dense), Qwen3.5-4B (hybrid) and Gemma 4 E4B (iSWA):
  schema A, schema B, schema A; the third answer equals the first within max abs diff
  1e-3 with the same argmax and reports a cache hit. Timing table of 20 A/B alternations
  with `--decide-prefix-cache 0` vs 8 (prefill_ms, prefix_ms, total_ms medians) on
  Qwen3.5-9B and Gemma 4 E4B.
- **Debias**: K = 1 gives the same probabilities as K = 0 (≤ 1e-6); K = 2 answers on 5
  states match, under the statistic, the per-key average of two separate K = 0 requests,
  the second with the choice options declared in the rotated order (its catalogue text
  equals variant 1's).
- **Fault hook**: CLI runs with `LLAMA_DECIDE_FAULT=rc1` (b5, spr ≥ 2, and b1, spr = 1) and
  `rc1-phase1` (with `after`) complete with `retries` ≥ 1 and answers matching a normal
  run under the statistic; on a server started with `rc1-twice`, a K = 2 first request
  returns 500, `/v1/decide/info` read right after it lists only slot 0 in `prefixes`, and
  the next K = 1 request matches a normal run under the statistic.
- **All switches**: one request with full-path, `after`, K = 4 and T = 2 runs on
  Qwen2.5-0.5B with `--decide-seqs 64` (answers sum to 1; `given`, `coverage`,
  `order_spread` present); on Qwen3.5-4B with `--decide-seqs 32` the same request returns
  the 400 `budget` naming 44 sequences per state against 28 available — never a 500.
- **Backend sampling**: a CLI full-path run with `--backend-sampling` completes.

### 12.4 Evaluation (live)

On the test (80) and train (320) sets, b5, Qwen3.5-9B and Gemma 4 E4B, client default
prompt variant: tree (the baseline of §12.2), full-path, `after` (config A: urgency after
queue; config B: urgency and angry after queue), debias `K = 4`, then temperature
(calibrate.py, last). Per run: queue/urgency/urg±1/angry accuracy, ECE, NLL, Brier, p50
per ticket, rounds; exact sign tests vs tree on per-ticket correctness (train and test
pooled, 400 tickets). Full-path adds the coverage distribution; debias adds the
`order_spread` distribution.

### 12.5 Report

REPORT-ENGINE.md section "Engine improvements (sub-project 3)": what each switch does,
its cost, the integrity results, the evaluation tables, and a recommendation per switch
(keep off / use when / default on) bound to the numbers. README: the switches table.
`patches/decide/` regenerated.
