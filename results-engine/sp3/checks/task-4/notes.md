# Sub-project 3 Task 4 live checks

Builds: Task 4 = engine 175b89656 + 819f5e6f3; the second build = engine d18f0a3e1 (protected cache trim, `restore`
fault mode, one-shot fault flag consumed only by its own mode). Rows marked (rebuild) and every file they name were
produced by the second build; so were the reruns of `aba-qwen3.5-4b-*`, `aba-gemma-4-e4b-*` (references for the restore
fault), `lru-qwen2.5-0.5b-cache2.*`, `server-rc1-twice-*`, `server-normal-*` and `tree-regression-*` (byte-identical
to the Task 4 output). Exception: the two `lru-*-cache2` sequences ran on the second build before its last edit (the
one-shot fault-flag change, which acts only when `LLAMA_DECIDE_FAULT` is set); their results match a hand trace of
d18f0a3e1 and its `test-decide-cache` cases. The other rows come from the Task 4 build; the second build does not change their paths (no trim below
capacity, no fault variable).

Helper: `checks.py` (request lines, A/B/A and sequence summaries, response comparisons with `sp3_metrics.compare`,
server drivers). CLI flags per preset (`models-engine.ini`): `-c 4096 -ngl 99 -fa on`, KV f16 for qwen2.5-0.5b,
q8_0 otherwise. "The statistic" = per field median abs diff <= 0.01 and no argmax disagreement over a 0.15 margin.

| check | files | result |
|---|---|---|
| tree regression (0.5B, `reference_check.py dump --tag task4`, 20 tickets, `--decide-seqs 9`) | `tree-regression-*` | max abs diff 0.0 vs `sp3/baseline/dump-qwen2.5-0.5b.jsonl` |
| A/B/A, cache 8 (A = keywords prompt, B = default prompt, 5 states each, `--decide-seqs 9`) | `aba-<model>-cache8.*` | third = first (max abs diff 0.0) on qwen2.5-0.5b, qwen3.5-4b, gemma-4-e4b; third `hits` 1, `cached_tokens` = prefix |
| A/B/A, cache 0 | `aba-<model>-cache0.*` | third = first (0.0), hits = misses = 0, prefix rebuilt (prompt_tokens as the first) |
| LRU, cache 2, A,B,C,B,A (0.5B) | `lru-qwen2.5-0.5b-cache2.*` | (hits, misses, entries) = (0,1,0) (0,1,1) (0,1,2) (1,0,2) (0,1,2): B restored on the 4th, A dropped as least recently used; repeats equal (0.0) |
| (rebuild) LRU, cache 2, A,B,C,A (0.5B) | `lru-abca-*` | (0,1,0) (0,1,1) (0,1,2) (1,0,2): the last request restores A (`hits` 1, `cached_tokens` 435); the trim dropped B, not the wanted A; answer equal to the first A (0.0). Before the fix A was dropped by the snapshot of C and rebuilt. Model-free: `test-decide-cache` (ctest) |
| fault `rc1`, b5, `--decide-seqs 9` (spr 2) | `fault-rc1-b5-*`, `normal-b5-*` | retries 5 per request (spr falls to 1), statistic passes vs the spr-2 normal run (max 0.068); vs a normal spr-1 run (`--decide-seqs 5`, `normal-b5-spr1-*`) max abs diff 0.0 |
| fault `rc1`, b1, `--decide-seqs 5` (spr 1) | `fault-rc1-b1-*`, `normal-b1-*` | retries 1 per request, max abs diff 0.0 |
| server `rc1-twice` (0.5B, `--decide-seqs 9 -c 2048`) | `server-rc1-twice-*`, `server-normal-*` | first request 500 `no KV cell space (rc=1)`; info right after: `prefixes` = slot 0 only, valid; next request 200, max abs diff 0.0 vs a normal server |
| (rebuild) fault `restore`, CLI A/B/A, cache 8, qwen3.5-4b (hybrid) and gemma-4-e4b (iSWA) | `restore-fault-aba-*` | third request `hits` 0, `misses` 1, `entries` 1 (normal run: 2); all answers equal to the cache-0 run (max abs diff 0.0); llama.cpp logs `state_seq_set_data: error loading state: failed to restore kv cache` once (the recurrent part fails after the attention part, the SWA part after the base part) |
| (rebuild) fault `restore` on a server (`-np 1 --decide-seqs 9 -c 4096`), A, B, A, then info | `server-restore-fault-*` | 3 x 200, third `hits` 0 / `misses` 1; info: `prefixes` = slot 0 only, valid (448 tokens on qwen3.5-4b, 438 on gemma-4-e4b), `prefix_cache.entries` 1 |
| (rebuild) a 400 budget leaves slot 0 resident (0.5B server `-c 2048`) | `server-budget-keeps-slot0-*` | A (200), then B with one 3176-token state: 400 `state 0 needs 3446 cells (...), n_ctx - 16 is 2032`; info before and after: slot 0 valid with A's hash `840fb64afac58bb3`, nothing else, no cache entry (B's prefix differs, so an eviction before the check would have snapshotted A) |
| `--backend-sampling` CLI tree run (b5) | `backend-sampling-*` | completes, max abs diff 0.0 vs normal; `-v` log: samplers attached to seqs 0-9, decide seqs 1-9 detached |
| spr = 0 budget message (`--decide-seqs 3`, triage B = 3) | `budget-spr0-qwen2.5-0.5b.txt` | 400 `schema needs 4 sequences per state (K_eff 1 × (1 trunk + 3 branch sequences, tree)); --decide-seqs 3 leaves 2 after 1 prefix slots` (spec 7.3) |
| `-np -1 --decide-seqs 253` | `np-auto-decide-seqs-253.txt` | exit 1 before loading: `--decide-seqs 253 plus 4 slot sequence(s) (-np) needs 257 sequences, the maximum is 256` |
| flags on a server (`LLAMA_ARG_DECIDE_TEMPERATURE=0.7`, `--decide-prefix-cache 3`) | `server-flags-*` | info `temperature` 0.7, `prefix_cache.capacity` 3; `/v1/decide` 500 `not implemented yet: temperature` (guard until Task 7) |
| extra: restore onto moved cells (server `-np 1`, a `/completion` of 888 tokens between B and the second A) | `server-moved-*` | see below |

Restore onto moved cells (corrected with the second build; an earlier version of this note said "argmax kept everywhere", which hid that two
of the three comparisons fail the statistic on `angry`). Per field median / max abs diff, 5 tickets, no argmax
disagreement anywhere:

| comparison | queue | urgency | angry | statistic |
|---|---|---|---|---|
| restore (cache 8, hit): third vs first | 0.0098 / 0.0386 | 0.0046 / 0.0273 | 0.0079 / 0.0129 | pass |
| rebuild (cache 0): third vs first | 0.0078 / 0.0256 | 0.0078 / 0.0740 | 0.0139 / 0.0249 | fail (angry median 0.0139 > 0.01) |
| restored vs rebuilt third answer | 0.0079 / 0.0295 | 0.0055 / 0.0644 | 0.0129 / 0.0209 | fail (angry median 0.0129 > 0.01) |

This is the 0.5B's sensitivity to where its cells sit in the KV cache, not a restore error: the cache-off rebuild
moves further from the first answer than the restore does on the max (0.074 vs 0.039) and on the urgency and angry
medians (the queue median is the exception: 0.0078 vs 0.0098), and the restore is exact whenever the cells land where
they were (every A/B/A above: 0.0). The 0.5B already failed the batch-invariance statistic in sub-project 1
(`results-engine/batch-invariance-qwen2.5-0.5b.json`, angry median 0.0117).
