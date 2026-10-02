# Follow-up Task 2: minor fixes and fault modes

Engine commits: `742e391e3` (`decide: coverage clamp, cache trim before copy, one cell formula, info slots`) and `a4d3e7c73`
(`decide: fault modes for the cell-estimate, lines-join and tail-mismatch paths`). The runs used the build at `23957f14c`,
which adds a later fix (`decide: no SIMD dispatch on MSVC targets; small-n tests`; it does not touch these paths).
Old build: `engine-63ea2c51a/build/bin`. Test machine: RTX 3090. Every run goes through `check.py` (see its header).

## Unit tests

`./test-engine.sh`: 100% tests passed out of 10.
- `test-decide-score` 22: coverage 1 + 3.1e-7 is reported as 1.0 and 1 + 4.5e-5 stays; debias mean of 1.0000015 and 1.0 is
  1.0, of 1.00001 twice stays, NaN stays.
- `test-decide-cache`: `plan_cache_store` is a no-op for capacity 0, an already-cached hash, no bytes, and a snapshot alone
  over the byte limit. Cases covered:
  - a capacity trim (the sp3 A,B,C,A case: B goes, not the wanted A);
  - every older entry wanted: nothing stored, nothing erased;
  - byte trims with one, two and three victims (ascending indices);
  - only wanted entries left over the limit: nothing erased, not even the unwanted entry that would go first;
  - both limits together.

## 2.2 cache sequences and 2.3/2.4 regression, CLI old vs new (`sequences`)

Flags: `-c 4096 -ngl 99 -fa on`, `-ctk/-ctv` as the preset says. `--decide-seqs`: 21 for 0.5B (the preset), 32 for 4B.
Request files: A/B/A (both models) and the 0.5B A,B,C,A are byte-identical to sub-project 3's Task 4 files; the Qwen3.5-4B
A,B,C,A file is new (sub-project 3 ran that sequence only on the 0.5B).

| | Qwen2.5-0.5B (dense) | Qwen3.5-4B (hybrid) |
|---|---|---|
| A/B/A, cache 8: (hits, misses, entries) per line, old = new | (0,1,0) (0,1,1) (1,0,2) | (0,1,0) (0,1,1) (1,0,2) |
| A,B,C,A, cache 2, old = new | (0,1,0) (0,1,1) (0,1,2) (1,0,2) | (0,1,0) (0,1,1) (0,1,2) (1,0,2) |
| answers new vs old, both sequences | max abs diff 0.0 | 0.0 |
| `cached_tokens` per line, old = new | A/B/A 0, 0, 435; A,B,C,A 0, 0, 0, 435 | 0, 0, 448; 0, 0, 0, 448 |
| tree b5 `--dump-tokens` (20 tickets): results / `tokens` / `engine` / rounds | 0.0 / identical / identical / identical | 0.0 / identical / identical / identical |

The tree run, `rounds` and `engine.states_per_round` show that `need_for` through `state_cells` (2.3) changes nothing.

## 2.4 info after A/B/A, server old vs new (`info`)

`llama-server` with the preset flags, `--jinja --reasoning off -np 1`, the same `--decide-seqs`. Both builds: 3 × 200, and
the `/v1/decide/info` bodies are equal.
- Qwen2.5-0.5B: `prefixes` = slot 0 only (435 tokens, valid); `prefix_cache` 2 entries, 8402100 bytes.
- Qwen3.5-4B: `prefixes` = slot 0 only (448 tokens, valid); 2 entries, 117620776 bytes.

This shows only that nothing regressed. No engine path leaves cells in only one memory part, so the new `seq_pos_min` listing
has nothing to show here.

## 2.5 fault modes, server F (fault) vs N (no fault), Qwen2.5-0.5B, `--decide-seqs 64` (`fault`)

R0 = 5 plain states; R1 = R0 plus `urgency` after `queue` (cell-estimate, lines-join) or R0 with `order_debias` 2
(tail-mismatch); R2 = R0's body again. Each server gets R0, R1, GET info, R2.

| mode | F: R1 | F: info after R1 | F: R2 | N: R0, R1, R2 | R2 F vs N | R2 `prefix_cache` F / N (hits, misses, entries) |
|---|---|---|---|---|---|---|
| cell-estimate | 500 `engine` `cell estimate exceeded: field "urgency" needs 13 branch cells per state with its answer lines, 15 are reserved` | slot 0 only, valid (248 tokens) | 200 | 200, 200, 200 | max abs diff 0.0 | (0,0,0) / (0,0,0) |
| lines-join | 500 `engine` `answer lines change the tokenisation of field "urgency"; remove its "after" (injected by LLAMA_DECIDE_FAULT=lines-join)` | slot 0 only, valid | 200 | 200 × 3 | 0.0 | (0,0,0) / (0,0,0) |
| tail-mismatch | 500 `engine` `catalogue variant 1 renders another chat template tail` | slot 0 only, valid | 200 | 200 × 3 | 0.0 | (0,0,0) / (0,0,1) |

cell-estimate and lines-join fire in `set_contexts` before phase 1. That is after the round's trunk and phase-0 branch
decodes, so their 500 goes through the per-round `cleanup` and `end_request`; info then lists slot 0 only.
The cell-estimate message carries the real counts (13 needed, 15 reserved): the fault forces the branch, not the numbers.
On N the debias R1 leaves a snapshot of slot 1 in the cache (1 entry); R2 then finds slot 0 resident on both servers.
