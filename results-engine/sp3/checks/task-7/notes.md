# Sub-project 3 Task 7 live checks (option-order debiasing and temperature)

Written by `checks.py notes` from the check files of this directory; `run.sh` produced every file (commands per check in
its functions) with the engine build of 8daa120b5 (the `c1-*` files: 3a43e2e62), except the second-build logs
`rebuild-build.log`, `rebuild-ctest.log` and `rebuild-test-engine.log` (build 5e9e030d3, model-free). Engine log up to 8daa120b5:
```
8daa120b5 decide: option-order debiasing and temperature
3a43e2e62 decide: one value rule for answers, given and lines; keep join-check detail
694ae5934 decide: sequential fields (after)
```
CLI = `engine/build/bin/llama-decide -c 4096 -ngl 99 -fa on`, KV f16 for qwen2.5-0.5b, q8_0 otherwise (`models-engine.ini`);
`--decide-seqs` 64 (Qwen2.5-0.5B, Gemma 4 E4B), 32 (Qwen3.5-4B), 21 (Qwen3.5-9B). Servers: the same flags plus `-np 1 --jinja
--port 8097`. Requests: `decide_client.build_request` (prompt variant `default`) on the first test tickets; b5 = lines of 5
states. "The statistic": per field median abs diff <= 0.01 and no argmax disagreement over a 0.15 reference margin
(`sp3_metrics.compare`); medians are given as queue / urgency / angry.

## Checks

| check | files | result |
|---|---|---|
| K = 1 vs K = 0, Qwen2.5-0.5B, 20 tickets b5 (bound 1e-6) | `k1-vs-k0-*` | max abs diff 0; results, usage and engine blocks identical: PASS |
| K = 2 vs the per-key mean of rotated K = 0 and K = 0 (5 states), qwen2.5-0.5b | `k2-qwen2.5-0.5b.jsonl`, `k2-check-qwen2.5-0.5b.json` | medians 0.00442 / 0.00403 / 0.00207, max 0.0441, statistic PASS (queue / urgency / angry); K = 2 `hits` 1, `misses` 0, `cached_tokens` 496; prefixes, dump variants, `order_spread`: ok |
| K = 2 vs the per-key mean of rotated K = 0 and K = 0 (5 states), qwen3.5-4b | `k2-qwen3.5-4b.jsonl`, `k2-check-qwen3.5-4b.json` | medians 0.00336 / 0.000745 / 0.000159, max 0.0561, statistic PASS (queue / urgency / angry); K = 2 `hits` 1, `misses` 0, `cached_tokens` 508; prefixes, dump variants, `order_spread`: ok |
| K = 2 vs the per-key mean of rotated K = 0 and K = 0 (5 states), gemma-4-e4b | `k2-gemma-4-e4b.jsonl`, `k2-check-gemma-4-e4b.json` | medians 0.000326 / 0.000267 / 5.79e-05, max 0.0168, statistic PASS (queue / urgency / angry); K = 2 `hits` 1, `misses` 0, `cached_tokens` 494; prefixes, dump variants, `order_spread`: ok |
| `rc1-twice` server, K = 2 first request, qwen2.5-0.5b | `server-rc1-twice-qwen2.5-0.5b.json`, `server-rc1-twice-next-vs-normal-qwen2.5-0.5b.json` | first 500 `no KV cell space (rc=1)`; info right after: `prefixes` slots [0] (valid [True]), cache entries 1; next K = 1 request 200, vs a normal server medians 0 / 0 / 0, max 0, statistic PASS |
| `rc1-twice` server, K = 2 first request, qwen3.5-4b | `server-rc1-twice-qwen3.5-4b.json`, `server-rc1-twice-next-vs-normal-qwen3.5-4b.json` | first 500 `no KV cell space (rc=1)`; info right after: `prefixes` slots [0] (valid [True]), cache entries 1; next K = 1 request 200, vs a normal server medians 0 / 0 / 0, max 0, statistic PASS |
| K = 4, Qwen3.5-9B, `--decide-seqs 21`, the same 5 states twice | `k4-qwen3.5-9b.jsonl`, `k4-check-qwen3.5-9b.json` | `seqs_per_state` 16, `states_per_round` 1; `prefix_cache` hits/misses 0/4 then 3/0; max abs(sum - 1) 2.22e-16; second vs first max abs diff 0; `prefix_ms` 802.153 then 20.772 |
| T = 2 vs the T = 1 answers scaled post hoc, Qwen2.5-0.5B, 20 tickets b5 (bound 1e-9) | `t2-vs-t1-posthoc-*` | 60 answers: max abs diff 6.66e-16 over probabilities, `expected`, `confidence`, `p_true`; values equal: PASS |
| server `--decide-temperature 2` vs a default server, same request sequence (Qwen2.5-0.5B) | `server-temp-*` | info `temperature` 2.0; `/v1/decide` without `temperature`: `engine.temperature` 2.0, post-hoc max abs diff 4.44e-16; with `temperature: 1`: 1.0, vs the default server 0; `/v1/systemone` vs post-hoc scaling of the default server's 2.22e-16: PASS |
| every switch: full-path + urgency, angry after queue + K = 4 + T = 2, Qwen2.5-0.5B `--decide-seqs 64`, 5 states | `allswitch-*` | 200; engine full_path, `order_debias` 4, `temperature` 2.0, `phases` 2, `seqs_per_state` 44, `states_per_round` 1; 15 answers sum to 1 within 2.22e-16, each with `coverage` and `order_spread`, dependents with `given` = the queue value: PASS; per-variant recomputation from the dump (mean, `order_spread`, mean coverage, then T): max abs diff 1.11e-16 (T = 2), 1.11e-16 (T = 1); T = 2 vs the T = 1 request scaled post hoc 4.44e-16, `coverage`, `order_spread`, `given` equal |
| the same request on Qwen3.5-4B, server `--decide-seqs 32`, after a plain request | `server-budget-allswitch-qwen3.5-4b.json` | plain 200; all-switch 400 `budget`: `schema needs 44 sequences per state (K_eff 4 × (1 trunk + 10 branch sequences, full_path)); --decide-seqs 32 leaves 28 after 4 prefix slots`; info before: slot 0 6462224defad50d8 254 tokens valid True, cache entries 0; after: slot 0 6462224defad50d8 254 tokens valid True, cache entries 0: PASS |

## Extra checks

| check | files | result |
|---|---|---|
| shared value rule: tree dump vs the Task 0 baseline (bound 1e-6) | `c1-tree-regression-*` | max abs diff 0 |
| shared value rule: the task-6/ `after` tree b5 run again | `c1-after-tree-*` | results, usage, engine and dump identical to the task-6/ file, max abs diff 0 |
| shared value rule: the task-6/ `after` fp b5 run again | `c1-after-fp-*` | results, usage, engine and dump identical to the task-6/ file, max abs diff 0 |
| final build: tree dump vs the Task 0 baseline | `tree-regression-*` | max abs diff 0 |
| final build: the task-6/ `after` tree b5 run again | `regress-after-tree-*` | identical to the task-6/ file apart from the new keys (engine `order_debias`, `temperature`; `tokens.prefixes`; entry `variant`), max abs diff 0 |
| final build: the task-6/ `after` fp b5 run again | `regress-after-fp-*` | identical to the task-6/ file apart from the new keys (engine `order_debias`, `temperature`; `tokens.prefixes`; entry `variant`), max abs diff 0 |
| K = 2 sequence in full-path mode (per-variant rows in the dump), qwen2.5-0.5b | `k2fp-*-qwen2.5-0.5b*` | variant 0 (slot 0, resident) vs plain K = 0: medians 0.00525 / 0.00603 / 0.0118, max 0.0359, statistic FAIL; variant 1 (slot 1, restored from slot 0's snapshot) vs rotated K = 0: medians 0.00281 / 0.00443 / 0.0131, max 0.022, statistic FAIL; K = 2 vs the K = 0 mean: medians 0.00449 / 0.00287 / 0.00517, max 0.0189, statistic PASS; `order_spread` vs the dump 0; hits 1 |
| K = 2 sequence in full-path mode (per-variant rows in the dump), qwen3.5-4b | `k2fp-*-qwen3.5-4b*` | variant 0 (slot 0, resident) vs plain K = 0: medians 0.00394 / 0.00323 / 5.22e-05, max 0.0694, statistic PASS; variant 1 (slot 1, restored from slot 0's snapshot) vs rotated K = 0: medians 0.00484 / 0.0023 / 0.000154, max 0.0773, statistic PASS; K = 2 vs the K = 0 mean: medians 0.00476 / 0.00325 / 0.000143, max 0.0404, statistic PASS; `order_spread` vs the dump 0; hits 1 |
| K = 2 sequence in full-path mode (per-variant rows in the dump), gemma-4-e4b | `k2fp-*-gemma-4-e4b*` | variant 0 (slot 0, resident) vs plain K = 0: medians 0.00041 / 7.2e-05 / 4.43e-05, max 0.0144, statistic PASS; variant 1 (slot 1, restored from slot 0's snapshot) vs rotated K = 0: medians 0.000216 / 0.000204 / 0.000166, max 0.0312, statistic PASS; K = 2 vs the K = 0 mean: medians 0.000179 / 8.02e-05 / 9.53e-05, max 0.0187, statistic PASS; `order_spread` vs the dump 0; hits 1 |
| `--info` with a body: K = 0, 2, 4 tree and K = 4 full-path + after + T = 2 (`--decide-seqs 64`) | `info-*` | (`seqs_per_state`, `states_per_round`) = [[4, 15], [8, 7], [16, 3], [44, 1]]: PASS |
| per-state cell check of a K = 2 request (one state of all 80 tickets, `-c 4096`) | `cells-k2-*` | budget: `state 0 needs 6906 cells (held 0, prefix 496, K_eff 2 × (state 3183, tail 6, branches 16)), n_ctx - 16 is 4080`; need = held + prefixes (496 as in the K = 2 dump) + 2 x (state + tail + branches): PASS |
| `LLAMA_DECIDE_FAULT=rc1`, K = 2, 20 tickets b5, `--decide-seqs 17` (1 state per round) vs a normal run | `fault-rc1-k2-*`, `k2-b5-seqs17-*` | retries per request [5, 5, 5, 5], rounds [5, 5, 5, 5]; answers max abs diff 0: PASS |

Qwen2.5-0.5B `angry` in the per-variant full-path comparisons (medians 0.0118, 0.0131) is the size of that model's known
batch-composition effect on `angry` (tree b5 vs b1: 0.0117, sub-project 1, `results-engine/batch-invariance-qwen2.5-0.5b.json`; Tasks 5, 6); Qwen3.5-4B and Gemma 4 E4B are the
binding models for the statistic here.

## Cost of K = 2 (the three lines of the K = 2 check, 5 states each; ms)

| model | line | prefix_ms | trunks (prefill - prefix) | scoring_ms | total_ms | rounds | decodes | prompt_tokens | cached_tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | rotated K = 0 (prefix built) | 19.3 | 12.0 | 10.7 | 46.5 | 1 | 3 | 524 | 0 |
| qwen2.5-0.5b | K = 0 (prefix built) | 16.0 | 12.2 | 12.8 | 45.2 | 1 | 3 | 524 | 0 |
| qwen2.5-0.5b | K = 2 (slot 0 resident, slot 1 restored) | 1.2 | 19.5 | 15.1 | 40.9 | 1 | 2 | 552 | 496 |
| qwen3.5-4b | rotated K = 0 (prefix built) | 113.5 | 166.9 | 103.1 | 386.3 | 1 | 3 | 555 | 0 |
| qwen3.5-4b | K = 0 (prefix built) | 131.7 | 166.4 | 103.6 | 404.5 | 1 | 3 | 555 | 0 |
| qwen3.5-4b | K = 2 (slot 0 resident, slot 1 restored) | 7.1 | 284.5 | 210.3 | 504.4 | 2 | 4 | 602 | 508 |
| gemma-4-e4b | rotated K = 0 (prefix built) | 133.2 | 90.1 | 50.2 | 277.5 | 1 | 3 | 524 | 0 |
| gemma-4-e4b | K = 0 (prefix built) | 96.7 | 91.0 | 50.4 | 242.2 | 1 | 3 | 524 | 0 |
| gemma-4-e4b | K = 2 (slot 0 resident, slot 1 restored) | 2.0 | 160.3 | 81.2 | 246.9 | 1 | 2 | 554 | 494 |
