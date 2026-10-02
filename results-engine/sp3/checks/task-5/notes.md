# Sub-project 3 Task 5 live checks (full-path scoring)

Build: engine `decide: full-path scoring` (base d18f0a3e1). CLI = `engine/build/bin/llama-decide -c 4096 -ngl 99 -fa on`,
KV f16 for qwen2.5-0.5b, q8_0 for gemma-4-e4b and qwen3.5-9b (`models-engine.ini`). Requests: the triage schema of
`decide_client.build_request` (prompt variant `default`) on the first 20 test tickets, `options.scoring = "full_path"`
unless named `tree-*`; b5 = 4 request lines of 5 states, b1 = 20 lines of 1 state. Helper: `checks.py` (subcommands in its
docstring). "The statistic" = per field median abs diff <= 0.01 and no argmax disagreement over a 0.15 margin
(`sp3_metrics.compare`). GPU: 22 MiB before every model run and after the last, 50-59 C, port 8097 only, the one server
stopped by PID. The row cap is `llama_n_seq_max` = 1 + `--decide-seqs` (CLI and server `-np 1`).

## Checks

| check | files | result |
|---|---|---|
| tree unchanged (0.5B, `reference_check.py dump --tag task5`, `--decide-seqs 9`, 20 tickets) | `tree-regression-*` | max abs diff 0.0 vs `sp3/baseline/dump-qwen2.5-0.5b.jsonl` (bound 1e-6); node records (prompt tokens, children, options) identical |
| full-path triage, 0.5B, `--decide-seqs 64`, b5 and b1 | `fp-b5-qwen2.5-0.5b.*`, `fp-b1-qwen2.5-0.5b.*` | 20 states each: max abs(sum - 1) 1.1e-16; coverage in (0, 1]: b5 min 0.9867 / 0.9974 / 0.9934, median 0.9941 / 0.9996 / 0.9977 (queue / urgency / angry); b1 min 0.9867 / 0.9969 / 0.9942 |
| full-path triage, Gemma 4 E4B, `--decide-seqs 64`, b5 and b1 | `fp-b5-gemma-4-e4b.*`, `fp-b1-gemma-4-e4b.*` | max abs(sum - 1) 1.1e-16; coverage min 0.9935 / 0.99999 / 0.9946, median 0.99986 / 0.999999 / 0.9989 |
| `sale`/`sales` (choice field `q`), 0.5B and Gemma, `--decide-seqs 64` | `sale-*` | full-path 200: 0.5B sale 0.916 / sales 0.084, coverage 0.283; Gemma 0.760 / 0.240, coverage 0.9997. Tree: 400 `invalid_schema`, field `q`, `options "sale" and "sales" share a token prefix ("sale" vs "sales"); rename one of them` |
| row-cap split, Gemma, b5 vs b1, `--decide-seqs 64` (cap 65, 85 rows per round) | `split-gemma-4-e4b.json` | 2 branch decodes per b5 request (predicted chunks 2); statistic PASS: medians 1.1e-5 / 8.0e-5 / 2.7e-4, max 0.046, no argmax disagreement over the margin |
| row-cap split, 0.5B, b5 vs b1, `--decide-seqs 64` | `split-qwen2.5-0.5b.json` | 2 branch decodes per b5 request; statistic FAILS on angry only: medians 0.0041 / 0.0032 / 0.0125, max 0.065, no argmax disagreement over the margin. Not caused by the split: tree b5 vs tree b1 on the same tickets fails the same way (angry median 0.0117, `tree-b5-vs-b1-*`), and so does full-path b5 without any split (`--decide-seqs 84`, one branch decode: angry 0.0104, `split-seqs84-*`). The 0.5B's angry field is the known batch-sensitive case (sub-project 1 batch invariance: 0.0117) |
| Qwen3.5-9B full-path, `--decide-seqs 21`, b5 | `fp-b5-qwen3.5-9b.*`, `rounds-*` | B = 10 options, `seqs_per_state` 11, `states_per_round` 1, 5 rounds per 5-state request; sums 1 (1.1e-16), coverage median 0.9997 / 0.9994 / 0.9975 |
| `--backend-sampling` full-path CLI run (0.5B, b5, `--decide-seqs 64`) | `backend-sampling-*` | completes; max abs diff 0.0 vs the normal run; context `n_outputs_max_per_seq = 1`, samplers attached to seqs 0-64, the engine detached 1-64 (64 `(nil)` lines) |

At `--decide-seqs 64` the 65th row of a round is the last row of urgency option 3 of state 3 (17 rows per state), so the
boundary falls between sequences. Two more caps put it inside a sequence:

| split inside a sequence | files | result |
|---|---|---|
| `--decide-seqs 60` (cap 61): state 3 urgency option 0 cut after its root row; its leaf row (END) comes from the second decode | `split-seqs60-*`, `rows-seqs60-vs-seqs84-*` | 1 split sequence per request; the rows after the boundary equal the unsplit run (`--decide-seqs 84`) within 4.3e-8 (0.5B) and 2.1e-4 (Gemma). Gemma vs b1: statistic PASS (angry median 3.0e-4); 0.5B: angry median 0.0119 (as above) |
| `--decide-seqs 68` (cap 69): state 4 queue option 0 (`billing`) cut after its root row; its value-node row and leaf row come from the second decode | `split-seqs68-*`, `rows-seqs68-vs-seqs84-*` | rows after the boundary (values of request line 0): 0.5B value node `"` 0.0 / merged 0.999995 (diff 4.6e-7), leaf END 0.227 (diff 6.7e-3); Gemma `"` 0.000125 / merged 0.99987 (1.1e-5), END 0.996 (6.1e-4). Max over the 8 rows after the boundary (4 lines): 0.033 (0.5B), 6.1e-4 (Gemma). A row read at another batch index would differ by about 1. Statistic vs b1 PASS on both models (0.5B angry 0.0084, Gemma 2.6e-4) |
| same b5 batches, split (`--decide-seqs 64`) vs one decode (84), 0.5B | `split64-vs-nosplit84-*` | statistic PASS: medians 0.0023 / 0.0025 / 0.0086, max 0.032 (batch composition noise of the 0.5B; median over all 680 dumped row values split-60 vs 84: 2.1e-7, largest values END after a lone `"` of probability ~e^-17) |

## Extra checks

| check | files | result |
|---|---|---|
| dump self-consistency: log P, distribution and coverage recomputed from each full-path entry's `tokens`, `rows`, `logp` alone (spec 4.1) vs the answers | `*.dumpcheck.json` | max abs diff 1.1e-16 (0.5B b5 and b1, Gemma b5 and b1, 9B b5; 60 fields each); every trie node read once per state and field, rows = `usage.scored_tokens` (17 per state) |
| `--dump-tokens` does not change answers (0.5B b5) | `fp-b5-dump-vs-nodump-*` | max abs diff 0.0 |
| spr = 0 budget message (0.5B, `--decide-seqs 10`, B = 10) | `budget-spr0-fp-qwen2.5-0.5b.err.txt` | 400 `budget`: `schema needs 11 sequences per state (K_eff 1 × (1 trunk + 10 branch sequences, full_path)); --decide-seqs 10 leaves 9 after 1 prefix slots` |
| `--info` with a full-path body (`--decide-seqs 64`) | `info-fp-qwen2.5-0.5b.json` | `scoring_modes` `["tree", "full_path"]`, `schema.seqs_per_state` 11, `states_per_round` 5 |
| server, 0.5B, `-np 1 --decide-seqs 64`, the 4 full-path b5 requests | `server-fp-b5-*`, `server-vs-cli-*`, `server-n-outputs-max.txt` | 4 x 200, 2 branch decodes each, answers equal to the CLI (0.0); info `scoring_modes` both modes. The server context has `n_seq_max = 65`, `n_outputs_max = 65` (the CLI: 2048), so an 85-row decode would hit the `n_outputs_max` assert there |
| fault `rc1`, full-path b5, `--decide-seqs 64` | `fault-rc1-fp-b5-*` | retries 4 per request (spr 5 -> 2 -> 1), 4 rounds; vs b1: statistic PASS (medians 0.0, max 0.048: only the 2-state retry round differs) |
| tree `prompt_tokens` = tokens decoded (0.5B, the b5 requests of task-4/, `--decide-seqs 9`) | `tree-normal-b5-*`, `tree-fault-rc1-b5-*` | normal run: answers 0.0 vs Task 4, `prompt_tokens` [524, 317, 340, 397] and `scored_tokens` 15 as in Task 4. `rc1` run: answers 0.0 vs the rc1 run of task-4/; `prompt_tokens` [758, 623, 643, 769] (Task 4 formula: [524, 317, 340, 397]); the difference 234 / 306 / 303 / 372 equals the trunk tokens of the 6 failed-attempt trunks per request (states 0, 1, 1, 2, 3, 4 at state + 6 tail tokens) |
| tree vs full-path distributions (informational; different scoring) | `tree-vs-fp-b5-*` | Gemma medians 1e-5 / 1.2e-4 / 1.6e-4, max 0.027; 0.5B 0.0030 / 0.0047 / 0.0103, max 0.045; no argmax disagreement on either |

## Cost (b5, lines 2-4, prefix resident, medians, ms)

| run | prefill | scoring | total | decodes | rows per state | prompt_tokens |
|---|---|---|---|---|---|---|
| 0.5B tree, `--decide-seqs 64` | 17.7 | 9.9 | 30.6 | 2 | 3 | 317-397 |
| 0.5B full-path | 13.7 | 65.0 | 81.9 | 3 | 17 | 577-657 |
| Gemma tree | 112.6 | 46.9 | 161.8 | 2 | 3 | 321-404 |
| Gemma full-path | 106.3 | 213.5 | 321.6 | 3 | 17 | 576-659 |
| 9B full-path, `--decide-seqs 21` (5 rounds) | 295.8 | 788.0 | 1085.9 | 10 | 17 | 602-682 |

The full-vocabulary log-sum-exp is plain scalar code on the CPU: 0.76 ms per row for a 262144-token vocabulary in an
isolated benchmark (85 rows = 64 ms of Gemma's 167 ms extra scoring time); the rest is the larger branch decode
(50 sequences, about 400 tokens and 85 output rows per round instead of 15 sequences and 15 rows).

## Second build (engine bbe4ba438: helpers moved to decide-fullpath, mode check, +inf test)

| check | files | result |
|---|---|---|
| the move is behaviour-neutral: the same 0.5B full-path b5 request, `--decide-seqs 64 --dump-tokens` | `rebuild-fp-b5-*` | max abs diff 0.0 vs the committed `fp-b5-qwen2.5-0.5b.jsonl`; `results`, `engine`, `usage` and the whole `tokens` dump identical on all 4 lines; decodes 4 / 3 / 3 / 3 as before |
| tree regression (`reference_check.py dump --tag task5fix`) | `rebuild-tree-regression-*` | max abs diff 0.0 vs `sp3/baseline/dump-qwen2.5-0.5b.jsonl`; the dump file is byte-identical to `tree-regression-dump-qwen2.5-0.5b.jsonl` |
