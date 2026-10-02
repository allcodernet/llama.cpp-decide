# Own decide engine vs the baseline fork — parity results (2026-09-25)

Phase 2 of this project: our own `/v1/decide` engine inside llama.cpp, run on the same 80 tickets, models and machine as the baseline: the baseline fork, a third-party llama.cpp fork with a `POST /v1/decision` endpoint, measured in an earlier phase of this work; neither the fork nor the scripts that ran it are part of this repository. Its recorded predictions and run summaries are under `results/`; the engine here was built separately. Design: `docs/specs/2026-09-25-own-decide-engine-design.md`.

## Setup
- Engine: upstream ggml-org/llama.cpp at `60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`, plus our branch `decide` (unpushed). The branch ships in this repo as a patch series, `patches/decide/` (`git -C engine format-patch 60b06ab9a..decide`). `build-engine.sh` clones upstream into `engine/` (git-ignored), creates `decide` at the pinned commit and applies the series with `git am`.
  - Engine tree after the published series (0001-0045): `c3daa53ab47099be8a69e6d81749f52072768fa7` (`git -C engine rev-parse HEAD^{tree}`; `git am` of the series on a scratch clone at the pinned commit gives it, checked on the laptop and on the test machine). The last measured build, `185e9e0a7`, had the tree `ff5f3f0d577cbcb635937e733ddf6bdf67826700` (see "Commit ids in this report" below). After 0044 (sub-project 4 after the rev 6 fixes) the tree was `695f2dc4b87bfc3d3c34271dbee929b86b0ec125`. After 0043 (sub-project 4 before the rev 6 fixes) the tree was `87b0cbb3714edbfea6f680a5bc71680f8260fe16`. After 0038 (the follow-ups' last fixes) the tree was `32d63de3ad1b00c086fb4abd2b9da53631d8d043`. After 0037 (the follow-ups before their last fixes) the tree was `baf95845e9c265e676bd1564837bddd6d98eca72`. `git am` gives new commit hashes, so compare trees, not commits. After 0032 (the end of sub-project 3) the tree was `85b062337db46796e1cd2969733a4edee31f6be3`: `build-engine.sh`'s clone-and-`git am` path, run on a clean clone of the local seed at the pinned commit, gave that tree (checked after the sub-project 3 follow-up fixes, without a build; after 0031 it was `a210c649206dc275f4ac66a78e9d8f8f98802728`). Earlier, after 0014, the tree was `7eed1fccb22fff52045c16c238b48db122bc2605`, and a fresh clone of this repo, built with `BUILD_CUDA=0 TARGETS="llama-decide-cli test-decide-schema" ./build-engine.sh`, reproduced it.
  - Series: 0001 decide: schema compilation, validation, catalogue rendering, hash; 0002 decide-schema: reject empty field names, split levels error message; 0003 decide: aggregation of node logits into option distributions; 0004 decide-score: hard-validate node_scores coverage against branch_nodes; 0005 decide-trie: tokenisation boundary rule, sibling check, quote-split fallback; 0006 common: --decide-seqs, unified KV and output reservation for decision sequences; 0007 decide: engine, api and llama-decide CLI; 0008 server: /v1/decide and /v1/decide/info backed by the decide engine; 0009 decide: runtime tokenisation-join self-check; 0010 decide: check every state's cell budget before any decode; 0011 decide-score: reject NaN and +inf logits with a 500 engine error; 0012 common: reject --decide-seqs above the sequence limit; 0013 decide: option indices per branch child in --dump-tokens; 0014 decide: cut join error text at UTF-8 boundaries. Sub-project 2 (`/v1/systemone`): 0015 decide-compat: Jev /v1/systemone translation library and error.field; 0016 decide-compat: empty instructions fall back to the default description; 0017 server: POST /v1/systemone Jev-compatible route; 0018 decide-compat: bound nesting, size and question count; status from map_decide_error; 0019 server: /v1/systemone request id header, body limit; field in /v1/decide pre-task errors. Sub-project 3 (engine improvements, section "Engine improvements (sub-project 3)"): 0020 decide-schema: after, scoring/order_debias/temperature options, catalogue variants; 0021 decide-score: full-path aggregation, temperature, variant averaging; 0022 decide-trie: full-path plan, vocabulary indexes, context tries; 0023 decide: prefix slots, prefix cache, temperature flag, sampler detach, fault hook; 0024 server: check --decide-seqs after -np resolution; 0025 decide: protect wanted cache entries from trimming; restore fault mode; 0026 decide: full-path scoring; 0027 decide: move full-path helpers to decide-fullpath; mode assert; +inf test; 0028 decide: sequential fields (after); 0029 decide: one value rule for answers, given and lines; keep join-check detail; 0030 decide: option-order debiasing and temperature; 0031 decide: clamp free sequences in the budget message; cell message helper; 0032 tests: run test-decide-trie on the in-repo vocab files (follow-up fix, test registration only). Follow-ups (section "Follow-ups (2026-10-02)"): 0033 decide: faster full-vocabulary log-sum-exp for full-path rows; 0034 decide: coverage clamp, cache trim before copy, one cell formula, info slots; 0035 decide: fault modes for the cell-estimate, lines-join and tail-mismatch paths; 0036 decide: no SIMD dispatch on MSVC targets; small-n tests; 0037 decide: mark the injected cell-estimate fault in its message; 0038 decide: fault message suffix for tail-mismatch; header note (follow-up fix). Sub-project 4 (section "Image input (sub-project 4)"): 0039 decide: states as text and image parts, strict validation, vision status (no vision yet); 0040 decide: image states: stb_image pixels, mtmd trunks, cell budget, held cells from the server; 0041 decide: llama-decide --mmproj and image-token limits; dump of image states for string replays; 0042 decide: non-causal projectors off after the failed gate (spec 9.5); 0043 decide: cell budget bounded by the SWA cache; 0044 decide: image aspect-ratio limit, early image budget check, small fixes (spec rev 6); 0045 decide: sequence budget before prefix preparation.
  - The parity, gate and triage integrity runs below used 0001-0008 (HEAD `5ffb34671`). 0009-0014 are follow-up fixes made after a code review of 0001-0008. After 0009-0013 (0014 changes only error text), the 0.5B triage dump (20 tickets) gave the same token ids and bit-identical engine answers as before (`results-engine/integrity/reference-diag-triage-qwen2.5-0.5b*`).
- Commit ids in this report: the ids quoted here and in the result file names (`5ffb34671`, `63ea2c51a`, `49785a95d`, `28a924b55`, `69756d495`, `185e9e0a7`, …) name the development builds the runs were made on. After the last run the series was regenerated with changes to code comments, commit messages and the text of two test inputs only, so `git am` of the published patches gives other commit ids. The quoted builds are these patch ranges of the published series: `5ffb34671` 0001-0008, `f042104fa` 0001-0017, `694ae5934` 0001-0028, `8daa120b5` 0001-0030, `5e9e030d3` 0001-0031, `63ea2c51a` 0001-0032, `387b4848d` 0001-0033, `742e391e3` 0001-0034, `a4d3e7c73` 0001-0035, `23957f14c` 0001-0036, `a32ee701e` 0001-0037, `49785a95d` 0001-0038, `0d192078a` 0001-0039, `b36eccdec` 0001-0040, `9bae50c77` 0001-0041, `28a924b55` 0001-0043, `69756d495` 0001-0044, `185e9e0a7` 0001-0045. The tree of the last measured build (`ff5f3f0d577cbcb635937e733ddf6bdf67826700`) and the published tree (`c3daa53ab47099be8a69e6d81749f52072768fa7`) differ in 14 comment lines of five test files (`tests/test-decide-{after,debias,image,schema,trie}.cpp`) and in the example state text of `tests/data/decide-triage.json` and `tests/test-decide-compat.cpp` (sample inputs that no test assertion reads), and in nothing else; the 27B text regression on the published tree against the recorded run of `185e9e0a7` (88 requests) has max abs diff 0, and the server smoke passes 11 of 11 (`results-engine/image/publish/recheck/`; both ran on the development build `d5cdf773a` of the published tree, which the smoke JSON does not record).
- Build flags (`build-engine.sh`): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON`; targets `llama-server llama-cli llama-decide-cli` and the decide/tokenizer tests. Machine: RTX 3070 Ti Laptop 8 GB, 31 GB RAM (the machine of the baseline runs).
- Server: `llama-server --models-preset models-engine.ini --models-max 1 --port 8097`. `[*]` settings are the same as the baseline runs' presets: ctx 4096, q8_0 KV, fa on, jinja, reasoning off, ngl 99, parallel 1; plus `decide-seqs = 21`. Models: `Qwen3.5-9B-Q4_K_M.gguf` (preset `qwen3.5-9b`) and `gemma-4-E4B-it-Q4_0.gguf` (preset `gemma-4-e4b`), the same files as the baseline. Every run of Phase 2 and sub-projects 2-3 in this report except the sub-project 3 preset check used these presets without `decide-temperature` (the follow-ups and Qwen3.8-27B ran on the test machine with `models-engine-3090.ini`, and the 27B's preset check at T = 0.8992: see their sections); since the sub-project 3 calibration, `[gemma-4-e4b]` also sets `decide-temperature = 1.6947`, so a Gemma request without `options.temperature` now gets tempered probabilities (same argmax), and the Gemma steps in "How to reproduce" pin T = 1 (`decide_client.py --ensure-t1`).
- Sequence budget: the triage schema needs one branch node per field, so `seqs_per_state` = 4 (trunk + 3 branches) and `states_per_round` = 21 // 4 = 5, which matches the baseline's N = 5. The engine returned `quote_split_fields: []` in all runs.
- Memory type from `/v1/decide/info`: qwen3.5-9b `recurrent: true, swa: false` (hybrid); gemma-4-e4b `recurrent: false, swa: true`. GPU memory in use after each run (nvidia-smi, from the run JSONs): qwen3.5-9b 6466 MiB (batch 1) / 6466 MiB (batch 5), gemma-4-e4b 3382 / 3384 MiB. Baseline: 6462 / 6466 and 3383 / 3385 MiB.
- Assistant prefixes from `/v1/decide/info`: qwen3.5-9b `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, gemma-4-e4b `<turn|>\n<|turn>model\n`. Both match the tails the baseline observed (`prefix_ok: true` in all five runs).
- Client: `decide_client.py`. Question and option texts are the same as in the baseline runs (from `experiment/common.py` of the earlier experiment, not in this repository), passed as a `choice` field with per-option descriptions, a `score` field with four level descriptions, and a `bool` field. One warmup request, then the 80 tickets in order, in requests of 1 or 5 states.
- Baseline rows: `results/score-table.txt` and `results/runs/` (the baseline fork with a small local patch that returned the probability of every option, not only the winner's; port 8096).
- Key names in recorded files: the keys that name the baseline (`baseline` and `baseline_margin` in `results-engine/compare-*.json`, `baseline_prompt` in the run JSONs of `decide_client.py`) were renamed after the runs, mechanically and with every value unchanged.

## Accuracy and calibration
| Run | Queue | Urgency | Urg±1 | Angry | ECE (queue) | Brier (angry) | NLL (queue) | p50 ms / ticket |
|---|---|---|---|---|---|---|---|---|
| Baseline, Qwen3.5-9B, batch 1 | 92.5% | 77.5% | 97.5% | 97.5% | 0.040 | 0.032 | 0.19 | 99 |
| Baseline, Qwen3.5-9B, batch 5 | 95.0% | 78.8% | 97.5% | 97.5% | 0.070 | 0.032 | 0.19 | 89 |
| Baseline, Gemma 4 E4B, batch 1 | 90.0% | 81.2% | 96.2% | 95.0% | 0.074 | 0.038 | 0.19 | 60 |
| Baseline, Gemma 4 E4B, batch 5 | 92.5% | 81.2% | 96.2% | 95.0% | 0.073 | 0.039 | 0.19 | 32 |
| Ours, Qwen3.5-9B, batch 1 | 91.2% | 77.5% | 96.2% | 97.5% | 0.065 | 0.031 | 0.26 | 120 |
| Ours, Qwen3.5-9B, batch 5 | 88.8% | 76.2% | 96.2% | 97.5% | 0.074 | 0.031 | 0.27 | 97 |
| Ours, Gemma 4 E4B, batch 1 | 91.2% | 81.2% | 98.8% | 96.2% | 0.052 | 0.033 | 0.30 | 59 |
| Ours, Gemma 4 E4B, batch 5 | 90.0% | 81.2% | 98.8% | 96.2% | 0.035 | 0.033 | 0.30 | 32 |

From `results-engine/score-table.txt` (`uv run score.py --preds-dir results-engine/preds`). 80 tickets: 1 ticket = 1.25 points; the table rounds 91.25 to 91.2.

Difference from the baseline row for the same configuration, in tickets:

| Configuration | Queue | Urgency | Urg±1 | Angry |
|---|---|---|---|---|
| Qwen3.5-9B, batch 1 | −1 | 0 | −1 | 0 |
| Qwen3.5-9B, batch 5 | −5 | −2 | −1 | 0 |
| Gemma 4 E4B, batch 1 | +1 | 0 | +2 | +1 |
| Gemma 4 E4B, batch 5 | −2 | 0 | +2 | +1 |

## Gate (spec 8.4)
| Configuration | Baseline p50 | Limit (1.5×) | Our p50 | Ratio | Rounds | Result |
|---|---|---|---|---|---|---|
| Qwen3.5-9B, batch 1 | 99 ms | 148.5 ms | 119.8 ms | 1.21× | 1 (80 requests) | pass |
| Qwen3.5-9B, batch 5 | 89 ms | 133.5 ms | 97.2 ms | 1.09× | 1 (16 requests) | pass |
| Gemma 4 E4B, batch 1 | 60 ms | 90 ms | 59.3 ms | 0.99× | 1 (80 requests) | pass |
| Gemma 4 E4B, batch 5 | 32 ms | 48 ms | 31.7 ms | 0.99× | 1 (16 requests) | pass |

All four runs completed, every batch-5 request in one round. Accuracy is reported, not gated. The ablation rule (a metric more than 5 points below the baseline row) fired once: Qwen3.5-9B batch 5 queue, 88.8% vs 95.0% (−6.25 points, 5 tickets). No other accuracy column is more than 5 points below its baseline row. The rule was applied to the four accuracy columns; the calibration columns are not in points. An ablation with the baseline's own prompt text was run during development and is not included.

## Latency breakdown
| Run | wall p50 / p90 per ticket | server total_ms p50 per request | total_ms / n p50 (per ticket) | prefill / scoring per ticket, p50 | rounds | decodes per request | cached / all prompt tokens |
|---|---|---|---|---|---|---|---|
| Baseline, Qwen3.5-9B, batch 1 | 99.4 / 115.4 ms | 95.8 ms | 95.8 ms | 53.7 / 41.6 ms | 1 | — | 84% (21520 / 25579) |
| Baseline, Qwen3.5-9B, batch 5 | 89.0 / 92.3 ms | 439.1 ms | 87.8 ms | 62.1 / 24.5 ms | 1 | — | 51% (4304 / 8363) |
| Baseline, Gemma 4 E4B, batch 1 | 60.1 / 67.6 ms | 56.0 ms | 56.0 ms | 32.5 / 23.5 ms | 1 | — | 85% (21280 / 25091) |
| Baseline, Gemma 4 E4B, batch 5 | 31.8 / 35.6 ms | 151.0 ms | 30.2 ms | 20.1 / 10.0 ms | 1 | — | 53% (4256 / 8067) |
| Ours, Qwen3.5-9B, batch 1 | 119.8 / 135.5 ms | 117.9 ms | 117.9 ms | 56.5 / 61.1 ms | 1 | 2 | 79% (20320 / 25659) |
| Ours, Qwen3.5-9B, batch 5 | 97.2 / 102.6 ms | 484.0 ms | 96.8 ms | 62.9 / 32.9 ms | 1 | 2 | 43% (4064 / 9403) |
| Ours, Gemma 4 E4B, batch 1 | 59.3 / 68.3 ms | 57.3 ms | 57.3 ms | 34.1 / 23.5 ms | 1 | 2 | 80% (19760 / 24771) |
| Ours, Gemma 4 E4B, batch 5 | 31.7 / 36.1 ms | 156.6 ms | 31.3 ms | 20.8 / 10.3 ms | 1 | 2 | 44% (3952 / 8963) |

Computed from `results/runs/*.json` and `results-engine/runs/*.json` with one method for both engines: per-ticket values are request values divided by the request's ticket count, one value per ticket (80 values; each batch-5 request counted 5 times), p50 = `statistics.median` and p90 = `statistics.quantiles(n=10)[8]` over those 80 values. This is why our p50 can differ slightly from the run JSON's `latency_s_per_ticket.p50`, which takes the upper-middle value (98.4 vs 97.2 ms for Qwen batch 5). The baseline wall p50s therefore read 99.4 / 89.0 / 60.1 / 31.8 ms here, where the baseline run JSONs' `latency_s_per_ticket.p50` holds the upper-middle value (100 ms for Qwen batch 1). "Per-decision equivalent" is `total_ms / n`, the same as the baseline fork's `per_decision_ms`.

The token columns count different things in the two engines. Our `prompt_tokens` excludes the cached prefix and includes the branch tokens (16 per ticket for Qwen), so the table shows cached / (cached + prompt). The baseline fork's `prompt_tokens` includes the cached prefix but not its branch rows. Cached prefix per request: ours 254 tokens (Qwen) and 247 (Gemma); baseline 269 and 266. `scored_tokens` (ours, 3 per ticket = logit rows read) and `scored_rows` (baseline, 16 / 15 per ticket = branch tokens in the batch) are also different quantities.

## Integrity (spec 8.3)
Statistic: per field, the median absolute difference between two probability vectors over all options (bar ≤ 0.01), and argmax disagreements where the reference top-2 margin exceeds 0.15 (bar: none).

What the reference check covers: it replays the engine's own token ids (from `llama-decide --dump-tokens`) through `/completion`. It therefore validates the KV/sequence choreography and the logit read-out, but it cannot detect tokenisation-boundary errors, because both sides use the same ids. Tokenisation boundaries are covered by the join self-check that the engine runs at start-up and on every request (patch 0009), and by `test-decide-trie` on the Qwen and Gemma tokenizers. No SentencePiece GGUF with `add_space_prefix` is on this machine, so rejection of such a model was not run live. The check did fire on the one `add_space_prefix` vocabulary available: the UGM tokenizer of `multilingual-e5-base` (an embedding model, loaded with `vocab_only` by `test-decide-trie`), at the state/tail join. The engine side of the check is the CLI path (`llama-decide`). Since the follow-up fixes (0009-0014), the check also covers fields with more than one branch node: it replays every branch node, renormalises over that node's children and multiplies along each option's path. The "multi-branch" rows use `tests/data/decide-multibranch.json`, whose `topic` field has 7 options and 3 branch nodes on both Qwen tokenizers (root; `refund` -> `_request`/`_status`; `login` -> `_problem`/`_security`), plus `angry`. That is 4 branch sequences per state, and 60 `topic` node replays per model. In these rows the queue column holds `topic`.

| Check | Model, KV | n | queue median / max | urgency median / max | angry median / max | argmax agreement q / u / a | over-margin disagreements | Result |
|---|---|---|---|---|---|---|---|---|
| Reference (`/completion` on the same token ids) | Qwen3.5-9B, q8_0 (as served) | 20 states × 3 fields | 0.0001 / 0.065 | 0.0041 / 0.073 | 0.0049 / 0.042 | 100 / 100 / 100% | 0 | pass |
| Reference | Qwen2.5-0.5B, q8_0 | 20 × 3 | 0.0070 / 0.078 | 0.0059 / 0.050 | 0.0216 / 0.058 | 100 / 100 / 95% | 0 | fail (angry median) |
| Reference | Qwen2.5-0.5B, f16 (preset override) | 20 × 3 | 0.0044 / 0.033 | 0.0057 / 0.042 | 0.0094 / 0.081 | 100 / 95 / 100% | 0 | pass |
| Batch invariance (alone vs batch of 5) | Qwen2.5-0.5B, f16 | 20 | 0.0031 / 0.025 | 0.0038 / 0.052 | 0.0117 / 0.035 | 100 / 100 / 100% | 0 | known miss (angry median) |
| Batch invariance | Qwen3.5-4B (hybrid), q8_0 | 20 | 0.0009 / 0.066 | 0.0036 / 0.053 | 0.0032 / 0.045 | 100 / 95 / 95% | 0 | pass |
| Rounds (12 states, `rounds 3`, vs 5/5/2 requests) | Qwen2.5-0.5B, f16 | 12 | 0 / 0 | 0 / 0 | 0 / 0 | 100 / 100 / 100% | 0 | pass, bit-identical |
| Batch invariance, parity runs (b1 vs b5) | Qwen3.5-9B, q8_0 | 80 | 0.0002 / 0.126 | 0.0032 / 0.093 | 0.0012 / 0.065 | 97.5 / 96.2 / 100% | 0 | pass |
| Batch invariance, parity runs (b1 vs b5) | Gemma 4 E4B, q8_0 | 80 | 0.0000 / 0.077 | 0.0001 / 0.029 | 0.0002 / 0.061 | 98.8 / 97.5 / 100% | 0 | pass |
| Reference, multi-branch schema (follow-up fixes) | Qwen3.5-4B (hybrid), q8_0 | 20 × 2 fields | topic 0.0005 / 0.088 | — | 0.0011 / 0.062 | 90 / — / 100% | 0 | pass |
| Reference, multi-branch schema (follow-up fixes) | Qwen2.5-0.5B, f16 (preset override) | 20 × 2 | topic 0.0035 / 0.079 | — | 0.0248 / 0.095 | 100 / — / 95% | 0 | fail (angry median) |
| Reference, `angry` only, same instructions (diagnostic) | Qwen2.5-0.5B, f16 | 20 × 1 | — | — | 0.0155 / 0.057 | — / — / 95% | 0 | fail (angry median) |

Files: `results-engine/reference-*.json`, `batch-invariance-*.json`, `rounds-equivalence-qwen2.5-0.5b.json`, the capped runs under `results-engine/integrity/`. The last two rows come from the parity runs (`batch-invariance-*-parity.json`).

- Decisions on the reference check: the 0.5B failed at q8_0 KV (angry median 0.0216). The 0.01 bar was derived from 9B measurements, and the q8_0 deviation on the 0.5B is uniform across fields when measured at the reference argmax (0.017 / 0.010 / 0.022), so it is not specific to the bool field. The 0.5B preset therefore carries an f16 KV override (`cache-type-k/v = f16` in `models-engine.ini`), under which it passes. The 9B passes at q8_0 KV as served. Both 0.5B results are committed; the q8_0 run has the suffix `-q8kv`.
- Known gate miss: 0.5B batch invariance, `angry` median 0.0117 against the 0.01 bar, with 100% argmax agreement and zero over-margin disagreements. By decision it is treated as small-model numerics and parked; the 4B and 9B pass the same gate. The same model passes the independent reference check for `angry` (0.0094), and the hybrid 4B passes batch invariance.
- Rounds equivalence was bit-identical, although the spec did not expect bit-identity because cell layout differs between the two runs. The 12-state request reported `rounds 3`.
- Multi-branch reference (follow-up fixes, `results-engine/reference-multibranch-*.json`):
  - The multi-branch field passes on both models: `topic` median 0.0005 (4B) and 0.0035 (0.5B), with zero over-margin disagreements. The 4B's two `topic` argmax disagreements have a reference top-2 margin below 0.15.
  - The 0.5B fails on `angry`, the single-branch field (median 0.0248). The deviations are not systematic: signed differences range from -0.041 to +0.095 (`p_true` engine minus reference).
  - Diagnostic: the same instructions with `angry` as the only field (1 branch node per state, no multi-branch sibling in the batch) also fail, at 0.0155. With the triage prompt, the same build still passes at 0.0094 (`results-engine/integrity/reference-diag-*`). The 0.5B `angry` deviation therefore depends on the prompt, not on multi-branch scoring, and is the same small-model numerics already seen at q8_0 KV and in 0.5B batch invariance. Under the spec 8.3 bar this 0.5B row is a fail; it is reported, not ruled away. The 4B hybrid passes every field.

## Per-ticket disagreements with the baseline fork
Argmax per field, ours vs the baseline for the same configuration (`uv run compare_engines.py`, `results-engine/compare-*.json`). "Confident" means both engines' top-2 margins exceed 0.15.

| Configuration | Queue: n (tickets) | ours right / baseline right | confident | Urgency: n (tickets) | ours / baseline right | confident | Angry: n (tickets) |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B, batch 1 | 3 (42, 61, 79) | 1 / 2 | 42, 61 | 6 (2, 19, 46, 64, 68, 79) | 3 / 3 | — | 0 |
| Qwen3.5-9B, batch 5 | 9 (42, 50, 52, 59, 61, 64, 65, 71, 79) | 2 / 7 | 42, 61, 71 | 6 (14, 19, 36, 46, 64, 79) | 2 / 4 | 64 | 0 |
| Gemma 4 E4B, batch 1 | 5 (20, 25, 50, 59, 70) | 3 / 2 | 50, 59 | 8 (1, 20, 25, 32, 33, 44, 57, 73) | 3 / 3 | 1, 25, 57 | 1 (32; ours right) |
| Gemma 4 E4B, batch 5 | 6 (16, 20, 25, 50, 59, 70) | 2 / 4 | 50, 59 | 8 (1, 20, 25, 32, 44, 57, 72, 73) | 3 / 3 | 1, 25, 57 | 1 (32; ours right) |

Argmax changes between batch 1 and batch 5 within each engine (queue / urgency / angry):

| Model | Ours | Baseline |
|---|---|---|
| Qwen3.5-9B | 65, 71 (both now wrong) / 2, 36, 68 / none | 50, 52, 64 (now right), 59 (now wrong) / 14 / none |
| Gemma 4 E4B | 25 (now wrong) / 33, 72 / none | 16, 25 (now right) / none / none |

## Observations
- Latency, dense vs hybrid:
  - Gemma 4 E4B (dense, SWA): our engine matches the baseline fork within 2 ms in every phase at both batch sizes (prefill 34.1 vs 32.5 ms, scoring 23.5 vs 23.5 ms at batch 1; 20.8 / 10.3 vs 20.1 / 10.0 ms at batch 5).
  - Qwen3.5-9B (hybrid): prefill is within 3 ms, but scoring takes 61.1 vs 41.6 ms at batch 1 and 32.9 vs 24.5 ms at batch 5. That puts our p50 at 1.21× and 1.09× the baseline, inside the 1.5× gate. The cause was not isolated. The candidates are how the three branch sequences take over the trunk's recurrent state, and how the hybrid memory's equal-split batching handles branches of unequal token length (spec section 10 risk). Our engine used 2 decodes per request and `rounds 1` in every request.
  - Batching gain: ours Qwen 119.8 → 97.2 ms (−19%) vs the baseline's 99.4 → 89.0 ms (−10%); Gemma 59.3 → 31.7 ms (−47%) in both engines. The hybrid model gains less from batching than the dense one in both engines.
- Accuracy:
  - At batch 1 both models are within 1 ticket of the baseline on queue and urgency. Gemma is 1–2 tickets higher on urg±1 and angry.
  - The Qwen batch 5 queue gap (5 tickets) is mostly batch-shape noise from both engines. At batch 1 the gap is 1 ticket. From batch 1 to 5 the baseline gained 2 net tickets (50, 52 and 64 right, 59 wrong) and ours lost 2 (65, 71). All six flips were below the 0.15 margin threshold at batch 1: our margins 0.13 and 0.04, the baseline's 0.014–0.090 (from `results/preds/`). Our own batch 1 vs batch 5 predictions pass the batch-invariance statistic at 80 tickets (queue median 0.0002, 0 over-margin flips).
  - Confident disagreements (both margins above 0.15): Qwen batch 1 queue tickets 42 and 61 (margins 0.45–0.64 on both sides), Qwen batch 5 queue 42, 61 and 71 and urgency 64, Gemma queue 50 and 59 and urgency 1, 25 and 57. The runs here do not separate the prompt text from the engine as their cause.
  - Calibration: queue NLL is higher with our prompt on both models (0.26–0.30 vs 0.19). Queue ECE is worse for Qwen (0.065 / 0.074 vs 0.040 / 0.070) and better for Gemma (0.052 / 0.035 vs 0.074 / 0.073). Angry Brier is lower in all four configurations (0.031–0.033 vs 0.032–0.039).
- Warnings: the server log showed the same benign lines as the baseline: `common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort` at Qwen load, and the Gemma GGUF token-type notices (`</s>`, `<|tool_response>`). There were no error-level lines. Each model load logged `decide: 21 sequences reserved above 1 slots` with `recurrent=1 swa=0` (Qwen) and `recurrent=0 swa=1` (Gemma).

## MoE models on this laptop (CPU experts)
Three Mixture-of-Experts GGUFs, none of which fits in 8 GB of VRAM, run with the expert weights on the CPU (`cpu-moe = 1` in the preset, spawned as `--cpu-moe`) and everything else on the GPU (`ngl 99` from `[*]`). All other settings are the `[*]` defaults (ctx 4096, q8_0 KV, fa on, `decide-seqs = 21`). There is no baseline run for these models: the rows below stand alone, and the dense rows above are the only comparison. Setup record: `results-engine/moe-setup.json` (verbatim server-log excerpts, `/v1/decide/info`, VRAM and RSS readings).

### Setup
| Preset | GGUF | Architecture (log) | Memory type | GPU / CPU model buffers (log) | VRAM after load / after runs | Model process RSS | Ready | Assistant prefix (`/v1/decide/info`) |
|---|---|---|---|---|---|---|---|---|
| `gemma-4-26b-a4b` | `gemma-4-26b-a4b-multilingual-Q4_K_M.gguf` (15.63 GiB; `general.name` "Gemma 4 26b A4B It Multilingual Merged") | `gemma4`, 25.23 B params, 128 experts, 8 used | `recurrent: false, swa: true` | CUDA0 1573.63 MiB / CPU_Mapped 15999.57 MiB | 2626 / 2660 (b5), 2766 (b1) MiB | 13.0 GiB after load, 16.4 GiB after the reference run | 16 s | `<turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>` |
| `qwen3.5-35b-a3b` | `Qwen3.5-35B-A3B-Q4_K_M.gguf` (19.71 GiB; no mmproj loaded) | `qwen35moe`, 34.66 B, 256 experts, 8 used | `recurrent: true, swa: false` (hybrid) | CUDA0 1305.15 MiB / CPU_Mapped 19779.53 MiB | 3270 / 3322 (b5), 3352 (b1) MiB | 13.7 GiB after load, 18.8 GiB after the runs | 17 s | `<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| `qwen3-30b-a3b` | `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (17.28 GiB) | `qwen3moe`, 30.53 B, 128 experts, 8 used | `recurrent: false, swa: false` | CUDA0 784.42 MiB / CPU_Mapped 17447.91 MiB | 1450 / 1514 (b5), 1530 (b1) MiB | 14.2 GiB after load | 19 s | `<\|im_end\|>\n<\|im_start\|>assistant\n` |

- Placement: every model logged `offloaded N/N layers to GPU` (31/31, 41/41, 49/49) with the expert tensors in the `CPU_Mapped` buffer, which is almost the whole file. Every load warned `tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance`. The GGUFs stay mmap'd: RSS counts their page-cache pages, and `free -m` showed 9.1–10.1 GiB used and 21–22 GiB available after each load. No `n-cpu-moe` or lower `ngl` was needed.
- KV / state on the GPU (log): Gemma KV 42.50 + 425.00 MiB, compute 337.84 MiB; Qwen3.5-35B KV 42.50 MiB plus recurrent state `RS buffer size = 1381.88 MiB`, compute 288.39 MiB; Qwen3-30B KV 204.00 MiB, compute 223.59 MiB.
- Each load logged `decide: 21 sequences reserved above 1 slots` with `memory recurrent=0 swa=1` (Gemma), `recurrent=1 swa=0` (Qwen3.5-35B) and `recurrent=0 swa=0` (Qwen3-30B). The engine returned `seqs_per_state 4`, `states_per_round 5` and `quote_split_fields: []` in every run.
- "Ready" is server start to the first `/v1/decide/info` answer, the request that triggers the load, with the GGUF largely in the page cache. It is not a cold-disk load time.
- Assistant prefixes, compared with the expected tails in `EXPECTED_PREFIX`:
  - Qwen3.5-35B: the same tail as `qwen3.5-9b`.
  - Gemma 26B-A4B: the tail is not the `gemma-4-e4b` one (`<|turn>model\n`). This GGUF's chat template appends `<|channel>thought\n<channel|>` after `<|turn>model\n` when `enable_thinking` is false, i.e. an empty, closed thought channel (the Gemma counterpart of Qwen's empty `<think>` block). `EXPECTED_PREFIX` holds the observed tail.
  - Qwen3-30B (Instruct-2507, no thinking mode): the tail `<|im_start|>assistant\n` was observed and added.
  - `prefix_ok: true` in every run.

### Accuracy (80 tickets, batch 5)
| Run | Queue | Urgency | Urg±1 | Angry | ECE (queue) | Brier (angry) | NLL (queue) | p50 ms / ticket |
|---|---|---|---|---|---|---|---|---|
| Ours, Gemma 4 26B-A4B, batch 5, CPU experts | 90.0% | 83.8% | 98.8% | 95.0% | 0.071 | 0.031 | 0.45 | 344 |
| Ours, Qwen3.5-35B-A3B, batch 5, CPU experts | 91.2% | 85.0% | 98.8% | 92.5% | 0.050 | 0.043 | 0.21 | 614 |
| Ours, Qwen3-30B-A3B-Instruct-2507, batch 5, CPU experts | 93.8% | 76.2% | 100.0% | 91.2% | 0.056 | 0.086 | 0.52 | 404 |

From `results-engine/score-table.txt`. Only batch 5 ran on all 80 tickets. Batch 1 ran on the first 20 tickets, for batch invariance, and is not scored. For reference, the dense rows at batch 5: Gemma 4 E4B 90.0 / 81.2 / 98.8 / 96.2%, Qwen3.5-9B 88.8 / 76.2 / 96.2 / 97.5%. With 80 tickets, 1 ticket = 1.25 points.

### Latency
| Run | wall p50 / p90 per ticket | server total_ms p50 per request | total_ms / n p50 | prefill / scoring per ticket, p50 | rounds | decodes per request | cached / all prompt tokens |
|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B, batch 5 (80) | 344.5 / 391.0 ms | 1719.7 ms | 343.9 ms | 230.4 / 113.1 ms | 1 (16 requests) | 2 | 43% (3952 / 9283) |
| Gemma 4 26B-A4B, batch 1 (first 20) | 1001.8 / 1112.0 ms | 999.3 ms | 999.3 ms | 783.5 / 216.3 ms | 1 (20 requests) | 2 | 78% (4940 / 6364) |
| Qwen3.5-35B-A3B, batch 5 (80) | 614.0 / 768.1 ms | 3067.9 ms | 613.6 ms | 494.9 / 121.1 ms | 1 (16 requests) | 2 | 43% (4064 / 9403) |
| Qwen3.5-35B-A3B, batch 1 (first 20) | 987.4 / 1187.6 ms | 985.3 ms | 985.3 ms | 790.0 / 190.5 ms | 1 (20 requests) | 2 | 78% (5080 / 6510) |
| Qwen3-30B-A3B, batch 5 (80) | 404.1 / 601.8 ms | 2018.2 ms | 403.6 ms | 272.0 / 128.1 ms | 1 (16 requests) | 2 | 45% (3968 / 8909) |
| Qwen3-30B-A3B, batch 1 (first 20) | 1144.1 / 1298.1 ms | 1141.6 ms | 1141.6 ms | 847.4 / 291.4 ms | 1 (20 requests) | 2 | 79% (4960 / 6290) |

Same method as the latency breakdown above (per-ticket values = request values / n, `statistics.median`, `statistics.quantiles(n=10)[8]`), from `results-engine/runs/decide-*-b5.json` and `results-engine/integrity/decide-*-b1-lim20-run.json`. The batch 1 rows cover only the first 20 tickets.

### Integrity (spec 8.3)
Same statistic and bars as the integrity section above. Reference = `reference_check.py dump` (`llama-decide --dump-tokens --cpu-moe`, server stopped) then `compare` against `/completion` on the same token ids. Batch invariance = batch 1 on the first 20 tickets (`decide_client.py --batch 1 --limit 20`) against the first 20 records of the 80-ticket batch 5 run. Those 20 tickets are the first 4 batch-5 requests, so they are the same requests a `--batch 5 --limit 20` run would send.

| Check | Model, KV | n | queue median / max | urgency median / max | angry median / max | argmax agreement q / u / a | over-margin disagreements | Result |
|---|---|---|---|---|---|---|---|---|
| Reference | Gemma 4 26B-A4B, q8_0, CPU experts | 20 × 3 | 0.0001 / 0.026 | 0.0012 / 0.085 | 0.0033 / 0.305 | 100 / 100 / 95% | 1 (angry, ticket 17) | fail |
| Batch invariance (b1 vs b5) | Gemma 4 26B-A4B | 20 | 0.0002 / 0.041 | 0.0008 / 0.084 | 0.0019 / 0.278 | 100 / 100 / 90% | 1 (angry, ticket 17) | fail |
| Reference | Qwen3.5-35B-A3B (hybrid), q8_0, CPU experts | 20 × 3 | 0.0002 / 0.353 | 0.0091 / 0.147 | 0.0041 / 0.046 | 95 / 90 / 100% | 1 (queue, ticket 5) | fail |
| Batch invariance (b1 vs b5) | Qwen3.5-35B-A3B | 20 | 0.0002 / 0.140 | 0.0047 / 0.071 | 0.0019 / 0.075 | 100 / 100 / 100% | 0 | pass |
| Reference | Qwen3-30B-A3B-Instruct-2507, q8_0, CPU experts | 20 × 3 | 0.0000 / 0.0004 | 0.0000 / 0.220 | 0.0000 / 0.297 | 100 / 100 / 100% | 0 | pass |
| Batch invariance (b1 vs b5) | Qwen3-30B-A3B-Instruct-2507 | 20 | 0.0000 / 0.0004 | 0.0000 / 0.211 | 0.0000 / 0.301 | 100 / 100 / 100% | 0 | pass |
| Reference, diagnostic: all expert matmuls on the CPU | Gemma 4 26B-A4B | 20 × 3 | 0.0001 / 0.051 | 0.0011 / 0.037 | 0.0009 / 0.166 | 100 / 100 / 95% | 1 (angry, ticket 17) | fail |
| Reference, diagnostic: all expert matmuls on the CPU | Qwen3.5-35B-A3B | 20 × 3 | 0.0001 / 0.339 | 0.0072 / 0.134 | 0.0038 / 0.098 | 95 / 90 / 100% | 2 (queue 5, urgency 16) | fail |

Files: `results-engine/reference-{gemma-4-26b-a4b,qwen3.5-35b-a3b,qwen3-30b-a3b}.json` (+ `-dump.jsonl`), `results-engine/batch-invariance-*.json` for the three presets, and the capped runs and diagnostics under `results-engine/integrity/` (`decide-*-b1-lim20*`, `decide-*-b5-first20.jsonl`, `reference-diag-cpuexperts-*`). Qwen3-30B's zero medians are below 5e-5 (e.g. queue 4e-10). Most of its option probabilities sit at 0 or 1, so the median falls on saturated options.

- The failing items:
  - Gemma ticket 17 `angry` (label false): `p_true` is 0.376 in the engine at batch 1, 0.681 in the reference, and 0.654 in the engine at batch 5 (one of 5 states in its request). The same ticket is the over-margin flip in both Gemma checks. A second Gemma batch flip, ticket 10 `angry` (0.443 → 0.607), is under the margin.
  - Qwen3.5-35B ticket 5 `queue` (label feedback): the engine says technical 0.672 / feedback 0.307, the reference feedback 0.657 / technical 0.318, and the engine at batch 5 technical 0.532 / feedback 0.432. The Qwen3.5-35B urgency median (0.0091) is just under the 0.01 bar.
- CLI and server agree exactly: for all three models, the `llama-decide` dump and the server's batch 1 run gave identical distributions (max difference 0 over 20 tickets × 3 fields). The reference deviations therefore lie between the engine's decode layout (cached prefix, trunk decode, branch decode) and `/completion`'s single prefill of the same token ids, not in a CLI/server difference.
- Diagnostic, op offload (`results-engine/integrity/reference-diag-cpuexperts-*`, `reference-diag-cpuexperts-spread.json`). ggml-cuda runs a CPU-resident matmul on the GPU (weights copied over PCIe) when the batch has ≥ 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32). So the long prefills run the expert matmuls on the GPU, and the short batch-1 branch decode runs them on the CPU. The dump and the server were rerun with `GGML_OP_OFFLOAD_MIN_BATCH=1000000`, which keeps every expert matmul on the CPU.
  - The same items still fail: Gemma angry 17, reference 0.659 vs engine 0.493; Qwen3.5 queue 5, reference feedback 0.643 vs engine technical 0.663. Qwen3.5 adds urgency 16 (reference margin 0.187). Mixed CPU/GPU expert backends are therefore not the only cause.
  - The offload setting alone moves results about as much as the gate allows. Engine vs engine across the two settings, for Qwen3.5-35B: urgency median 0.0075, max 0.203, argmax agreement 85%. Reference vs reference: urgency median 0.0063, max 0.167, 95%. For Gemma both spreads are smaller (medians ≤ 0.0035, max 0.176 on angry).

### Observations
- MoE routing works through the branch decode. All three models ran every request in one round with 2 decodes. The branch tokens are routed through the CPU experts like any other token, and every median is small (≤ 0.0091). Nothing failed to load or run. No request returned an error, and the only non-benign-looking log line was a Gemma shutdown warning `CUDA0 compute buffer size of 443.5261 MiB, does not match expectation of 337.8386 MiB` after the batch 5 run (VRAM stayed ≤ 2766 MiB).
- MoE outliers: in each MoE reference check a few single values deviate by 0.30–0.35 (max abs diff 0.305, 0.353, 0.297), against ≤ 0.13 max in every dense check above. Qwen3-30B passes only because its outliers keep the argmax (e.g. ticket 10 angry, reference 0.124 vs engine 0.421 at batch 1, 0.120 at batch 5). Top-k expert routing is discontinuous: a small numerical difference in a hidden state can change the selected experts of a token and then shift the output by much more than the difference itself. This is the likely cause, but not a proven one. It fits the pattern of medians near zero, rare large jumps, flips that follow batch shape and backend placement, and dense models that do not show such jumps. The engine agrees with itself exactly (CLI = server). What differs is the decode layout, and that is exactly what the engine changes by design.
- Hybrid MoE (Qwen3.5-35B-A3B): the recurrent state copy from prefix to trunk to branch works with MoE layers (`recurrent=1`, 1381.88 MiB RS buffer). Batch invariance passes. The reference check fails on one confident queue item and has the highest urgency median of the three (0.0091). The dense-attention MoE (Qwen3-30B) passes both checks, and the dense hybrid Qwen3.5-9B passed the reference at 0.0041 urgency. The hybrid MoE is therefore the model where the engine's split decode and the single prefill drift apart most. That can be the recurrent chunked scan over different chunk boundaries combined with routing flips; the two were not separated.
- Latency is dominated by the CPU experts: 344–614 ms per ticket at batch 5 and about 1 s at batch 1, against 32–97 ms for the dense models on the GPU. Per ticket, batch 5 is 2.9× (Gemma), 1.6× (Qwen3.5-35B) and 2.8× (Qwen3-30B) faster than batch 1 (batch 1 measured on the first 20 tickets), plausibly because the expert weights are read once per decode for all states of the request. Prefill takes roughly 67–81% of the per-ticket server time (p50 prefill / p50 total).
- Nothing was tuned: no tolerance, no preset beyond `cpu-moe = 1`. The Gemma 26B GGUF is a community "Multilingual Merged" build, not Google's release file, so its numbers describe that file.

## Prompt variants
Which catalogue/instruction text gives the best accuracy on the 80 tickets, for Qwen3.5-9B and Gemma 4 E4B, and is one text best for both? Each variant sets the request's top-level `instructions` and a description mode for the fields (`decide_client.py --prompt-variant NAME`, texts in `prompt_variants.py`). The engine always renders its own header and catalogue block from the fields after the instructions. `full` = the descriptions used in the parity runs (one line per queue option and urgency level); `short` = one-word descriptions (`Team.`, `Urgency.`, `Anger.`, queue options as bare keys, urgency levels `Wait / Normal / Today / Immediately`).

- `default`: the parity runs' request (not re-run; `decide-<preset>-b<N>.jsonl`).
- `merged`: the default line, then a compact one-line-per-field catalogue with the full option and level descriptions (`billing = …; technical = …`); one-word schema.
- `framing`: a two-sentence task framing instead of the default line; full schema.
- `keywords`: the default line plus keyword cues per queue option and one-line urgency and anger rules, written from the descriptions in `decide_client.py`; full schema. It differs from `default` only by the cue block. The cues come from the descriptions only; while checking the test file's format the author saw the first 200 characters of tickets 0–2, and no cue was taken from them.

Texts, verbatim:

`default` (description mode `full`):
```
Answer each question about this support ticket from its text.
```

`merged` (description mode `short`):
```
Answer each question about this support ticket from its text.

queue (which team must act first): billing = Payments, refunds, invoices, charges; technical = Bugs, crashes, outages, errors, performance, login problems; sales = Contracts, plan changes, pricing questions, cancellation threats, procurement; feedback = Praise, feature requests, general non-problem questions
urgency (how urgent): 0 = Can wait: praise, feature requests, idle questions, no problem; 1 = Normal: a real problem for one user without time pressure; 2 = Should be handled today: user is blocked, money wrongly taken, a deadline within days, or repeated contact; 3 = Drop everything: outage affecting many users, security incident, data loss, or enterprise customer about to churn
angry: true if the text itself shows anger (hostile wording, shouting, threats, sarcasm aimed at the company)
```

`framing` (description mode `full`):
```
You are triaging customer support tickets for a software company. Read the ticket and answer each question from the text alone.
```

`keywords` (description mode `full`):
```
Answer each question about this support ticket from its text.

Queue cues:
- billing: payment, refund, invoice, charge, billed
- technical: bug, crash, outage, error, slow, login
- sales: contract, plan change, pricing, cancel, procurement
- feedback: praise, thanks, feature request, suggestion, general question without a problem
Urgency rules:
- 0: nothing is wrong (praise, a feature request, an idle question).
- 1: one user has a real problem, with no time pressure.
- 2: the user is blocked, money was wrongly taken, a deadline is days away, or they are writing again.
- 3: an outage for many users, a security incident, data loss, or an enterprise customer about to leave.
Anger: true only when the wording itself is angry: hostile words, shouting (ALL CAPS, !!!), threats, or sarcasm aimed at the company.
```

### Results
80 tickets, all runs `rounds 1`, 2 decodes per request. From `results-engine/score-table.txt` (`uv run score.py --preds-dir results-engine/preds`); files `results-engine/preds/decide-<preset>-b<N>-<variant>.jsonl` and the run JSONs under `results-engine/runs/`. 1 ticket = 1.25 points.

| Model | Variant | Batch | Queue | Urgency | Urg±1 | Angry | ECE (queue) | NLL (queue) | p50 ms / ticket |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | `default` | 1 | 91.2% | 77.5% | 96.2% | 97.5% | 0.065 | 0.26 | 120 |
| Qwen3.5-9B | `default` | 5 | 88.8% | 76.2% | 96.2% | 97.5% | 0.074 | 0.27 | 97 |
| Qwen3.5-9B | `merged` | 1 | 95.0% | 73.8% | 97.5% | 97.5% | 0.061 | 0.21 | 124 |
| Qwen3.5-9B | `merged` | 5 | 92.5% | 75.0% | 97.5% | 97.5% | 0.060 | 0.21 | 97 |
| Qwen3.5-9B | `framing` | 1 | 93.8% | 75.0% | 95.0% | 97.5% | 0.050 | 0.21 | 123 |
| Qwen3.5-9B | `framing` | 5 | 92.5% | 76.2% | 95.0% | 97.5% | 0.061 | 0.22 | 97 |
| Qwen3.5-9B | `keywords` | 1 | 95.0% | 80.0% | 96.2% | 97.5% | 0.042 | 0.18 | 123 |
| Qwen3.5-9B | `keywords` | 5 | 95.0% | 78.8% | 95.0% | 97.5% | 0.061 | 0.18 | 98 |
| Gemma 4 E4B | `default` | 1 | 91.2% | 81.2% | 98.8% | 96.2% | 0.052 | 0.30 | 59 |
| Gemma 4 E4B | `default` | 5 | 90.0% | 81.2% | 98.8% | 96.2% | 0.035 | 0.30 | 32 |
| Gemma 4 E4B | `merged` | 1 | 91.2% | 83.8% | 97.5% | 96.2% | 0.040 | 0.27 | 58 |
| Gemma 4 E4B | `merged` | 5 | 91.2% | 82.5% | 97.5% | 96.2% | 0.048 | 0.27 | 31 |
| Gemma 4 E4B | `framing` | 1 | 92.5% | 77.5% | 97.5% | 95.0% | 0.025 | 0.20 | 59 |
| Gemma 4 E4B | `framing` | 5 | 92.5% | 77.5% | 97.5% | 95.0% | 0.045 | 0.21 | 31 |
| Gemma 4 E4B | `keywords` | 1 | 93.8% | 82.5% | 98.8% | 96.2% | 0.042 | 0.20 | 60 |
| Gemma 4 E4B | `keywords` | 5 | 93.8% | 82.5% | 98.8% | 95.0% | 0.053 | 0.20 | 31 |

Cached system prefix per request (tokens, Qwen / Gemma): `default` 254 / 247, `merged` 297 / 288, `framing` 267 / 260, `keywords` 448 / 438.[^prefix-info]

[^prefix-info]: The committed variant runs were recorded before `decide_client.py` re-fetched `/v1/decide/info` after its warm-up request, so the `info` block of each run JSON holds the prefix cached by the previous invocation on the server. Each variant's own count sits in the `info` of the next run of the same variant, the b5 file (b1 ran just before it): e.g. Qwen `keywords` = 448 comes from `decide-qwen3.5-9b-b5-keywords.json`, while its b1 file shows 267, the `framing` prefix of the run before. The per-request `usage.cached_tokens` in every run JSON is the run's own count and gives the same numbers. Runs made after the fix (the held-out check below) record their own prefix in `info`.

Queue answers that change against `default` at the same batch size (tickets; "right" = the variant now matches the label, "wrong" = `default` matched it):

| Variant | Qwen b1 | Qwen b5 | Gemma b1 | Gemma b5 |
|---|---|---|---|---|
| `merged` | 5: 4 right (42, 52, 64, 79), 1 wrong (65) | 5: 4 right (42, 64, 71, 79), 1 wrong (41) | 4: 2 right (16, 65), 2 wrong (25, 29) | 3: 2 right (16, 65), 1 wrong (29) |
| `framing` | 2: 2 right (50, 79) | 3: 3 right (65, 71, 79) | 3: 2 right (16, 20), 1 wrong (70) | 4: 3 right (16, 20, 25), 1 wrong (70) |
| `keywords` | 3: 3 right (50, 64, 79) | 5: 5 right (50, 64, 65, 71, 79) | 2: 2 right (16, 65) | 5: 4 right (16, 25, 50, 65), 1 wrong (4) |

Every changed ticket is either right→wrong or wrong→right; no change went from one wrong answer to another. Exact two-sided sign test on the `keywords` right/wrong counts: p = 0.25 (3–0), 0.063 (5–0), 0.50 (2–0), 0.38 (4–1); none is below 0.05 on its own. Batch 1 and batch 5 score the same tickets, so the four columns are not independent.

Latency: the first two passes over the variants ran on a hot GPU (SM clock 510 MHz at 80 °C after the runs). A default-prompt control run at the same time measured 110 ms (Gemma b1) and 135–202 ms (Qwen b1), against 59 / 120 ms in the parity runs. The committed runs are a third pass, with a cool-down to ≤ 60 °C before every run and default-prompt controls before and after each model (outputs kept outside the repo): Qwen 122.6 / 123.7 ms (b1) and 97.4 / 97.4 ms (b5), Gemma 58.7 / 58.4 ms (b1) and 30.6 / 30.6 ms (b5). The variant p50s in the table lie within 2 ms of these controls. The predictions were bit-identical in all three passes (queue probabilities compared per ticket), so accuracy does not depend on the pass. `compare_engines.py` was not run: unchanged, it stops at the first MoE preds file, which has no baseline run (`results/preds/decision-gemma-4-26b-a4b-b5.jsonl` does not exist); the changes against `default` above come from a join of the preds files.

### Findings
- `keywords` is the only variant with the highest queue accuracy of the four (alone or tied) on both models at both batch sizes: Qwen 95.0 / 95.0% (default 91.2 / 88.8%), Gemma 93.8 / 93.8% (default 91.2 / 90.0%). Against `default` it changes 2–5 queue answers per configuration, 14 of 15 to the label. Urgency is also higher (Qwen +2 / +2 tickets, Gemma +1 / +1), queue NLL falls from 0.26–0.27 to 0.18 (Qwen) and from 0.30 to 0.20 (Gemma). It loses 1 ticket on Qwen b5 urg±1 and on Gemma b5 angry, and Gemma b5 queue ECE rises from 0.035 to 0.053. p50 is within 3 ms of `default`.
- `merged` reaches 95.0% queue on Qwen at batch 1 only (92.5% at b5) and has the lowest Qwen urgency (73.8 / 75.0%); on Gemma its queue equals `default` at b1. `framing` gains 1–3 queue tickets but loses urgency on Qwen b1 (−2) and on Gemma (77.5% vs 81.2%, −3 at both batch sizes); Qwen b5 urgency is unchanged.

### Recommendation
Make `keywords` the default for both models. It is the only text that is best (or tied best) on queue for both models at both batch sizes and does not lower queue or urgency accuracy in any of the four configurations; the cost is a cached prefix about 190 tokens longer; p50 stays within 3 ms. The evidence is small: the gains are 2–5 tickets of 80, no single configuration passes a sign test at 0.05, and the best variant was picked on the same 80 tickets it is scored on, so the measured margin over `default` likely overstates the true one. A held-out ticket set is needed to confirm it: see the held-out check below.

### Held-out check (320 train tickets)
The comparison above picked `keywords` on the same 80 tickets it was scored on. This check runs the variants on a second labelled set, `data/triage_train.jsonl`: 320 tickets in the same format (`text`, `queue`, `urgency`, `angry`), 80 per queue, no text shared with the 80-ticket test set, not used before in this project; the variant texts were written without looking at it. The 320 tickets select the variant; the 80-ticket results above are the confirmation.

Selection rule, written here before the runs:

> Primary metric: queue accuracy on the 320 train tickets at batch 5; tie-break: mean of urgency and angry accuracy. The winner is compared with `default` by an exact two-sided sign test on per-ticket queue correctness (discordant pairs only). The client default changes to the winner only if the winner is the same for both models and its sign test against `default` has p < 0.05 on both models; otherwise `default` stays and the report says so.

Implementation notes, also fixed before the runs: per-ticket correctness as in `score.py` (queue and urgency: argmax of the distribution; angry: `p_true ≥ 0.5`); two-sided p = twice the smaller tail of Binomial(discordant pairs, 0.5), capped at 1; a tie that the tie-break does not resolve leaves that model without a single winner, and then `default` stays.

Runs: `decide_client.py --batch 5 --test-set …/triage_train.jsonl --out-dir results-engine/heldout --prompt-variant V` for each variant (and the ablation variant of the parity section, not included), all Qwen runs first, then all Gemma runs; every request ran in 1 round with 2 decodes. Scores from `results-engine/heldout/score-table.txt` (`score.py` on `results-engine/heldout/preds`), sign tests and the rule from `results-engine/heldout/stats.json` (`heldout_stats.py`); run JSONs under `results-engine/heldout/runs/`. 1 ticket = 0.31 points. The GPU was at 59–60 °C before every run (Qwen `default`: 55 °C). That cool-down (below 60 °C before each run) is stricter than the planned "above 75 °C, wait until below 65 °C": the first Qwen `merged` run, started at 74 °C under that plan, ran up to 76 °C with the SM clock down to 510 MHz (nvidia-smi reported thermal slowdown at 75 °C) and p50 142 ms; it was run again cool, and its predictions were bit-identical to the hot run.

| Model | Variant | Queue | Urgency | Urg±1 | Angry | ECE (queue) | NLL (queue) | p50 ms / ticket |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | `default` | 89.7% | 80.3% | 98.4% | 98.4% | 0.029 | 0.23 | 94 |
| Qwen3.5-9B | `merged` | 94.1% | 75.9% | 98.1% | 97.8% | 0.037 | 0.17 | 96 |
| Qwen3.5-9B | `framing` | 91.2% | 79.4% | 97.8% | 98.1% | 0.034 | 0.21 | 96 |
| Qwen3.5-9B | `keywords` | 94.7% | 78.4% | 97.8% | 98.4% | 0.028 | 0.18 | 96 |
| Gemma 4 E4B | `default` | 89.4% | 84.4% | 99.1% | 97.5% | 0.068 | 0.38 | 29 |
| Gemma 4 E4B | `merged` | 90.6% | 85.6% | 98.4% | 97.5% | 0.059 | 0.31 | 29 |
| Gemma 4 E4B | `framing` | 91.9% | 80.3% | 98.4% | 96.6% | 0.050 | 0.29 | 29 |
| Gemma 4 E4B | `keywords` | 94.4% | 84.4% | 98.4% | 97.5% | 0.041 | 0.25 | 29 |

Exact two-sided sign tests against `default` on the discordant pairs (320 tickets per model). The rule uses only the winner's queue test; the other rows are descriptive.

| Model | Comparison | Field | Variant right, `default` wrong | `default` right, variant wrong | p |
|---|---|---|---|---|---|
| Qwen3.5-9B | `keywords` (winner) vs `default` | queue | 16 | 0 | 3.1e-05 |
| Qwen3.5-9B | `keywords` (winner) vs `default` | urgency | 12 | 18 | 0.36 |
| Qwen3.5-9B | `keywords` (winner) vs `default` | angry | 1 | 1 | 1 |
| Gemma 4 E4B | `keywords` (winner) vs `default` | queue | 17 | 1 | 1.4e-04 |
| Gemma 4 E4B | `keywords` (winner) vs `default` | urgency | 16 | 16 | 1 |
| Gemma 4 E4B | `keywords` (winner) vs `default` | angry | 2 | 2 | 1 |

Decision under the rule: `keywords` has the most queue tickets right on both models (Qwen 303 / 320, next `merged` 301; Gemma 302 / 320, next `framing` 294; the tie-break was not needed), and its queue sign test against `default` gives p = 3.1e-05 (16–0) on Qwen and p = 1.4e-04 (17–1) on Gemma, both below 0.05. The client default therefore changes to `keywords`: `decide_client.py` without `--prompt-variant` now sends the `keywords` request (outputs `decide-<preset>-b<N>-keywords.*`), and `--prompt-variant default` gives the request of every run before this check. `keywords` does not gain on the other two fields: Qwen urgency is 6 tickets lower than `default` (78.4% vs 80.3%, 12–18, p = 0.36); Qwen angry and Gemma urgency and angry have as many tickets right as `default` (1–1, 16–16, 2–2).

Against the 80-ticket results: there `keywords` beat `default` on queue at batch 5 by 5–0 (Qwen, 95.0% vs 88.8%) and 4–1 (Gemma, 93.8% vs 90.0%); on the 320 held-out tickets the direction is the same, 16–0 (94.7% vs 89.7%) and 17–1 (94.4% vs 89.4%), while the Qwen urgency gain of the 80 tickets (+2 tickets) is not repeated (−6).

## Jev compatibility (`POST /v1/systemone`)

Sub-project 2 (spec `docs/specs/2026-09-25-systemone-compat-design.md`): the
engine's `llama-server` answers TypeSafe AI's System One shape by translating the request
into a `/v1/decide` body (`tools/decide/decide-compat.{h,cpp}`, ctest `test-decide-compat`)
and the decide result back into Jev answers. No new scoring: the numbers are the engine's.
Known differences (no rounding/clipping, `output_tokens` 0, visible question ids, limits,
router-mode error bodies, ...) are listed in `README.md`, section "Jev compatibility".

Setup: `./serve-engine.sh` (router mode, preset `qwen3.5-9b`, Qwen3.5-9B Q4_K_M, :8097), engine
branch `decide` at `f042104fa`.

### Smoke: `/v1/systemone` vs the Python oracle through `/v1/decide`
`systemone_smoke.py` rebuilds the translated `/v1/decide` body in Python (spec 3.2; its text
helpers match nlohmann `dump()`/`dump(2)` for strings, objects, arrays and ordinary numbers, UTF-8 kept; floats of 1e15 and above and integers beyond 64 bits are written differently, and no test input uses them) and posts it to
`/v1/decide`, and the Jev request to `/v1/systemone`. Cases: the three examples of Jev's public API docs merged
into one request (`sales: null` and an object score level added to exercise the helpers) and
triage ticket 0 with the `decide_client.py` questions (choice `queue` with the four option
descriptions, score `urgency` with the four levels, noul `angry`). Every probability (and the
score's expected value) is identical through both endpoints: max abs diff 0.

```
$ uv run systemone_smoke.py --url http://127.0.0.1:8097
== docs example
{
  "model": "qwen3.5-9b",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.9989880051195126
    },
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.699858256827507,
        "technical": 0.26296746587565506,
        "sales": 0.03717427729683782
      },
      "confidence": 0.5497873852412605
    },
    "frustration": {
      "type": "score",
      "score": 1.1310731131754708,
      "legend": {
        "0": "Calm",
        "1": {
          "level": "Frustrated"
        },
        "2": "Very angry"
      },
      "probabilities": {
        "0": 0.09709657975557312,
        "1": 0.6747337273133829,
        "2": 0.22816969293104397
      },
      "confidence": 0.5121005909700744
    }
  },
  "usage": {
    "input_tokens": 193,
    "output_tokens": 0
  }
}
max abs diff vs /v1/decide oracle body: 0
== triage ticket 0
{
  "model": "qwen3.5-9b",
  "answers": {
    "queue": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.8469228376491807,
        "technical": 0.050744615439838464,
        "sales": 0.04753659344437321,
        "feedback": 0.05479595346660762
      },
      "confidence": 0.7958971168655743
    },
    "urgency": {
      "type": "score",
      "score": 0.09410470728164672,
      "legend": {
        "0": "Can wait: praise, feature requests, idle questions, no problem",
        "1": "Normal: a real problem for one user without time pressure",
        "2": "Should be handled today: user is blocked, money wrongly taken, a deadline within days, or repeated contact",
        "3": "Drop everything: outage affecting many users, security incident, data loss, or enterprise customer about to churn"
      },
      "probabilities": {
        "0": 0.9115927888461662,
        "1": 0.08324431541307024,
        "2": 0.004628295353714387,
        "3": 0.0005346003870492324
      },
      "confidence": 0.882123718461555
    },
    "angry": {
      "type": "noul",
      "noul": 0.00036706759957805073
    }
  },
  "usage": {
    "input_tokens": 309,
    "output_tokens": 0
  }
}
max abs diff vs /v1/decide oracle body: 0
OK: identical probabilities through both endpoints (max abs diff 0)
```

### Official SDK acceptance
`tests/test_systemone_sdk.py` with the unmodified TypeSafe AI Python SDK 0.7.0
(`TypeSafeClient(api_key="local", base_url=..., retry=RetryPolicy(max_retries=0), timeout=180)`,
`system_one(..., model="qwen3.5-9b")`): one `Choice` (one option with a `None` description), one
`Score` (one object level) and one `Noul` with criteria on a support ticket. The SDK's strict
models parse the response; the test checks typed answers, question order, probabilities summing
to 1 (±1e-6), `legend` = the original criteria (object kept), `noul` in [0, 1] and integer usage
with `output_tokens` 0. A second test sends a one-level `Score` (valid for the SDK, not for our
compiler) and gets the SDK's typed `TypeSafeUnprocessableEntityError`
(`422 questions.one: field "one": score needs 2 to 10 levels`).

```
$ SYSTEMONE_URL=http://127.0.0.1:8097 uv run --with <SDK checkout> pytest -q tests/test_systemone_sdk.py
..                                                                       [100%]
2 passed in 0.21s
```
With `-s` the first test prints `SDK 0.7.0: model=qwen3.5-9b queue=billing urgency=2.079 angry=0.520 usage=232/0`.

## Engine improvements (sub-project 3)

Spec `docs/specs/2026-09-25-engine-improvements-design.md`; engine patches 0020-0032 (branch `decide` at
`63ea2c51a`; 0032, a follow-up fix, only registers `test-decide-trie` with ctest, so the binaries are those of `5e9e030d3`). Five switches, all off by default except the prefix cache (on, 8 entries), plus hardening. Every number below comes from committed files under
`results-engine/sp3/`: `baseline/` (Task 0: tree b5, prompt `keywords`, test and train, both models), `checks/task-4..7/`
(live checks while the engine was built), `integrity/` (Task 8, final build) and `eval/` (Task 9: evaluation,
calibration; `eval/run.sh` and `eval/analysis.py` produce every file there, `eval/summary.md` holds the tables below).

### The switches

A request that uses none of the per-request switches gets the probabilities of the build before sub-project 3
(regression in the integrity table below: max abs diff 0) when its prefix is resident or built into the same cells
(spec 1: a prefix restored from the cache, or rebuilt after other cells were allocated, can land in other KV cells, which
moves the answers of placement-sensitive models such as Qwen2.5-0.5B, see Integrity), with one exception since the
calibration below: the `gemma-4-e4b` preset applies `decide-temperature = 1.6947`, so its answers keep every argmax but
their probabilities are tempered (`options.temperature: 1` returns the untempered ones). `decide_client.py` sets the
switches with `--scoring`, `--after FIELD=PARENT[,PARENT]`, `--order-debias K` and `--temperature T`; each switch that
changes the request adds a suffix to the output names, and so does a temperature other than 1.0 in the answers'
`engine.temperature` (`-t<T>`, e.g. `-t1.6947` for a Gemma run without `--temperature`). `--ensure-t1` runs at T = 1
whatever the preset; without `--force` the client stops before writing over existing outputs.

| # | switch | set with | default | what it does | response additions |
|---|---|---|---|---|---|
| 1 | full-path scoring | request `options.scoring: "full_path"` | `"tree"` | one branch sequence per option; P(option) = the product of its path tokens' full-vocabulary probabilities times the probability that the value ends there (a value-end token such as `,` or `}` next, or one token that merges the last piece with the end, e.g. `",`), renormalised over the options. Lifts tree mode's sibling-prefix rule (`sale`/`sales` is a 400 in tree mode) | per field `coverage` = Σ P(option) before renormalisation (stem-relative; not comparable across fields) |
| 2 | sequential fields | field `"after": [names]` (up to 8 levels) | none | the field is scored after its ancestors, with their answers written into its prompt (`  "queue": "billing",\n` between `{` and its key, in declaration order); one more branch decode phase per level | dependent fields: `given` = the ancestors' values |
| 3 | prefix cache | server/CLI `--decide-prefix-cache N` (preset key `decide-prefix-cache`) | 8 | LRU of up to N prefix snapshots (≤ 2 GiB) in host RAM: a prefix evicted by another schema, prompt or catalogue variant is restored instead of decoded again | `engine.prefix_cache` hits/misses, `timings.prefix_ms` |
| 4 | option-order debiasing | request `options.order_debias: K` (0-8) | 0 (off; 1 is also off) | K_eff = min(K, largest choice option count) catalogues, choice options rotated left by ⌊v·n/K_eff⌋ in variant v; every state is scored under each and the distributions are averaged | per field `order_spread` = max over options of (max − min over variants) |
| 5 | temperature | request `options.temperature: T`; server/CLI `--decide-temperature T` (preset key `decide-temperature`) | 1.0 | p_T(o) ∝ exp(log p(o) / T) after debias averaging; `value` (argmax) unchanged; `coverage` and `order_spread` before temperature | `engine.temperature` |

Hardening beside them (spec 9): decide sequences are detached from backend samplers (`--backend-sampling` works with
several rows per sequence), `LLAMA_DECIDE_FAULT` injects `rc == 1` for the retry path (`rc1`, `rc1-phase1`, `rc1-twice`,
`restore`; since the follow-ups also `cell-estimate`, `lines-join` and `tail-mismatch`, section "Follow-ups (2026-10-02)"), and `--decide-seqs` is checked after `-np` is resolved.

### Integrity (spec 12.3)

Details, commands and every output: `results-engine/sp3/checks/task-{4,5,6,7}/notes.md` (checks during the engine tasks)
and `results-engine/sp3/integrity/notes.md` (Task 8, engine `5e9e030d3`, the final build). "The statistic" is the
parent spec's (per field: median abs diff over options ≤ 0.01, no argmax disagreement where the reference's top-2
margin > 0.15). Qwen3.5-4B, Gemma 4 E4B and Qwen3.5-9B are the binding models, designated during execution (decision after the
`after` checks: the 0.5B's angry failures are its known batch sensitivity, and Gemma 4 E4B binds the row-cap split checks); spec
12.3 lists Qwen2.5-0.5B without an exemption. Qwen2.5-0.5B is reported beside its own tree-mode numbers (below).

| check | result |
|---|---|
| Regression (spec 12.2): the Task 0 baseline configurations (tree b5 `keywords`, test and train, both models) on freshly started routers, and the 0.5B tree dump | max abs diff 0 in all five against the Task 0 baseline (`integrity/regression-*.json`); `decide_client.py`'s requests without switches byte-identical to the baseline's (164 bodies, `regression-request-identity.json`). Produced before the calibration preset change below |
| Full-path reference (replay of every node through `/completion`, top-512), 20 states b1 | Qwen3.5-4B, Gemma 4 E4B, Qwen3.5-9B pass (probability medians ≤ 0.0082, coverage medians ≤ 0.0005, node medians ≤ 0.0004); no unverifiable edge on any model; coverage gate (median ≥ 0.8) 0.9997 / 0.9994 / 0.9975 (9B) and 0.9999 / 1.0000 / 0.9990 (Gemma), queue / urgency / angry. Qwen2.5-0.5B fails only on angry (0.0144; its tree replay on the same states 0.0094) |
| Row-cap splits: full-path b5 whose branch rows need two decodes vs the same states as b1 | Gemma 4 E4B passes (medians 1.1e-05 / 7.5e-05 / 0.0003); b5 references read the rows after the chunk boundary correctly (Gemma chunk-1 nodes 7.4e-06 median); the 0.5B's angry 0.0125 with the split, 0.0104 without it, tree 0.0117 |
| `after` reference (urgency and angry after queue, full-path, dependent prompts include the answer lines) | Qwen3.5-4B passes (0.0010 / 0.0019 / 0.0022); every `given` equals the returned queue value and every `lines` text is built from it (120 entries, 0 problems, both models); 0.5B angry 0.0126 |
| Prefix cache A/B/A (Task 4) | third answer = first (max abs diff 0) with a cache hit on Qwen2.5-0.5B (dense), Qwen3.5-4B (hybrid) and Gemma 4 E4B (iSWA); an injected restore failure falls back to a rebuild (answers 0 vs cache off); a failed request (400 budget, injected 500) leaves only slot 0 |
| Debias (Task 7) | K = 1 = K = 0 (max abs diff 0); K = 2 = the per-key mean of two K = 0 requests with rotated options, under the statistic, on the 0.5B, the 4B and Gemma (medians ≤ 0.0044); `after` + K = 2 with ≥ 2 states per round (Task 8) passes on the 4B |
| Temperature (Task 7) | T = 2 answers = the T = 1 answers scaled post hoc (max abs diff 6.7e-16); server default `--decide-temperature 2` and `/v1/systemone` likewise (≤ 4.4e-16) |
| Fault hook (injected `rc == 1`) | every retried request completes (`retries` ≥ 1): tree `rc1` b5 and b1, `rc1-phase1` tree (with `after`) and `rc1` with K = 2 equal a normal run with the same states per round (max abs diff 0); full-path `rc1` and `rc1-phase1` pass the statistic against normal runs with more states per round (max 0.048, 0.069); `rc1-twice`: the first request 500, then only slot 0 in `/v1/decide/info` and the next request equal to a normal server's (0) |
| Every switch at once (full-path, `after`, K = 4, T = 2) | runs on Qwen2.5-0.5B at `--decide-seqs 64` (15 answers sum to 1, `coverage`, `order_spread`, `given` present; recomputed from the dump within 1.1e-16); on Qwen3.5-4B at 32 the same request is a 400 `budget`: `schema needs 44 sequences per state (K_eff 4 × (1 trunk + 10 branch sequences, full_path)); --decide-seqs 32 leaves 28 after 4 prefix slots`, slot 0 untouched |
| `--backend-sampling` | tree (Task 4) and full-path (Task 5) CLI runs complete with answers equal to normal runs (0) |

Qwen2.5-0.5B (non-binding by that decision) is sensitive to batch composition and KV cell placement, independent of the new features:
its tree-mode b5 vs b1 angry median is 0.0117 (sub-project 1, `results-engine/batch-invariance-qwen2.5-0.5b.json`), and
in Task 4 the same request after other cells had been allocated moved by up to 0.039 when its prefix was restored (the
statistic passed) and by up to 0.074 when it was rebuilt (angry median 0.0139, the statistic failed), while restores into
the original cells were exact (0). Its failures above are all on angry and of that size, but only the row-cap split is
matched by tree mode on the same states (split 0.0125 and unsplit 0.0104 vs tree b5 vs b1 0.0117). The full-path
reference (0.0144) is further from the replay than tree mode's replay of the same 20 states (0.0094, which passes), so
on this model full-path's angry answer deviates more than tree's; the `after` reference (0.0126, also full-path) covers
the same tickets, but its dependent angry prompt differs from tree's (it includes the queue answer line). The models
designated binding stay at or below 0.0082.

### Evaluation (spec 12.4)

Runs: the test (80) and train (320) sets, b5, both models, `decide_client.py`'s default prompt variant `keywords`, presets
of `models-engine.ini` (`--decide-seqs 21`); full-path, after A (urgency after queue), after B (urgency and angry after
queue) and debias K = 4 (= the queue field's option count, so every option takes every position once): 16 runs, each on a
freshly started preset router, all at T = 1 (every run JSON's engine blocks report `temperature` 1.0, checked by
`analysis.py check`, together with scoring, effective K and phases; answer keys and `given` checked on every row:
0 problems). The tree rows are the Task 0 baseline (scores) and the regression runs of `results-engine/sp3/integrity/` (cost: the same requests on the
same build, answers identical to the baseline). Tables from `results-engine/sp3/eval/summary.md`.

Scores, test set (80 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | tree (baseline) | 95.0% | 78.8% | 95.0% | 97.5% | 0.182 | 0.535 | 0.122 | 0.839 | 0.061 | 0.061 | 0.063 | 0.033 | 92.7 | 1 |
| qwen3.5-9b | full-path | 93.8% | 81.2% | 96.2% | 97.5% | 0.182 | 0.522 | 0.126 | 0.830 | 0.055 | 0.072 | 0.061 | 0.034 | 217.3 | 5 |
| qwen3.5-9b | after A (urgency after queue) | 95.0% | 85.0% | 96.2% | 97.5% | 0.175 | 0.555 | 0.123 | 0.853 | 0.052 | 0.121 | 0.065 | 0.034 | 144.9 | 1 |
| qwen3.5-9b | after B (urgency, angry after queue) | 95.0% | 85.0% | 96.2% | 98.8% | 0.181 | 0.546 | 0.138 | 0.865 | 0.045 | 0.120 | 0.090 | 0.038 | 174.3 | 1 |
| qwen3.5-9b | debias K = 4 | 95.0% | 78.8% | 95.0% | 98.8% | 0.178 | 0.508 | 0.124 | 0.809 | 0.047 | 0.055 | 0.076 | 0.034 | 513.4 | 5 |
| gemma-4-e4b | tree (baseline) | 93.8% | 82.5% | 98.8% | 95.0% | 0.197 | 0.582 | 0.131 | 0.909 | 0.053 | 0.110 | 0.031 | 0.031 | 31.4 | 1 |
| gemma-4-e4b | full-path | 92.5% | 82.5% | 98.8% | 95.0% | 0.200 | 0.580 | 0.131 | 0.911 | 0.054 | 0.106 | 0.027 | 0.031 | 154.2 | 5 |
| gemma-4-e4b | after A (urgency after queue) | 93.8% | 86.2% | 98.8% | 96.2% | 0.202 | 0.612 | 0.131 | 0.945 | 0.041 | 0.081 | 0.039 | 0.031 | 56.1 | 1 |
| gemma-4-e4b | after B (urgency, angry after queue) | 93.8% | 86.2% | 98.8% | 96.2% | 0.201 | 0.607 | 0.125 | 0.933 | 0.042 | 0.082 | 0.042 | 0.031 | 56.8 | 1 |
| gemma-4-e4b | debias K = 4 | 92.5% | 83.8% | 98.8% | 96.2% | 0.203 | 0.565 | 0.131 | 0.900 | 0.046 | 0.099 | 0.028 | 0.031 | 232.5 | 5 |

Scores, train set (320 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | tree (baseline) | 94.7% | 78.4% | 97.8% | 98.4% | 0.176 | 0.552 | 0.092 | 0.820 | 0.028 | 0.085 | 0.054 | 0.022 | 94.8 | 1 |
| qwen3.5-9b | full-path | 95.0% | 78.1% | 97.8% | 98.4% | 0.177 | 0.548 | 0.093 | 0.818 | 0.021 | 0.081 | 0.054 | 0.022 | 259.6 | 5 |
| qwen3.5-9b | after A (urgency after queue) | 94.7% | 82.2% | 98.4% | 98.4% | 0.178 | 0.539 | 0.092 | 0.809 | 0.031 | 0.079 | 0.058 | 0.022 | 146.6 | 1 |
| qwen3.5-9b | after B (urgency, angry after queue) | 94.7% | 82.2% | 98.4% | 98.4% | 0.177 | 0.533 | 0.098 | 0.808 | 0.027 | 0.094 | 0.059 | 0.024 | 181.2 | 1 |
| qwen3.5-9b | debias K = 4 | 95.3% | 78.8% | 97.8% | 98.4% | 0.175 | 0.530 | 0.093 | 0.797 | 0.029 | 0.071 | 0.055 | 0.022 | 494.8 | 5 |
| gemma-4-e4b | tree (baseline) | 94.4% | 84.4% | 98.4% | 97.5% | 0.248 | 0.523 | 0.056 | 0.827 | 0.041 | 0.087 | 0.011 | 0.017 | 29.7 | 1 |
| gemma-4-e4b | full-path | 94.4% | 84.7% | 98.4% | 97.5% | 0.246 | 0.524 | 0.055 | 0.825 | 0.039 | 0.090 | 0.011 | 0.017 | 152.1 | 5 |
| gemma-4-e4b | after A (urgency after queue) | 94.7% | 85.0% | 98.8% | 97.8% | 0.246 | 0.559 | 0.055 | 0.861 | 0.044 | 0.079 | 0.010 | 0.017 | 54.5 | 1 |
| gemma-4-e4b | after B (urgency, angry after queue) | 94.4% | 85.3% | 98.8% | 97.8% | 0.247 | 0.560 | 0.053 | 0.860 | 0.040 | 0.080 | 0.016 | 0.016 | 69.4 | 1 |
| gemma-4-e4b | debias K = 4 | 94.7% | 84.7% | 98.4% | 97.5% | 0.253 | 0.514 | 0.055 | 0.822 | 0.046 | 0.079 | 0.013 | 0.017 | 227.8 | 5 |

Scores, test and train pooled (400 tickets; `sp3_metrics.score_preds` on the 400 rows, so NLL is the mean over the 400 tickets and ECE is binned over them):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | tree (baseline) | 94.8% | 78.5% | 97.2% | 98.2% | 0.1773 | 0.5483 | 0.0981 | 0.8237 | 0.032 | 0.078 | 0.054 | 0.024 |
| qwen3.5-9b | full-path | 94.8% | 78.8% | 97.5% | 98.2% | 0.1780 | 0.5428 | 0.0996 | 0.8204 | 0.023 | 0.067 | 0.054 | 0.025 |
| qwen3.5-9b | after A (urgency after queue) | 94.8% | 82.8% | 98.0% | 98.2% | 0.1777 | 0.5420 | 0.0979 | 0.8175 | 0.033 | 0.087 | 0.055 | 0.024 |
| qwen3.5-9b | after B (urgency, angry after queue) | 94.8% | 82.8% | 98.0% | 98.5% | 0.1776 | 0.5357 | 0.1061 | 0.8194 | 0.030 | 0.098 | 0.065 | 0.027 |
| qwen3.5-9b | debias K = 4 | 95.2% | 78.8% | 97.2% | 98.5% | 0.1752 | 0.5253 | 0.0990 | 0.7995 | 0.029 | 0.066 | 0.058 | 0.024 |
| gemma-4-e4b | tree (baseline) | 94.2% | 84.0% | 98.5% | 97.0% | 0.2378 | 0.5344 | 0.0708 | 0.8430 | 0.038 | 0.081 | 0.011 | 0.020 |
| gemma-4-e4b | full-path | 94.0% | 84.2% | 98.5% | 97.0% | 0.2370 | 0.5349 | 0.0701 | 0.8419 | 0.035 | 0.085 | 0.011 | 0.020 |
| gemma-4-e4b | after A (urgency after queue) | 94.5% | 85.2% | 98.8% | 97.5% | 0.2375 | 0.5700 | 0.0702 | 0.8777 | 0.040 | 0.076 | 0.009 | 0.020 |
| gemma-4-e4b | after B (urgency, angry after queue) | 94.2% | 85.5% | 98.8% | 97.5% | 0.2379 | 0.5691 | 0.0673 | 0.8743 | 0.037 | 0.074 | 0.013 | 0.019 |
| gemma-4-e4b | debias K = 4 | 94.2% | 84.5% | 98.5% | 97.2% | 0.2428 | 0.5240 | 0.0704 | 0.8372 | 0.043 | 0.076 | 0.011 | 0.020 |

Exact sign tests against tree on per-ticket correctness, test and train pooled (400 tickets). Cells: tickets right tree / configuration (discordant pairs: only tree right - only the configuration right, two-sided p; Holm = Holm-adjusted over all 24 tests of this table):

| model | configuration | queue | urgency | angry |
|---|---|---|---|---|
| qwen3.5-9b | full-path | 379 / 379 (1-1, p 1, Holm 1) | 314 / 315 (2-3, p 1, Holm 1) | 393 / 393 (0-0, p 1, Holm 1) |
| qwen3.5-9b | after A (urgency after queue) | 379 / 379 (0-0, p 1, Holm 1) | 314 / 331 (8-25, p 0.0046, Holm 0.1) | 393 / 393 (0-0, p 1, Holm 1) |
| qwen3.5-9b | after B (urgency, angry after queue) | 379 / 379 (0-0, p 1, Holm 1) | 314 / 331 (7-24, p 0.0033, Holm 0.08) | 393 / 394 (0-1, p 1, Holm 1) |
| qwen3.5-9b | debias K = 4 | 379 / 381 (1-3, p 0.62, Holm 1) | 314 / 315 (1-2, p 1, Holm 1) | 393 / 394 (0-1, p 1, Holm 1) |
| gemma-4-e4b | full-path | 377 / 376 (1-0, p 1, Holm 1) | 336 / 337 (0-1, p 1, Holm 1) | 388 / 388 (0-0, p 1, Holm 1) |
| gemma-4-e4b | after A (urgency after queue) | 377 / 378 (0-1, p 1, Holm 1) | 336 / 341 (5-10, p 0.3, Holm 1) | 388 / 390 (0-2, p 0.5, Holm 1) |
| gemma-4-e4b | after B (urgency, angry after queue) | 377 / 377 (0-0, p 1, Holm 1) | 336 / 342 (4-10, p 0.18, Holm 1) | 388 / 390 (1-3, p 0.62, Holm 1) |
| gemma-4-e4b | debias K = 4 | 377 / 377 (2-2, p 1, Holm 1) | 336 / 338 (0-2, p 0.5, Holm 1) | 388 / 389 (0-1, p 1, Holm 1) |

Agreement with tree over the 400 tickets (`sp3_metrics.compare`, tree as the reference). Cells: median / max abs diff over options, argmax changes (of them where tree's top-2 margin > 0.15); decisions changed = all fields:

| model | configuration | queue | urgency | angry | decisions changed |
|---|---|---|---|---|---|
| qwen3.5-9b | full-path | 0.00016 / 0.17, 2 (0) | 0.0012 / 0.38, 5 (1) | 0.00022 / 0.075, 0 (0) | 7 of 1200 (1) |
| qwen3.5-9b | after A (urgency after queue) | 0.0001 / 0.051, 0 (0) | 0.0065 / 0.61, 38 (27) | 0.00021 / 0.06, 0 (0) | 38 of 1200 (27) |
| qwen3.5-9b | after B (urgency, angry after queue) | 8.4e-05 / 0.042, 0 (0) | 0.0069 / 0.62, 36 (26) | 0.0011 / 0.22, 1 (0) | 37 of 1200 (26) |
| qwen3.5-9b | debias K = 4 | 0.0006 / 0.37, 4 (1) | 0.0013 / 0.21, 3 (1) | 0.00028 / 0.043, 1 (0) | 8 of 1200 (2) |
| gemma-4-e4b | full-path | 3.4e-06 / 0.073, 1 (0) | 3.5e-05 / 0.062, 1 (0) | 4e-05 / 0.034, 0 (0) | 2 of 1200 (0) |
| gemma-4-e4b | after A (urgency after queue) | 2.7e-06 / 0.057, 1 (0) | 0.00026 / 0.62, 16 (8) | 3.8e-05 / 0.034, 2 (0) | 19 of 1200 (8) |
| gemma-4-e4b | after B (urgency, angry after queue) | 2.6e-06 / 0.046, 0 (0) | 0.00026 / 0.61, 15 (8) | 0.0001 / 0.31, 4 (1) | 19 of 1200 (9) |
| gemma-4-e4b | debias K = 4 | 8.5e-06 / 0.19, 4 (2) | 6.4e-05 / 0.11, 2 (0) | 4.3e-05 / 0.07, 1 (0) | 7 of 1200 (2) |

Distribution of coverage (full-path) per field, test and train pooled (400 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | queue | 0.987746 | 0.997279 | 0.999173 | 0.999757 | 0.999936 | 0.999985 | 0.999995 | 0.999331 |
| qwen3.5-9b | urgency | 0.999401 | 0.999572 | 0.999758 | 0.999875 | 0.999947 | 0.999970 | 0.999983 | 0.999836 |
| qwen3.5-9b | angry | 0.974898 | 0.992154 | 0.995527 | 0.996967 | 0.997946 | 0.998920 | 0.999390 | 0.996419 |
| gemma-4-e4b | queue | 0.979687 | 0.996910 | 0.999767 | 0.999953 | 0.999986 | 0.999996 | 0.999999 | 0.999421 |
| gemma-4-e4b | urgency | 0.999991 | 0.999997 | 0.999998 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999999 |
| gemma-4-e4b | angry | 0.994860 | 0.997513 | 0.999005 | 0.999713 | 0.999871 | 0.999919 | 0.999969 | 0.999285 |

Distribution of order_spread (debias K = 4) per field, test and train pooled (400 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | queue | 5.3e-07 | 5.807e-05 | 0.001796 | 0.01495 | 0.1165 | 0.4621 | 0.6779 | 0.09012 |
| qwen3.5-9b | urgency | 1.646e-05 | 0.0001629 | 0.005066 | 0.01973 | 0.06165 | 0.1546 | 0.2435 | 0.04263 |
| qwen3.5-9b | angry | 7.208e-06 | 2.396e-05 | 8.471e-05 | 0.001057 | 0.02142 | 0.05231 | 0.08328 | 0.01295 |
| gemma-4-e4b | queue | 2.269e-07 | 1.156e-06 | 1.65e-05 | 0.0003787 | 0.006411 | 0.3092 | 0.7559 | 0.03862 |
| gemma-4-e4b | urgency | 4.618e-06 | 4.248e-05 | 0.0005851 | 0.005202 | 0.03613 | 0.1397 | 0.2655 | 0.02859 |
| gemma-4-e4b | angry | 1.661e-06 | 3.733e-06 | 1.429e-05 | 0.0001217 | 0.001929 | 0.04746 | 0.1172 | 0.006731 |

Cost on the train set (64 requests of 5 states, `--decide-seqs 21`; tree = the regression run of sp3/integrity/, the same requests on this build). Per request medians in ms; snapshots = prefix cache entries and MB after the warm-up:

| model | configuration | p50 ms / ticket | tickets / s | vs tree | total | prefill | prefix | scoring | decodes | rounds | seqs per state | states per round | cache hits / misses | snapshots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | tree (baseline) | 93.0 | 10.7 | 1.00x | 462.9 | 302.1 | 0.1 | 159.5 | 2 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| qwen3.5-9b | full-path | 259.6 | 3.6 | 0.34x | 1295.9 | 336.4 | 0.1 | 958.8 | 10 | 5 | 11 | 1 | 0 / 0 | 0, 0.0 |
| qwen3.5-9b | after A (urgency after queue) | 146.6 | 7.1 | 0.66x | 730.5 | 449.5 | 0.1 | 278.8 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| qwen3.5-9b | after B (urgency, angry after queue) | 181.2 | 5.5 | 0.52x | 902.5 | 495.6 | 0.1 | 405.1 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| qwen3.5-9b | debias K = 4 | 494.8 | 2.0 | 0.19x | 2471.8 | 1360.4 | 22.9 | 1106.2 | 10 | 5 | 16 | 1 | 192 / 0 | 3, 181.5 |
| gemma-4-e4b | tree (baseline) | 29.8 | 32.6 | 1.00x | 146.8 | 94.5 | 0.2 | 49.6 | 2 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | full-path | 152.1 | 6.6 | 0.20x | 757.8 | 294.2 | 0.2 | 456.9 | 10 | 5 | 11 | 1 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | after A (urgency after queue) | 54.5 | 19.1 | 0.58x | 270.0 | 143.1 | 0.2 | 125.1 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | after B (urgency, angry after queue) | 69.4 | 15.2 | 0.47x | 344.6 | 161.0 | 0.2 | 182.8 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | debias K = 4 | 227.8 | 4.4 | 0.13x | 1136.9 | 753.8 | 7.5 | 371.0 | 10 | 5 | 16 | 1 | 192 / 0 | 3, 40.1 |

Findings:
- **Full-path** shows no significant accuracy difference from tree on this schema (pooled sign tests p = 1 on every
  field of both models). 7 of 1200 field decisions change on the 9B (queue 2, urgency 5; 1 of them where tree's top-2
  margin exceeds 0.15) and 2 of 1200 on Gemma (none over the margin); the summed NLL stays close (9B 0.839 → 0.830 test,
  0.820 → 0.818 train; Gemma 0.909 → 0.911, 0.827 → 0.825). Coverage is close to 1 (median ≥ 0.9969 on every field,
  minimum 0.9749): both models answer inside the option set, so renormalising over it changes little. The triage
  options share no token prefixes, so tree mode loses nothing here.
- **After** (urgency after queue) on Qwen3.5-9B: +17 of 400 urgency answers right, 314 → 331 (8-25; unadjusted
  p = 0.0046, Holm 0.10: not significant over the 24 sign tests of the table), 78.5% → 82.8% over the 400 tickets
  (78.8% → 85.0% on test, 78.4% → 82.2% on train). Config B, which makes both urgency and angry dependent on queue, also
  reaches 331 (7-24, unadjusted p = 0.0033, Holm 0.080). On Gemma the gain is smaller (336 → 341, 5-10, p = 0.30; B 342,
  p = 0.18). The urgency probabilities move differently per model: on the 9B the urgency NLL rises on the test set
  (config A 0.535 → 0.555, ECE 0.061 → 0.121; B 0.546) and falls on the train set (A 0.552 → 0.539, ECE 0.085 → 0.079;
  B 0.533); over the 400 tickets it falls, 0.5483 → 0.5420 (A) and 0.5357 (B), and so does the summed NLL, 0.8237 →
  0.8175 (A) and 0.8194 (B).
  On Gemma the urgency NLL rises on both sets (test 0.582 → 0.612, train 0.523 → 0.559; 400 tickets 0.5344 → 0.5700)
  while its ECE improves (0.110 → 0.081 test, 0.087 → 0.079 train), and the summed NLL rises (400 tickets 0.8430 →
  0.8777 A, 0.8743 B). Angry after queue (config B) adds nothing: 393 → 394 (9B), 390 in A and B (Gemma).
- **After shifts the level-0 fields slightly.** Queue (level 0 in A and B) and angry (level 0 in A) keep their prompts, but
  the phases change the composition of the branch decodes: queue median abs diff 1.0e-04 (9B) and 2.7e-06 (Gemma), max
  0.051 / 0.057; the 9B's queue decisions are unchanged while its test queue NLL moves 0.182 → 0.175; one Gemma queue
  decision flips (inside tree's margin; train queue 94.4% → 94.7%), and so do two Gemma angry decisions in config A (388 →
  390 right, max diff 0.034). This is the effect Task 6 measured with identical prompt tokens (queue median 0.0036 on the
  0.5B, 0.0003 on the 4B, `checks/task-6/notes.md`).
- **Debias K = 4** shows no significant accuracy difference (pooled sign tests p ≥ 0.62 on the 9B, e.g. queue 379 → 381,
  1-3; p ≥ 0.5 on Gemma); 8 of 1200 field decisions change on the 9B and 7 on Gemma (2 of them over tree's margin on
  each). It lowers the summed NLL on the 9B (0.839 → 0.809 test, 0.820 → 0.797 train, the lowest of all runs; urgency 0.535 → 0.508 and 0.552
  → 0.530) and slightly on Gemma (0.909 → 0.900, 0.827 → 0.822). `order_spread` shows that the queue answer depends on the
  option order for a minority of tickets: median 0.015 (9B) and 0.0004 (Gemma), p95 0.46 and 0.31, max 0.68 and 0.76.
  Urgency and angry keep their order in every variant, but the catalogue text around them changes, so they move too (p95
  0.15 / 0.14 and 0.05 / 0.05).
- **Cost.** Throughput relative to tree at `--decide-seqs 21`: full-path 0.34x (9B) and 0.20x (Gemma), after A 0.66x /
  0.58x, after B 0.52x / 0.47x, debias K = 4 0.19x / 0.13x. Full-path needs 11 sequences per state (1 trunk + one per
  option) and debias 16 (4 variants x (1 trunk + 3 branches)), so only one state fits a round and a b5 request takes 5
  rounds (10 decodes). Full-path scoring also reads 17 rows per state instead of 3, each with a full-vocabulary
  log-sum-exp in plain scalar code (Task 5 at 64 sequences, one round: Gemma scoring 213.5 vs 46.9 ms per 5 states, 0.76
  ms per row for its 262144-token vocabulary). `after` adds one branch decode per round (3 decodes instead of 2) and
  longer dependent branches (the answer lines). Debias restores its K − 1 = 3 extra prefixes from the prefix cache on
  every request (192 hits over 64 requests, prefix 22.9 ms on the 9B and 7.5 ms on Gemma); the three snapshots, taken
  at the warm-up, hold 181.5 MB of host RAM on the 9B (60.5 MB each: attention KV plus recurrent state) and 40.1 MB on
  Gemma (13.4 MB each); Task 4 measured 57.1 MB per snapshot on the hybrid Qwen3.5-4B. More `--decide-seqs` would fit
  more states per round at the price of VRAM (one recurrent state per sequence on hybrid models).

The prefix cache itself (Task 8, `integrity/notes.md` section 5, `integrity/cache-timing-*.json`: 20 alternations of two prompts, A = `keywords`,
B = `default`, 5 states each, `--decide-seqs 21`; medians per prompt over its 20 requests, 19 hits and 1 miss each with the
cache): Qwen3.5-9B total 1251.5 → 692.3 ms (`keywords`, prefix 564.8 → 7.8 ms) and 1000.6 → 690.4 ms (`default`, prefix
315.9 → 7.3 ms) per request with `--decide-prefix-cache` 0 → 8; Gemma 4 E4B 325.5 → 150.3 ms (prefix 171.9 → 2.6) and 251.5
→ 149.2 ms (102.1 → 1.7). A single schema keeps its prefix resident in slot 0, so the cache only acts when schemas,
prompts or debias variants alternate.

### Temperature calibration (spec 8)

`calibrate.py` fits one T per model on the stored probabilities of the Task 0 baseline train run (320 tickets, tree b5,
`keywords`) by minimising the summed NLL of queue, urgency and angry (`sp3_metrics.nll`, the engine's scaling applied
post hoc): 64 log-spaced grid points over [0.25, 4], then golden-section refinement; T is kept to 4 decimals and every
"after" number uses that value. Rule, fixed before looking at the test set: T goes into the model's preset as
`decide-temperature` only if it lowers the test-set summed NLL. Files: `results-engine/sp3/eval/calibration-<model>.json`.

| model | T | train NLL Σ before → after | test NLL Σ before → after | test ECE queue / urgency / angry before → after | test Brier angry | rule |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 1.1187 | 0.820 → 0.814 | 0.839 → 0.845 | 0.061 / 0.061 / 0.063 → 0.066 / 0.043 / 0.071 | 0.033 → 0.036 | not applied (test NLL rises) |
| Gemma 4 E4B | 1.6947 | 0.827 → 0.699 | 0.909 → 0.780 | 0.053 / 0.110 / 0.031 → 0.027 / 0.038 / 0.041 | 0.031 → 0.035 | applied: `decide-temperature = 1.6947` in `[gemma-4-e4b]` |

Gemma's single temperature trades fields. On the test set queue and urgency improve (NLL 0.197 → 0.183 and 0.582 → 0.464,
ECE 0.053 → 0.027 and 0.110 → 0.038), angry gets slightly worse (NLL 0.131 → 0.132, ECE 0.031 → 0.041, Brier 0.031 →
0.035): angry was already the best-calibrated field (train ECE 0.011), and T = 1.69 flattens it together with the
over-confident urgency. The rule looks at the summed NLL (−0.129), so it applies; per-field temperatures are out of scope
(spec 1). Accuracy does not change (T keeps every argmax). Qwen3.5-9B's T helps urgency (test ECE 0.061 → 0.043) but raises
the summed test NLL, so its preset stays at 1.0.

The rule's change is the last step of the evaluation: `decide-temperature = 1.6947` in `[gemma-4-e4b]` of
`models-engine.ini`. Every run above, and the spec 12.2 regression of Task 8, ran before it with T = 1. Check after the
change (`eval/run.sh preset`, `eval/preset/`, files `decide-gemma-4-e4b-b5-keywords-t1.6947.*`): the Gemma test run
without switches on a fresh router reports
`engine.temperature` 1.6947 on every request, its answers equal the baseline test answers scaled post hoc to T = 1.6947
within 2.0e-15 (`sp3_metrics.py compare --max-diff 1e-9`), and its metrics are the "after" row above (NLL Σ 0.780, ECE
0.027 / 0.038 / 0.041, Brier 0.035, accuracy unchanged, p50 30.8 ms). Since then a Gemma request without
`options.temperature`, including every `/v1/systemone` request to that preset, is answered at T = 1.6947; to reproduce the
T = 1 numbers send `options.temperature: 1` (`decide_client.py --ensure-t1` or `--temperature 1`) or remove the line
(the steps concerned: "How to reproduce").

### Recommendations

| switch | recommendation | because |
|---|---|---|
| full-path scoring | keep off (tree stays the default); use it when option keys share token prefixes (a 400 in tree mode, e.g. `sale`/`sales`) or when `coverage` is wanted | no significant accuracy difference from tree here (sign tests p = 1; 7 (9B) and 2 (Gemma) of 1200 field decisions change; summed NLL within 0.01) at 0.34x (9B) and 0.20x (Gemma) throughput |
| sequential fields (`after`) | use when one answer should inform another and a gain is observed on labelled tickets, e.g. urgency after queue on Qwen3.5-9B; keep off by default | 9B urgency +17 of 400 (unadjusted p = 0.0046, Holm 0.10: not significant over the 24 sign tests; 78.5% → 82.8%) at 0.66x throughput; Gemma +5 (p = 0.30); urgency NLL over the 400 tickets: 9B 0.548 → 0.542 (higher on test, lower on train), Gemma 0.534 → 0.570 (higher on both sets, with lower ECE), so re-fit a temperature for that mode when calibrated probabilities matter; angry after queue gains nothing |
| prefix cache | default on (`--decide-prefix-cache 8`, as shipped) | a prefix of another schema, prompt or debias variant is restored instead of rebuilt; per request with two alternating prompts, cache 0 → 8: 9B 1251.5 → 692.3 ms (`keywords`) and 1000.6 → 690.4 ms (`default`), Gemma 325.5 → 150.3 and 251.5 → 149.2 ms; host RAM per snapshot 60.5 MB (9B), 13.4 MB (Gemma); needed by debias |
| option-order debiasing | keep off; use K = the largest choice option count as a diagnostic (`order_spread`) or where a small NLL gain is worth the cost | no significant accuracy difference (p ≥ 0.5; 8 (9B) and 7 (Gemma) of 1200 field decisions change); 9B summed NLL −0.030 test / −0.023 train, Gemma −0.009 / −0.005; 0.19x (9B) and 0.13x (Gemma) throughput |
| temperature | per model by the calibration rule: `decide-temperature = 1.6947` for Gemma 4 E4B, none for Qwen3.5-9B; requests can override it | Gemma test summed NLL 0.909 → 0.780 (queue and urgency better, angry slightly worse); Qwen's T raised its test NLL; no runtime cost (p50 30.8 vs 31.4 ms) |

## Follow-ups (2026-10-02)

Engine patches 0033-0038 on top of sub-project 3 (branch `decide`
at `49785a95d`, tree `32d63de3ad1b…`). Testing moved to a second machine with an RTX 3090 (section "The test machine"
below). Outputs: `results-engine/followups/task-0..3/`. The engine checks compare the sub-project 3 build (`63ea2c51a`,
built on the test machine as `engine-63ea2c51a/`) with the new build on the same GPU.

### Faster full-path rows (0033, 0036)

Each full-path row needs the log-sum-exp over the whole vocabulary. Before this change that was a scalar loop over
float `exp`. `full_vocab_lse` now sums with an SSE2 or an AVX2+FMA path chosen at run time (`__builtin_cpu_supports`;
GCC or Clang on x86-64). Both paths use a float exp and add every term in double. The old loop stays as the scalar
fallback, and the build flags are unchanged. 0036 turns the dispatch off for MSVC targets and adds tests for short rows.

From `results-engine/followups/task-1/notes.md`, measured on the test machine's Ryzen 9 3950X:
- **Unit test, accuracy.** Max abs error against an all-double reference: scalar 7.44e-09, SSE2 3.47e-09, AVX2
  3.40e-09 (bound 2e-7).
- **Unit test, speed.** A 262144-token row, median of 41: scalar 0.753 ms, SSE2 0.335 ms (2.25x), AVX2 0.163 ms
  (4.62x). Over five runs the AVX2 speedup was 4.54-4.62x.
- **Edge cases pinned by the tests.** NaN or +inf anywhere gives NaN, a row of only −inf gives NaN, and one finite entry
  among −inf gives exactly that entry.

Live comparison, `llama-decide` old vs new, full-path b5, `--decide-seqs 64` (all 5 states in one round), 4 request
lines of 5 states, median of 5 repetitions (`qwen2.5-0.5b.json`, `gemma-4-e4b.json`):

| | Qwen2.5-0.5B (151936 tokens) | Gemma 4 E4B (262144 tokens) |
|---|---|---|
| answers new vs old, max abs diff (bound 1e-6) | 2.1e-8 | 6.0e-8 |
| scoring ms per request, old → new | 57.8-61.3 → 30.0-33.1 (1.85-1.93x) | 131.7-136.5 → 85.2-87.2 (1.54-1.57x) |
| saved per full-path row (85 rows per request) | 0.33 ms | 0.55-0.58 ms |
| total ms per request, old → new (line 1) | 71.3 → 44.6 | 174.9 → 128.1 |
| tree b5: results max abs diff / token dump | 0 / identical | 0 / identical |

The full-path throughput figures of sub-project 3 (0.34x / 0.20x on the laptop at `--decide-seqs 21`) were measured
before this change, and nothing was re-measured on the laptop. On the 3090, Qwen3.8-27B's full-path runs reach 0.54x of
tree at `--decide-seqs 64` (next section).

### Minor fixes (0034)

From `results-engine/followups/task-2/notes.md`:
- **Coverage clamp.** A coverage in (1, 1 + 1e-6] is reported as exactly 1, both for a single answer and for the debias
  mean. Larger values stay. Unit tests: 1 + 3.1e-7 → 1.0; 1 + 4.5e-5 stays.
- **Cache decision before the copy.** `plan_cache_store` decides whether a new prefix snapshot is stored, and which
  entries it evicts, before anything is copied off the GPU.
  - If the snapshot would not be stored, nothing is erased or copied (only wanted entries left over a limit, or a
    snapshot alone over 2 GiB).
  - Otherwise the victims are erased first and the snapshot is copied after, so host memory stays within the limits.
  - Unit tests cover: capacity 0, an already-cached hash, the sub-project 3 A,B,C,A trim, wanted entries, one to three
    byte victims, and both limits together.
  - The path where the copy fails after victims were erased (the entry is not stored, the victims stay erased) was
    checked by reading the code only.
  - Live, old vs new build: identical (hits, misses, entries) per request on Qwen2.5-0.5B and Qwen3.5-4B for A/B/A at
    cache 8 and A,B,C,A at cache 2. Answers max abs diff 0 and `cached_tokens` equal.
- **One cell formula.** `need_for` now calls `state_cells`. The tree b5 dumps (20 tickets) are identical old vs new,
  including rounds and `states_per_round`.
- **Info slots.** `/v1/decide/info` lists prefix slots by `seq_pos_min`, so cells left in only one memory part of a
  hybrid model would show. After A/B/A the info bodies are equal old vs new on the 0.5B (dense) and the 4B (hybrid).
  No engine path leaves such cells, so this shows only that nothing regressed.

### Fault modes for the remaining 500 paths (0035, 0037, 0038)

`LLAMA_DECIDE_FAULT` gains three one-shot modes (spec §9; `llama-decide --help`):
- `cell-estimate`: the first exact cell check before a dependent phase reports "cell estimate exceeded";
- `lines-join`: the first context join check of a dependent field reports "answer lines change the tokenisation";
- `tail-mismatch`: the first tail comparison of a catalogue variant v ≥ 1 fails.

These are the three engine errors that sub-project 3 could only unit-test or not test at all. The live check
(`fault-<mode>.json`) used Qwen2.5-0.5B on `llama-server` at `--decide-seqs 64`. Server F (fault armed) and server N (not
armed) each received the same sequence:
- R0: 5 plain states;
- R1: R0 with urgency after queue (`cell-estimate`, `lines-join`), or R0 with `order_debias` 2 (`tail-mismatch`);
- a GET of `/v1/decide/info`;
- R2: R0's body again.

| mode | F: R1 | F: info after R1 | F: R2 | N: R0, R1, R2 | R2 F vs N |
|---|---|---|---|---|---|
| cell-estimate | 500 `engine`, `cell estimate exceeded: field "urgency" needs 13 branch cells per state with its answer lines, 15 are reserved` | slot 0 only, valid | 200 | 200 × 3 | max abs diff 0 |
| lines-join | 500 `engine`, `answer lines change the tokenisation of field "urgency"; remove its "after" (injected by LLAMA_DECIDE_FAULT=lines-join)` | slot 0 only, valid | 200 | 200 × 3 | 0 |
| tail-mismatch | 500 `engine`, `catalogue variant 1 renders another chat template tail` | slot 0 only, valid | 200 | 200 × 3 | 0 |

The first two faults fire after the round's trunk and phase-0 decodes, so their 500 goes through the per-round cleanup.
The info afterwards lists slot 0 only. The cell-estimate message carries the real counts (the fault forces the branch,
not the numbers). Since 0037 that message ends in `(injected by LLAMA_DECIDE_FAULT=cell-estimate)`; the run above used
the build before 0037 (`23957f14c`). 0037 changes only that text. 0038 (a follow-up fix) does the same for the tail-mismatch
message, which now ends in `(injected by LLAMA_DECIDE_FAULT=tail-mismatch)`; a real tail mismatch keeps the bare text.
0038 also routes the older `restore` fault through the same one-shot helper (same behaviour).

### Portability (Task 3)

New environment variables. Unset, every default is the previous behaviour; empty gives the same, except for
`CUDA_BIN`, `CUDA_ARCH` and `CUDA_HOST_COMPILER`, where empty drops the setting. A re-run of
`eval/run.sh score calibration`, `analysis.py summary`, `integrity/checks.py run-params` and `notes` in a scratch copy
gave byte-identical outputs to the committed files. Relative paths are taken from the project directory.

| variable | default | read by |
|---|---|---|
| `TRIAGE_DATA` | `data` (the bundled set; until the repository bundled it, the author's copy of the same files) | `score.TEST_SET`, `decide_client.TEST_SET`, `heldout_stats.TRAIN_SET` (and every script that imports them); the run scripts' `DATA` default (`DATA` still wins), exported to their Python steps |
| `ENGINE_PRESETS` | `models-engine.ini` | `serve-engine.sh`, `decide_client._preset_settings` (the run JSON's `preset_settings`, `model_file`, `decide_seqs`), `reference_check._engine_settings` (`decide_command`, `cache_timing.py`, `checks.py debias-run`) |
| `ENGINE_BIN` | `engine/build/bin` | `serve-engine.sh` and `reference_check.py` (e.g. `engine-63ea2c51a/build/bin`) |
| `MODELS`, `EVAL_DIR`, `BASELINE_DIR`, `COST_DIR` | the two sp3 presets; `results-engine/sp3/eval`, `…/baseline`, `…/integrity/regression` (`COST_DIR` follows `BASELINE_DIR` when only that is set) | `results-engine/sp3/eval/run.sh` and `analysis.py` (new `baseline` step, refused with the default `BASELINE_DIR` or `EVAL_DIR`; per-model `preset` step that prints the line to add and never edits a presets file; summary sentences derived from the runs) |
| `GPU_VRAM_MAX`, `GPU_WAIT_MIN` | 200 MiB, 0 min | `gpu_ready` in both sp3 run scripts (re-check once a minute for up to `GPU_WAIT_MIN` minutes); `router_stop` stops only the router the script started |
| `CUDA_BIN`, `CUDA_ARCH`, `CUDA_HOST_COMPILER` | `/opt/cuda/bin`, `86`, `/usr/bin/g++-15` | `build-engine.sh` (an empty value drops the setting) |

`reference_check.decide_command` also passes the placement keys `n-cpu-moe`, `tensor-split`, `override-tensor`,
`main-gpu`, `split-mode` and `device` from a preset to `llama-decide`.

Smoke on the test machine (`results-engine/followups/task-3/notes.md`). Settings: Gemma 4 E4B, the 3090 presets,
10-ticket scratch test and train sets, and scratch `EVAL_DIR` / `BASELINE_DIR`. The command was
`results-engine/sp3/eval/run.sh baseline runs score calibration preset`. Result:
- exit 0 after 64 s;
- nothing written inside the repository (`git status` unchanged; no repository path newer than the start);
- every run JSON records the 3090 presets' settings, and the summary derives "10 tickets" and the scratch cost
  directory.

### The test machine

The test machine (`user@gpu-host`) has an RTX 3090 24 GB, a Ryzen 9 3950X (32 threads), 62 GB RAM, CUDA 13.3 and `g++-15`. The
desktop holds about 612 MiB of VRAM. The laptop stays the git authority: edit and commit there, then sync to
the repository copy there (`<repo>`), build and run there, copy results back, and commit on the laptop.

`remote-sync.sh` (no git remote involved):
- **`push [--take-over] [--force] [project|engine|all]`.**
  - Fetches a `git bundle` of the commits the test machine lacks and checks out the same commit id there.
  - Then copies this side's tracked-modified and untracked-unignored files with `tar` over ssh.
  - It stops and lists the files there that differ from every version here (a run's rewrite, or a foreign untracked
    file); `--force` prints the same list and then replaces them. Commits made there are discarded (reflog only).
- **`pull PATH...`** copies results back to the same relative paths.
- **`run CMD...`** runs a command in the remote directory.
- **Safeguards.** It refuses foreign targets: the remote directory must be absent or empty, or a repository this
  script pushes to. Every command prints its target and source, and each remote repository records the checkout that
  pushes to it; a push from another checkout needs `--take-over`, which still stops on files that would be lost.

Results from the two GPUs are compared under the statistic, never for equality. Task 0 found both small models pass the
statistic between the 3090 runs and the laptop's committed baseline (`results-engine/followups/task-0/notes.md`):

| model | median abs diff q / u / a | max abs diff | argmax disagreements |
|---|---|---|---|
| Qwen3.5-9B | 0.00047 / 0.00155 / 0.00046 | 0.130 | 2 (none over margin) |
| Gemma 4 E4B | 0.00001 / 0.00004 / 0.00005 | 0.087 | 2 (none over margin) |

The same runs give accuracy within one ticket of the laptop's. The 3090's p50 per ticket is 35.3 ms (9B) and 13.9 ms
(Gemma), against 92.7 and 31.4 ms on the laptop.

## Qwen3.8-27B on the RTX 3090

Every number in this section comes from `results-engine/3090/notes.md` and the files it names (written by
`results-engine/3090/notes.py`), and from `results-engine/followups/task-0/notes.md` for the model facts. The runs used
engine `23957f14c`, which is 0001-0036. 0037 and 0038 change only fault messages, a header comment and how the
`restore` fault disarms (same behaviour), so the results hold for the current tree.

### Model and preset

- **Model and preset.** Preset `qwen3.8-27b` of `models-engine-3090.ini`:
  - model file `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB, IQ3_S);
  - `ngl 999`, `fa on`, KV q8_0, ctx 4096, `decide-seqs 64`;
  - no mmproj and no speculative decoding. The MTP layer (`blk.64.*`, about 0.35 GB) is reported unused and ignored.
- **Hybrid model.** `/v1/decide/info` reports `recurrent: true`.
- **VRAM.**
  - The model idles at 15412 MiB with `--decide-seqs 21` and at 22114 MiB with 64 (whole GPU, desktop included), so
    each decide sequence holds about 156 MiB of recurrent state.
  - After a run, 22316-22332 MiB of 24 GB are in use.
  - 64 sequences fit a full-path batch of 5 (55 sequences) in one round. No run hit a budget or memory error.
- **Prompt and data.** The thinking-off assistant prefix is the Qwen3.5 tail (`EXPECTED_PREFIX` entry). The test set is
  80 tickets and the train set 320, prompt `keywords`, b5, every run at T = 1 except the preset check.

### Scores, beside the laptop's small models

Tree mode, b5, prompt `keywords`, T = 1. The 27B was measured on the RTX 3090 at `--decide-seqs 64`; Qwen3.5-9B
(Q4_K_M) and Gemma 4 E4B (Q4_0) on the RTX 3070 Ti laptop at 21. That makes this a comparison across GPU, model size
and quantisation, not of the engine. Sources: `results-engine/3090/eval/summary.md` and `results-engine/sp3/eval/summary.md`.

| set | model (GPU) | queue | urgency | urg±1 | angry | NLL Σ | p50 ms / ticket |
|---|---|---|---|---|---|---|---|
| test (80) | Qwen3.8-27B IQ3_S (3090) | 98.8% | 88.8% | 98.8% | 100.0% | 0.531 | 101.3 |
| test (80) | Qwen3.5-9B Q4_K_M (3070 Ti) | 95.0% | 78.8% | 95.0% | 97.5% | 0.839 | 92.7 |
| test (80) | Gemma 4 E4B Q4_0 (3070 Ti) | 93.8% | 82.5% | 98.8% | 95.0% | 0.909 | 31.4 |
| train (320) | Qwen3.8-27B | 96.6% | 85.3% | 99.7% | 100.0% | 0.484 | 101.4 |
| train (320) | Qwen3.5-9B | 94.7% | 78.4% | 97.8% | 98.4% | 0.820 | 94.8 |
| train (320) | Gemma 4 E4B | 94.4% | 84.4% | 98.4% | 97.5% | 0.827 | 29.7 |
| pooled (400) | Qwen3.8-27B | 97.0% | 86.0% | 99.5% | 100.0% | 0.4933 | — |
| pooled (400) | Qwen3.5-9B | 94.8% | 78.5% | 97.2% | 98.2% | 0.8237 | — |
| pooled (400) | Gemma 4 E4B | 94.2% | 84.0% | 98.5% | 97.0% | 0.8430 | — |

### Switches

Switch runs against the 27B's own tree baseline. The numbers are pooled over 400 tickets (`eval/summary.md`). Sign
tests are exact and two-sided; Holm is over this model's 12 tests. Cells give tickets right, tree → switch, then the
discordant pairs, p and Holm.

| configuration | queue | urgency | angry | decisions changed (of 1200; over margin) | summed NLL (400) | throughput vs tree |
|---|---|---|---|---|---|---|
| tree | 388 | 344 | 400 | — | 0.4933 | 1.00x (9.9 tickets/s) |
| full-path | 388 → 389 (0-1, p 1, Holm 1) | 344 → 345 (0-1, p 1, Holm 1) | 400 → 400 | 2 (0) | 0.4915 | 0.54x |
| after A (urgency after queue) | 388 → 389 (0-1, p 1) | 344 → 348 (8-12, p 0.50, Holm 1) | 400 → 400 | 21 (10) | 0.4790 | 0.96x |
| after B (urgency, angry after queue) | 388 → 389 (0-1, p 1) | 344 → 350 (7-13, p 0.26, Holm 1) | 400 → 396 (4-0, p 0.12, Holm 1) | 25 (13) | 0.4760 | 0.82x |
| debias K = 4 | 388 → 389 (0-1, p 1) | 344 → 342 (2-0, p 0.50, Holm 1) | 400 → 400 | 3 (0) | 0.4956 | 0.33x |

No sign test is significant: the smallest unadjusted p is 0.12, and every Holm-adjusted p is 1.

Per set (`eval/summary.md`, test and train tables), both `after` runs are worse on the test set and better on train:
urgency on test 71 → 69 of 80 for both (88.8% → 86.2%), summed NLL 0.531 → 0.560 (A) / 0.556 (B); on train urgency
273 → 279 / 281 of 320, summed NLL 0.484 → 0.459 / 0.456. The pooled gains above come from the train set.

**Coverage (full-path, 400 tickets), medians:** queue 0.999808, urgency 0.999701, angry 0.996858 (minimum 0.987).

**`order_spread` (K = 4):**

| field | median | p95 | max |
|---|---|---|---|
| queue | 0.0023 | 0.126 | 0.285 |
| urgency | 0.0093 | 0.071 | 0.201 |
| angry | 0.0018 | 0.017 | 0.115 |

**Cost per request on the train set** (64 requests of 5 states, medians, `--decide-seqs 64`):

| mode | decodes | rounds | seqs per state | states per round |
|---|---|---|---|---|
| tree | 2 | 1 | 4 | 15 |
| full-path | 3 | 1 | 11 | 5 |
| after A / B | 3 | 1 | 4 | 15 |
| debias K = 4 | 4 | 2 | 16 | 3 |

Debias restores 3 prefixes per request from the cache (192 hits, prefix 50.7 ms). Its 3 snapshots hold 517.5 MB of
host RAM, 172.5 MB each.

### Calibration

`eval/calibration-qwen3.8-27b.json`: T = 0.8992, fitted on the train baseline.

| test set | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a |
|---|---|---|---|---|---|---|---|---|
| T = 1 | 0.0716 | 0.3817 | 0.0778 | 0.5312 | 0.0237 | 0.0843 | 0.0671 | 0.0165 |
| T = 0.8992 | 0.0690 | 0.3898 | 0.0685 | 0.5273 | 0.0246 | 0.0742 | 0.0590 | 0.0144 |

The rule (lower test summed NLL) applies, so `decide-temperature = 0.8992` is in `[qwen3.8-27b]` of
`models-engine-3090.ini`, added after every other run. The gain comes from angry (NLL and ECE) and queue NLL. Urgency
NLL gets worse (0.382 → 0.390) while its ECE improves, and queue ECE is slightly worse. The live preset check matches the
baseline scaled post hoc with max abs diff 2.2e-16 (`eval/preset/`).

### Integrity

Reference checks used 20 test tickets, one state per line, `--decide-seqs 64`. `llama-decide` produced the dump with
the server stopped; the replay went through `/completion`. The replays are kept and recompute byte for byte offline
(`integrity/offline-recompute.txt`).

| check | median abs diff q / u / a | max abs diff | disagreements over margin | verdict |
|---|---|---|---|---|
| reference, tree | 3.6e-05 / 0.0007 / 0.0009 | 0.022 | 0 | PASS |
| reference, full-path | 2.3e-05 / 0.0008 / 0.0010 | 0.022 | 0 | PASS |
| batch invariance, b1 vs b5 (first 20 test tickets) | 4.3e-05 / 0.0007 / 0.0014 | 0.021 | 0 | PASS |

Full-path detail:
- **Coverage.** Engine coverage medians are 0.99975 / 0.99936 / 0.99852; coverage median abs diff is at most 0.0002
  (PASS).
- **Node comparison.** Medians 2.8e-05 / 0.0002 / 0.0001 (PASS).
- **Unverifiable edges.** 0 of 280.

### /v1/systemone

`systemone_smoke.py --model qwen3.8-27b` passes: identical probabilities through both endpoints (max abs diff 0), the
request id header, and the depth-40 422. The official SDK test with `SYSTEMONE_MODEL=qwen3.8-27b` gives 2 passed
(`systemone/`).

### Latency

Tree mode, test set (80 tickets). The two p50 definitions over the same per-ticket values (a request's wall time divided
by its tickets) differ:
- the score tables' `p50 ms` is the median (`statistics.median`; 101.3 ms at b5);
- the run JSON's `latency_s_per_ticket.p50` is the upper middle value (102.9 ms at b5).

| batch | p50, median (score table) | p50, run JSON | p90, run JSON | server p50 | VRAM after run |
|---|---|---|---|---|---|
| 1 (`latency/`) | 141.4 ms | 142.7 ms | 155.7 ms | 141.1 ms | 22332 MiB |
| 5 (`baseline/test/`) | 101.3 ms | 102.9 ms | 108.1 ms | 102.5 ms | 22316 MiB |

At b1 the 141.4 ms median is computed from `latency/preds/` the same way as the score tables. It is not printed in
`notes.md`, which gives the run JSON values.

### Recommendations for this model

| switch | recommendation | because (pooled, 400 tickets) |
|---|---|---|
| full-path scoring | off unless option keys share token prefixes | 2 of 1200 decisions change, sign tests p = 1, summed NLL 0.4933 → 0.4915, at 0.54x |
| `after` | off by default; re-check on labelled tickets before using | urgency +4 (A) and +6 (B) of 400, p 0.50 / 0.26, not significant; after B loses 4 angry tickets (p 0.12); summed NLL 0.479 / 0.476 vs 0.493; worse on test (urgency 88.8% → 86.2%, summed NLL 0.531 → 0.560 / 0.556), better on train; 0.96x / 0.82x |
| prefix cache | on (default 8) | debias reuses 3 restored prefixes per request (192 hits); 172.5 MB host RAM per snapshot |
| order debiasing | off | 3 of 1200 decisions change, summed NLL 0.4956 vs 0.4933, at 0.33x |
| temperature | the preset's T = 0.8992 (rule applied) | test summed NLL 0.5312 → 0.5273; angry better, urgency NLL worse |

## Image input (sub-project 4)

A `/v1/decide` state can now be an ordered list of text parts and JPEG or PNG images (base64 data URIs), answered in the
same single pass as a text state; a server without a projector behaves as before and answers image requests with 400
`vision_unavailable`. Every binding check passed: old and new builds give identical text answers (max abs diff 0) with
and without the projector, the no-vision checks pass on Qwen2.5-0.5B and on Qwen3.8-27B without `--mmproj`, and every
integrity check of spec 9.4 passes on Qwen3.8-27B. After the rev 6 fixes (spec rev 6: images at most 200:1, an early
budget check), the text regression, the server smoke and every integrity check of the binding model were run again on
the final engine and pass (section "Final engine"). The non-causal gate of spec 9.5 failed on gemma-3-4b, so projectors
with non-causal image attention report `vision.available = false` (`non_causal_projector`): Gemma-3-style vision is not
supported.

Every number in this section comes from `results-engine/image/notes.md` (written by `results-engine/image/notes.py`
from the files under `results-engine/image/`) and the files it names; the tables are copied from it. Design: spec
`docs/specs/2026-10-02-image-input-design.md` (rev 6; rev 7 moves the sequence-budget check, section "Sequence budget first (0045)").
Every run was on the RTX 3090 test machine.

### Setup

- **Engine.** Patches 0039-0044 (subjects in "Setup" above):
  - 0039 parses and validates image parts and reports the vision status;
  - 0040 decodes the pixels (stb_image), builds trunks with libmtmd, counts image cells in the budget and takes the
    held cells from the server's slots;
  - 0041 gives `llama-decide` `--mmproj`, the image-token limits and the dump of image states;
  - 0042 switches non-causal projectors off after the failed gate;
  - 0043 bounds the cell budget by the sliding-window cache (Task 6b);
  - 0044 (spec rev 6): an image's longer side may be at most 200 times its
    shorter side, the cells of a state are checked right after each of its images is tokenised, the engine rejects an
    image part itself when vision is not available, and the image SHA-256 is computed only for a dump.
- **Which engine each result comes from.** The live results of Task 1 (`task-1/`) ran on `0d192078a` (0001-0039),
  of Task 2 (`task-2/`) on `b36eccdec` (0001-0040) and of Task 3 (`task-3/`) on `9bae50c77` (0001-0041); the later
  checks below supersede them (the server smoke was repeated on the final engine). Tasks 5 and 6 (regression, no
  vision, integrity, gate) ran on 0001-0041 (`9bae50c77`); `gate-off` ran on 0001-0042. 0042 changes only the
  availability of non-causal projectors. 0043 changes the cell limit only on models with sliding-window layers: the
  27B regression was repeated on `28a924b55` (0001-0043) with max abs diff 0, and gemma-4-e4b-vision's held pair was
  re-run on it. The SWA checks (`swa/`, Task 6b) and the COCO evaluation (Task 7) ran on `28a924b55`. The rev 6 engine is
  `69756d495` (0001-0044): on it the 27B text regression, the server smoke, the checks of spec rev 6 and every
  integrity check of the binding model were run again (`final/`, section "Final engine" below). The final engine is
  `185e9e0a7` (0001-0045), which only moves the sequence-budget check: on it the 27B text regression, the server smoke
  and a double-fault check were run (`publish/`, section "Sequence budget first (0045)" below); the other results were
  not re-run. The
  old build for every old-vs-new check is
  `engine-49785a95d/` on the test machine (the head before this work, tree `32d63de3ad1b00c086fb4abd2b9da53631d8d043`,
  built with `llama-server` and `llama-decide-cli`, same flags as `build-engine.sh`).
- **Models** (spec 8):

  | model | preset | role | ran |
  |---|---|---|---|
  | Qwen3.8-27B IQ3_S + BF16 projector | `qwen3.8-27b-vision` | main: M-RoPE positions, hybrid (recurrent) model; binding for the statistic | yes |
  | SmolVLM-500M-Instruct Q8_0 + Q8_0 projector | `smolvlm-500m` | sequential positions, several image chunks per image; non-binding | no: skipped. The engine's tokenisation-join check rejects its tokenizer, so the engine refuses the model (`integrity/smolvlm-500m/SKIPPED.txt`; its words "does not load" come from Task 0 and mean this refusal) |
  | Gemma 4 E4B Q4_0 + BF16 projector (`gemma4v`) | `gemma-4-e4b-vision` | added during execution as the substitute for SmolVLM: sequential positions, causal attention, one image chunk per image; non-binding | yes |
  | gemma-3-4b-it Q4_K_M + f16 projector | `gemma-3-4b-vision` | non-causal image attention; decides the gate | yes; the gate failed |
  | Qwen3.6-35B-A3B UD-IQ4_XS + F32 projector | `qwen3.6-35b-a3b-vision` | MoE with a projector; non-binding | yes, fully on the GPU (no placement keys) |
  | Qwen2.5-0.5B Q8_0 (no projector); Qwen3.8-27B without `--mmproj` | `qwen2.5-0.5b`, `qwen3.8-27b` | no-vision checks (spec 9.3) | yes |

- **Presets** (`models-engine-3090.ini`; `[*]`: q8_0 KV, fa on, jinja, reasoning off, ngl 99, parallel 1). Every
  vision preset sets `ctx-size = 16384` and `decide-seqs = 16`; `qwen3.8-27b-vision` and `qwen3.6-35b-a3b-vision` also
  `ngl = 999`, and `qwen3.8-27b-vision` `image-min-tokens = 1024` (the spike's setting: a COCO image takes 1026-1090
  cells but 34-49 positions). All six reference tags fit at `decide-seqs 16` on every preset that loads
  (`task-0/fit-*.json`). Task 6b added the text presets `gemma-4-e4b-16k` (ctx 16384, `decide-seqs 16`) and
  `gemma-4-e4b-16k-swa-full` (the same with `swa-full = 1`). The existing presets are unchanged.
- **Integrity states** (`states-integrity.jsonl`): 8 states over four COCO JPEGs and a PNG copy of one of them: four
  image-only states, text before an image, text after an image, two images with text between them, and the PNG with text
  after it. Fields (`schema-multi.json`): bool `cat`, `dog`, `indoors` and a 5-option choice `animal`; `tree-one` asks
  `cat` only. The statistic is the text one (per field: median abs diff at most 0.01, no argmax disagreement where the
  reference's top-2 margin is over 0.15).
- **COCO subset** (spec 8): 200 images of COCO val2017 chosen with seed 20261002, 20 positives and 20 negatives for each
  of person, car, bicycle, cat and dog, every image used once, split 100/100 into calibration and test; 972 of the 1000
  image-field pairs have a label (485 calibration, 487 test). The repository holds the list, the checksums and
  `results-engine/image/coco/NOTICE.md` (COCO Consortium, annotations CC BY 4.0; each image keeps its Flickr licence);
  the images are downloaded into the git-ignored `local/`. `results-engine/image/NOTICE.md` credits the four integrity
  images (ids, URLs, licence ids) and points to the subset's notice.

### Text regression and no vision

Old build `engine-49785a95d` (before this work) against the new build, same preset, same request sequence on a fresh router each.

| preset | run | requests | max abs diff | differences | result |
|---|---|---|---|---|---|
| qwen3.8-27b | Task 5 | 88 | 0.0 | 0 | pass |
| qwen3.8-27b-vision | Task 5 | 88 | 0.0 | 0 | pass |
| qwen3.8-27b | Task 6b, new build `28a924b55` | 88 | 0.0 | 0 | pass |

Existing integrity scripts on the new build (qwen3.8-27b, 20 tickets): tree reference pass, full-path reference pass, batch invariance pass, offline recompute `tree identical fp identical`.

The new build answers text requests exactly as the old one: the triage requests with each switch of sub-project 3
(debiasing at K = 2), max abs diff 0 over every key except `timings`, with and without the projector loaded, and again
on the final engine. The existing reference, batch-invariance and offline-recompute scripts pass on the new build.

| preset | info, both 400s, log, serving | text answers equal the run without the rejected requests | CLI 400 |
|---|---|---|---|
| qwen2.5-0.5b | pass | pass (max abs diff 0.0) | pass |
| qwen3.8-27b | pass | pass (max abs diff 0.0) | pass |

Without a projector, info reports `no_projector`; an image request and a mixed request get 400 `vision_unavailable`
(`field` `states[0]` and `states[1]`); info is unchanged after them; the text requests that follow give the same
probabilities and prefix-cache counts as a run without the rejected requests; `llama-decide` without `--mmproj` exits 1
with the same 400. The log check is weak: the instance prints nothing for a 400 at default verbosity, so it shows that
no error line was printed, not more.

### Integrity of image states

Reference cells: result (worst per-field median abs diff; argmax disagreements over the margin, all fields). Held cells: n/a where cells equal positions.

| preset | tree-one | tree-multi | fp | after | debias2 | t0.5 | batch | retry | malformed (engine unchanged) | held cells (old → new) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-vision | pass (0.00048; 0) | pass (0.0025; 0) | pass (0.0014; 0) | pass (0.00096; 0) | pass (0.00077; 0) | pass (0.00026; 0) | pass | pass | pass (pass) | pass (58 → 1057) |
| smolvlm-500m | skipped: fit check: does not load: tokenizer does not tokenise the trunk/branch join independently (add_space_prefix?); model not supported: llama 8B Q8_0, vocab type BPE: prefix/state join "...end_of_utterance>\nUser: " + "Hi" tokenises jointly as [21041], separately as [216 26843] (from token 8, left part has 9) | | | | | | | | | |
| gemma-4-e4b-vision | pass (0.00061; 0) | pass (0.0007; 0) | pass (0.002; 0) | pass (0.0011; 0) | pass (0.00079; 0) | **FAIL** (3.9e-05; 1) | pass | pass | pass (pass) | pass, n/a (148 = 148) |
| qwen3.6-35b-a3b-vision | pass (0.00041; 0) | pass (0.0033; 0) | pass (0.0029; 0) | pass (0.0097; 0) | pass (0.007; 0) | pass (0.00073; 0) | **FAIL** | pass | pass (pass) | pass (41 → 321) |
| gemma-3-4b-vision | pass (9.7e-09; 0) | **FAIL** (1.5e-06; 1) | pass (5e-07; 0) | pass (1.3e-06; 0) | pass (5.5e-07; 0) | **FAIL** (7.2e-12; 1) | **FAIL** | **FAIL** | pass (pass) | pass, n/a (274 = 274) |

Kept failures (`failures.txt`): gemma-4-e4b-vision: reference t0.5: not pass: True | qwen3.6-35b-a3b-vision: batch | gemma-3-4b-vision: retry; reference tree-multi: not pass: True; reference t0.5: not pass: True; batch.

Held cells, answer to the sized state, old build → new build: qwen3.8-27b-vision 500 engine → 400 budget; gemma-4-e4b-vision 500 engine → 400 budget; qwen3.6-35b-a3b-vision 500 engine → 400 budget; gemma-3-4b-vision 200 → 200.

Non-causal gate (gemma-3-4b, spec 9.5): **failed** (tree-multi, t0.5, batch); non-causal projectors are switched off (check of the rebuilt engine: pass, info `{"available": false, "reason": "non_causal_projector"}`).

Ticket 1, P(dog = true): reference 0.653 (tree-multi) / 0.653 (fp); engine 0.103 (tree-multi) / 0.906 (fp).

Engine against itself on gemma-3-4b: retry (rc1 vs clean) max abs diff 0.850; batch together max abs diff 0.348, mixed 0.070.

n_ubatch rule (gemma-3-4b, `-ub 128`): pass.

- **Binding model.** Qwen3.8-27B passes every check: the six reference tags (largest per-field median 0.0025, no
  disagreement over the margin), batch invariance (together and mixed with text states), the `rc1` retry (8 retries,
  same image usage), the 27 malformed cases (26 × 400 `invalid_states` at the expected field, one 400 `budget` for an
  image state over the budget) with info and the next answers unchanged, and held cells: with an image chat resident
  the old build counts 58 held cells and answers the sized state with 500 `engine` ("no KV cell space"); the new build
  counts 1057 and answers 400 `budget`.
- **Non-binding failures, kept as results.**
  - gemma-4-e4b-vision fails `t0.5` on one record (ticket 1, image only, field `indoors`): reference 0.607 / 0.393,
    engine 0.448 / 0.552, reference margin 0.214. At T = 1 the same record already disagrees (0.554 vs 0.474) under
    the margin (0.108); T = 0.5 sharpens a near tie past it. Its retry also has one argmax disagreement on `indoors`
    under the margin (max abs diff 0.028), which the statistic allows.
  - qwen3.6-35b-a3b-vision fails batch invariance `mixed` on `indoors`: median 0.0116 against the bound 0.01 (max
    0.022, no argmax disagreement). It is a MoE model; the cause was not investigated.
- **Held cells** are tested only where an image's cells exceed its positions (the two M-RoPE Qwen models: 41 → 321 on
  the 35B). On gemma-4-e4b-vision and gemma-3-4b-vision both counts agree and the check is not applicable;
  gemma-4-e4b-vision's sized state is now over its SWA limit and gets 400 `budget`. The held-count comparison therefore
  ran live only on the two Qwen models.
- **The gate failed.** On gemma-3-4b every failure is the field `dog` on the two states of one image (ticket 1, image
  only; ticket 5, the same image with text after it); all other records agree. The engine disagrees with itself: for
  ticket 1 it gives P(dog = true) 0.103 in `tree-multi` and 0.906 in `fp`, against a reference of 0.653 in both, and the
  retry and batch checks, which compare the engine with itself without a replay, differ by up to 0.850 and 0.348. So
  the answer depends on how the round is laid out. This fits the projector's non-causal switch acting on the whole
  context while other sequences are in the KV cache, which was not proven. As spec 9.5 specifies, 0042 makes
  non-causal projectors unavailable: info reports `non_causal_projector` and image requests get 400 `vision_unavailable`
  (`gate-off.json`). The `n_ubatch` rule of spec 5.3 passed before the switch and cannot be reached after it.
- **Several image chunks per image.** No model that ran splits an image into several image chunks (spec 8: SmolVLM was
  the model for that case), so the chunk loop of spec 5.2 ran live only with one image chunk per image.

### Sliding-window cache budget (Task 6b)

| preset | n_ctx | info `memory.swa_cells` | SWA cache in llama.cpp's log | result |
|---|---|---|---|---|
| gemma-4-e4b | 4096 | 4096 | 4096, 4096 | pass |
| gemma-4-e4b-16k | 16384 | 9216 | 9216, 9216 | pass |
| gemma-4-e4b-16k-swa-full | 16384 | 16384 | 16384, 16384 | pass |
| gemma-3-4b-vision | 16384 | 16384 | 16384, 16384 | pass |

| run | held | prefix | K | limit | largest state answered 200 | +1 token | any 500 | result |
|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b-16k-idle | 0 | 68 | 1 | 9200 | 9120 | 400 budget | no | pass |
| gemma-4-e4b-16k-debias2 | 0 | 136 | 2 | 9200 | 4520 | 400 budget | no | pass |
| gemma-4-e4b-16k-chat | 16 | 68 | 1 | 9200 | 9104 | 400 budget | no | pass |
| gemma-4-e4b-16k-swa-full-idle | 0 | 68 | 1 | 16368 | 16288 | 400 budget | no | pass |
| gemma-3-4b-vision-idle | 0 | 64 | 1 | 16368 | 16292 | 400 budget | no | pass |

Prefix of 10000 words: gemma-4-e4b-16k exit 1, `prefix needs 10061 cells with 0 held by slots, the sliding-window cache holds 9216 cells (limit 9200); start with --swa-full to use n_ctx`; gemma-4-e4b-16k-swa-full exit 0, prefix 10061 tokens.

Round sizing (gemma-4-e4b-16k, three states in one request; gate: each round of the split equals the same composition sent on its own):

| case | state tokens | need of 2 / 3 states | limit | status, rounds, retries | gate: round 1 vs pair, round 2 vs alone (max abs diff) | recorded, pair vs each alone: median / max abs diff, argmax disagreements |
|---|---|---|---|---|---|---|
| same | 4554, 4554, 4554 | 9200 / 13766 | 9200 | 200, 2, 0 | 0.0, 0.0: pass | 0.0285 / 0.0353, 0 |
| distinct | 4654, 4454, 3000 | 9200 / 12212 | 9200 | 200, 2, 0 | 0.0, 0.0: pass | 0.0273 / 0.1038, 1 |

Controls for the pair against each text alone (`swa/control/`, two states in one request vs each alone, answers repeated twice):

| control | sizes | state 0: max abs diff | state 1: max abs diff, argmax disagreements | repeat equals first run |
|---|---|---|---|---|
| gemma-4-e4b-16k-full | 4654, 4454 | 0.0296 | 0.1038, 1 | yes |
| gemma-4-e4b-16k-swa-full-full | 4654, 4454 | 0.0296 | 0.1038, 1 | yes |
| gemma-4-e4b-16k-full-old | 4654, 4454 | 0.0296 | 0.1038, 1 | yes |
| gemma-4-e4b-16k-quarter | 1163, 1113 | 0.0239 | 0.0188, 0 | yes |

Pair and alone answers equal bit for bit on the limited preset, its `--swa-full` twin and the old build: yes.

- **The defect predates this work.** On models with sliding-window attention layers the budget allowed `n_ctx − 16`
  cells, but the SWA cache can hold fewer (Gemma 4 E4B at ctx 16384: n_swa 512 × 17 sequences (16 decide sequences + 1
  server slot) + n_ubatch 512 = 9216). A state between the
  two passed the budget and ended in 500 `engine`, text or image, on the old build too. The held check on
  gemma-4-e4b-vision found it (Task 6, `integrity/gemma-4-e4b-vision/held-control/`).
- **The fix (0043).** Every cell limit is `min(n_ctx, SWA cells) − 16`, with the SWA size computed as llama.cpp
  computes it; info's `memory.swa_cells` equals the size in llama.cpp's log on all four presets. The sweeps answer 200
  up to the limit and 400 `budget` one token over it, never 500, idle, with K = 2 and with a chat resident. With
  `--swa-full` the limit is `n_ctx − 16` again. gemma-3-4b's SWA cache already has `n_ctx` cells, so its limit is
  unchanged. Round sizing uses the same limit: three states that fit `n_ctx` but not the SWA cache are answered in 2
  rounds without a retry, and each round answers exactly (max abs diff 0) as the same states sent on their own. Models
  without SWA layers are unchanged (the 27B regression above). Gemma 4 E4B's text integrity at ctx 4096 (where the SWA
  cache is not the bound) passes (`swa/text-integrity/`).
- **Recorded, not a gate: a batching gap on Gemma 4 E4B that predates this work.** Two long synthetic states decoded in
  one batch differ from the same states decoded alone beyond the statistic (pair vs alone median 0.027-0.029, max 0.104,
  one argmax flip at a reference margin of 0.053). The answers are the same bit for bit with `--swa-full` and on the old
  build, so neither the SWA bound nor this work causes it. A quarter-size pair with other texts differs less (max
  0.024), which suggests but does not show a dependence on length. Real triage tickets pass batch invariance on this
  model (medians 9e-06 to 5.0e-05, `swa/text-integrity/`). The model is not binding.

### Evaluation

One decide run and two chat runs (run 1 with `cache_prompt` false, run 2 with `cache_prompt` true; otherwise the same requests). Chat: one question per call, `max_tokens` 1, temperature 0, thinking off.

| split | field | n | accuracy (decide, T = 1) | NLL (T = 1) | accuracy (chat run 1, strict) |
|---|---|---|---|---|---|
| calibration | person | 91 | 0.978 | 0.114 | 0.956 |
| calibration | car | 96 | 0.979 | 0.099 | 0.938 |
| calibration | bicycle | 98 | 1.000 | 0.031 | 0.990 |
| calibration | cat | 100 | 0.980 | 0.046 | 0.940 |
| calibration | dog | 100 | 0.990 | 0.082 | 0.960 |
| calibration | all | 485 | 0.986 | 0.074 | 0.957 |
| test | person | 93 | 1.000 | 0.011 | 0.978 |
| test | car | 96 | 0.990 | 0.082 | 0.969 |
| test | bicycle | 98 | 0.990 | 0.037 | 0.939 |
| test | cat | 100 | 1.000 | 0.022 | 0.990 |
| test | dog | 100 | 1.000 | 0.057 | 0.970 |
| test | all | 487 | 0.996 | 0.042 | 0.969 |

Errors (wrong answers) per reading. Strict: the chat answer must start with yes or no. `p_yes`: the yes/no log-probabilities among the top 20, renormalised, read at 0.5.

| split | n | decide | chat strict, run 1 / run 2 | chat `p_yes`, run 1 / run 2 |
|---|---|---|---|---|
| calibration | 485 | 7 (0.9856) | 21 / 21 | 11 / 12 |
| test | 487 | 2 (0.9959) | 15 / 17 | 5 / 6 |
| all | 972 | 9 (0.9907) | 36 / 38 | 16 / 18 |

Chat answers that are neither yes nor no: run 1 20 (labels: false 20; p_yes on the label's side of 0.5: 19; answers: "```" 13, "Based" 6, "Looking" 1); run 2 21 (labels: false 21; p_yes on the label's side of 0.5: 19; answers: "```" 12, "Based" 7, "" 1, "<tool_call>" 1).

Run 1 against run 2, per field (1000): first tokens differ 79, parsed yes/no/other differs 18, `p_yes` on the other side of 0.5 2.

Temperature fitted on the calibration half: T = 0.7086 (calibration NLL 0.0736 → 0.0637, test NLL 0.0420 → 0.0263); reported only, no preset change.

Time (medians, client wall-clock around each POST, requests one at a time):

| condition | per image | per call | first call | calls 2-5 |
|---|---|---|---|---|
| decide, one request with five fields | 1.321 s (server 1312.4 ms) | | | |
| chat run 1, `cache_prompt` false | 6.813 s | 1.345 s | 1.506 s | 1.329 s |
| chat run 2, `cache_prompt` true | 6.661 s | 1.323 s | 1.405 s | 1.317 s |

Run 2, calls 2-5: 800 calls, median `prompt_n` 1081, median `cache_n` 0, calls with `cache_n` > 0: 0.

- **Decide** at T = 1 gets 7 of 485 scored fields wrong on the calibration half and 2 of 487 on the test half (NLL
  0.074 / 0.042). The temperature fitted on the calibration half, T = 0.709, lowers the test NLL from 0.042 to 0.026; it
  is reported only and no preset changed. There is one decide run, so its run-to-run reproducibility was not measured.
- **Chat baseline** (one call per field, `max_tokens` 1). Read strictly it has 36 and 38 errors of 972 in its two runs.
  20 and 21 of its answers are neither yes nor no but the first token of a longer reply; all are on fields labelled
  false, and 19 of them in each run have `p_yes` on the right side of 0.5. Read by `p_yes`, chat has 16 and 18 errors
  against decide's 9; on the test half 5 and 6 against 2. The strict gap is mostly a format effect of the 1-token
  baseline, not a difference in perception. By the model's own yes/no preference the two are close, decide slightly
  ahead on both halves; the counts are too small to call decide more accurate.
- **Chat is not reproducible run to run on this server.** Between the two runs, which differ only in `cache_prompt`
  and reuse no image prefix, 79 of 1000 greedy first tokens differ (60 of them `No` against `no`), 18 parsed answers
  and 2 `p_yes` sides change. The chat error counts carry about 2 errors of noise.
- **Time.** One decide request with five fields takes 1.32 s per image (median); five chat calls take 6.81 s with
  `cache_prompt` false and 6.66 s with `cache_prompt` true. On this hybrid model the server did not reuse the image
  prefix (`cache_n` 0 in all 800 calls 2-5), so every chat call encodes the image again, and one chat call costs about
  what one five-field decide request costs (1.32-1.35 s). The ratio on a model whose prefix reuse works was not
  measured.

### Final engine (spec rev 6)

On the final engine `69756d495`: the text regression against the old build, the server smoke, the limits of rev 6 and every integrity check of the binding model (`final/`).

| check | result |
|---|---|
| text regression qwen3.8-27b, old vs new: 88 requests, max abs diff 0.0, 0 differences | pass |
| smoke qwen3.8-27b-vision: 11 of 11 checks | pass |
| limits qwen3.8-27b-vision: info: vision.max_image_aspect_ratio 200 | pass |
| limits qwen3.8-27b-vision: 200 x 1: 200 with one image | pass |
| limits qwen3.8-27b-vision: 1 x 200: 200 with one image | pass |
| limits qwen3.8-27b-vision: 201 x 1: 400 invalid_states naming the aspect-ratio limit | pass |
| limits qwen3.8-27b-vision: 1 x 201: 400 invalid_states naming the aspect-ratio limit | pass |
| limits qwen3.8-27b-vision: probes: the image alone 200, a long text 400 budget with the cell counts | pass |
| limits qwen3.8-27b-vision: control within the budget: the undecodable PNG gives 400 invalid_states (pass 3) | pass |
| limits qwen3.8-27b-vision: over the limit after image 1: 400 budget with the cells of 1 image(s), before the undecodable PNG | pass |
| limits qwen3.8-27b-vision: over the limit after image 2: 400 budget with the cells of 2 image(s), before the undecodable PNG | pass |
| limits qwen3.8-27b-vision: info unchanged by the rejected requests | pass |

Early budget check: one image has 1038 cells, the largest state the budget admits is 16301; over after image 1 (text 15782 tokens): 400 budget; over after image 2 (text 14744 tokens): 400 budget.

| preset | tree-one | tree-multi | fp | after | debias2 | t0.5 | batch | retry | malformed (engine unchanged) | held cells (old → new) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-vision | pass (0.00048; 0) | pass (0.0025; 0) | pass (0.0014; 0) | pass (0.00096; 0) | pass (0.00077; 0) | pass (0.00026; 0) | pass | pass | pass (pass) | pass (58 → 1057) |

Malformed cases: 32 of 32 checks pass (30 requests, the aspect-ratio cases among them).

Failures (`failures.txt`): none.

- **The finding (code review of the final engine).** The pixel limit did not bound memory for extreme aspect ratios.
  libmtmd's resize (`calc_size_preserved_ratio`, `tools/mtmd/mtmd-image.cpp:122-157`) rounds the short side up to the
  projector's alignment after scaling to the pixel budget, and `prepare_inputs` tokenised every image of a request
  before any budget check and kept the chunks (with their f32 pixels) until the request ended. The claim holds by
  reading and arithmetic, without a model: with the 27B preset's values (alignment 32, `image-min-tokens` 1024 →
  1,048,576 minimum pixels, 4096 tokens → 4,194,304 maximum pixels) a 16,000,000 × 2 PNG (within the 32,000,000-pixel
  limit, about 93 KB) resizes to 5,792,608 × 32 = 185,363,456 pixels, 2.07 GiB of f32 RGB and 181,019 cells per image;
  32 such images would hold about 66 GiB, more than the test machine's 62 GB. No such image was sent to a server
  before the fix was built.
- **The fix (0044, spec rev 6).** Pass 1 rejects an image whose longer side is more than 200 times its shorter side
  (400 `invalid_states` naming the sizes and the limit; integer arithmetic). At 200:1 the scaled short side stays
  above the alignment (at the 27B's maximum pixels: sqrt(4,194,304 / 200) ≈ 145 > 32), so the pixel budget bounds the
  preprocessed image again. info reports `vision.max_image_aspect_ratio` (200). `prepare_inputs` checks the state's
  cells so far (text tokens before the image plus the image cells, with the same held, prefix, K, tail and branch terms
  as the per-state check) right after each image is tokenised and ends the request with the usual 400 `budget` at
  once; the per-state check after all parts stays. The live check shows the order: a state whose cells cross the limit
  at its first (or second) image gets 400 `budget` naming the cells of one (or two) images, although an undecodable
  PNG follows, which within the budget gives 400 `invalid_states`. To compute the budget terms before the images,
  `run_request` now prepares the prefixes before tokenising the states (same work, earlier); a text request sees no
  difference (the 27B regression above: max abs diff 0).
- **Smaller fixes in 0044.** The engine rejects an image part itself (400 `vision_unavailable`) when vision is not
  available, so a caller that skips `states_of` cannot reach libmtmd without a projector (not reachable without a model
  in a ctest: it needs an engine instance); the image SHA-256 is computed only when the request dumps tokens; comments
  say that the `n_ubatch` rule cannot fire while non-causal projectors are off and what `slot_seqs` still feeds.
- **Results.** Every check passed on `69756d495`: the 27B text regression against the old build (88 requests, max abs
  diff 0), the server smoke (11 of 11), the rev 6 limits and early budget checks, and every integrity check of the
  binding model. Its reference files, dumps, replays and `batch.json` are byte-equal to those of Task 6 on 0001-0041
  (`integrity/qwen3.8-27b-vision/`); `retry.json` has the same probabilities and differs in its timing values; the held
  files differ only in keys the scripts added since then. The malformed set now has 29
  cases (the 26 before plus 201 × 1, 16,000 × 2 and 16,000,000 × 2, each 400 `invalid_states` naming the 200:1 limit).
- **Documentation fixes.** The earlier limitation "after a server sleep and resume the held-cells count can be higher"
  was wrong: `load_model` clears and re-creates the server slots on every load, resume included
  (`tools/server/server-context.cpp:1255,1272`), so it is removed. `results-engine/image/NOTICE.md` credits the four
  integrity images.

### Sequence budget first (0045, spec rev 7)

The rev 6 fixes (0044) moved the preparation of the prefixes before the images are tokenised, so that the early budget check has
its numbers. That also moved it before the sequence-budget check (too few decide sequences for the schema, 400
`budget`), so a text request with two faults at once (a chat template or join failure in the prefix, a 500, and too few
decide sequences) got the 500 where it used to get the 400. 0045 checks the sequence budget first: it needs only the
number of catalogue variants and the branches per state, and now runs before the prefixes are prepared and before any
image is loaded. The early image budget check is unchanged; an image request with too few sequences is now rejected
before its images are decoded (spec 3.2 pass 3 rev 7). No ctest reaches `run_request` (it needs a model), and the text
double fault cannot be built with a real tokenizer, so the order is shown by reading and by a live image double fault
(`checks.py double-fault`): a schema with as many bool fields as decide sequences together with a state whose cells
cross the limit at its image.

| check (qwen3.8-27b-vision, `decide-seqs` 16) | `69756d495` (0001-0044) | `185e9e0a7` (0001-0045) |
|---|---|---|
| too few sequences alone: 400 `budget` naming the sequences | pass | pass |
| over the cell limit at the image alone: 400 `budget` naming the image cells | pass | pass |
| both at once: the sequence-budget 400 answers | fail (the image-cells 400 answered) | pass |
| info unchanged by the rejected requests | pass | pass |

On `185e9e0a7` also: the text regression of qwen3.8-27b against the old build (`engine-49785a95d`), 88 requests, max
abs diff 0.0, 0 differences; the server smoke on qwen3.8-27b-vision, 11 of 11 pass (`publish/smoke-qwen3.8-27b-vision.json`).
Build: zero compiler warnings; the 11 decide ctests pass.

Labels in recorded files were renamed after the runs, mechanically and with every value unchanged: the smoke check "media marker in a text part stays text" lost a parenthetical suffix (`task-2/`, `final/`, `publish/`), the directory and step name of the rev 6 checks became `final` (`final/regression/qwen3.8-27b.json`, `gpu-log.txt`), and the skip reason in `integrity/smolvlm-500m/SKIPPED.txt` starts with "fit check:". The sub-project 3 checks were relabelled the same way: the files of their second build got the prefix `rebuild-` (`sp3/checks/task-5/`, `sp3/checks/task-7/`, and the paths inside the two task-5 JSONs), and the summary command note in `sp3/integrity/run-params.json` reads "second pass".

### Limits and open points

- Out of scope (spec 10): images in the instructions or the prefix, and an image shared by several states; remote
  URLs, `file:` URLs and server paths; audio and video; a cache of image trunks or embeddings (an image is encoded again
  for every debias variant and every retry); decoding the images of several states in one batch; images in
  `/v1/systemone`; a calibrated temperature preset for image requests.
- Not supported: projectors with non-causal image attention (Gemma 3), after the failed gate.
- Images whose longer side is more than 200 times their shorter side (very long strips or panoramas) are rejected
  (spec rev 6); crop or pad them before sending.
- Not covered live: an image split into several image chunks (SmolVLM skipped); the held-count comparison outside the
  two Qwen models.
- Kept failures of non-binding models: gemma-4-e4b-vision `t0.5` (one borderline record), qwen3.6-35b-a3b-vision batch
  `mixed` (median 0.0116 against 0.01). The batching gap on Gemma 4 E4B with two long states predates this work.
- On SWA models the held cells of chat slots count in full against the SWA cache, though their cells outside the window
  could be reused: such a model with a long chat resident rejects a little earlier than it must. A model that sets
  `n_swa` without building a sliding-window cache would get a lower limit (no model used here does).
- `llama-decide` writes a plain-text error and exits 2 if an error body cannot be dumped as strict JSON (the server
  uses a tolerant dump); no known input reaches it.
- Base64 pad bits are not checked (RFC 4648 §3.5 allows decoders to ignore them).
- The offline recompute of an image replay needs the images in `local/` (not committed; `integrity-images.sha256`,
  `coco/images.sha256`).
- Behaviour that changed through decisions made during the work:
  - a state that is neither a string nor an array gets 400 `invalid_states` "states[i]: states entries must be strings
    or arrays of parts" with `field` `states[i]` (a new message for text-only requests too);
  - a part with an unknown key (for example OpenAI's `detail`) is a 400 (spec rev 4);
  - a failed `mtmd_tokenize` is a 500 `engine` (spec rev 4);
  - the server's held cells come from `slot.prompt.n_tokens()`, for text requests too (the regression shows no change);
  - a state over the SWA cache is a 400 `budget` instead of a 500 (spec rev 5);
  - an image over 200:1 is a 400 `invalid_states`, and a state's cells are checked after each of its images, so a
    state over the budget can get 400 `budget` before a later image of the request is decoded (spec rev 6).
- `LLAMA_ARG_MMPROJ`, `LLAMA_ARG_IMAGE_MIN_TOKENS` and `LLAMA_ARG_IMAGE_MAX_TOKENS` now also apply to `llama-decide` and
  `llama-completion` (the three options are registered for `LLAMA_EXAMPLE_COMPLETION`); `llama-completion --help`
  lists them, and `llama-completion` ignores them.

## How to reproduce
```bash
# models-engine.ini sets decide-temperature = 1.6947 for gemma-4-e4b (sub-project 3 calibration): a Gemma request to the
# server (/v1/decide, /v1/systemone) without options.temperature is answered at that T. Not affected: llama-decide CLI
# runs (reference_check.py dump and split, cache_timing.py, the CLI live checks), servers started with -m instead of the
# presets (checks/task-7/run.sh) and /completion replays (reference_check.py compare). Every committed Gemma run in this
# report except the preset check (results-engine/sp3/eval/preset/) ran at T = 1 (every argmax, hence accuracy, is the
# same at any T; NLL, ECE and Brier are not). decide_client.py names its outputs by the temperature the engine applied (engine.temperature of the
# warm-up answer): -t<T> when it is not 1.0, so a Gemma command without a temperature flag now writes
# decide-gemma-4-e4b-...-t1.6947.* beside the committed T = 1 files. --ensure-t1 gives the T = 1 files: it sends
# options.temperature 1 only when the server default is not 1.0 (else the request is the committed one, byte for byte)
# and stops before writing if the engine applied another T; --temperature 1 gives the same answers and names but always
# sends the option. decide_client.py also stops before writing when a preds or run file of its name exists: --force
# replaces the committed files, another --out-dir keeps them (then compare with sp3_metrics.py compare).
# Steps with Gemma runs: the parity, prompt-variant, held-out and Task 0 baseline commands below carry --ensure-t1 (no
# change to the request on the other presets); results-engine/sp3/integrity/run.sh regression and
# results-engine/sp3/eval/run.sh runs pass --ensure-t1 --force themselves (their Gemma requests now carry
# options.temperature 1; the committed runs, from before the line, carry none); results-engine/sp3/eval/run.sh preset
# passes --force and writes the -t1.6947 files.
./build-engine.sh                      # clones upstream 60b06ab9a into engine/, branch decide + git am patches/decide/*.patch, CUDA build
./test-engine.sh                       # every decide test through ctest (schema, score, compat, trie, cache, fullpath, debias, after, image; trie and after on the in-repo Qwen2 and Gemma 4 vocab files)
./serve-engine.sh                      # separate terminal; presets from models-engine.ini on :8097
# decide_client.py uses --prompt-variant keywords by default since the held-out check; the parity, integrity and MoE runs used default
uv run decide_client.py --model qwen3.5-9b --batch 1 --prompt-variant default; uv run decide_client.py --model qwen3.5-9b --batch 5 --prompt-variant default
uv run decide_client.py --model gemma-4-e4b --batch 1 --prompt-variant default --ensure-t1; uv run decide_client.py --model gemma-4-e4b --batch 5 --prompt-variant default --ensure-t1
# prompt variants: for P in qwen3.5-9b gemma-4-e4b, V in merged framing keywords, N in 1 5:
uv run decide_client.py --model P --batch N --prompt-variant V --ensure-t1
uv run score.py --preds-dir results-engine/preds | tee results-engine/score-table.txt
# held-out check: for P in qwen3.5-9b gemma-4-e4b (all Qwen runs first), V in default merged framing keywords, GPU below 60 °C before each run:
uv run decide_client.py --model P --batch 5 --prompt-variant V --test-set data/triage_train.jsonl --out-dir results-engine/heldout --ensure-t1
uv run score.py --preds-dir results-engine/heldout/preds --test-set data/triage_train.jsonl | tee results-engine/heldout/score-table.txt
uv run heldout_stats.py                                                   # sign tests and the rule: results-engine/heldout/stats.json
uv run compare_engines.py                                                 # compare-*.json
uv run batch_invariance.py results-engine/preds/decide-qwen3.5-9b-b1.jsonl results-engine/preds/decide-qwen3.5-9b-b5.jsonl \
    --out results-engine/batch-invariance-qwen3.5-9b-parity.json
# reference check: dump with the server stopped, compare with it running
uv run reference_check.py dump --model ./models/Qwen3.5-9B-Q4_K_M.gguf --preset qwen3.5-9b --n 20
uv run reference_check.py compare --preset qwen3.5-9b --url http://127.0.0.1:8097
# Jev compatibility (server running): smoke, then the SDK acceptance test (SDK not a project dependency)
uv run systemone_smoke.py --url http://127.0.0.1:8097
SYSTEMONE_URL=http://127.0.0.1:8097 uv run --with local/typesafe-sdk-python pytest -q tests/test_systemone_sdk.py   # SDK v0.7.0 cloned there (README)
# multi-branch reference (follow-up fixes): same two phases with the multi-branch schema
uv run reference_check.py dump --model ~/.lmstudio/models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf \
    --preset qwen3.5-4b --n 20 --schema tests/data/decide-multibranch.json --tag multibranch
uv run reference_check.py compare --preset qwen3.5-4b --tag multibranch
# batch invariance / rounds (0.5B, 4B): decide_client.py --prompt-variant default --limit 20 at --batch 1 and 5, --batch 12 --limit 12 vs --batch 5 --limit 12,
# then batch_invariance.py on each pair (inputs kept under results-engine/integrity/)
# MoE with CPU experts (presets gemma-4-26b-a4b, qwen3.5-35b-a3b, qwen3-30b-a3b; cpu-moe = 1); for each preset P with GGUF G:
uv run decide_client.py --model P --batch 5 --prompt-variant default; uv run decide_client.py --model P --batch 1 --limit 20 --prompt-variant default
#   move the b1 preds/run JSON to results-engine/integrity/decide-P-b1-lim20{.jsonl,-run.json}, head -20 of the b5 preds to decide-P-b5-first20.jsonl
uv run batch_invariance.py results-engine/integrity/decide-P-b1-lim20.jsonl results-engine/integrity/decide-P-b5-first20.jsonl \
    --out results-engine/batch-invariance-P.json
uv run reference_check.py dump --model G --preset P --n 20          # server stopped; passes --cpu-moe from the preset
uv run reference_check.py compare --preset P                        # server running
# op-offload diagnostic: the same dump/compare with --tag diag-cpuexperts and GGML_OP_OFFLOAD_MIN_BATCH=1000000 in the
# environment of both llama-decide and serve-engine.sh (outputs moved to results-engine/integrity/)
# sub-project 3 (engine improvements); every output under results-engine/sp3/; Gemma at T = 1 through --ensure-t1 (first
# lines of this block; eval/run.sh runs also checks engine.temperature 1.0 in every run).
# baseline (Task 0): for P in qwen3.5-9b gemma-4-e4b, S in test train, each on a freshly started ./serve-engine.sh:
uv run decide_client.py --model P --batch 5 --test-set data/triage_S.jsonl --out-dir results-engine/sp3/baseline/S --ensure-t1
uv run sp3_metrics.py score --preds-dir results-engine/sp3/baseline/S/preds --test-set data/triage_S.jsonl
# live checks during the engine tasks: results-engine/sp3/checks/task-N/ (checks.py; task 7 also run.sh, one function per
# check; the task 4-6 notes.md give the CLI template, flags and checks.py subcommands, not a command per output file)
results-engine/sp3/integrity/run.sh dumps replays split cache debias regression nosplit   # Task 8; starts and stops its own servers
results-engine/sp3/eval/run.sh runs score calibration     # Task 9: 16 runs on fresh routers, scores, sign tests, summary, calibrate.py
# then the calibration rule: decide-temperature = 1.6947 in [gemma-4-e4b], and the check of the preset:
results-engine/sp3/eval/run.sh preset
# follow-ups: on the test machine (user@gpu-host:<repo>; export REMOTE and REMOTE_DIR), from the laptop through
# ./remote-sync.sh push project|engine, ./remote-sync.sh run '<command>' and ./remote-sync.sh pull <path>; engine build there:
# ./build-engine.sh with its defaults (engine/ already pushed), the old build in engine-63ea2c51a/ (git worktree, same flags):
# git -C engine worktree add ../engine-63ea2c51a 63ea2c51a, then in engine-63ea2c51a/ with PATH=/opt/cuda/bin:$PATH:
# cmake -B build <the build-engine.sh flags> && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-decide-cli
uv run results-engine/followups/task-1/check.py run qwen2.5-0.5b; uv run results-engine/followups/task-1/check.py run gemma-4-e4b
uv run results-engine/followups/task-2/check.py sequences qwen2.5-0.5b --seqs 21; uv run results-engine/followups/task-2/check.py sequences qwen3.5-4b --seqs 32
uv run results-engine/followups/task-2/check.py info qwen2.5-0.5b --seqs 21; uv run results-engine/followups/task-2/check.py info qwen3.5-4b --seqs 32
uv run results-engine/followups/task-2/check.py fault cell-estimate   # also lines-join, tail-mismatch (LLAMA_DECIDE_FAULT on server F)
# Qwen3.8-27B (test machine): results-engine/3090/run.sh exports ENGINE_PRESETS=models-engine-3090.ini TRIAGE_DATA=data
# GPU_VRAM_MAX=1500 GPU_WAIT_MIN=20 MODELS=qwen3.8-27b EVAL_DIR=results-engine/3090/eval BASELINE_DIR=results-engine/3090/baseline
# and writes into results-engine/3090/ (move it aside before a re-run, or set OUT_DIR; PRESET, OUT_DIR, SEQS: its header);
# every step at T = 1 until preset. models-engine-3090.ini now holds decide-temperature = 0.8992, and systemone (no
# --ensure-t1) runs at the preset's T: remove that line first for a from-scratch reproduction
results-engine/3090/run.sh eval dumps replays offline batch latency systemone
# then the rule's change, decide-temperature = 0.8992 in [qwen3.8-27b] of models-engine-3090.ini, and:
results-engine/3090/run.sh preset notes
# sub-project 4 (image input), test machine, from the laptop through ./remote-sync.sh run '…'; results under results-engine/image/
# setup (Task 0, once): the models of results-engine/image/models.sha256 under models/ and ~/.lmstudio/models/;
# the four integrity JPEGs from their COCO URLs into local/image/integrity/ (checked by integrity-images.sha256; credits
# in results-engine/image/NOTICE.md) and the PNG copy of image 39769:
mkdir -p local/image/integrity
for i in 000000039769 000000000776 000000000139 000000000632; do curl -fsSL --retry 3 -o local/image/integrity/coco-$i.jpg http://images.cocodataset.org/val2017/$i.jpg; done
sha256sum -c results-engine/image/integrity-images.sha256
uv run --with pillow python -c "from PIL import Image; Image.open('local/image/integrity/coco-000000039769.jpg').save('local/image/integrity/coco-000000039769.png')"
# the old build for every old-vs-new check: a worktree of the head before this work, built with build-engine.sh's flags
git -C engine worktree add ../engine-49785a95d 49785a95d63510972b4ff0d42dbbe80052cb2fea
#   then in engine-49785a95d/ with PATH=/opt/cuda/bin:$PATH: cmake -B build <the build-engine.sh flags> -DLLAMA_BUILD_TESTS=ON
#   && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-decide-cli
# which reference tags fit each preset's decide-seqs (task-0/fit-<preset>.json, read by run.sh integrity):
uv run results-engine/image/fit_check.py <preset>        # for each vision preset; --unavailable REASON for one that cannot run
results-engine/image/run.sh smoke cli regression text-integrity no-vision    # Tasks 2, 3, 5 (PRESET default qwen3.8-27b-vision)
PRESET=<qwen3.8-27b-vision|smolvlm-500m|gemma-4-e4b-vision|qwen3.6-35b-a3b-vision> results-engine/image/run.sh integrity
# gemma-3-4b-vision's integrity, ubatch and gate need the engine at 0001-0041 (9bae50c77), where NON_CAUSAL_ENABLED in
# tools/decide/decide-image.h is true (or a build with it set back to true): from 0042 on it is false and every gemma-3
# image request gets 400 vision_unavailable, so integrity, ubatch and gate refuse to run for it (FORCE=1 overrides
# and overwrites the committed gate evidence); on the final engine only gate-off applies to gemma-3-4b-vision
results-engine/image/run.sh gate-off                                           # after the failed gate
# Task 6 diagnosis of gemma-4-e4b-vision's 500 (integrity/gemma-4-e4b-vision/held-control/): the commands are in
# results-engine/image/held_control.py's docstring (server, bisect, cli), run by hand on a router of that preset
PRESET=gemma-4-e4b-vision results-engine/image/run.sh held; results-engine/image/run.sh swa swa-regression swa-integrity swa-control   # Task 6b
results-engine/image/run.sh coco-data coco coco-cached                         # Task 7
# the spec rev 6 checks on the final engine, into results-engine/image/final/:
results-engine/image/run.sh final
INTEGRITY_DIR=results-engine/image/final/integrity/qwen3.8-27b-vision results-engine/image/run.sh integrity
# then on the laptop: uv run results-engine/image/notes.py
```
Models are under `models/` (the files of the baseline runs: `Qwen3.5-9B-Q4_K_M.gguf`, `gemma-4-E4B-it-Q4_0.gguf`), plus `Qwen2.5-0.5B-Instruct-Q8_0.gguf` and `Qwen3.5-4B-Q4_K_M.gguf` under `~/.lmstudio/models/lmstudio-community/` for the integrity checks (the sub-project 3 run scripts take that directory from `LMSTUDIO_MODELS` and the test and train sets' directory from `DATA`, with these defaults). The test set is `data/triage_test.jsonl`; the held-out check used `data/triage_train.jsonl` (the runs read byte-identical copies of these files outside the repository: on the laptop the author's experiment directory, on the test machine `local/data/`). On the test machine the small GGUFs are in `models/` (presets `models-engine-3090.ini`) and the SDK checkout in `local/typesafe-sdk-python/`.
