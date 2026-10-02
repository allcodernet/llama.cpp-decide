# Qwen3.8-27B on the RTX 3090 (follow-ups Task 5)

Written by `notes.py` from the files of this directory. Measured on an RTX 3090; the sub-project 3 numbers in `REPORT-ENGINE.md` come from an RTX 3070 Ti laptop GPU.

## Setup

Engine `23957f14c` (`engine/build` on the test machine). Preset `qwen3.8-27b` of `models-engine-3090.ini`: `/home/marduk/.lmstudio/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, `ngl 999`, `fa on`, KV `q8_0`/`q8_0`, `ctx-size 4096`, `decide-seqs 64` (`/v1/decide/info`: `decide_seqs 64`, `n_ctx 4096`, `recurrent true`, server default temperature 1.0). Data: `local/data/triage_test.jsonl` (80) and `triage_train.jsonl` (320). Every run except the preset check at T = 1 (`--ensure-t1`); the preset check runs at the preset's fitted T (Calibration below). Commands: `results-engine/3090/run.sh` (its header lists the environment).

## Evaluation (`eval/summary.md`, written by `results-engine/sp3/eval/analysis.py summary`)

Tree = this model's own baseline in `baseline/`; sign tests and Holm over this model's 12 tests only.

Scores, test set (80 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | tree (baseline) | 98.8% | 88.8% | 98.8% | 100.0% | 0.072 | 0.382 | 0.078 | 0.531 | 0.024 | 0.084 | 0.067 | 0.016 | 101.3 | 1 |
| qwen3.8-27b | full-path | 98.8% | 88.8% | 98.8% | 100.0% | 0.071 | 0.382 | 0.078 | 0.530 | 0.023 | 0.087 | 0.067 | 0.016 | 185.4 | 1 |
| qwen3.8-27b | after A (urgency after queue) | 98.8% | 86.2% | 98.8% | 100.0% | 0.071 | 0.411 | 0.077 | 0.560 | 0.024 | 0.079 | 0.067 | 0.016 | 106.3 | 1 |
| qwen3.8-27b | after B (urgency, angry after queue) | 98.8% | 86.2% | 98.8% | 98.8% | 0.071 | 0.416 | 0.069 | 0.556 | 0.023 | 0.079 | 0.047 | 0.015 | 123.5 | 1 |
| qwen3.8-27b | debias K = 4 | 98.8% | 88.8% | 98.8% | 100.0% | 0.073 | 0.372 | 0.079 | 0.525 | 0.025 | 0.098 | 0.068 | 0.017 | 309.8 | 2 |

Scores, train set (320 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | tree (baseline) | 96.6% | 85.3% | 99.7% | 100.0% | 0.089 | 0.352 | 0.043 | 0.484 | 0.021 | 0.035 | 0.039 | 0.006 | 101.4 | 1 |
| qwen3.8-27b | full-path | 96.9% | 85.6% | 99.7% | 100.0% | 0.089 | 0.350 | 0.043 | 0.482 | 0.019 | 0.038 | 0.039 | 0.006 | 185.9 | 1 |
| qwen3.8-27b | after A (urgency after queue) | 96.9% | 87.2% | 99.4% | 100.0% | 0.088 | 0.327 | 0.043 | 0.459 | 0.021 | 0.058 | 0.039 | 0.006 | 106.4 | 1 |
| qwen3.8-27b | after B (urgency, angry after queue) | 96.9% | 87.8% | 99.4% | 99.1% | 0.088 | 0.326 | 0.041 | 0.456 | 0.021 | 0.053 | 0.025 | 0.008 | 123.3 | 1 |
| qwen3.8-27b | debias K = 4 | 96.9% | 84.7% | 99.7% | 100.0% | 0.093 | 0.351 | 0.044 | 0.488 | 0.016 | 0.035 | 0.040 | 0.007 | 304.4 | 2 |

Scores, test and train pooled (400 tickets; `sp3_metrics.score_preds` on the 400 rows, so NLL is the mean over the 400 tickets and ECE is binned over them):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | tree (baseline) | 97.0% | 86.0% | 99.5% | 100.0% | 0.0852 | 0.3582 | 0.0500 | 0.4933 | 0.021 | 0.025 | 0.045 | 0.008 |
| qwen3.8-27b | full-path | 97.2% | 86.2% | 99.5% | 100.0% | 0.0852 | 0.3562 | 0.0501 | 0.4915 | 0.020 | 0.030 | 0.045 | 0.008 |
| qwen3.8-27b | after A (urgency after queue) | 97.2% | 87.0% | 99.2% | 100.0% | 0.0851 | 0.3439 | 0.0500 | 0.4790 | 0.021 | 0.059 | 0.045 | 0.008 |
| qwen3.8-27b | after B (urgency, angry after queue) | 97.2% | 87.5% | 99.2% | 99.0% | 0.0849 | 0.3443 | 0.0468 | 0.4760 | 0.022 | 0.054 | 0.030 | 0.010 |
| qwen3.8-27b | debias K = 4 | 97.2% | 85.5% | 99.5% | 100.0% | 0.0890 | 0.3556 | 0.0510 | 0.4956 | 0.016 | 0.028 | 0.046 | 0.009 |

Exact sign tests against tree on per-ticket correctness, test and train pooled (400 tickets). Cells: tickets right tree / configuration (discordant pairs: only tree right - only the configuration right, two-sided p; Holm = Holm-adjusted over all 12 tests of this table):

| model | configuration | queue | urgency | angry |
|---|---|---|---|---|
| qwen3.8-27b | full-path | 388 / 389 (0-1, p 1, Holm 1) | 344 / 345 (0-1, p 1, Holm 1) | 400 / 400 (0-0, p 1, Holm 1) |
| qwen3.8-27b | after A (urgency after queue) | 388 / 389 (0-1, p 1, Holm 1) | 344 / 348 (8-12, p 0.5, Holm 1) | 400 / 400 (0-0, p 1, Holm 1) |
| qwen3.8-27b | after B (urgency, angry after queue) | 388 / 389 (0-1, p 1, Holm 1) | 344 / 350 (7-13, p 0.26, Holm 1) | 400 / 396 (4-0, p 0.12, Holm 1) |
| qwen3.8-27b | debias K = 4 | 388 / 389 (0-1, p 1, Holm 1) | 344 / 342 (2-0, p 0.5, Holm 1) | 400 / 400 (0-0, p 1, Holm 1) |

Agreement with tree over the 400 tickets (`sp3_metrics.compare`, tree as the reference). Cells: median / max abs diff over options, argmax changes (of them where tree's top-2 margin > 0.15); decisions changed = all fields:

| model | configuration | queue | urgency | angry | decisions changed |
|---|---|---|---|---|---|
| qwen3.8-27b | full-path | 4.5e-06 / 0.028, 1 (0) | 0.00039 / 0.066, 1 (0) | 0.00021 / 0.018, 0 (0) | 2 of 1200 (0) |
| qwen3.8-27b | after A (urgency after queue) | 4.4e-06 / 0.03, 1 (0) | 0.0075 / 0.41, 20 (10) | 0.00029 / 0.02, 0 (0) | 21 of 1200 (10) |
| qwen3.8-27b | after B (urgency, angry after queue) | 4.5e-06 / 0.027, 1 (0) | 0.0075 / 0.42, 20 (10) | 0.0031 / 0.28, 4 (3) | 25 of 1200 (13) |
| qwen3.8-27b | debias K = 4 | 5.5e-05 / 0.15, 1 (0) | 0.00073 / 0.087, 2 (0) | 0.00058 / 0.031, 0 (0) | 3 of 1200 (0) |

Distribution of coverage (full-path) per field, test and train pooled (400 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | queue | 0.987856 | 0.998611 | 0.999601 | 0.999808 | 0.999897 | 0.999956 | 0.999982 | 0.999595 |
| qwen3.8-27b | urgency | 0.998200 | 0.999257 | 0.999566 | 0.999701 | 0.999781 | 0.999865 | 0.999945 | 0.999644 |
| qwen3.8-27b | angry | 0.987040 | 0.992957 | 0.995302 | 0.996858 | 0.997807 | 0.998593 | 0.999612 | 0.996385 |

Distribution of order_spread (debias K = 4) per field, test and train pooled (400 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | queue | 9.485e-08 | 1.362e-06 | 0.0001303 | 0.002296 | 0.01355 | 0.1258 | 0.2849 | 0.02009 |
| qwen3.8-27b | urgency | 0.0003272 | 0.001142 | 0.003313 | 0.009331 | 0.02593 | 0.07138 | 0.2012 | 0.02011 |
| qwen3.8-27b | angry | 3.635e-05 | 0.0001375 | 0.000412 | 0.001757 | 0.006596 | 0.01746 | 0.1151 | 0.004937 |

Cost on the train set (64 requests of 5 states, `--decide-seqs 64`; tree = the tree runs in `results-engine/3090/baseline`). Per request medians in ms; snapshots = prefix cache entries and MB after the warm-up:

| model | configuration | p50 ms / ticket | tickets / s | vs tree | total | prefill | prefix | scoring | decodes | rounds | seqs per state | states per round | cache hits / misses | snapshots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b | tree (baseline) | 101.4 | 9.9 | 1.00x | 505.2 | 343.1 | 0.3 | 160.9 | 2 | 1 | 4 | 15 | 0 / 0 | 0, 0.0 |
| qwen3.8-27b | full-path | 185.9 | 5.4 | 0.54x | 927.6 | 346.5 | 0.3 | 580.8 | 3 | 1 | 11 | 5 | 0 / 0 | 0, 0.0 |
| qwen3.8-27b | after A (urgency after queue) | 106.4 | 9.5 | 0.96x | 529.9 | 348.7 | 0.3 | 177.4 | 3 | 1 | 4 | 15 | 0 / 0 | 0, 0.0 |
| qwen3.8-27b | after B (urgency, angry after queue) | 123.3 | 8.1 | 0.82x | 614.5 | 346.2 | 0.3 | 267.1 | 3 | 1 | 4 | 15 | 0 / 0 | 0, 0.0 |
| qwen3.8-27b | debias K = 4 | 304.4 | 3.3 | 0.33x | 1519.8 | 949.5 | 50.7 | 563.0 | 4 | 2 | 16 | 3 | 192 / 0 | 3, 517.5 |

Checks (engine blocks, answer keys, given): 0 problems.

## Calibration (`eval/calibration-qwen3.8-27b.json`, `calibrate.py`, spec 8)

Fitted on the train baseline: T = 0.8992 (grid best 0.8958, refined 0.899247).

| set | T | nll_queue | nll_urgency | nll_angry | nll_sum | ece_queue | ece_urgency | ece_angry | brier_angry |
|---|---|---|---|---|---|---|---|---|---|
| train | 1.0 | 0.0886 | 0.3523 | 0.0430 | 0.4839 | 0.0209 | 0.0346 | 0.0391 | 0.0064 |
| train | 0.8992 | 0.0875 | 0.3572 | 0.0352 | 0.4800 | 0.0242 | 0.0473 | 0.0320 | 0.0052 |
| test | 1.0 | 0.0716 | 0.3817 | 0.0778 | 0.5312 | 0.0237 | 0.0843 | 0.0671 | 0.0165 |
| test | 0.8992 | 0.0690 | 0.3898 | 0.0685 | 0.5273 | 0.0246 | 0.0742 | 0.0590 | 0.0144 |

Rule: the fitted T goes into the model's preset in models-engine.ini as decide-temperature only if it lowers the test-set summed NLL. Test summed NLL 0.5312 -> 0.5273: apply.

On this machine the presets file is `models-engine-3090.ini`: `[qwen3.8-27b]` holds `decide-temperature = 0.8992` (added after every run above).

Preset check (`eval/preset/`): the live test run at the preset's T against the baseline test preds scaled post hoc: max abs diff 2.22e-16, within 1e-9: True.

## Integrity (`integrity/`)

`reference_check.py` on the first 20 test tickets, one state per line, `--decide-seqs 64`; dump with `llama-decide` (server stopped), replay through the router's `/completion` (n_probs 512; the replays are kept as `*-replay.jsonl.gz`, so `compare --offline` recomputes them). Statistic: per field median abs diff <= 0.01 and no argmax disagreement where the reference's top-2 margin > 0.15.

| check | field | median abs diff | max abs diff | argmax agreement | disagreements over margin | pass |
|---|---|---|---|---|---|---|
| reference tree | queue | 3.6e-05 | 0.0130 | 1 | 0 | PASS |
| reference tree | urgency | 0.0007 | 0.0166 | 1 | 0 | PASS |
| reference tree | angry | 0.0009 | 0.0217 | 1 | 0 | PASS |
| reference fp | queue | 2.3e-05 | 0.0224 | 1 | 0 | PASS |
| reference fp | urgency | 0.0008 | 0.0149 | 0.95 | 0 | PASS |
| reference fp | angry | 0.0010 | 0.0114 | 1 | 0 | PASS |
| batch b1 vs b5 (n 20) | queue | 4.3e-05 | 0.0085 | 1 | 0 | PASS |
| batch b1 vs b5 (n 20) | urgency | 0.0007 | 0.0208 | 1 | 0 | PASS |
| batch b1 vs b5 (n 20) | angry | 0.0014 | 0.0196 | 1 | 0 | PASS |

Overall: reference tree PASS; reference full-path PASS (statistic, coverage and node comparison per field, unverifiable edges); batch invariance PASS.

Full-path detail:

| field | engine coverage median | reference coverage median | coverage median abs diff | node median / max abs diff (values) | pass |
|---|---|---|---|---|---|
| queue | 0.999747 | 0.999778 | 1.2e-05 (PASS) | 2.8e-05 / 0.0719 (320) PASS | PASS |
| urgency | 0.999363 | 0.999280 | 6.5e-05 (PASS) | 0.0002 / 0.0149 (160) PASS | PASS |
| angry | 0.998521 | 0.998558 | 0.0002 (PASS) | 0.0001 / 0.0115 (80) PASS | PASS |

Unverifiable edges: 0 of 280 (0.00%; bound 5 %, each within 2x the list's smallest probability: True). Coverage gate (median engine coverage >= 0.8): pass.

Recorded `/completion` requests: tree 60, full-path 340. Offline recomputation (`integrity/offline-recompute.txt`): tree identical fp identical

## Latency and cost

Tree, test set (80 tickets), per ticket: wall time of a request / its tickets (p50, p90), server time (timings.total_ms / tickets); VRAM in use on the GPU after the run (desktop about 612 MiB included).

| batch | run | p50 ms | p90 ms | server p50 ms | rounds per request | VRAM after run MiB |
|---|---|---|---|---|---|---|
| 1 | `latency/runs/decide-qwen3.8-27b-b1-keywords.json` | 142.7 | 155.7 | 141.1 | 1 | 22332 |
| 5 | `baseline/test/runs/decide-qwen3.8-27b-b5-keywords.json` | 102.9 | 108.1 | 102.5 | 1 | 22316 |

Per mode on the train set (64 requests of 5 states, `--decide-seqs 64`; medians per request; from `eval/summary.json`):

| mode | p50 ms / ticket | tickets / s | vs tree | total ms | decodes | rounds | seqs per state | states per round | VRAM after run MiB |
|---|---|---|---|---|---|---|---|---|---|
| tree | 101.4 | 9.9 | 1.00x | 505.2 | 2 | 1 | 4 | 15 | 22316 |
| full-path | 185.9 | 5.4 | 0.54x | 927.6 | 3 | 1 | 11 | 5 | 22316 |
| after-a | 106.4 | 9.5 | 0.96x | 529.9 | 3 | 1 | 4 | 15 | 22316 |
| after-b | 123.3 | 8.1 | 0.82x | 614.5 | 3 | 1 | 4 | 15 | 22316 |
| debias4 | 304.4 | 3.3 | 0.33x | 1519.8 | 4 | 2 | 16 | 3 | 22318 |

## /v1/systemone

`systemone_smoke.py --model qwen3.8-27b` (`systemone/smoke.txt`), last line: `OK: identical probabilities through both endpoints (max abs diff 0); request id header and depth-40 422 checked`

Official SDK test, `SYSTEMONE_MODEL=qwen3.8-27b` (`systemone/sdk.txt`), last line: `2 passed in 0.62s`
