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
