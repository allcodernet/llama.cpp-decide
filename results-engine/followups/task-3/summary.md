Scores, test set (10 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | tree (baseline) | 90.0% | 90.0% | 100.0% | 100.0% | 0.448 | 0.107 | 0.014 | 0.569 | 0.136 | 0.081 | 0.013 | 0.002 | 13.8 | 1 |
| gemma-4-e4b | full-path | 90.0% | 90.0% | 100.0% | 100.0% | 0.410 | 0.117 | 0.013 | 0.540 | 0.131 | 0.086 | 0.012 | 0.001 | 43.0 | 5 |
| gemma-4-e4b | after A (urgency after queue) | 90.0% | 100.0% | 100.0% | 100.0% | 0.438 | 0.069 | 0.015 | 0.521 | 0.136 | 0.060 | 0.014 | 0.002 | 16.8 | 1 |
| gemma-4-e4b | after B (urgency, angry after queue) | 90.0% | 100.0% | 100.0% | 100.0% | 0.440 | 0.071 | 0.008 | 0.519 | 0.136 | 0.062 | 0.008 | 0.001 | 18.3 | 1 |
| gemma-4-e4b | debias K = 4 | 80.0% | 100.0% | 100.0% | 100.0% | 0.489 | 0.094 | 0.015 | 0.598 | 0.152 | 0.076 | 0.014 | 0.002 | 58.7 | 5 |

Scores, train set (10 tickets, b5, prompt `keywords`, T = 1):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a | p50 ms | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | tree (baseline) | 100.0% | 70.0% | 100.0% | 100.0% | 0.001 | 1.048 | 0.003 | 1.051 | 0.001 | 0.295 | 0.003 | 0.000 | 14.5 | 1 |
| gemma-4-e4b | full-path | 100.0% | 70.0% | 100.0% | 100.0% | 0.001 | 1.064 | 0.003 | 1.067 | 0.001 | 0.296 | 0.003 | 0.000 | 42.8 | 5 |
| gemma-4-e4b | after A (urgency after queue) | 100.0% | 70.0% | 100.0% | 100.0% | 0.001 | 1.007 | 0.003 | 1.011 | 0.001 | 0.250 | 0.003 | 0.000 | 17.8 | 1 |
| gemma-4-e4b | after B (urgency, angry after queue) | 100.0% | 70.0% | 100.0% | 100.0% | 0.001 | 1.017 | 0.002 | 1.020 | 0.001 | 0.251 | 0.002 | 0.000 | 19.6 | 1 |
| gemma-4-e4b | debias K = 4 | 100.0% | 70.0% | 100.0% | 100.0% | 0.000 | 1.023 | 0.003 | 1.026 | 0.000 | 0.241 | 0.003 | 0.000 | 63.2 | 5 |

Scores, test and train pooled (20 tickets; `sp3_metrics.score_preds` on the 20 rows, so NLL is the mean over the 20 tickets and ECE is binned over them):

| model | configuration | queue | urgency | urg±1 | angry | NLL q | NLL u | NLL a | NLL Σ | ECE q | ECE u | ECE a | Brier a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | tree (baseline) | 95.0% | 80.0% | 100.0% | 100.0% | 0.2241 | 0.5776 | 0.0086 | 0.8102 | 0.068 | 0.179 | 0.008 | 0.001 |
| gemma-4-e4b | full-path | 95.0% | 80.0% | 100.0% | 100.0% | 0.2051 | 0.5907 | 0.0079 | 0.8037 | 0.065 | 0.182 | 0.008 | 0.001 |
| gemma-4-e4b | after A (urgency after queue) | 95.0% | 85.0% | 100.0% | 100.0% | 0.2192 | 0.5379 | 0.0088 | 0.7659 | 0.068 | 0.126 | 0.008 | 0.001 |
| gemma-4-e4b | after B (urgency, angry after queue) | 95.0% | 85.0% | 100.0% | 100.0% | 0.2200 | 0.5440 | 0.0053 | 0.7693 | 0.067 | 0.128 | 0.005 | 0.000 |
| gemma-4-e4b | debias K = 4 | 90.0% | 85.0% | 100.0% | 100.0% | 0.2448 | 0.5585 | 0.0090 | 0.8123 | 0.076 | 0.130 | 0.008 | 0.001 |

Exact sign tests against tree on per-ticket correctness, test and train pooled (20 tickets). Cells: tickets right tree / configuration (discordant pairs: only tree right - only the configuration right, two-sided p; Holm = Holm-adjusted over all 12 tests of this table):

| model | configuration | queue | urgency | angry |
|---|---|---|---|---|
| gemma-4-e4b | full-path | 19 / 19 (0-0, p 1, Holm 1) | 16 / 16 (0-0, p 1, Holm 1) | 20 / 20 (0-0, p 1, Holm 1) |
| gemma-4-e4b | after A (urgency after queue) | 19 / 19 (0-0, p 1, Holm 1) | 16 / 17 (0-1, p 1, Holm 1) | 20 / 20 (0-0, p 1, Holm 1) |
| gemma-4-e4b | after B (urgency, angry after queue) | 19 / 19 (0-0, p 1, Holm 1) | 16 / 17 (0-1, p 1, Holm 1) | 20 / 20 (0-0, p 1, Holm 1) |
| gemma-4-e4b | debias K = 4 | 19 / 18 (1-0, p 1, Holm 1) | 16 / 17 (0-1, p 1, Holm 1) | 20 / 20 (0-0, p 1, Holm 1) |

Agreement with tree over the 20 tickets (`sp3_metrics.compare`, tree as the reference). Cells: median / max abs diff over options, argmax changes (of them where tree's top-2 margin > 0.15); decisions changed = all fields:

| model | configuration | queue | urgency | angry | decisions changed |
|---|---|---|---|---|---|
| gemma-4-e4b | full-path | 3e-06 / 0.043, 0 (0) | 7.7e-05 / 0.034, 0 (0) | 1.2e-05 / 0.011, 0 (0) | 0 of 60 (0) |
| gemma-4-e4b | after A (urgency after queue) | 7e-06 / 0.0097, 0 (0) | 0.00066 / 0.2, 1 (0) | 2.4e-05 / 0.0048, 0 (0) | 1 of 60 (0) |
| gemma-4-e4b | after B (urgency, angry after queue) | 7.7e-06 / 0.0039, 0 (0) | 0.00085 / 0.17, 1 (0) | 7.7e-05 / 0.052, 0 (0) | 1 of 60 (0) |
| gemma-4-e4b | debias K = 4 | 1.7e-05 / 0.14, 1 (0) | 0.00011 / 0.047, 1 (0) | 3.2e-05 / 0.0069, 0 (0) | 2 of 60 (0) |

Distribution of coverage (full-path) per field, test and train pooled (20 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | queue | 0.998326 | 0.999370 | 0.999909 | 0.999963 | 0.999987 | 0.999995 | 0.999997 | 0.999830 |
| gemma-4-e4b | urgency | 0.999996 | 0.999997 | 0.999998 | 0.999999 | 0.999999 | 1.000000 | 1.000000 | 0.999999 |
| gemma-4-e4b | angry | 0.997277 | 0.998759 | 0.999273 | 0.999725 | 0.999884 | 0.999919 | 0.999926 | 0.999460 |

Distribution of order_spread (debias K = 4) per field, test and train pooled (20 tickets):

| model | field | min | p5 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | queue | 6.592e-07 | 1.367e-06 | 5.502e-05 | 0.0004507 | 0.004044 | 0.06468 | 0.5178 | 0.03114 |
| gemma-4-e4b | urgency | 0.0001452 | 0.0001818 | 0.0009067 | 0.006753 | 0.02949 | 0.1589 | 0.1673 | 0.03332 |
| gemma-4-e4b | angry | 3.987e-06 | 9.342e-06 | 1.799e-05 | 0.0001599 | 0.0004141 | 0.004341 | 0.02073 | 0.001519 |

Cost on the train set (2 requests of 5 states, `--decide-seqs 21`; tree = the tree runs in `/tmp/t3-smoke/baseline`). Per request medians in ms; snapshots = prefix cache entries and MB after the warm-up:

| model | configuration | p50 ms / ticket | tickets / s | vs tree | total | prefill | prefix | scoring | decodes | rounds | seqs per state | states per round | cache hits / misses | snapshots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e4b | tree (baseline) | 14.5 | 69.2 | 1.00x | 70.6 | 43.6 | 0.3 | 23.6 | 2 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | full-path | 42.8 | 23.4 | 0.34x | 211.5 | 87.2 | 0.2 | 119.5 | 10 | 5 | 11 | 1 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | after A (urgency after queue) | 17.8 | 56.2 | 0.81x | 87.2 | 42.3 | 0.3 | 41.5 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | after B (urgency, angry after queue) | 19.6 | 51.0 | 0.74x | 96.2 | 45.1 | 0.3 | 47.7 | 3 | 1 | 4 | 5 | 0 / 0 | 0, 0.0 |
| gemma-4-e4b | debias K = 4 | 63.2 | 15.8 | 0.23x | 313.5 | 195.7 | 5.9 | 108.9 | 10 | 5 | 16 | 1 | 6 / 0 | 3, 40.1 |

Checks (engine blocks, answer keys, given): 0 problems.
