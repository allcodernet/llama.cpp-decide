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

Checks (engine blocks, answer keys, given): 0 problems.
