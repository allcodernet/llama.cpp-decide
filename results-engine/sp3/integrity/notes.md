# Sub-project 3 Task 8: integrity runs (spec 12.2, 12.3)

Engine `5e9e030d3 decide: clamp free sequences in the budget message; cell message helper`: llama-decide and llama-server built from its sources (their build
info, `--version`, names the parent `8daa120b5`: they were linked 15 s before the commit, with the
same sources). Run parameters per output file: `run-params.json`. Tools: `reference_check.py` (dump, compare, split;
compare keeps every request with its answer in `reference-<tag>-<preset>-replay.jsonl.gz`, and `--offline` recomputes a
result from that file alone), `cache_timing.py`, and in this directory `checks.py` (the extra `after` + debias check,
`rowcap-rows`, `chunk-answers`, `request-identity`, `offline-check`, `run-params`, `notes`) and `run.sh` (steps dumps, replays, split,
cache, debias, regression, in that order, then the diagnostic step nosplit). Requests: the triage schema of `decide_client.build_request` (prompt variant `default`) on
the first 20 test tickets; b1 = one state per request line, b5 = five. KV cache as the presets of `models-engine.ini`
(f16 for qwen2.5-0.5b, q8_0 otherwise); dumps with `llama-decide`, the server stopped; replays through the preset
router (`serve-engine.sh`, port 8097), `/completion` with `n_probs` 512 and `cache_prompt` off. "The statistic" = per
field median abs diff over options <= 0.01 and no argmax disagreement where the reference's top-2 margin > 0.15
(`sp3_metrics.compare`, with the replay or the one-state run as the reference). Cell format below: median / max /
disagreements over the margin; nodes: median / max (values compared).

## 1. Full-path reference, b1 (spec 12.3)

| model (`--decide-seqs`) | per field | unverifiable edges | pass | file |
|---|---|---|---|---|
| qwen2.5-0.5b (64) | queue: statistic 0.0047 / 0.0336 / 0 PASS; coverage 0.0004 PASS; nodes 3.7e-05 / 0.0610 (320) PASS<br>urgency: statistic 0.0061 / 0.0498 / 0 PASS; coverage 1.6e-05 PASS; nodes 4.7e-05 / 0.0498 (160) PASS<br>angry: statistic 0.0144 / 0.0971 / 0 FAIL; coverage 0.0003 PASS; nodes 0.0003 / 0.0971 (80) PASS | 0 of 280 | FAIL | `reference-fp-qwen2.5-0.5b.json` |
| qwen3.5-4b (32) | queue: statistic 0.0010 / 0.0635 / 0 PASS; coverage 4.2e-05 PASS; nodes 0.0003 / 0.1228 (320) PASS<br>urgency: statistic 0.0047 / 0.0469 / 0 PASS; coverage 1.8e-05 PASS; nodes 0.0001 / 0.0468 (160) PASS<br>angry: statistic 0.0038 / 0.0465 / 0 PASS; coverage 0.0005 PASS; nodes 6.1e-05 / 0.0468 (80) PASS | 0 of 280 | PASS | `reference-fp-qwen3.5-4b.json` |
| gemma-4-e4b (64) | queue: statistic 1.5e-05 / 0.0529 / 0 PASS; coverage 1.3e-05 PASS; nodes 4.2e-06 / 0.0530 (320) PASS<br>urgency: statistic 0.0001 / 0.0173 / 0 PASS; coverage 2.3e-07 PASS; nodes 3.1e-06 / 0.0173 (160) PASS<br>angry: statistic 0.0004 / 0.0564 / 0 PASS; coverage 0.0002 PASS; nodes 2.2e-05 / 0.0564 (80) PASS | 0 of 280 | PASS | `reference-fp-gemma-4-e4b.json` |
| qwen3.5-9b (21) | queue: statistic 0.0002 / 0.0644 / 0 PASS; coverage 1.9e-05 PASS; nodes 4.9e-05 / 0.0645 (320) PASS<br>urgency: statistic 0.0037 / 0.0590 / 0 PASS; coverage 8.1e-05 PASS; nodes 0.0004 / 0.0590 (160) PASS<br>angry: statistic 0.0082 / 0.0404 / 0 PASS; coverage 0.0003 PASS; nodes 0.0002 / 0.0404 (80) PASS | 0 of 280 | PASS | `reference-fp-qwen3.5-9b.json` |

Qwen3.5-4B, Gemma 4 E4B and Qwen3.5-9B are binding. Qwen2.5-0.5B beside its tree-mode reference on the same 20
states at 64 sequences (`reference-tree-b1-qwen2.5-0.5b.json`, the tree replay of `reference_check.py`, renormalised
children): queue full-path 0.0047 vs tree 0.0044, urgency full-path 0.0061 vs tree 0.0057, angry full-path 0.0144 vs tree 0.0094 (tree `pass` True). Its known `angry` batch noise (sub-project 1, `results-engine/batch-invariance-qwen2.5-0.5b.json`: 0.0117 b5 vs b1) is of the size of both.

Coverage sanity gate (median engine coverage per field >= 0.8): qwen3.5-9b 0.9997 / 0.9994 / 0.9975 PASS; gemma-4-e4b 0.9999 / 1.0000 / 0.9990 PASS (queue / urgency / angry).

Unverifiable edges (a path token absent from the replay's top-512): none on any model or run (`unverifiable` in every
`reference-*.json`), so the forced-token check reserved for that case was not needed. Engine rows recomputed with spec 4.1
reproduce the answers within 1.1e-16 (`engine_recompute_max_abs_diff`).

The largest node differences: qwen2.5-0.5b 0.0971 (angry children); qwen3.5-4b 0.1228 (queue end); gemma-4-e4b 0.0564 (angry children); qwen3.5-9b 0.0645 (queue children).
The largest END differences (in parentheses: the field; the path token into the row; that token's engine
probability at the row before): qwen2.5-0.5b 0.0610 (queue; `"`; 1.9e-08); qwen3.5-4b 0.1228 (queue; `"`; 4.3e-04); gemma-4-e4b 0.0060 (queue; `"`; 1.8e-05); qwen3.5-9b 0.0137 (queue; `"`; 3.6e-05).
These rows follow the lone closing quote of a queue value, a non-canonical continuation (the models write the
merged `",`), and enter P(o) only multiplied by that token's probability: with every replayed END value in place
of the engine's (spec 4.1 on the engine's rows otherwise) no option probability moves by more than 9.6e-08 (qwen2.5-0.5b), 6.3e-06 (qwen3.5-4b), 8.2e-07 (gemma-4-e4b), 1.2e-04 (qwen3.5-9b). END/MERGED pairs that are -inf on
both sides (digits and ` true`/` false` have no MERGED tokens) are counted, not compared: 120 per 20-state run.

## 2. Extra: b5 references, rows after the row-cap chunk boundary

| model | per field | chunk rows (0 / 1) | nodes chunk 0 | nodes chunk 1 | sequences cut | pass | files |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | queue: statistic 0.0046 / 0.0428 / 0 PASS; coverage 0.0003 PASS; nodes 4.7e-05 / 0.0626 (320) PASS<br>urgency: statistic 0.0052 / 0.0380 / 0 PASS; coverage 2.1e-05 PASS; nodes 2.6e-05 / 0.0379 (160) PASS<br>angry: statistic 0.0076 / 0.0626 / 0 PASS; coverage 0.0004 PASS; nodes 0.0006 / 0.0625 (80) PASS | 260 / 80 | 4.7e-05 / 0.0626 (432) | 3.4e-05 / 0.0595 (128) | 0 | PASS | `reference-fp-b5-qwen2.5-0.5b.json`, `rowcap-rows-fp-b5-qwen2.5-0.5b.json` |
| gemma-4-e4b | queue: statistic 1.3e-05 / 0.0726 / 0 PASS; coverage 7.1e-06 PASS; nodes 4.6e-06 / 0.0726 (320) PASS<br>urgency: statistic 0.0001 / 0.0121 / 0 PASS; coverage 2.2e-07 PASS; nodes 3.5e-06 / 0.0121 (160) PASS<br>angry: statistic 0.0003 / 0.0615 / 0 PASS; coverage 0.0003 PASS; nodes 8.9e-06 / 0.0619 (80) PASS | 260 / 80 | 4.4e-06 / 0.0455 (432) | 7.4e-06 / 0.0726 (128) | 0 | PASS | `reference-fp-b5-gemma-4-e4b.json`, `rowcap-rows-fp-b5-gemma-4-e4b.json` |

Row cap 65 (`--decide-seqs 64` + the slot sequence), 17 rows per state, 85 per
5-state round: rows 66-85 of each round were read from the second branch decode with
chunk-relative batch indices (two branch decodes per b5 line in `split-rowcap-*.json`). Fields with rows in both
chunks: 0, so the boundary falls between two fields and cuts no sequence (Task 5 cut sequences mid-way at 60
and 68). Tree b5 beside the 0.5B: queue 0.0035, urgency 0.0050, angry 0.0110 (`reference-tree-b5-qwen2.5-0.5b.json`).

Answers by the chunk that read their field's rows (`chunk-answers-<preset>.json`; per field the max abs diff over
options, median / max over the chunk's fields). b5 split and b1: `reference_check.py split` at 64 sequences (branch decodes per b5 line [2, 2, 2, 2]); b5 unsplit and its b1: the same at 84 sequences, where a round's rows fit one chunk (branch decodes [1, 1, 1, 1]; `split-s84-*.json`, a diagnostic run after the regression on the same build). The dumped b5 and b1 answers equal
the split run's (max abs diff 0.0000 qwen2.5-0.5b, 0.0000 gemma-4-e4b).

| model | chunk (fields) | b5 vs replay | b1 vs replay | b5 vs b1 | b5 split vs b5 unsplit | b5 unsplit vs b1 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0 (44) | 0.0143 / 0.0626 | 0.0145 / 0.0971 | 0.0108 / 0.0635 | 0.0067 / 0.0324 | 0.0105 / 0.0635 |
| qwen2.5-0.5b | 1 (16) | 0.0098 / 0.0380 | 0.0143 / 0.0579 | 0.0111 / 0.0652 | 0.0077 / 0.0277 | 0.0083 / 0.0526 |
| gemma-4-e4b | 0 (44) | 0.0011 / 0.0417 | 0.0009 / 0.0529 | 0.0003 / 0.0235 | 0.0003 / 0.0192 | 0.0005 / 0.0313 |
| gemma-4-e4b | 1 (16) | 0.0005 / 0.0726 | 0.0004 / 0.0564 | 0.0004 / 0.0459 | 0.0002 / 0.0189 | 0.0003 / 0.0608 |

Between the split and the unsplit run, chunk-1 fields move about as much as chunk-0 fields (medians qwen2.5-0.5b 0.0077 vs 0.0067; gemma-4-e4b 0.0002 vs 0.0003). The Gemma fields farthest from their replay (rows in chunk 1) are about as far without the split: ticket 4 `queue` 0.0726 split, 0.0914 unsplit, 0.0306 b1; ticket 18 `angry` 0.0615 split, 0.0514 unsplit, 0.0156 b1; ticket 14 `angry` 0.0596 split, 0.0412 unsplit, 0.0564 b1.

## 3. After reference: full-path, urgency and angry after queue, b1 (spec 12.3)

| model (`--decide-seqs`) | per field | unverifiable | given/lines | pass | file |
|---|---|---|---|---|---|
| qwen2.5-0.5b (64) | queue: statistic 0.0047 / 0.0336 / 0 PASS; coverage 0.0004 PASS; nodes 3.7e-05 / 0.0610 (320) PASS<br>urgency: statistic 0.0037 / 0.0844 / 0 PASS; coverage 3.3e-05 PASS; nodes 4.6e-05 / 0.0844 (160) PASS<br>angry: statistic 0.0126 / 0.0952 / 0 FAIL; coverage 4.1e-05 PASS; nodes 0.0010 / 0.0953 (80) PASS | 0 of 280 | 40 records, 120 entries, 0 problems | FAIL | `reference-fp-after-qwen2.5-0.5b.json` |
| qwen3.5-4b (32) | queue: statistic 0.0010 / 0.0681 / 0 PASS; coverage 6.7e-05 PASS; nodes 0.0003 / 0.1545 (320) PASS<br>urgency: statistic 0.0019 / 0.0605 / 0 PASS; coverage 2.7e-06 PASS; nodes 7.9e-06 / 0.0605 (160) PASS<br>angry: statistic 0.0022 / 0.0589 / 0 PASS; coverage 0.0032 PASS; nodes 0.0002 / 0.0589 (80) PASS | 0 of 280 | 40 records, 120 entries, 0 problems | PASS | `reference-fp-after-qwen3.5-4b.json` |

The dependent entries' tokens hold the lines, so their replay prompts do; every dependent answer and entry has
`given` = the queue value the state returned and `lines` = `  "queue": <value>,\n` built from it. Qwen3.5-4B is binding.
The 4B's largest END difference (as in section 1): 0.1545 (queue; `"`; 5.0e-04); the replayed END values move no option probability by more than 1.6e-05.

## 4. Row-cap split: full-path b5 vs b1 at 64 sequences, tree b5 vs b1 beside (spec 12.3)

| model (`--decide-seqs`) | branch decodes per b5 line | b1 lines unsplit | queue: full-path / tree | urgency: full-path / tree | angry: full-path / tree | pass | file |
|---|---|---|---|---|---|---|---|
| gemma-4-e4b (64) | [2, 2, 2, 2] (rounds [1, 1, 1, 1], rows 85) | True | 1.1e-05 PASS / 1.4e-05 PASS | 7.5e-05 PASS / 0.0001 PASS | 0.0003 PASS / 0.0005 PASS | PASS | `split-rowcap-gemma-4-e4b.json` |
| gemma-4-e4b (84) | [1, 1, 1, 1] (rounds [1, 1, 1, 1], rows 85) | True | 1.1e-05 PASS / 1.4e-05 PASS | 5.7e-05 PASS / 0.0001 PASS | 0.0004 PASS / 0.0005 PASS | diagnostic (no split) | `split-s84-gemma-4-e4b.json` |
| qwen2.5-0.5b (64) | [2, 2, 2, 2] (rounds [1, 1, 1, 1], rows 85) | True | 0.0041 PASS / 0.0031 PASS | 0.0032 PASS / 0.0038 PASS | 0.0125 FAIL / 0.0117 FAIL | FAIL | `split-rowcap-qwen2.5-0.5b.json` |
| qwen2.5-0.5b (84) | [1, 1, 1, 1] (rounds [1, 1, 1, 1], rows 85) | True | 0.0049 PASS / 0.0031 PASS | 0.0033 PASS / 0.0038 PASS | 0.0104 FAIL / 0.0117 FAIL | diagnostic (no split) | `split-s84-qwen2.5-0.5b.json` |

Gemma 4 E4B is binding. The rows at 84 sequences are the diagnostic of section 2 (a round's rows in one chunk).
The 0.5B's full-path `angry` median is 0.0125 with the split and 0.0104 without it, beside tree mode's 0.0117 and 0.0117 (no split at either count): the batch-composition effect, not the split.

## 5. Prefix cache timing: 20 A/B alternations (spec 12.3)

| model | schema (prompt variant) | `--decide-prefix-cache` | prefill_ms | prefix_ms | total_ms | hits / misses | file |
|---|---|---|---|---|---|---|---|
| qwen3.5-9b | A (`keywords`) | 0 | 995.9 | 564.8 | 1251.5 | 0 / 0 | `cache-timing-qwen3.5-9b.json` |
| qwen3.5-9b | A (`keywords`) | 8 | 439.0 | 7.8 | 692.3 | 19 / 1 | `cache-timing-qwen3.5-9b.json` |
| qwen3.5-9b | B (`default`) | 0 | 746.1 | 315.9 | 1000.6 | 0 / 0 | `cache-timing-qwen3.5-9b.json` |
| qwen3.5-9b | B (`default`) | 8 | 437.4 | 7.3 | 690.4 | 19 / 1 | `cache-timing-qwen3.5-9b.json` |
| gemma-4-e4b | A (`keywords`) | 0 | 270.5 | 171.9 | 325.5 | 0 / 0 | `cache-timing-gemma-4-e4b.json` |
| gemma-4-e4b | A (`keywords`) | 8 | 97.9 | 2.6 | 150.3 | 19 / 1 | `cache-timing-gemma-4-e4b.json` |
| gemma-4-e4b | B (`default`) | 0 | 198.7 | 102.1 | 251.5 | 0 / 0 | `cache-timing-gemma-4-e4b.json` |
| gemma-4-e4b | B (`default`) | 8 | 96.4 | 1.7 | 149.2 | 19 / 1 | `cache-timing-gemma-4-e4b.json` |

Medians per schema over its 20 requests of a run (A and B alternate; 5 states each, `--decide-seqs` 21, `llama-decide`). The two prefixes differ in
length, so the schemas' timings form two groups: a median over both, as first reported, describes neither.
With the cache the first A and B miss and every later request restores its prefix (after snapshotting the other one).

## 6. Extra (Task 7): `after` + order_debias 2 with more than one state per round

| run | states per round, rounds per b5 line | b5 vs b1 medians (q / u / a) | statistic | given/lines entries (b5, b1) | recomputation | pass | file |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B tree, `--decide-seqs 32` (binding) | 3, [2, 2, 2, 2] | 0.0004 / 0.0024 / 0.0019 | PASS | 80, 80 (0 problems) | n/a (a tree dump has no per-variant values) | PASS | `debias-tree-qwen3.5-4b.json` |
| Qwen2.5-0.5B full-path, `--decide-seqs 64` | 2, [3, 3, 3, 3] | 0.0037 / 0.0009 / 0.0082 | PASS | 240, 240 (0 problems) | 60 + 60 answers, max 2.2e-16; queue value != variant 0's argmax in 6 states | PASS | `debias-fp-qwen2.5-0.5b.json` |

Full-path K = 2 needs 22 sequences per state (`seqs_per_state` of the 0.5B run), one state per round at 32 sequences on the 4B; the averages, order_spread and coverage are therefore recomputed on the 0.5B at 64 sequences (2 states per round), and the 4B binding check runs in tree mode (8 sequences per state, 3 states per round).

## 7. Regression (spec 12.2), the last planned run, each on a freshly started router

`decide_client.py` at `826110a` (the commit used); its requests without switches are byte-identical to those of `f972d7e` (the Task 0 baseline commit): 164 request bodies and the output names, `identical` True (`regression-request-identity.json`).

| run | max abs diff vs the Task 0 baseline | within 1e-6 | file |
|---|---|---|---|
| qwen3.5-9b test (b5, keywords) | 0 | True | `regression-test-qwen3.5-9b.json` |
| gemma-4-e4b test (b5, keywords) | 0 | True | `regression-test-gemma-4-e4b.json` |
| qwen3.5-9b train (b5, keywords) | 0 | True | `regression-train-qwen3.5-9b.json` |
| gemma-4-e4b train (b5, keywords) | 0 | True | `regression-train-gemma-4-e4b.json` |
| qwen2.5-0.5b tree dump (`reference_check.py dump`, 20 states, `--decide-seqs` 9) | 0 | True | `regression-dump-qwen2.5-0.5b.json` |

The regression was the last planned run; the no-split diagnostic of section 2 (`split-s84-*`, `llama-decide` only, no server) followed it on the same build.

## 8. Offline recomputation

Every replay request and answer is saved (`reference-*-replay.jsonl.gz`: the prompt token ids and parameters, the
generated token, the listed ids and log-probs, the listed pieces); `reference_check.py compare --offline` recomputes each
result from its dump and replay alone: 10 of 10 results byte-identical to the committed ones (`offline-recompute.json`).

## 9. Run parameters and GPU log

`run-params.json`: per output file (85) the `run.sh` step, runner, preset, model, `--decide-seqs`, b, mode and command, reconstructed from `run.sh` by `checks.py run-params` and checked against the outputs that record their own parameters (the dumps' `--decide-seqs` and b are recorded by none: see its `notes`).

`gpu-log.txt`: VRAM and temperature at each step start and model load as `run.sh` wrote them; its last two lines, marked `(diagnostic split s84, ...)`, were written by hand.
