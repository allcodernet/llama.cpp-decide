# Follow-up Task 1: faster full-path rows

Engine commit `387b4848d` (`decide: faster full-vocabulary log-sum-exp for full-path rows`, parent 63ea2c51a).
`full_path_row` takes its normaliser from `full_vocab_lse`: the code of before as the scalar fallback; on x86-64 (GCC or
Clang) an SSE2 path and an AVX2+FMA path chosen at run time (`__builtin_cpu_supports`), both a Cephes-style float exp
with every term added in double. NaN, +inf or only -inf in a row return the scalar result (NaN). No build flag changes.

Test machine: Ryzen 9 3950X (AVX2, FMA), RTX 3090. Old build = `engine-63ea2c51a/build/bin/llama-decide`, new build =
`engine/build/bin/llama-decide`, both run from the repository root.

## Unit test (`test-decide-fullpath`, on the test machine)

`./test-engine.sh`: 100% tests passed out of 10. The test's printed lines:

```
full_vocab_lse max abs error vs the all-double reference: scalar 7.44e-09 sse2 3.47e-09 avx2 3.40e-09 dispatch 3.40e-09
full_vocab_lse timing, 262144 tokens, median of 41: scalar 0.753 ms (1.00x) sse2 0.335 ms (2.25x) avx2 0.163 ms (4.62x) dispatch 0.163 ms (4.62x)
```

"scalar" is the code before this change; targets were >= 2x (SSE2) and >= 3x (AVX2). Over five runs of the final code the timings were
scalar 0.75-0.80 ms, SSE2 0.335-0.358 ms (2.22-2.26x), AVX2 0.163-0.175 ms (4.54-4.62x). Accuracy cases: random rows at
1000, 151936 and 262144 tokens, ranges +-5 and +-40, one dominant token, 10 % -inf; a 262144-token row with 3 huge and many
tiny terms (each about 4e-8 of a huge one); a 1003-token row (vector tails). Bound 2e-7 for `full_vocab_lse`,
`1e-6 * max(1, |v|)` for `full_path_row`'s values. Pins: NaN / +inf at the start, inside and last give NaN; a row of only
-inf gives NaN; one finite entry among -inf gives exactly that entry.

## Live: old vs new (`check.py run PRESET`)

Flags from `models-engine-3090.ini` plus `--decide-seqs 64`: `-c 4096 -ngl 99 -fa on`, `-ctk/-ctv f16` (0.5B) and
`q8_0` (Gemma). Requests: `checks.py requests --preset P --n 20 --per-request 5` (4 lines of 5 states; identical to the
sp3 Task 5 request files). Full-path: 5 repetitions per build, alternating old/new; scoring time = median of the 5.
Tree: one `--dump-tokens` run per build.

| | Qwen2.5-0.5B (151936 tokens) | Gemma 4 E4B (262144 tokens) |
|---|---|---|
| full-path answers new vs old, max abs diff (bound 1e-6) | 2.1e-8 | 6.0e-8 |
| scoring ms per request, old -> new (lines 0-3) | 61.3 -> 33.1, 58.0 -> 30.3, 57.8 -> 30.2, 57.9 -> 30.0 | 136.5 -> 87.2, 131.7 -> 85.2, 132.2 -> 85.4, 132.9 -> 86.1 |
| scoring speedup | 1.85-1.93x | 1.54-1.57x |
| saved per full-path row (85 rows per request) | 0.33 ms | 0.55-0.58 ms |
| total ms per request, old -> new (line 1) | 71.3 -> 44.6 | 174.9 -> 128.1 |
| tree b5 `results` new vs old, max abs diff | 0.0 | 0.0 |
| tree b5 `tokens` dump | identical | identical |

Line 0 also builds the prefix (cache miss). Per-request and per-repetition numbers: `qwen2.5-0.5b.json`,
`gemma-4-e4b.json`; responses: `fp-b5-{old,new}-P.jsonl` (first repetition), `tree-b5-dump-{old,new}-P.jsonl`.
