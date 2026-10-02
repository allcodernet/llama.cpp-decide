# Sub-project 3 Task 6 live checks (sequential fields, `after`)

Build: engine `694ae5934` `decide: sequential fields (after)` (base bbe4ba438). CLI = `engine/build/bin/llama-decide -c 4096
-ngl 99 -fa on`, KV f16 for qwen2.5-0.5b, q8_0 for qwen3.5-4b (`models-engine.ini`); `--decide-seqs 64` (0.5B) and 32 (4B)
unless named `seqsN`. Requests: the triage schema of `decide_client.build_request` (prompt variant `default`) on the first
20 test tickets; b5 = 4 lines of 5 states, b1 = 20 lines of 1 state; `after-*` = urgency and angry after queue, `plain-*` =
no `after`, `qs-*` = queue, urgency (after queue), `tag` (choice `["a", "<b"]`, after queue), angry (after tag, so its
ancestors are queue and tag): 3 phases. Helper: `checks.py` (subcommands in its docstring; `dump-diff`, `records-compare`,
`cost` and `lines` work on the committed outputs alone, without a model). "The statistic" = per field median
abs diff <= 0.01 and no argmax disagreement over a 0.15 margin (`sp3_metrics.compare`). GPU (nvidia-smi readings during the session, not check results): under 200 MiB before every model run (22 MiB,
once 184 MiB while a vocab-only ctest process held a CUDA context), 22 MiB after the last, 50-74 C (the 4B server session peaked at 74 C, then stopped), port 8097 only, every server stopped by PID.

## Checks

| check | files | result |
|---|---|---|
| `after`: every dependent answer's and dump entry's `given` = the queue answer; `lines` = `  "queue": "<value>",\n`; entries carry their level as `phase`; `engine.phases` 2 | `after-{tree,fp}-b5-{qwen2.5-0.5b,qwen3.5-4b}.check.json` | PASS on all 4: 20 states each; 40 dependent entries (tree) / 120 (full-path: 4 urgency + 2 angry options per state); level-0 entries have no `lines`/`given`; answers sum to 1 (max abs(sum - 1) 2.2e-16); full-path coverage in (0, 1] |
| the dependent entries' tokens hold the lines (server `/detokenize`, `/tokenize`) | `*.detok.json` | PASS on all 8 dumps (after and qs, tree and full-path, both models): the text is lines + stem text + the option's value (full-path) or a prefix of it (tree); on both tokenizers the entry tokens start with the lines' own tokens (`lines_tokens_kept_as_prefix`), so no token spans the lines and the next key |
| the same, without a model: every dependent entry ends with the tokens of its lines-free entry (same field and node or option, from the `plain-*` dump), the tokens before them are the lines | `lines-{tree,fp}-b5-{qwen2.5-0.5b,qwen3.5-4b}.json` | PASS on all 4: lines of 7 tokens (Qwen2.5, 40 tree / 120 full-path entries) and 8 (Qwen3.5); full-path: the lines are queue's entry for the answered option without its closing-quote token, then the text `",\n` after the value: one token on Qwen2.5 (id 756, 120 entries), two on Qwen3.5 (ids 487, 198, 120 entries); per request line prompt_tokens(after) - prompt_tokens(plain) = the lines tokens of all dependent entries (0.5B 70 tree / 210 full-path, 4B 80 / 240) |
| full-path dependent rows (dump self-consistency, Task 5 `dumpcheck`: log P, distribution, coverage recomputed from `tokens`, `rows`, `logp`) | `after-fp-*.dumpcheck.json` | max abs diff 1.1e-16 on both models (60 fields each): the plan rebuilt on the context trie reads its rows at the shifted indices |
| a request without `after` unchanged: `reference_check.py dump --tag task6` (0.5B, `--decide-seqs 9`) vs `sp3/baseline/dump-qwen2.5-0.5b.jsonl` | `tree-regression-*` | max abs diff 0.0 (bound 1e-6); the dump is byte-identical to the regression dump of task-5/ |
| `LLAMA_DECIDE_FAULT=rc1-phase1`, retries >= 1 | `fault-rc1-phase1-*.retries.json` | retries per request: 0.5B tree b5 at `--decide-seqs 9` (spr 2) 5, 5, 5, 5 (5 rounds, each retried once, `states_per_round` 1 at the end); 0.5B tree b1 at 5: 1 per request (20); 0.5B full-path b5: 4 per request (4 rounds); 4B tree b5: 3 per request (3 rounds, each retried once) |
| `rc1-phase1` answers vs a normal run | `fault-rc1-phase1-*-vs-normal-*.json` | 0.5B tree b5 vs a normal spr-1 run (`--decide-seqs 5`): max abs diff 0.0; b1 vs normal b1: 0.0; 0.5B full-path vs normal: statistic PASS (medians 0.0040 / 0.0038 / 0.0093, max 0.069); 4B tree vs normal (spr 7): statistic PASS (0.0008 / 0.0022 / 0.0033, max 0.064). 0.5B tree b5 vs the normal spr-2 run: angry median 0.0105 (FAIL); the same 0.0105 separates the two normal runs spr 2 vs spr 1 (`normal-after-tree-b5-seqs9-vs-seqs5-*`) and a plain request at spr 2 vs spr 15 (`normal-plain-tree-seqs9-vs-seqs64-*`: angry 0.0105): the 0.5B's batch sensitivity on angry, not the retry |
| `rc1-phase1` on a request without `after` | `fault-rc1-phase1-plain-*` | retries 0, answers 0.0 vs normal (nothing to fail) |
| a cycle -> 400 (urgency after angry, angry after urgency) | `cycle-cli-*.err.txt`, `cycle-server-*.jsonl` | CLI and server (both models): HTTP 400 `{"error":{"message":"field \"urgency\": after: cycle urgency -> angry -> urgency","code":"invalid_schema","field":"urgency"}}` |
| a quote-split choice field as dependent (tree mode: the first children `" "` and `" "<` share a prefix) | `qs-{tree,fp}-b5-*.check.json` | runs on both models: tree `engine.quote_split_fields` `["tag"]`, 3 phases; 60 dependent entries (tree) / 160 (full-path, where `tag` is not quote-split); tag's tree entries detokenize to its lines followed by a prefix of `  "tag": "<option>"` (`qs-tree-*.detok.json`); angry (phase 2) has the lines of queue and of the quote-split tag (e.g. `  "queue": "billing",\n  "tag": "a",\n` in `qs-tree-b5-qwen3.5-4b.jsonl`) and `given` {queue, tag} |

## Extra checks

| check | files | result |
|---|---|---|
| independent replay of the tree-mode prompts (`reference_check.py compare` on the dump entries; a dependent prompt = prefix + state + tail + lines + stem), Qwen3.5-4B b5 | `replay-*-tree-b5-qwen3.5-4b.json` | statistic PASS: after 0.0008 / 0.0038 / 0.0026 (queue / urgency / angry medians); plain 0.0010 / 0.0038 / 0.0015; qs 0.0006 / 0.0017 / 0.0022 (the `tag` field is left out: its `<b` child, engine probability 1.2e-6 to 7.9e-6 over the 20 states of `qs-tree-b5-qwen3.5-4b.jsonl`, was outside the replay's top-512 in this session (not recorded in a committed file: the committed `replay-qs-*` dumps omit the `tag` records); the 0.5B `<b` child has engine probability 4.5e-6 to 1.8e-4 in `qs-tree-b5-qwen2.5-0.5b.jsonl` and was left out the same way) |
| the same replay, Qwen2.5-0.5B | `replay-*-tree-{b1,b5}-qwen2.5-0.5b.json`, `queue-plain-vs-after-tree-b1-qwen2.5-0.5b.json` | b1: plain 0.0044 / 0.0057 / 0.0094 (PASS, as sub-project 1), after 0.0040 / 0.0051 / 0.0115, qs 0.0039 / 0.0063 / 0.0157; b5: plain 0.0044 / 0.0076 / 0.0160, after 0.0061 / 0.0063 / 0.0185, qs 0.0040 / 0.0047 / 0.0259. Angry fails in every 0.5B b5 replay, the plain one included; no argmax disagreement over the margin anywhere. Pure batch effect on the 0.5B: queue in plain b1 vs after b1 (`records-compare` of the two replay dumps: 20 of 20 records with identical prompt tokens; phase-0 batch 3 fields vs 1): median 0.0036, max 0.0478, no argmax disagreement |
| server `/v1/decide` vs the CLI, after tree b5 | `server-*` | max abs diff 0.0 on both models |
| `--info` with an `after` body | `info-after-qwen2.5-0.5b.jsonl` | 200; `schema.seqs_per_state` / `states_per_round` 4 / 15 (tree) and 11 / 5 (full-path), the same as without `after` |
| worst-case budget (spec 5.4) in the per-state message, `-c 256` | `budget-c256-*.err.txt`, `lines-{tree,fp}-b5-qwen2.5-0.5b.json` | `branches` 16 -> 34 (tree, 2 dependent sequences per state) and 68 -> 122 (full-path, 6): 9 reserved cells per dependent sequence in both modes (the longest queue line of 7 tokens + 2); the exact lines are 7 tokens, so each sequence keeps 2 cells of slack |
| full-path without `after` vs the committed run of task-5/ | `fp-regression-*` (`dump-diff`) | results, usage and decodes identical; the dump is identical apart from `"phase": 0` on every entry; the engine block adds `"phases": 1` |
| `after` vs plain (informational: queue moves by the phase batch only, urgency and angry by the lines) | `plain-vs-after-*.json` | queue median 0.0035 (0.5B) / 0.0003 (4B); urgency 0.0996 / 0.0201, angry 0.169 / 0.0127; argmax changed on 9 + 6 (0.5B) and 5 + 1 (4B, tree) of 20 tickets |

## Cost (b5, lines 2-4, prefix resident, medians; `cost-b5.json`)

| run | scoring ms | total ms | decodes | prompt_tokens | phases | spr |
|---|---|---|---|---|---|---|
| 0.5B tree plain / after / qs | 9.5 / 19.7 / 31.7 | 29.4 / 37.6 / 52.7 | 2 / 3 / 4 | 340 / 410 / 505 | 1 / 2 / 3 | 15 / 15 / 12 |
| 0.5B full-path plain / after / qs | 63.6 / 72.2 / 117.6 | 80.4 / 92.7 / 140.8 | 3 / 3 / 8 | 600 / 810 / 1020 | 1 / 2 / 3 | 5 / 5 / 4 |
| 4B tree plain / after / qs | 99.9 / 151.3 / 227.4 | 319.2 / 366.9 / 437.3 | 2 / 3 / 4 | 365 / 445 / 550 | 1 / 2 / 3 | 7 / 7 / 6 |
| 4B full-path plain / after / qs | 473.4 / 556.4 / 632.0 | 669.3 / 754.4 / 827.8 | 8 / 9 / 12 | 625 / 865 / 1095 | 1 / 2 / 3 | 2 / 2 / 2 |

Each phase adds at least one branch decode per round, but the totals also depend on the row cap: plain full-path runs
can already need several branch decodes per round (the 0.5B for its 85 rows, the 4B: 8 decodes plain vs 9 with `after`
over 3 rounds, `cost-b5.json`), and the after run splits the same rows by phase; `prompt_tokens` grows by exactly the lines tokens of every
dependent branch sequence (`lines-*.json`).
