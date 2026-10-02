# Task 0 probe on the RTX 3090 (2026-10-02)

Test machine `user@gpu-host` (RTX 3090 24 GB, about 612 MiB held by the desktop), engine `63ea2c51a` (tree
`85b062337db4…`) built there with `./build-engine.sh`. Server: `llama-server --models-preset models-engine-3090.ini
--models-max 1 --host 127.0.0.1 --port 8097`. Client: `uv run decide_client.py --model P --batch 5 --test-set
local/data/triage_test.jsonl --out-dir /tmp/t0-probe/<dir>` (80 test tickets, `keywords`, tree mode, T = 1; the 3090
presets carry no temperature). Scores: `sp3_metrics.py score`; comparisons: `sp3_metrics.py compare <laptop baseline>
<3090 run>` (the `b` paths in the compare files point to the scratch directory; the same files are under `q9/`, `g4/`).

## 9B and Gemma: 3090 vs the laptop's committed baseline (`results-engine/sp3/baseline/test/preds/`)

| preset | statistic | median abs diff q / u / a | max abs diff | argmax disagreements (over margin) |
|---|---|---|---|---|
| qwen3.5-9b | pass | 0.00047 / 0.00155 / 0.00046 | 0.130 (urgency) | 2 urgency (0) |
| gemma-4-e4b | pass | 0.00001 / 0.00004 / 0.00005 | 0.087 (queue) | 1 queue, 1 angry (0) |

| preset | GPU | queue | urgency | urg±1 | angry | NLL Σ | p50 ms / ticket |
|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 3070 Ti laptop (committed) | 95.0 % | 78.75 % | 95.0 % | 97.5 % | 0.839 | 92.7 |
| qwen3.5-9b | 3090 | 95.0 % | 78.75 % | 96.25 % | 97.5 % | 0.862 | 35.3 |
| gemma-4-e4b | 3070 Ti laptop (committed) | 93.75 % | 82.5 % | 98.75 % | 95.0 % | 0.909 | 31.4 |
| gemma-4-e4b | 3090 | 95.0 % | 82.5 % | 98.75 % | 96.25 % | 0.895 | 13.9 |

## Qwen3.8-27B (`qwen3.8-27b`, IQ3_S GSQ-RCO with an MTP head)

- Loads in our engine. Architecture `qwen35` (hybrid): `/v1/decide/info` reports `memory.recurrent: true`,
  `swa: false`, `n_ctx 4096`, `decide_seqs 64` (`info-27b-64.json`). The MTP layer (`blk.64.*`, about 0.35 GB) is
  reported as unused and ignored (no speculative decoding, no mmproj).
- Assistant prefix with thinking off: `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (the template's
  `enable_thinking is false` branch, the same tail as Qwen3.5); `EXPECTED_PREFIX["qwen3.8-27b"]` added.
- VRAM in use on the GPU (desktop included) with the model loaded and idle: 15412 MiB at `--decide-seqs 21`, 22114 MiB
  at 64, so about 156 MiB per decide sequence (the recurrent state); 22316 MiB after the batch-5 run. The two idle
  readings were read from `nvidia-smi` during the session (no committed capture; the load logs `load-27b-seqs21.log`
  and `load-27b-seqs64.log` are from those two loads); 22316 MiB is the run file's `vram_mib_after_run`. 64 fits with
  about 2.2 GB free, so the preset keeps `decide-seqs = 64` (a full-path batch of 5 needs 55 sequences in one round).
- `common_fit_params: failed to fit params to free device memory ... abort` appears only in the 64-sequence load log:
  at 64 sequences the fit's projection did not fit within its free-memory margin (at 21 it did, and no line is
  logged). Because the preset sets `ngl`, the fit only gives up; the model loads fully on the GPU and runs.
- Batch-5 run, 80 test tickets, tree, T = 1: queue 98.75 %, urgency 88.75 %, urg±1 98.75 %, angry 100 %, NLL Σ 0.531
  (q 0.072, u 0.382, a 0.078); p50 101.6 ms per ticket (server 102.8 ms), 1 round per request, 0.454–0.574 s per
  5-ticket request (`wall_s` of the 16 requests; median 0.508, upper middle 0.516).
- `runs/decide-qwen3.8-27b-b5-keywords.json` records `decide_seqs: 21` and `model_file: null`: `decide_client.py`
  reads `models-engine.ini` for those fields (the `ENGINE_PRESETS` setting of follow-ups Task 3); the engine used 64 (see the info file).
