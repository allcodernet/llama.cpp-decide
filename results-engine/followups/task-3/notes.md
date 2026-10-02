# Follow-up Task 3: portability smoke on the RTX 3090 test machine (2026-10-02)

Project `f9a0b1f`, engine `engine/build` at `23957f14c` (Tasks 1–2). Command (`smoke.sh`, run from `/tmp/t3-smoke`
through `./remote-sync.sh run`): the first 10 lines of `local/data/triage_{test,train}.jsonl` as a scratch
`TRIAGE_DATA`, then

```
TRIAGE_DATA=/tmp/t3-smoke/data MODELS=gemma-4-e4b ENGINE_PRESETS=models-engine-3090.ini GPU_VRAM_MAX=1500 GPU_WAIT_MIN=20 \
EVAL_DIR=/tmp/t3-smoke/eval BASELINE_DIR=/tmp/t3-smoke/baseline LOGS=/tmp/t3-smoke/logs \
results-engine/sp3/eval/run.sh baseline runs score calibration preset
```

Result: **exit 0 after 64 s** (one invocation; `set -e`, so every step exited 0). 10 router starts on 127.0.0.1:8097
(2 baseline + 8 runs), each `analysis.py check` ok, summary "0 problems". Before: no `llama-server`, 612 MiB, 53 C.
After: no `llama-server`, nothing listening on 8097, 612 MiB (`smoke-status.txt`).

Nothing written inside the repository: `git status --porcelain --untracked-files=all` empty before and after
(identical), and no repository path (`.git`, `.venv`, `__pycache__` pruned) newer than the start marker. Every output
is under `/tmp/t3-smoke/{baseline,eval}` (listing in `smoke-status.txt`; `gpu-log.txt` in EVAL_DIR).

Settings reached every reader:
- the router served `models-engine-3090.ini` (`run-settings.txt`, last line); every run JSON records
  `model_file ./models/gemma-4-E4B-it-Q4_0.gguf`, `decide_seqs 21` (the 3090 file's `[*]`; `[gemma-4-e4b]` has no
  override) and `preset_settings` without `decide-temperature` (`models-engine.ini` holds 1.6947), engine temperature
  1.0, `fork_commit 23957f14c`;
- `summary.md` derives its sentences: "10 tickets" per set, "20 tickets" pooled, "Cost on the train set (2 requests of
  5 states, `--decide-seqs 21`; tree = the tree runs in `/tmp/t3-smoke/baseline`)";
- `calibration-gemma-4-e4b.json` read the scratch data and baseline (`train_set`, `test_set`, `train_preds` under
  `/tmp/t3-smoke`, n 10 / 10): T = 1.9346, test summed NLL 0.5691 → 0.5902, `rule.apply` false;
- `preset` step: "gemma-4-e4b: the rule keeps the preset without decide-temperature; no preset check" (skip). The
  `add` branch (rule applies, no T in the presets file) was checked model-free in part A.

These are plumbing checks on 10 tickets; the scores and the fitted T mean nothing.

Files: `smoke.sh`, `smoke-status.txt` (before/after, exit, git status, written paths, output and log listing),
`run.log` (run.sh's stdout and stderr), `run-settings.txt`, `summary.md`, `calibration-gemma-4-e4b.json`.
