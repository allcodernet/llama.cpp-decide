#!/usr/bin/env bash
# Follow-up Task 3 smoke (test machine): eval/run.sh baseline runs score calibration preset on 10-ticket test and train
# files, every output under T, nothing inside the repository.
set -uo pipefail
T=/tmp/t3-smoke
cd "$(dirname "$0")/../../.."
rm -rf "$T/data" "$T/eval" "$T/baseline" "$T/logs"
mkdir -p "$T/data"
head -n 10 data/triage_test.jsonl > "$T/data/triage_test.jsonl"
head -n 10 data/triage_train.jsonl > "$T/data/triage_train.jsonl"
{ echo "head: $(git log --oneline -1)"; echo "engine: $(git -C engine log --oneline -1)"
  echo "llama-server processes: $(pgrep -x llama-server | wc -l)"
  echo "gpu: $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"; } > "$T/before.txt"
git status --porcelain --untracked-files=all > "$T/status-before.txt"
touch "$T/marker"
export TRIAGE_DATA="$T/data" MODELS=gemma-4-e4b ENGINE_PRESETS=models-engine-3090.ini GPU_VRAM_MAX=1500 GPU_WAIT_MIN=20 \
    EVAL_DIR="$T/eval" BASELINE_DIR="$T/baseline" LOGS="$T/logs"
start=$SECONDS
results-engine/sp3/eval/run.sh baseline runs score calibration preset > "$T/run.log" 2>&1
echo "run.sh baseline runs score calibration preset: exit $? after $((SECONDS - start)) s" > "$T/exit.txt"
sleep 3
{ echo "llama-server processes: $(pgrep -x llama-server | wc -l)"
  echo "port 8097 listening: $(ss -ltn 'sport = :8097' | tail -n +2 | wc -l)"
  echo "gpu: $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"; } > "$T/after.txt"
git status --porcelain --untracked-files=all > "$T/status-after.txt"
find . \( -path ./.git -o -path ./.venv -o -name __pycache__ \) -prune -o -newer "$T/marker" -print > "$T/repo-written.txt"
cat "$T/exit.txt"
