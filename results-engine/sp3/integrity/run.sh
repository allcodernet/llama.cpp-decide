#!/usr/bin/env bash
# Sub-project 3 Task 8 part B: integrity runs (spec 12.2, 12.3). Every output lands in this directory; llama.cpp logs in $LOGS.
# Usage: results-engine/sp3/integrity/run.sh STEP...
#   dumps      full-path reference dumps (b1: 0.5B 64, 4B 32, Gemma 64, 9B 21 sequences), the b5 extras (0.5B, Gemma at 64),
#              the `after` dumps (0.5B 64, 4B 32) and the 0.5B tree references (b1, b5 at 64)          [CLI, server stopped]
#   replays    reference_check.py compare of every dump above through one preset router on :8097  [server]
#   split      row-cap split check on the 0.5B and Gemma at 64 sequences                            [CLI]
#   cache      cache_timing.py on the 9B and Gemma                                                  [CLI]
#   debias     `after` + order_debias 2 with >= 2 states per round: 4B tree (32), 0.5B full-path (64) [CLI]
#   regression Task 0 baseline configurations (tree b5 keywords, test and train, 9B and Gemma), each on a freshly started
#              router, at T = 1 (decide_client.py --ensure-t1: options.temperature 1 only while the preset sets
#              decide-temperature, Gemma since the calibration; the runs of this script carry no temperature), replacing the
#              committed runs (--force); and the 0.5B tree dump; sp3_metrics.py compare --max-diff 1e-6   [server, CLI]
#   nosplit    diagnostic, run after the regression on the same build: the split check's requests at 84 sequences
#              (row cap 85, a b5 round's 85 rows in one chunk) on the 0.5B and Gemma, then checks.py chunk-answers
#              (answers by row-cap chunk: split vs unsplit, against the replays)                    [CLI]
# DATA: the directory of triage_test.jsonl and triage_train.jsonl (default TRIAGE_DATA, else
# data, the bundled set); passed to the Python steps as TRIAGE_DATA. ENGINE_PRESETS, ENGINE_BIN:
# serve-engine.sh and reference_check.py. GPU_VRAM_MAX, GPU_WAIT_MIN: gpu_ready (default 200 MiB, 0 min = give up at once).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
I=results-engine/sp3/integrity
LOGS=${LOGS:-/tmp/task8-logs}
mkdir -p "$LOGS"
PORT=8097
URL=http://127.0.0.1:$PORT
DATA=${DATA:-${TRIAGE_DATA:-data}}
export TRIAGE_DATA="$DATA"
RC=(uv run reference_check.py)
CK=(uv run "$I/checks.py")

# the Qwen2.5-0.5B and Qwen3.5-4B GGUFs (models-engine.ini's files) are under LMSTUDIO_MODELS
LMSTUDIO_MODELS=${LMSTUDIO_MODELS:-$HOME/.lmstudio/models/lmstudio-community}
model_of() {
    case "$1" in
        qwen2.5-0.5b) echo "$LMSTUDIO_MODELS/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf" ;;
        qwen3.5-4b) echo "$LMSTUDIO_MODELS/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" ;;
        qwen3.5-9b) echo models/Qwen3.5-9B-Q4_K_M.gguf ;;
        gemma-4-e4b) echo models/gemma-4-E4B-it-Q4_0.gguf ;;
    esac
}

GPU_VRAM_MAX=${GPU_VRAM_MAX:-200}
GPU_WAIT_MIN=${GPU_WAIT_MIN:-0}
# global constraints: no llama-server, VRAM under GPU_VRAM_MAX MiB (else wait up to GPU_WAIT_MIN minutes, re-checking
# once a minute); above 75 C wait until below 65 C
gpu_ready() {
    local used t why waited=0
    while :; do
        why=
        if pgrep -x llama-server >/dev/null; then
            why="a llama-server is running"
        else
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            if [ "$used" -ge "$GPU_VRAM_MAX" ]; then why="VRAM in use: $used MiB"; fi
        fi
        [ -n "$why" ] || break
        if [ "$waited" -ge "$GPU_WAIT_MIN" ]; then echo "$why" >&2; exit 1; fi
        echo "$why; re-checking in 1 min ($waited of $GPU_WAIT_MIN min waited)" >&2
        sleep 60
        waited=$((waited + 1))
    done
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)
    if [ "$t" -gt 75 ]; then
        while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)" -ge 65 ]; do sleep 10; done
    fi
    echo "gpu: ${used} MiB, ${t} C" | tee -a "$LOGS/gpu.log" >&2
}

# the preset router of the reference checks (README, serve-engine.sh), loaded preset by preset
router_start() {
    gpu_ready
    ./serve-engine.sh > "$LOGS/router-$1.log" 2>&1 &
    ROUTER_PID=$!
    until curl -sf "$URL/health" >/dev/null 2>&1 || curl -sf "$URL/v1/models" >/dev/null 2>&1; do
        kill -0 "$ROUTER_PID" 2>/dev/null || { echo "router did not start" >&2; exit 1; }
        sleep 1
    done
}
router_load() {  # load PRESET (the router starts its instance on the first request naming it); give up after 600 s
    local deadline=$((SECONDS + 600))
    until [ "$(curl -s -m 60 -o /dev/null -w '%{http_code}' "$URL/v1/decide/info?model=$1")" = 200 ]; do
        kill -0 "$ROUTER_PID" 2>/dev/null || { echo "router died" >&2; exit 1; }
        [ "$SECONDS" -lt "$deadline" ] || { echo "preset $1 did not load within 600 s" >&2; exit 1; }
        sleep 1
    done
    nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | sed "s/^/$1 loaded: /" | tee -a "$LOGS/gpu.log" >&2
}
router_stop() {  # stops the router this script started (serve-engine.sh execs it) and waits for it and its instances
    local pids p
    pids=$(pgrep -P "$ROUTER_PID" || true)
    kill "$ROUTER_PID" 2>/dev/null || true
    wait "$ROUTER_PID" 2>/dev/null || true
    for p in $pids; do while kill -0 "$p" 2>/dev/null; do sleep 1; done; done
    sleep 2
    ROUTER_PID=
}
# a failing step must not leave a router holding the GPU
ROUTER_PID=
trap '[ -n "$ROUTER_PID" ] && kill "$ROUTER_PID"; true' EXIT

dump() {  # dump PRESET TAG [reference_check dump args...]
    local p=$1 tag=$2
    shift 2
    gpu_ready
    "${RC[@]}" dump --model "$(model_of "$p")" --preset "$p" --n 20 --tag "$tag" --out-dir "$I" "$@"
}

step_dumps() {
    dump qwen2.5-0.5b fp --scoring full_path --decide-seqs 64
    dump qwen3.5-4b fp --scoring full_path --decide-seqs 32
    dump gemma-4-e4b fp --scoring full_path --decide-seqs 64
    dump qwen3.5-9b fp --scoring full_path --decide-seqs 21
    dump qwen2.5-0.5b fp-b5 --scoring full_path --decide-seqs 64 --batch 5
    dump gemma-4-e4b fp-b5 --scoring full_path --decide-seqs 64 --batch 5
    dump qwen2.5-0.5b fp-after --scoring full_path --decide-seqs 64 --after urgency=queue --after angry=queue
    dump qwen3.5-4b fp-after --scoring full_path --decide-seqs 32 --after urgency=queue --after angry=queue
    # the 0.5B tree references beside its full-path results (same states, same sequences, b1 and b5)
    dump qwen2.5-0.5b tree-b1 --decide-seqs 64
    dump qwen2.5-0.5b tree-b5 --decide-seqs 64 --batch 5
}

compare() {  # compare PRESET TAG
    "${RC[@]}" compare --preset "$1" --tag "$2" --out-dir "$I" --url "$URL" > "$LOGS/compare-$2-$1.log"
    tail -1 "$LOGS/compare-$2-$1.log" >&2
}

step_replays() {
    router_start replays
    router_load qwen2.5-0.5b
    for t in fp fp-b5 fp-after tree-b1 tree-b5; do compare qwen2.5-0.5b $t; done
    router_load gemma-4-e4b
    for t in fp fp-b5; do compare gemma-4-e4b $t; done
    router_load qwen3.5-4b
    for t in fp fp-after; do compare qwen3.5-4b $t; done
    router_load qwen3.5-9b
    compare qwen3.5-9b fp
    router_stop
}

step_split() {
    for p in qwen2.5-0.5b gemma-4-e4b; do
        gpu_ready
        "${RC[@]}" split --model "$(model_of $p)" --preset $p --decide-seqs 64 --n 20 --tag rowcap --out-dir "$I" > "$LOGS/split-$p.log"
        tail -3 "$LOGS/split-$p.log" >&2
    done
}

step_cache() {
    for p in qwen3.5-9b gemma-4-e4b; do
        gpu_ready
        uv run cache_timing.py --model "$(model_of $p)" --preset $p --alternations 20 --out "$I/cache-timing-$p.json" | tee "$LOGS/cache-$p.log"
    done
}

step_debias() {
    gpu_ready
    "${CK[@]}" debias-run --model "$(model_of qwen3.5-4b)" --preset qwen3.5-4b --decide-seqs 32 --scoring tree --tag tree
    "${CK[@]}" debias-check --preset qwen3.5-4b --tag tree --binding
    gpu_ready
    "${CK[@]}" debias-run --model "$(model_of qwen2.5-0.5b)" --preset qwen2.5-0.5b --decide-seqs 64 --scoring full_path --tag fp
    "${CK[@]}" debias-check --preset qwen2.5-0.5b --tag fp
}

step_regression() {
    local sets=(test train) p s
    for p in qwen3.5-9b gemma-4-e4b; do
        for s in "${sets[@]}"; do
            router_start "regression-$p-$s"
            router_load $p
            uv run decide_client.py --model $p --batch 5 --test-set "$DATA/triage_$s.jsonl" \
                --out-dir "$I/regression/$s" --ensure-t1 --force > "$LOGS/regression-$p-$s.log" 2>&1
            router_stop
            uv run sp3_metrics.py compare "results-engine/sp3/baseline/$s/preds/decide-$p-b5-keywords.jsonl" \
                "$I/regression/$s/preds/decide-$p-b5-keywords.jsonl" --max-diff 1e-6 --out "$I/regression-$s-$p.json" \
                | grep -E '"max_abs_diff": |within_bound' | tail -2 >&2
        done
    done
    dump qwen2.5-0.5b regression
    uv run sp3_metrics.py compare results-engine/sp3/baseline/dump-qwen2.5-0.5b.jsonl \
        "$I/reference-regression-qwen2.5-0.5b-dump.jsonl" --max-diff 1e-6 --out "$I/regression-dump-qwen2.5-0.5b.json" \
        | grep -E 'within_bound' >&2
}

step_nosplit() {
    for p in qwen2.5-0.5b gemma-4-e4b; do
        gpu_ready
        "${RC[@]}" split --model "$(model_of $p)" --preset $p --decide-seqs 84 --n 20 --tag s84 --out-dir "$I" > "$LOGS/split-s84-$p.log"
        tail -3 "$LOGS/split-s84-$p.log" >&2
        "${CK[@]}" chunk-answers --preset $p > /dev/null
    done
}

for s in "$@"; do "step_${s//-/_}"; done
