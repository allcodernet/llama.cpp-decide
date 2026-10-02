#!/usr/bin/env bash
# Sub-project 3 Task 9: evaluation runs, scores, sign tests and distributions (spec 12.4), calibration (spec 8) and the
# check of the calibrated preset. Every output lands in EVAL_DIR (default: this directory); llama.cpp and client logs in
# $LOGS.
# Usage: results-engine/sp3/eval/run.sh STEP...
#   baseline     the tree baseline runs into BASELINE_DIR (refused with the default BASELINE_DIR, the committed Task 0
#                baseline, and with the default EVAL_DIR, whose gpu-log.txt is committed): per
#                model test and train, b5, decide_client.py's default prompt variant, --ensure-t1, each on a freshly
#                started preset router, analysis.py check; then sp3_metrics.py score per set (score.json)   [server]
#   runs         the 16 evaluation runs: qwen3.5-9b and gemma-4-e4b x full-path, after A (urgency after queue), after B
#                (urgency and angry after queue), debias K = 4 (flags: analysis.py flags) x test (80) and train (320); b5,
#                decide_client.py's default prompt variant (keywords), T = 1 (--ensure-t1: options.temperature 1 only
#                while the preset sets decide-temperature, Gemma since the calibration; the committed runs predate it
#                and carry no temperature); every run on a freshly started preset router (serve-engine.sh, :8097),
#                then analysis.py check (engine blocks: scoring, effective K, phases, temperature 1.0)       [server]
#                With other MODELS: the same four configurations per model.
#   score        sp3_metrics.py score per directory, sp3_metrics.py signtest --pool against the Task 0 baseline (test +
#                train, 400 tickets) per run, analysis.py summary (summary.json, summary.md)                [no GPU]
#   calibration  calibrate.py on the Task 0 baseline runs: calibration-<model>.json (the rule's decision)     [no GPU]
#   preset       per model, from its calibration-<model>.json (analysis.py preset): the rule applies and the presets file
#                holds the fitted T as decide-temperature (Gemma: models-engine.ini since the rule's change) -> the
#                test run without switches on a fresh router (outputs named -t<T>; analysis.py check:
#                engine.temperature = that T),
#                compared with the baseline test preds scaled post hoc (analysis.py temper; sp3_metrics.py compare
#                --max-diff 1e-9), and scored; the rule applies and the presets file lacks that T -> the line to add
#                is printed (never edited here; add it, then run the step again); the rule does not apply -> skipped
#                                                                                                             [server]
# The runs and preset steps replace existing files (decide_client.py --force); the baseline step never does.
# Settings (environment):
#   DATA          the directory of triage_test.jsonl and triage_train.jsonl (default TRIAGE_DATA, else data, the
#                 bundled set); passed to the Python steps as TRIAGE_DATA
#   MODELS        space-separated presets (default "qwen3.5-9b gemma-4-e4b"). decide_client.py stops on a model
#                 without an EXPECTED_PREFIX entry (prefix mismatch): add the entry before its runs.
#   EVAL_DIR      default results-engine/sp3/eval; BASELINE_DIR default results-engine/sp3/baseline; COST_DIR (the tree
#                 runs of analysis.py summary's cost table) default BASELINE_DIR when that is set, else
#                 results-engine/sp3/integrity/regression. Relative paths from the project directory.
#   ENGINE_PRESETS, ENGINE_BIN    serve-engine.sh (and decide_client.py's preset settings)
#   GPU_VRAM_MAX  gpu_ready: the VRAM in use (MiB) at which a run does not start (default 200)
#   GPU_WAIT_MIN  gpu_ready: minutes to wait, re-checking once a minute, for a free GPU (no llama-server, VRAM under
#                 GPU_VRAM_MAX) before giving up (default 0: give up at once)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # BASH_SOURCE: also when sourced (results-engine/3090/run.sh)
cd "$ROOT"
E=${EVAL_DIR:-results-engine/sp3/eval}
B=${BASELINE_DIR:-results-engine/sp3/baseline}
LOGS=${LOGS:-/tmp/task9-logs}
mkdir -p "$LOGS" "$E"
PORT=8097
URL=http://127.0.0.1:$PORT
DATA=${DATA:-${TRIAGE_DATA:-data}}
export TRIAGE_DATA="$DATA"
AN=(uv run results-engine/sp3/eval/analysis.py)
read -ra MODEL_LIST <<< "${MODELS:-qwen3.5-9b gemma-4-e4b}"
SETS=(test train)
CONFIGS=(full-path after-a after-b debias4)
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
    echo "$(date +%T) $1: ${used} MiB, ${t} C before start" | tee -a "$E/gpu-log.txt" >&2
}

# the preset router (serve-engine.sh), started fresh for every run
router_start() {
    gpu_ready "$1"
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
}
router_stop() {  # stops the router this script started (serve-engine.sh execs it) and waits for it and its instances
    local pids p
    nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | sed "s/^/$(date +%T) $1: /; s/$/ at stop/" \
        | tee -a "$E/gpu-log.txt" >&2
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

step_baseline() {
    local p s run
    if [ "$(realpath -m "$B")" = "$(realpath -m results-engine/sp3/baseline)" ]; then
        echo "baseline: set BASELINE_DIR (the default holds the committed Task 0 baseline)" >&2
        exit 1
    fi
    if [ "$(realpath -m "$E")" = "$(realpath -m results-engine/sp3/eval)" ]; then
        echo "baseline: set EVAL_DIR too (gpu_ready logs to EVAL_DIR/gpu-log.txt; the default is committed)" >&2
        exit 1
    fi
    for p in "${MODEL_LIST[@]}"; do
        run=$("${AN[@]}" name "$p")
        for s in "${SETS[@]}"; do
            router_start "$p-baseline-$s"
            router_load "$p"
            uv run decide_client.py --model "$p" --batch 5 --test-set "$DATA/triage_$s.jsonl" --out-dir "$B/$s" \
                --ensure-t1 > "$LOGS/baseline-$p-$s.log" 2>&1
            router_stop "$p-baseline-$s"
            "${AN[@]}" check "$B/$s/runs/$run.json"
        done
    done
    for s in "${SETS[@]}"; do
        uv run sp3_metrics.py score --preds-dir "$B/$s/preds" --test-set "$DATA/triage_$s.jsonl" | tee "$LOGS/score-baseline-$s.txt"
    done
}

step_runs() {
    local p c s flags run
    for p in "${MODEL_LIST[@]}"; do
        for c in "${CONFIGS[@]}"; do
            read -ra flags <<< "$("${AN[@]}" flags "$c")"
            run=$("${AN[@]}" name "$p" "$c")
            for s in "${SETS[@]}"; do
                router_start "$p-$c-$s"
                router_load "$p"
                uv run decide_client.py --model "$p" --batch 5 --test-set "$DATA/triage_$s.jsonl" --out-dir "$E/$s" "${flags[@]}" \
                    --ensure-t1 --force > "$LOGS/run-$p-$c-$s.log" 2>&1
                router_stop "$p-$c-$s"
                "${AN[@]}" check "$E/$s/runs/$run.json"
            done
        done
    done
}

step_score() {
    local s p c base run
    for s in "${SETS[@]}"; do
        uv run sp3_metrics.py score --preds-dir "$E/$s/preds" --test-set "$DATA/triage_$s.jsonl" | tee "$LOGS/score-$s.txt"
    done
    for p in "${MODEL_LIST[@]}"; do
        base=$("${AN[@]}" name "$p")
        for c in "${CONFIGS[@]}"; do
            run=$("${AN[@]}" name "$p" "$c")
            uv run sp3_metrics.py signtest \
                --pool "$B/test/preds/$base.jsonl" "$E/test/preds/$run.jsonl" "$DATA/triage_test.jsonl" \
                --pool "$B/train/preds/$base.jsonl" "$E/train/preds/$run.jsonl" "$DATA/triage_train.jsonl" \
                --out "$E/signtest-${run#decide-}.json" > /dev/null
        done
    done
    "${AN[@]}" summary > "$LOGS/summary.txt"
    tail -1 "$E/summary.md" >&2
}

step_calibration() {
    uv run calibrate.py --models "${MODEL_LIST[@]}" --baseline-dir "$B" --out-dir "$E" | tee "$LOGS/calibrate.txt"
}

step_preset() {
    local p decision action t base run
    for p in "${MODEL_LIST[@]}"; do
        # the rule's decision and the fitted T ("t" of calibration-<model>.json)
        decision=$("${AN[@]}" preset "$p")
        read -r action t <<< "$decision"
        [ "$action" = check ] || continue
        base=$("${AN[@]}" name "$p")
        run=$("${AN[@]}" name "$p" --temperature "$t")
        router_start "preset-$p-test"
        router_load "$p"
        uv run decide_client.py --model "$p" --batch 5 --test-set "$DATA/triage_test.jsonl" --out-dir "$E/preset" --force \
            > "$LOGS/run-preset.log" 2>&1
        router_stop "preset-$p-test"
        "${AN[@]}" check "$E/preset/runs/$run.json" --temperature "$t"
        "${AN[@]}" temper "$B/test/preds/$base.jsonl" "$E/preset/tempered-baseline-test-$p.jsonl" --t "$t"
        uv run sp3_metrics.py compare "$E/preset/tempered-baseline-test-$p.jsonl" "$E/preset/preds/$run.jsonl" \
            --max-diff 1e-9 --out "$E/preset/compare-$p.json" | grep -E '"max_abs_diff": |within_bound' | tail -2 >&2
        uv run sp3_metrics.py score --preds-dir "$E/preset/preds" --test-set "$DATA/triage_test.jsonl" | tee "$LOGS/score-preset.txt"
    done
}

for s in "$@"; do "step_${s//-/_}"; done
