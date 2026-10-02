#!/usr/bin/env bash
# Sub-project 3 Task 7 live checks (option-order debiasing and temperature, spec 7, 8, 12.3).
# Usage: results-engine/sp3/checks/task-7/run.sh CHECK...   CHECK: c1 regress k1 k2 k2fp twice rc1k2 k4 temp server-temp allswitch budget info cells notes
# Every output lands in this directory (llama.cpp logs in $LOGS); checks.py does the request lines, server drivers and checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
S=results-engine/sp3/checks/task-7
T6=results-engine/sp3/checks/task-6
BIN=engine/build/bin
LOGS=${LOGS:-/tmp/task7-logs}
mkdir -p "$LOGS"
PORT=8097
URL=http://127.0.0.1:$PORT
CK=(uv run "$S/checks.py")

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
# models-engine.ini: KV f16 for qwen2.5-0.5b, q8_0 otherwise
kv_of() { if [ "$1" = qwen2.5-0.5b ]; then echo f16; else echo q8_0; fi; }

# global constraints: no llama-server, VRAM under 200 MiB; above 75 C wait until below 65 C
gpu_ready() {
    if pgrep -x llama-server >/dev/null; then echo "a llama-server is running" >&2; exit 1; fi
    local used t
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -ge 200 ]; then echo "VRAM in use: $used MiB" >&2; exit 1; fi
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)
    if [ "$t" -gt 75 ]; then
        while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -1)" -ge 65 ]; do sleep 10; done
    fi
    echo "gpu: ${used} MiB, ${t} C" >&2
}

# cli PRESET SEQS IN OUT [llama-decide args...]: one llama-decide process over the request lines of IN; the error JSON of a
# failing line goes to OUT.err.json (the CLI stops at the first error)
cli() {
    local p=$1 seqs=$2 in=$3 out=$4 kv
    shift 4
    kv=$(kv_of "$p")
    gpu_ready
    if ! "$BIN/llama-decide" -m "$(model_of "$p")" --decide-seqs "$seqs" -c 4096 -ngl 99 -fa on -ctk "$kv" -ctv "$kv" "$@" \
            < "$in" > "$out" 2> "$LOGS/$(basename "$out").log"; then
        grep '^{"error"' "$LOGS/$(basename "$out").log" > "${out%.jsonl}.err.json" || true
        echo "llama-decide failed on $in (see ${out%.jsonl}.err.json)" >&2
    fi
}

# server_start PRESET SEQS NAME [llama-server args...]; SERVER_ENV (e.g. LLAMA_DECIDE_FAULT=rc1-twice) goes into its environment
SERVER_ENV=
server_start() {
    local p=$1 seqs=$2 name=$3 kv
    shift 3
    kv=$(kv_of "$p")
    gpu_ready
    env ${SERVER_ENV:-} "$BIN/llama-server" -m "$(model_of "$p")" --decide-seqs "$seqs" -c 4096 -ngl 99 -fa on -ctk "$kv" -ctv "$kv" \
        -np 1 --jinja --port $PORT "$@" > "$LOGS/server-$name.log" 2>&1 &
    SERVER_PID=$!
    until curl -sf "$URL/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server $name did not start" >&2; exit 1; }
        sleep 1
    done
}
server_stop() {
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 2
    SERVER_PID=
}
# a failing check must not leave a server holding the GPU
SERVER_PID=
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; true' EXIT

req() { "${CK[@]}" requests "$@"; }

# shared value rule: the tree dump of the baseline and the `after` runs of task-6/ are unchanged
check_c1() {
    gpu_ready
    uv run reference_check.py dump --model "$(model_of qwen2.5-0.5b)" --preset qwen2.5-0.5b --n 20 --tag task7c1
    mv results-engine/reference-task7c1-qwen2.5-0.5b-dump.jsonl $S/c1-tree-regression-dump-qwen2.5-0.5b.jsonl
    uv run sp3_metrics.py compare results-engine/sp3/baseline/dump-qwen2.5-0.5b.jsonl $S/c1-tree-regression-dump-qwen2.5-0.5b.jsonl \
        --max-diff 1e-6 --out $S/c1-tree-regression-qwen2.5-0.5b.json || true
    for m in tree fp; do
        cli qwen2.5-0.5b 64 $T6/requests-after-$m-b5-qwen2.5-0.5b.jsonl $S/c1-after-$m-b5-qwen2.5-0.5b.jsonl --dump-tokens
        "${CK[@]}" same $T6/after-$m-b5-qwen2.5-0.5b.jsonl $S/c1-after-$m-b5-qwen2.5-0.5b.jsonl $S/c1-after-$m-b5-vs-task6-qwen2.5-0.5b.json
    done
}

# final build, requests without the new switches: the same two checks (the dump adds `variant` per entry and tokens.prefixes,
# the engine block order_debias and temperature)
check_regress() {
    gpu_ready
    uv run reference_check.py dump --model "$(model_of qwen2.5-0.5b)" --preset qwen2.5-0.5b --n 20 --tag task7
    mv results-engine/reference-task7-qwen2.5-0.5b-dump.jsonl $S/tree-regression-dump-qwen2.5-0.5b.jsonl
    uv run sp3_metrics.py compare results-engine/sp3/baseline/dump-qwen2.5-0.5b.jsonl $S/tree-regression-dump-qwen2.5-0.5b.jsonl \
        --max-diff 1e-6 --out $S/tree-regression-qwen2.5-0.5b.json || true
    for m in tree fp; do
        cli qwen2.5-0.5b 64 $T6/requests-after-$m-b5-qwen2.5-0.5b.jsonl $S/regress-after-$m-b5-qwen2.5-0.5b.jsonl --dump-tokens
        "${CK[@]}" same $T6/after-$m-b5-qwen2.5-0.5b.jsonl $S/regress-after-$m-b5-qwen2.5-0.5b.jsonl \
            $S/regress-after-$m-b5-vs-task6-qwen2.5-0.5b.json --ignore-engine order_debias,temperature --ignore-tokens prefixes --ignore-entry variant
    done
}

# K = 1 is K = 0 (spec 12.3): 20 tickets, b5, one process each
check_k1() {
    local p=qwen2.5-0.5b
    req --preset $p --n 20 --per-request 5 --debias 0 $S/requests-k0-b5-$p.jsonl
    req --preset $p --n 20 --per-request 5 --debias 1 $S/requests-k1-b5-$p.jsonl
    cli $p 64 $S/requests-k0-b5-$p.jsonl $S/k0-b5-$p.jsonl
    cli $p 64 $S/requests-k1-b5-$p.jsonl $S/k1-b5-$p.jsonl
    "${CK[@]}" same $S/k0-b5-$p.jsonl $S/k1-b5-$p.jsonl $S/k1-vs-k0-$p.json
}

# K = 2 vs the per-key mean of two K = 0 requests (spec 12.3): the rotated K = 0 request first, then the plain one, then K = 2,
# in one process, so slot 1 is restored from the snapshot slot 0 made of the rotated prefix
check_k2() {
    local p seqs
    for p in qwen2.5-0.5b qwen3.5-4b gemma-4-e4b; do
        seqs=64
        [ $p = qwen3.5-4b ] && seqs=32
        req --preset $p --n 5 --per-request 5 --debias 0 --rotate 1/2 $LOGS/k2-rot.jsonl
        req --preset $p --n 5 --per-request 5 --debias 0 $LOGS/k2-plain.jsonl
        req --preset $p --n 5 --per-request 5 --debias 2 $LOGS/k2-k2.jsonl
        cat $LOGS/k2-rot.jsonl $LOGS/k2-plain.jsonl $LOGS/k2-k2.jsonl > $S/requests-k2-$p.jsonl
        cli $p $seqs $S/requests-k2-$p.jsonl $S/k2-$p.jsonl --dump-tokens
        "${CK[@]}" k2 $S/k2-$p.jsonl $S/k2-check-$p.json
    done
}

# the same sequence in full-path mode: per-variant rows in the dump separate variant 1 (restored into slot 1) from variant 0
check_k2fp() {
    local p seqs
    for p in qwen2.5-0.5b qwen3.5-4b gemma-4-e4b; do
        seqs=64
        [ $p = qwen3.5-4b ] && seqs=32
        req --preset $p --n 5 --per-request 5 --debias 0 --rotate 1/2 --scoring full_path $LOGS/k2fp-rot.jsonl
        req --preset $p --n 5 --per-request 5 --debias 0 --scoring full_path $LOGS/k2fp-plain.jsonl
        req --preset $p --n 5 --per-request 5 --debias 2 --scoring full_path $LOGS/k2fp-k2.jsonl
        cat $LOGS/k2fp-rot.jsonl $LOGS/k2fp-plain.jsonl $LOGS/k2fp-k2.jsonl > $S/requests-k2fp-$p.jsonl
        cli $p $seqs $S/requests-k2fp-$p.jsonl $S/k2fp-$p.jsonl --dump-tokens
        "${CK[@]}" k2fp $S/k2fp-$p.jsonl $S/k2fp-check-$p.json
    done
}

# rc1-twice server (spec 12.3): a K = 2 first request is a 500, info right after lists only slot 0, the next K = 1 request
# matches a normal server's K = 1 request
check_twice() {
    local p seqs
    for p in qwen2.5-0.5b qwen3.5-4b; do
        seqs=64
        [ $p = qwen3.5-4b ] && seqs=32
        SERVER_ENV=LLAMA_DECIDE_FAULT=rc1-twice
        server_start $p $seqs twice-$p
        SERVER_ENV=
        "${CK[@]}" server-twice --preset $p --n 5 --url $URL $S/server-rc1-twice-$p.json
        server_stop
        server_start $p $seqs normal-$p
        "${CK[@]}" server-post --preset $p --n 5 --debias 1 --url $URL $S/server-normal-k1-$p.jsonl
        server_stop
        "${CK[@]}" compare $S/server-normal-k1-$p.jsonl $S/server-rc1-twice-$p.next.jsonl $S/server-rc1-twice-next-vs-normal-$p.json
    done
}

# rc1 with K = 2 blocks: every round's first attempt fails and is cleaned up; at --decide-seqs 17 a round holds one state, so the
# retries decode the batches of a normal run
check_rc1k2() {
    local p=qwen2.5-0.5b
    req --preset $p --n 20 --per-request 5 --debias 2 $S/requests-k2-b5-$p.jsonl
    cli $p 17 $S/requests-k2-b5-$p.jsonl $S/k2-b5-seqs17-$p.jsonl
    LLAMA_DECIDE_FAULT=rc1 cli $p 17 $S/requests-k2-b5-$p.jsonl $S/fault-rc1-k2-b5-seqs17-$p.jsonl
    "${CK[@]}" retries $S/fault-rc1-k2-b5-seqs17-$p.jsonl $S/fault-rc1-k2-b5-seqs17-$p.retries.json --min 1
    "${CK[@]}" compare $S/k2-b5-seqs17-$p.jsonl $S/fault-rc1-k2-b5-seqs17-$p.jsonl $S/fault-rc1-k2-vs-normal-$p.json --max-diff 0
}

# K = 4 on Qwen3.5-9B at --decide-seqs 21 (spec 12.3): the same 5 states twice (second: slots 1..3 restored)
check_k4() {
    local p=qwen3.5-9b
    req --preset $p --n 5 --per-request 5 --debias 4 $LOGS/k4.jsonl
    cat $LOGS/k4.jsonl $LOGS/k4.jsonl > $S/requests-k4-$p.jsonl
    cli $p 21 $S/requests-k4-$p.jsonl $S/k4-$p.jsonl
    "${CK[@]}" k4 $S/k4-$p.jsonl $S/k4-check-$p.json --k 4
}

# T = 2 is the T = 1 answer scaled post hoc (spec 8): 20 tickets, b5, one process each (identical decodes)
check_temp() {
    local p=qwen2.5-0.5b
    req --preset $p --n 20 --per-request 5 $S/requests-t1-b5-$p.jsonl
    req --preset $p --n 20 --per-request 5 --temperature 2 $S/requests-t2-b5-$p.jsonl
    cli $p 64 $S/requests-t1-b5-$p.jsonl $S/t1-b5-$p.jsonl
    cli $p 64 $S/requests-t2-b5-$p.jsonl $S/t2-b5-$p.jsonl
    "${CK[@]}" posthoc $S/t1-b5-$p.jsonl $S/t2-b5-$p.jsonl $S/t2-vs-t1-posthoc-$p.json --t 2
}

# --decide-temperature 2 is the default of /v1/decide and /v1/systemone (spec 8): the same request sequence on a default
# server and on a --decide-temperature 2 server
check_server_temp() {
    local p=qwen2.5-0.5b
    server_start $p 9 temp-default-$p
    "${CK[@]}" server-temp --preset $p --n 5 --url $URL $S/server-temp-default-$p.json
    server_stop
    server_start $p 9 temp-2-$p --decide-temperature 2
    "${CK[@]}" server-temp --preset $p --n 5 --url $URL $S/server-temp-2-$p.json
    server_stop
    "${CK[@]}" server-temp-check $S/server-temp-default-$p.json $S/server-temp-2-$p.json $S/server-temp-check-$p.json --t 2
}

# every switch in one request (spec 12.3): full-path, urgency and angry after queue, K = 4, T = 2 on Qwen2.5-0.5B at
# --decide-seqs 64; the same request at T = 1 in its own process for the post-hoc check
check_allswitch() {
    local p=qwen2.5-0.5b
    req --preset $p --n 5 --per-request 5 --debias 4 --scoring full_path --after urgency=queue --after angry=queue --temperature 2 \
        $S/requests-allswitch-t2-$p.jsonl
    req --preset $p --n 5 --per-request 5 --debias 4 --scoring full_path --after urgency=queue --after angry=queue --temperature 1 \
        $S/requests-allswitch-t1-$p.jsonl
    cli $p 64 $S/requests-allswitch-t2-$p.jsonl $S/allswitch-t2-$p.jsonl --dump-tokens
    cli $p 64 $S/requests-allswitch-t1-$p.jsonl $S/allswitch-t1-$p.jsonl --dump-tokens
    "${CK[@]}" allswitch $S/requests-allswitch-t2-$p.jsonl $S/allswitch-t2-$p.jsonl $S/allswitch-check-$p.json
    "${CK[@]}" dumpcheck $S/allswitch-t2-$p.jsonl $S/allswitch-t2-dumpcheck-$p.json --t 2
    "${CK[@]}" dumpcheck $S/allswitch-t1-$p.jsonl $S/allswitch-t1-dumpcheck-$p.json --t 1
    "${CK[@]}" posthoc $S/allswitch-t1-$p.jsonl $S/allswitch-t2-$p.jsonl $S/allswitch-t2-vs-t1-posthoc-$p.json --t 2
}

# the same request on Qwen3.5-4B at --decide-seqs 32: the 400 budget naming 44 vs 28, slot 0 untouched (server)
check_budget() {
    local p=qwen3.5-4b
    server_start $p 32 budget-$p
    "${CK[@]}" server-budget --preset $p --n 5 --url $URL $S/server-budget-allswitch-$p.json
    server_stop
}

# the per-state cell check of a K = 2 request names the K prefixes and K x (state + tail + branches): one state of all 80 tickets
check_cells() {
    local p=qwen2.5-0.5b
    req --preset $p --n 80 --per-request 80 --one-state --debias 2 $S/requests-cells-k2-$p.jsonl
    cli $p 64 $S/requests-cells-k2-$p.jsonl $S/cells-k2-$p.jsonl
    "${CK[@]}" cells $S/cells-k2-$p.err.json $S/k2-$p.jsonl $S/cells-k2-check-$p.json --k 2
}

# --info with a schema body reports seqs_per_state and states_per_round for its K, scoring and after (spec 3.3)
check_info() {
    local p=qwen2.5-0.5b
    req --preset $p --n 1 --per-request 1 --debias 0 $LOGS/i0.jsonl
    req --preset $p --n 1 --per-request 1 --debias 2 $LOGS/i2.jsonl
    req --preset $p --n 1 --per-request 1 --debias 4 $LOGS/i4.jsonl
    req --preset $p --n 1 --per-request 1 --debias 4 --scoring full_path --after urgency=queue --after angry=queue --temperature 2 $LOGS/i4f.jsonl
    cat $LOGS/i0.jsonl $LOGS/i2.jsonl $LOGS/i4.jsonl $LOGS/i4f.jsonl > $S/requests-info-$p.jsonl
    cli $p 64 $S/requests-info-$p.jsonl $S/info-$p.jsonl --info
    "${CK[@]}" info $S/info-$p.jsonl $S/info-check-$p.json --expect 4:15,8:7,16:3,44:1
}

for c in "$@"; do
    case "$c" in
        c1) check_c1 ;;
        regress) check_regress ;;
        k1) check_k1 ;;
        k2) check_k2 ;;
        k2fp) check_k2fp ;;
        twice) check_twice ;;
        rc1k2) check_rc1k2 ;;
        k4) check_k4 ;;
        temp) check_temp ;;
        server-temp) check_server_temp ;;
        allswitch) check_allswitch ;;
        budget) check_budget ;;
        info) check_info ;;
        cells) check_cells ;;
        notes) "${CK[@]}" notes ;;
        *) echo "unknown check $c" >&2; exit 2 ;;
    esac
done
