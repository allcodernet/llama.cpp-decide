#!/usr/bin/env bash
# Sub-project 4 (image input): live steps on the RTX 3090 test machine (spec docs/specs/2026-10-02-image-input-design.md).
# Usage: [PRESET=<vision preset>] results-engine/image/run.sh STEP...        (PRESET default qwen3.8-27b-vision)
# Outputs under results-engine/image/, logs in $LOGS (default /tmp/image-logs). Servers: the preset router of serve-engine.sh
# on 127.0.0.1:8097 with models-engine-3090.ini, started fresh for a step and stopped after it. GPU and router helpers
# (gpu_ready: no llama-server and VRAM under GPU_VRAM_MAX MiB, waiting up to GPU_WAIT_MIN minutes; router_start, router_load,
# router_stop, the EXIT trap) come from results-engine/sp3/eval/run.sh, sourced without steps; its gpu-log.txt lands here.
# LLAMA_MEDIA_MARKER is fixed to <__media__> so /completion replays know the marker (spec 7).
#   smoke     Task 2: smoke.py server on PRESET -> task-2/smoke-<preset>.json                                     [server]
#   cli       Task 3: smoke.py cli-text (old vs new llama-decide, Qwen2.5-0.5B) and cli-image (PRESET) -> task-3/       [CLI]
#   regression      spec 9.2: checks.py text-run on a fresh router of the old build (engine-49785a95d) and of the new
#                   build, for qwen3.8-27b (no projector) and qwen3.8-27b-vision (--mmproj); text-compare ->
#                   regression/<preset>.json                                                                          [server]
#   text-integrity  spec 9.2: results-engine/3090/run.sh dumps replays offline batch for qwen3.8-27b on the new build into
#                   text-integrity/ (its baseline input copied from results-engine/3090/baseline)                [CLI, server]
#   no-vision       spec 9.3 on qwen2.5-0.5b and qwen3.8-27b: no-vision/<preset>-with.json, -without.json, <preset>.json,
#                   cli-<preset>.json                                                                         [server, CLI]
#   integrity  spec 9.4 on PRESET: the six reference dumps (CLI) on states-integrity.jsonl (a tag without room at the preset's
#              decide-seqs, per task-0/fit-<preset>.json, is listed in left-out.txt instead), retry (CLI), then on a router
#              of the new build the compares and batch; malformed on a fresh router and its comparison run on another fresh
#              router; held on a router of the old build, then of the new build -> integrity/<preset>/; a failed check is
#              listed in failures.txt and the step exits 1 at its end; with integrity/<preset>/SKIPPED.txt the step only
#              prints that the preset is skipped; INTEGRITY_DIR=dir writes there instead of integrity/<preset>/; exits 1
#              without a run when llama-decide --info reports non_causal_projector for PRESET, unless FORCE=1  [CLI, server]
#   held       the held pair of integrity alone on PRESET (old build, then new build); its earlier held-old / held-new lines
#              leave failures.txt first, the other lines stay                                                     [server]
#   swa        Task 6b (spec 5.3 rev 5) -> swa/: checks.py swa-info (llama-decide --info) on gemma-4-e4b, gemma-4-e4b-16k,
#              gemma-4-e4b-16k-swa-full (both with a prefix of 10000 words) and gemma-3-4b-vision; checks.py swa on a
#              fresh router of the new build each:
#              gemma-4-e4b-16k idle (K = 1, --round), idle K = 2 and with a text chat resident, gemma-4-e4b-16k-swa-full,
#              gemma-3-4b-vision; a failed check is listed in swa/failures.txt and the step exits 1 at its end  [CLI, server]
#   swa-regression  Task 6b: checks.py text-run on qwen3.8-27b, old build then new build (each sidecar names its
#                   server binary), and text-compare -> swa/regression/                                       [server]
#   swa-integrity   Task 6b: a b5 run of the first 20 test tickets on gemma-4-e4b (the baseline of the batch step), then
#                   results-engine/3090/run.sh dumps replays offline batch for gemma-4-e4b into swa/text-integrity/
#                                                                                                         [CLI, server]
#   swa-control  Task 6b diagnostic: swa_control.py pair (the round-1 pair of swa --round "distinct" against each text alone,
#              twice) on gemma-4-e4b-16k at full size (4654, 4454) and at a quarter (1163, 1113), and on
#              gemma-4-e4b-16k-swa-full at full size, all on the new build; gemma-4-e4b-16k at full size on the old
#              build (engine-49785a95d, file suffix -old) -> swa/control/; a pair that fails is listed in
#              swa/control/failures.txt and the step exits 1 at its end                                          [server]
#   ubatch     spec 5.3 on PRESET (a non-causal projector): checks.py ubatch -> integrity/<preset>/ubatch.json; refuses as
#              integrity does when info reports non_causal_projector, unless FORCE=1                               [CLI]
#   gate       spec 9.5: checks.py gate on integrity/gemma-3-4b-vision -> gate.json; first, as integrity, exits 1 when
#              llama-decide --info reports non_causal_projector for gemma-3-4b-vision, unless FORCE=1            [CLI]
#   gate-off   spec 9.5, after a failed gate and the rebuild: checks.py gate-off on gemma-3-4b-vision -> gate-off.json [server];
#              with integrity/gemma-3-4b-vision/SKIPPED.txt it only writes {"skipped": <reason>, "pass": true}       [no GPU]
#   coco-data  spec 8: the COCO 2017 annotations (local/coco/, once; SHA-256 recorded in coco/annotations.sha256 on the first
#              run, checked on every later run), coco.py select -> coco/subset.jsonl, NOTICE.md, then
#              coco.py download -> local/image/coco/, coco/images.sha256                                       [no GPU]
#   coco       spec 9.6 on qwen3.8-27b-vision: coco.py decide and chat on one router, then score and readings ->
#              coco/{decide,chat,scores,chat-readings}-qwen3.8-27b-vision.*                                      [server]
#   coco-cached  Task 7 addition: coco.py chat --cache-prompt on a fresh router, then readings against the decide run ->
#              coco/chat-cached{,-readings}-qwen3.8-27b-vision.*                                                 [server]
#   Both coco steps refuse to overwrite an existing result file unless FORCE=1.
#   final      spec rev 6 on the final engine -> final/: smoke.py server on PRESET (smoke-<preset>.json), checks.py limits
#              on a fresh router (limits-<preset>.json), the text regression of swa-regression into final/regression/;
#              refuses to overwrite unless FORCE=1; a failed check is listed in final/failures.txt, exit 1 at the end
#              (the integrity checks of the final engine: INTEGRITY_DIR=results-engine/image/final/integrity/<preset>)
#                                                                                                                [server]
#   double-fault  checks.py double-fault on PRESET on a fresh router of the new build ->
#              publish/double-fault-<engine commit>-<preset>.json; refuses to overwrite unless FORCE=1           [server]
#   publish    the engine with the sequence budget before prefix preparation -> publish/: double-fault as above, smoke.py
#              server on PRESET (smoke-<preset>.json), the text regression of swa-regression into publish/regression/;
#              refuses to overwrite unless FORCE=1; a failed check is listed in publish/failures.txt, exit 1 at the end [server]
#   recheck    the engine regenerated with changes to comments, commit messages and test text only (same code) ->
#              publish/recheck/: the text-run of the 27B regression on the new build, compared with the recorded
#              publish/regression/ run of 185e9e0a7 (text-compare, max abs diff 0 required), and smoke.py server on
#              PRESET; refuses to overwrite unless FORCE=1                                                       [server]
set -euo pipefail
export ENGINE_PRESETS=models-engine-3090.ini TRIAGE_DATA=data GPU_VRAM_MAX=1500 GPU_WAIT_MIN=20 \
    EVAL_DIR=results-engine/image LOGS=${LOGS:-/tmp/image-logs} LLAMA_MEDIA_MARKER='<__media__>'
unset DATA COST_DIR MODELS BASELINE_DIR ENGINE_BIN
P=${PRESET:-qwen3.8-27b-vision}
STEPS=("$@")
set --
# shellcheck source=../sp3/eval/run.sh
source "$(dirname "$0")/../sp3/eval/run.sh"
O=results-engine/image
IMAGES=local/image/integrity
OLD_BIN=engine-49785a95d/build/bin

setting() {  # setting PRESET KEY: the key of models-engine-3090.ini ([*] overlaid with the preset)
    uv run python -c "from decide_client import _preset_settings; print(_preset_settings('$1')['$2'])"
}
start_on() {  # start_on old|new NAME PRESET: a fresh router of that build (engine-49785a95d or engine/) with PRESET loaded
    if [ "$1" = old ]; then export ENGINE_BIN=$OLD_BIN; else unset ENGINE_BIN; fi
    router_start "$2"
    router_load "$3"
}
server_bin() {  # the llama-server of the build start_on chose (serve-engine.sh's ENGINE_BIN, default engine/build/bin)
    echo "${ENGINE_BIN:-engine/build/bin}/llama-server"
}
stop() {  # stop NAME: router_stop, then back to the new build
    router_stop "$1"
    unset ENGINE_BIN
}

do_smoke() {
    start_on new "smoke-$P" "$P"
    uv run $O/smoke.py server --url "$URL" --model "$P" --images $IMAGES --out "$O/task-2/smoke-$P.json"
    stop "smoke-$P"
}

do_cli() {
    gpu_ready cli-text
    uv run $O/smoke.py cli-text --old-bin $OLD_BIN --model "$(setting qwen2.5-0.5b model)" --preset qwen2.5-0.5b \
        --out "$O/task-3/cli-text.json"
    gpu_ready "cli-image-$P"
    uv run $O/smoke.py cli-image --model "$(setting "$P" model)" --preset "$P" --images $IMAGES --out "$O/task-3/cli-image-$P.json"
}

do_regression() {
    local p b d=$O/regression
    mkdir -p "$d"
    for p in qwen3.8-27b qwen3.8-27b-vision; do
        for b in old new; do
            start_on $b "regression-$p-$b" $p
            uv run $O/checks.py text-run --url "$URL" --model $p --server-bin "$(server_bin)" --out "$d/$p-$b.jsonl"
            stop "regression-$p-$b"
        done
        uv run $O/checks.py text-compare "$d/$p-old.jsonl" "$d/$p-new.jsonl" --out "$d/$p.json"
    done
}

do_text_integrity() {
    local d=$O/text-integrity
    mkdir -p "$d/baseline/test/preds"
    cp results-engine/3090/baseline/test/preds/decide-qwen3.8-27b-b5-keywords.jsonl "$d/baseline/test/preds/"
    PRESET=qwen3.8-27b OUT_DIR=$d SEQS=64 LOGS=$LOGS/text-integrity results-engine/3090/run.sh dumps replays offline batch
}

do_no_vision() {
    local p d=$O/no-vision
    mkdir -p "$d"
    for p in qwen2.5-0.5b qwen3.8-27b; do
        start_on new "nv-$p-with" $p
        uv run $O/checks.py no-vision --url "$URL" --model $p --image $IMAGES/coco-000000039769.jpg \
            --log "$LOGS/router-nv-$p-with.log" --out "$d/$p-with.json"
        stop "nv-$p-with"
        start_on new "nv-$p-without" $p
        uv run $O/checks.py no-vision --url "$URL" --model $p --image $IMAGES/coco-000000039769.jpg --skip-rejected \
            --out "$d/$p-without.json"
        stop "nv-$p-without"
        uv run $O/checks.py no-vision-compare "$d/$p-with.json" "$d/$p-without.json" --out "$d/$p.json"
        gpu_ready "nv-cli-$p"
        uv run $O/checks.py cli-no-vision --model "$(setting $p model)" --preset $p --image $IMAGES/coco-000000039769.jpg \
            --out "$d/cli-$p.json"
    done
}

REF_TAGS=(tree-one tree-multi fp after debias2 t0.5)   # checks.py REF_TAGS
ref_args() {  # ref_args TAG: the reference_check.py dump arguments of that tag (the six integrity reference configurations of Task 6)
    case $1 in
        tree-one)   echo "--schema $O/schema-one.json" ;;
        tree-multi) echo "--schema $O/schema-multi.json" ;;
        fp)         echo "--schema $O/schema-multi.json --scoring full_path" ;;
        after)      echo "--schema $O/schema-multi.json --after dog=cat" ;;
        debias2)    echo "--schema $O/schema-multi.json --order-debias 2" ;;
        t0.5)       echo "--schema $O/schema-multi.json --temperature 0.5" ;;
    esac
}
fits() {  # fits PRESET TAG: True when the fit check (fit_check.py) found room for TAG (do_integrity checked that the fit file exists)
    uv run python -c "import json; print(json.load(open('$O/task-0/fit-$1.json')).get('$2', 0) >= 1)"
}
note_fail() {  # note_fail DIR CHECK: record a failed check and go on; the step exits 1 at its end
    echo "$2" >> "$1/failures.txt"
    echo "FAILED: $2" >&2
}

non_causal_guard() {  # non_causal_guard PRESET: exit 1 when llama-decide --info reports non_causal_projector, unless FORCE=1
    local reason
    [ "${FORCE:-0}" = 1 ] && return 0
    gpu_ready "vision-reason-$1"
    reason=$(uv run $O/checks.py vision-reason --model "$(setting "$1" model)" --preset "$1")   # a failed --info ends the step
    if [ "$reason" = non_causal_projector ]; then
        echo "$1: info reports non_causal_projector, so this engine answers its image states with 400 vision_unavailable;" \
             "its committed results came from engine 0001-0041 (9bae50c77). Set FORCE=1 to run anyway (overwrites them)." >&2
        exit 1
    fi
}

do_integrity() {
    local d=${INTEGRITY_DIR:-$O/integrity/$P} t gguf seqs img=$IMAGES/coco-000000039769.jpg fit=$O/task-0/fit-$P.json
    [ -e "$fit" ] || { echo "$fit is missing: run the fit check (results-engine/image/fit_check.py $P) first" >&2; exit 1; }
    if [ ! -e "$d/SKIPPED.txt" ] && grep -q '"unavailable"' "$fit"; then   # the fit check: download or load failed
        mkdir -p "$d"
        uv run python -c "import json; print('fit check: ' + json.load(open('$fit'))['unavailable'])" > "$d/SKIPPED.txt"
    fi
    if [ -e "$d/SKIPPED.txt" ]; then echo "$P: skipped: $(head -1 "$d/SKIPPED.txt")"; return 0; fi
    non_causal_guard "$P"
    mkdir -p "$d"
    rm -f "$d/failures.txt" "$d/left-out.txt"
    gguf=$(setting "$P" model)
    seqs=$(setting "$P" decide-seqs)
    for t in "${REF_TAGS[@]}"; do
        if [ "$(fits "$P" "$t")" != True ]; then
            echo "$t: no room at decide-seqs $seqs (task-0/fit-$P.json)" | tee -a "$d/left-out.txt"
            continue
        fi
        gpu_ready "dump-$t-$P"
        # shellcheck disable=SC2046  # ref_args is split on purpose
        uv run reference_check.py dump --model "$gguf" --preset "$P" --n 8 --states $O/states-integrity.jsonl \
            --decide-seqs "$seqs" --tag "$t" --out-dir "$d" $(ref_args "$t") > "$LOGS/dump-$t-$P.log" 2>&1 \
            || note_fail "$d" "dump $t (log $LOGS/dump-$t-$P.log)"
        tail -1 "$LOGS/dump-$t-$P.log"
    done
    gpu_ready "retry-$P"
    uv run $O/checks.py retry --model "$gguf" --preset "$P" --states $O/states-integrity.jsonl --schema $O/schema-multi.json \
        --out "$d/retry.json" || note_fail "$d" retry
    start_on new "integrity-$P" "$P"
    for t in "${REF_TAGS[@]}"; do
        [ -e "$d/reference-$t-$P-dump.jsonl" ] || continue
        uv run reference_check.py compare --preset "$P" --tag "$t" --out-dir "$d" --url "$URL" > "$LOGS/compare-$t-$P.log" 2>&1 \
            || note_fail "$d" "compare $t (log $LOGS/compare-$t-$P.log)"
        grep -q '^pass: True$' "$LOGS/compare-$t-$P.log" || note_fail "$d" "reference $t: not pass: True"
        grep '^pass: ' "$LOGS/compare-$t-$P.log" | sed "s/^/$t /" || true
    done
    uv run $O/checks.py batch --url "$URL" --model "$P" --states $O/states-integrity.jsonl --schema $O/schema-multi.json \
        --out "$d/batch.json" || note_fail "$d" batch
    stop "integrity-$P"
    start_on new "malformed-$P" "$P"
    uv run $O/checks.py malformed --url "$URL" --model "$P" --image $img --out "$d/malformed.json" || note_fail "$d" malformed
    stop "malformed-$P"
    start_on new "malformed-fresh-$P" "$P"
    uv run $O/checks.py malformed --url "$URL" --model "$P" --image $img --skip-cases --out "$d/malformed-fresh.json" \
        || note_fail "$d" malformed-fresh
    stop "malformed-fresh-$P"
    uv run $O/checks.py malformed-compare "$d/malformed.json" "$d/malformed-fresh.json" --out "$d/malformed-compare.json" \
        || note_fail "$d" "malformed-compare (a rejected request changed the engine)"
    held_pair "$d"
    if [ -e "$d/failures.txt" ]; then echo "$P: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "$P: every integrity check passed"
}

held_pair() {  # held_pair DIR: the held check on the old build, then on the new build
    local img=$IMAGES/coco-000000039769.jpg
    start_on old "held-old-$P" "$P"
    uv run $O/checks.py held --url "$URL" --model "$P" --image $img --out "$1/held-old.json" || note_fail "$1" held-old
    stop "held-old-$P"
    start_on new "held-new-$P" "$P"
    uv run $O/checks.py held --url "$URL" --model "$P" --image $img --old "$1/held-old.json" --out "$1/held-new.json" \
        || note_fail "$1" held-new
    stop "held-new-$P"
}

do_held() {
    local d=$O/integrity/$P
    mkdir -p "$d"
    if [ -e "$d/failures.txt" ]; then
        sed -i '/^held-old$/d; /^held-new$/d' "$d/failures.txt"
        [ -s "$d/failures.txt" ] || rm "$d/failures.txt"
    fi
    held_pair "$d"
    if grep -qx 'held-old\|held-new' "$d/failures.txt" 2>/dev/null; then echo "$P: held failed" >&2; return 1; fi
    echo "$P: held passed"
}

swa_run() {  # swa_run PRESET NAME ARGS...: checks.py swa ARGS on a fresh router of the new build -> swa/PRESET-NAME.json
    local p=$1 n=$2
    shift 2
    start_on new "swa-$p-$n" "$p"
    uv run $O/checks.py swa --url "$URL" --model "$p" --out "$O/swa/$p-$n.json" "$@" || note_fail $O/swa "swa $p-$n"
    stop "swa-$p-$n"
}

do_swa() {
    local d=$O/swa e p prefix
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    for e in gemma-4-e4b:4096 gemma-4-e4b-16k:9216 gemma-4-e4b-16k-swa-full:16384 gemma-3-4b-vision:16384; do
        p=${e%%:*} prefix=()
        case $p in gemma-4-e4b-16k*) prefix=(--prefix-words 10000) ;; esac   # over the SWA limit, within n_ctx - 16
        gpu_ready "swa-info-$p"
        uv run $O/checks.py swa-info --model "$(setting "$p" model)" --preset "$p" --expect-swa-cells "${e##*:}" "${prefix[@]}" \
            --out "$d/info-$p.json" || note_fail "$d" "swa-info $p"
    done
    swa_run gemma-4-e4b-16k idle --expect-swa-cells 9216 --round
    swa_run gemma-4-e4b-16k debias2 --expect-swa-cells 9216 --debias 2
    swa_run gemma-4-e4b-16k chat --expect-swa-cells 9216 --chat
    swa_run gemma-4-e4b-16k-swa-full idle --expect-swa-cells 16384
    swa_run gemma-3-4b-vision idle --expect-swa-cells 16384
    if [ -e "$d/failures.txt" ]; then echo "swa: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "swa: every check passed"
}

do_swa_control() {
    local d=$O/swa/control e b p n s x
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    for e in new:gemma-4-e4b-16k:full:4654,4454 new:gemma-4-e4b-16k:quarter:1163,1113 \
             new:gemma-4-e4b-16k-swa-full:full:4654,4454 old:gemma-4-e4b-16k:full:4654,4454; do
        IFS=: read -r b p n s <<< "$e"
        x=$p-$n
        [ "$b" = old ] && x=$x-old
        start_on "$b" "swa-control-$x" "$p"
        uv run $O/swa_control.py pair --url "$URL" --model "$p" --sizes "$s" --repeat 2 --out "$d/$x.json" \
            || note_fail "$d" "pair $x"
        stop "swa-control-$x"
    done
    if [ -e "$d/failures.txt" ]; then echo "swa-control: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "swa-control: every pair ran"
}

text_regression() {  # text_regression DIR NAME: checks.py text-run on qwen3.8-27b, old build then new build, and text-compare -> DIR
    local b d=$1
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    for b in old new; do
        start_on $b "$2-$b" qwen3.8-27b
        uv run $O/checks.py text-run --url "$URL" --model qwen3.8-27b --server-bin "$(server_bin)" --out "$d/qwen3.8-27b-$b.jsonl" \
            || note_fail "$d" "text-run $b"
        stop "$2-$b"
    done
    uv run $O/checks.py text-compare "$d/qwen3.8-27b-old.jsonl" "$d/qwen3.8-27b-new.jsonl" --out "$d/qwen3.8-27b.json" \
        || note_fail "$d" text-compare
    if [ -e "$d/failures.txt" ]; then echo "$2: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "$2: identical"
}

do_swa_regression() {
    text_regression $O/swa/regression swa-regression
}

do_final() {
    local d=$O/final img=$IMAGES/coco-000000039769.jpg
    fresh "$d/smoke-$P.json" "$d/limits-$P.json" "$d/regression/qwen3.8-27b.json"
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    start_on new "final-smoke-$P" "$P"
    uv run $O/smoke.py server --url "$URL" --model "$P" --images $IMAGES --out "$d/smoke-$P.json" || note_fail "$d" "smoke $P"
    stop "final-smoke-$P"
    start_on new "final-limits-$P" "$P"
    uv run $O/checks.py limits --url "$URL" --model "$P" --image $img --out "$d/limits-$P.json" || note_fail "$d" "limits $P"
    stop "final-limits-$P"
    text_regression "$d/regression" final-regression || note_fail "$d" "regression (see regression/failures.txt)"
    if [ -e "$d/failures.txt" ]; then echo "final: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "final: every check passed"
}

double_fault() {  # double_fault DIR: checks.py double-fault on a fresh router of the new build -> DIR/double-fault-<engine>-<preset>.json
    local f
    f=$1/double-fault-$(git -C engine rev-parse --short=9 HEAD)-$P.json
    fresh "$f"
    mkdir -p "$1"
    start_on new "double-fault-$P" "$P"
    uv run $O/checks.py double-fault --url "$URL" --model "$P" --image $IMAGES/coco-000000039769.jpg --out "$f" \
        || note_fail "$1" "double-fault $P"
    stop "double-fault-$P"
}

do_double_fault() {
    double_fault $O/publish
    if [ -e "$O/publish/failures.txt" ]; then echo "double-fault: failed checks:" >&2; cat "$O/publish/failures.txt" >&2; return 1; fi
}

do_publish() {
    local d=$O/publish
    fresh "$d/smoke-$P.json" "$d/regression/qwen3.8-27b.json"
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    double_fault "$d"
    start_on new "publish-smoke-$P" "$P"
    uv run $O/smoke.py server --url "$URL" --model "$P" --images $IMAGES --out "$d/smoke-$P.json" || note_fail "$d" "smoke $P"
    stop "publish-smoke-$P"
    text_regression "$d/regression" publish-regression || note_fail "$d" "regression (see regression/failures.txt)"
    if [ -e "$d/failures.txt" ]; then echo "publish: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "publish: every check passed"
}

do_recheck() {
    local d=$O/publish/recheck
    fresh "$d/qwen3.8-27b-new.jsonl" "$d/smoke-$P.json"
    mkdir -p "$d"
    rm -f "$d/failures.txt"
    start_on new recheck-regression qwen3.8-27b
    uv run $O/checks.py text-run --url "$URL" --model qwen3.8-27b --server-bin "$(server_bin)" --out "$d/qwen3.8-27b-new.jsonl" \
        || note_fail "$d" "text-run new"
    stop recheck-regression
    uv run $O/checks.py text-compare $O/publish/regression/qwen3.8-27b-new.jsonl "$d/qwen3.8-27b-new.jsonl" \
        --out "$d/qwen3.8-27b.json" || note_fail "$d" text-compare
    start_on new "recheck-smoke-$P" "$P"
    uv run $O/smoke.py server --url "$URL" --model "$P" --images $IMAGES --out "$d/smoke-$P.json" || note_fail "$d" "smoke $P"
    stop "recheck-smoke-$P"
    if [ -e "$d/failures.txt" ]; then echo "recheck: failed checks:" >&2; cat "$d/failures.txt" >&2; return 1; fi
    echo "recheck: every check passed"
}

do_swa_integrity() {
    local d=$O/swa/text-integrity
    mkdir -p "$d/baseline/test"
    start_on new swa-b5-gemma-4-e4b gemma-4-e4b
    uv run decide_client.py --model gemma-4-e4b --batch 5 --limit 20 --test-set "$TRIAGE_DATA/triage_test.jsonl" \
        --out-dir "$d/baseline/test" --ensure-t1 > "$LOGS/swa-b5-gemma-4-e4b.log" 2>&1
    stop swa-b5-gemma-4-e4b
    PRESET=gemma-4-e4b OUT_DIR=$d LOGS=$LOGS/swa-integrity results-engine/3090/run.sh dumps replays offline batch
}

do_ubatch() {
    if [ -e "$O/integrity/$P/SKIPPED.txt" ]; then echo "$P: skipped: $(head -1 "$O/integrity/$P/SKIPPED.txt")"; return 0; fi
    non_causal_guard "$P"
    mkdir -p "$O/integrity/$P"
    gpu_ready "ubatch-$P"
    uv run $O/checks.py ubatch --model "$(setting "$P" model)" --preset "$P" --image $IMAGES/coco-000000039769.jpg \
        --out "$O/integrity/$P/ubatch.json"
}

do_gate() {
    non_causal_guard gemma-3-4b-vision
    uv run $O/checks.py gate --dir $O/integrity/gemma-3-4b-vision --preset gemma-3-4b-vision --out $O/gate.json
}

do_gate_off() {
    local skipped=$O/integrity/gemma-3-4b-vision/SKIPPED.txt
    if [ -e "$skipped" ]; then   # gemma cannot run: no live check; the vision_reason_for ctest covers the flipped constant
        uv run python -c "import json, sys; json.dump({'skipped': open('$skipped').readline().strip(), 'pass': True}, open('$O/gate-off.json', 'w'), indent=1)"
        echo "gate-off: not checked live (gemma unavailable: $(head -1 "$skipped"))"
        return 0
    fi
    start_on new gate-off gemma-3-4b-vision
    uv run $O/checks.py gate-off --url "$URL" --model gemma-3-4b-vision --image $IMAGES/coco-000000039769.jpg --out $O/gate-off.json
    stop gate-off
}

do_coco_data() {
    local zip=local/coco/annotations_trainval2017.zip
    if [ ! -e $zip ]; then
        mkdir -p local/coco
        curl -fL --retry 3 -o $zip.part http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        mv $zip.part $zip
    fi
    # the first run records the archive's SHA-256 (committed); every later run checks it
    if [ -e $O/coco/annotations.sha256 ]; then
        sha256sum -c $O/coco/annotations.sha256
    else
        mkdir -p $O/coco
        sha256sum $zip > $O/coco/annotations.sha256
    fi
    uv run $O/coco.py select --annotations $zip --out-dir $O/coco
    uv run $O/coco.py download --subset $O/coco/subset.jsonl --dir local/image/coco --sums $O/coco/images.sha256
}

fresh() {  # fresh FILE...: exit 1 when one of the result files exists, unless FORCE=1
    local f
    if [ "${FORCE:-0}" = 1 ]; then return 0; fi
    for f in "$@"; do
        if [ -e "$f" ]; then echo "$f exists: set FORCE=1 to overwrite it" >&2; exit 1; fi
    done
}

do_coco() {
    local m=qwen3.8-27b-vision d=$O/coco
    fresh $d/decide-$m.jsonl $d/chat-$m.jsonl $d/scores-$m.json $d/chat-readings-$m.json
    start_on new "coco-$m" $m
    uv run $O/coco.py decide --url "$URL" --model $m --subset $d/subset.jsonl --images local/image/coco --out $d/decide-$m.jsonl
    uv run $O/coco.py chat --url "$URL" --model $m --subset $d/subset.jsonl --images local/image/coco --out $d/chat-$m.jsonl
    stop "coco-$m"
    uv run $O/coco.py score --model $m --subset $d/subset.jsonl --decide $d/decide-$m.jsonl --chat $d/chat-$m.jsonl \
        --out $d/scores-$m.json
    uv run $O/coco.py readings --model $m --subset $d/subset.jsonl --decide $d/decide-$m.jsonl --chat $d/chat-$m.jsonl \
        --out $d/chat-readings-$m.json
}

do_coco_cached() {
    local m=qwen3.8-27b-vision d=$O/coco
    fresh $d/chat-cached-$m.jsonl $d/chat-cached-readings-$m.json
    start_on new "coco-cached-$m" $m
    uv run $O/coco.py chat --url "$URL" --model $m --subset $d/subset.jsonl --images local/image/coco --cache-prompt \
        --out $d/chat-cached-$m.jsonl
    stop "coco-cached-$m"
    uv run $O/coco.py readings --model $m --subset $d/subset.jsonl --decide $d/decide-$m.jsonl --chat $d/chat-cached-$m.jsonl \
        --out $d/chat-cached-readings-$m.json
}

for s in "${STEPS[@]}"; do "do_${s//-/_}"; done
