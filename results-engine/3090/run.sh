#!/usr/bin/env bash
# Follow-ups Task 5: Qwen3.8-27B (preset qwen3.8-27b of models-engine-3090.ini) on the RTX 3090 test machine. Sets the
# environment of this evaluation (follow-ups Task 5) and calls the existing scripts; every output lands under $OUT_DIR, logs in $LOGS
# (default /tmp/task5-logs).
# Settings (unset: the Task 5 values): PRESET (default qwen3.8-27b), OUT_DIR (default results-engine/3090), SEQS (the
# dumps' --decide-seqs; default the preset's decide-seqs, 64). Another model: PRESET=<p> OUT_DIR=results-engine/<dir>.
# T: `eval`, `batch` and `latency` pin T = 1 (--ensure-t1); the dumps (llama-decide, no --decide-temperature) and the
# replays (/completion) do not read the preset's decide-temperature; `systemone` runs at the preset's temperature. The
# committed results came from runs at T = 1 before models-engine-3090.ini got decide-temperature = 0.8992, so a re-run
# of `systemone` now runs at 0.8992 (its checks hold at any T): remove that line first for T = 1, as in the original
# order (every step at T = 1, the rule's preset change last, then `preset`).
# Usage: [PRESET=<p> OUT_DIR=<dir> SEQS=<n>] results-engine/3090/run.sh STEP...
#   eval       results-engine/sp3/eval/run.sh baseline runs score calibration: tree baseline (b5, keywords, test 80 +
#              train 320) in baseline/, the four switch configurations in eval/, scores, pooled sign tests against the
#              own baseline, summary.json/md (cost: the baseline runs), calibration-<preset>.json               [server]
#   dumps      reference_check.py dump on the first 20 test tickets, one state per line, --decide-seqs $SEQS
#              (default the preset's): tree (tag tree) and full-path (tag fp) ->
#              integrity/reference-<tag>-<preset>-dump.jsonl                                   [CLI, server stopped]
#   replays    reference_check.py compare of both dumps through one router (/completion, n_probs 512): reference-<tag>-
#              <preset>.json and the replays -replay.jsonl.gz (compare --offline recomputes without a server)    [server]
#   offline    reference_check.py compare --offline of both dumps from their replays in a temporary directory; the
#              results must equal the committed ones byte for byte -> integrity/offline-recompute.txt     [no GPU]
#   batch      batch invariance: decide_client.py --batch 1 --limit 20 into integrity/batch-invariance/, then
#              batch_invariance.py against the first 20 rows of the b5 baseline test preds -> integrity/batch-invariance-
#              <preset>.json                                                                                    [server]
#   latency    decide_client.py --batch 1 on the 80 test tickets into latency/ (tree p50 per ticket at b1; b5 is the
#              baseline's)                                                                                      [server]
#   systemone  systemone_smoke.py and the official SDK test (local/typesafe-sdk-python) against the preset:
#              systemone/smoke.txt, systemone/sdk.txt                                                           [server]
#   preset     results-engine/sp3/eval/run.sh preset (after every step above; the live check only once
#              models-engine-3090.ini holds the fitted T, see that script)                                      [server]
#   notes      results-engine/3090/notes.py --preset --dir --seqs -> $OUT_DIR/notes.md (tables from the files above;
#              its prose names the model by --title, default the preset)                                        [no GPU]
# With OUT_DIR unset, a re-run writes into the committed results-engine/3090/: move it aside first, or set OUT_DIR.
# reference_check.py dump/compare has no overwrite guard, the runs and preset steps use decide_client.py --force, and
# the baseline step refuses existing files (decide_client.py stops on them).
# GPU and router helpers (gpu_ready with GPU_VRAM_MAX / GPU_WAIT_MIN, router_start/load/stop on 127.0.0.1:8097, the
# EXIT trap) come from results-engine/sp3/eval/run.sh, sourced without steps; gpu_ready logs to eval/gpu-log.txt.
set -euo pipefail
P=${PRESET:-qwen3.8-27b}
O=${OUT_DIR:-results-engine/3090}
export ENGINE_PRESETS=models-engine-3090.ini TRIAGE_DATA=data GPU_VRAM_MAX=1500 GPU_WAIT_MIN=20 MODELS=$P \
    EVAL_DIR=$O/eval BASELINE_DIR=$O/baseline LOGS=${LOGS:-/tmp/task5-logs}
unset DATA COST_DIR
STEPS=("$@")
set --
# shellcheck source=../sp3/eval/run.sh
source "$(dirname "$0")/../sp3/eval/run.sh"
I=$O/integrity
GGUF=$(uv run python -c "from decide_client import _preset_settings; print(_preset_settings('$P')['model'])")
SEQS=${SEQS:-$(uv run python -c "from decide_client import _preset_settings; print(_preset_settings('$P')['decide-seqs'])")}

do_eval() {
    results-engine/sp3/eval/run.sh baseline runs score calibration
}

do_dumps() {
    gpu_ready dump-tree
    uv run reference_check.py dump --model "$GGUF" --preset $P --n 20 --tag tree --decide-seqs $SEQS --out-dir "$I" \
        2>&1 | tee "$LOGS/dump-tree.log" | tail -1
    gpu_ready dump-fp
    uv run reference_check.py dump --model "$GGUF" --preset $P --n 20 --tag fp --scoring full_path --decide-seqs $SEQS \
        --out-dir "$I" 2>&1 | tee "$LOGS/dump-fp.log" | tail -1
}

do_replays() {
    local t
    router_start replays
    router_load $P
    for t in tree fp; do
        uv run reference_check.py compare --preset $P --tag $t --out-dir "$I" --url "$URL" > "$LOGS/compare-$t.log"
        tail -1 "$LOGS/compare-$t.log"
    done
    router_stop replays
}

do_offline() {
    local t tmp out=()
    tmp=$(mktemp -d)
    for t in tree fp; do
        cp "$I/reference-$t-$P-dump.jsonl" "$I/reference-$t-$P-replay.jsonl.gz" "$tmp/"
        uv run reference_check.py compare --preset $P --tag $t --out-dir "$tmp" --offline > /dev/null
        if cmp -s "$tmp/reference-$t-$P.json" "$I/reference-$t-$P.json"; then out+=("$t identical"); else out+=("$t DIFFERS"); fi
    done
    rm -rf "$tmp"
    echo "${out[*]}" | tee "$I/offline-recompute.txt"
    if [[ ${out[*]} == *DIFFERS* ]]; then echo "offline: a recomputed reference differs from the committed one" >&2; exit 1; fi
}

do_batch() {
    local d=$I/batch-invariance
    router_start batch-b1
    router_load $P
    uv run decide_client.py --model $P --batch 1 --limit 20 --test-set "$DATA/triage_test.jsonl" --out-dir "$d" \
        --ensure-t1 > "$LOGS/batch-b1.log" 2>&1
    router_stop batch-b1
    head -n 20 "$B/test/preds/decide-$P-b5-keywords.jsonl" > "$d/decide-$P-b5-keywords-first20.jsonl"
    uv run batch_invariance.py "$d/preds/decide-$P-b1-keywords.jsonl" "$d/decide-$P-b5-keywords-first20.jsonl" \
        --out "$I/batch-invariance-$P.json"
}

do_latency() {
    router_start latency-b1
    router_load $P
    uv run decide_client.py --model $P --batch 1 --test-set "$DATA/triage_test.jsonl" --out-dir "$O/latency" \
        --ensure-t1 > "$LOGS/latency-b1.log" 2>&1
    router_stop latency-b1
}

do_systemone() {
    mkdir -p "$O/systemone"
    router_start systemone
    router_load $P
    uv run systemone_smoke.py --url "$URL" --model $P 2>&1 | tee "$O/systemone/smoke.txt"
    SYSTEMONE_URL=$URL SYSTEMONE_MODEL=$P uv run --with local/typesafe-sdk-python pytest -q -p no:cacheprovider \
        tests/test_systemone_sdk.py 2>&1 | tee "$O/systemone/sdk.txt"
    router_stop systemone
}

do_preset() {
    results-engine/sp3/eval/run.sh preset
}

do_notes() {
    uv run results-engine/3090/notes.py --preset "$P" --dir "$O" --seqs "$SEQS"
}

for s in "${STEPS[@]}"; do "do_$s"; done
