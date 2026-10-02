#!/usr/bin/env bash
# Throwaway: per "SEQS:CTX" config start the spike server (image-min-tokens 1024), record idle VRAM, run the
# 3-image batch request (decide_image.py batch) while sampling nvidia-smi every 100 ms, record the peak, stop the server.
# Usage (test machine, in local/spike-image): vram_sweep.sh OUT.tsv SEQS:CTX...
set -uo pipefail
OUT=$1; shift
echo -e "seqs\tctx\tstatus\tidle_mib\tpeak_mib\tbatch_total_ms" > "$OUT"
for cfg in "$@"; do
    seqs=${cfg%%:*}; ctx=${cfg##*:}
    log=/tmp/spike-image-sweep-$seqs-$ctx.log
    pid=$(SEQS=$seqs CTX=$ctx ./serve-spike.sh "$log") || { echo "cannot start $cfg"; exit 1; }
    ok=0
    for i in $(seq 100); do
        if curl -sf http://127.0.0.1:8097/health >/dev/null; then ok=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    if [ $ok = 0 ]; then
        echo -e "$seqs\t$ctx\tload-failed\t-\t-\t-" >> "$OUT"
        grep -iE "out of memory|failed|error" "$log" | head -3
        kill "$pid" 2>/dev/null; sleep 5; continue
    fi
    idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ( while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 0.1; done ) > /tmp/spike-image-vram-$seqs.txt &
    sampler=$!
    if timeout 600 uv run --quiet decide_image.py batch . /tmp/spike-image-batch-$seqs-$ctx.json > /tmp/spike-image-batch-$seqs-$ctx.txt 2>&1; then
        st=ok
        ms=$(/usr/bin/python3 -c "import json;print(json.load(open('/tmp/spike-image-batch-$seqs-$ctx.json'))['batch_response']['timings']['total_ms'])")
    else
        st=request-failed; ms=-
        tail -3 /tmp/spike-image-batch-$seqs-$ctx.txt
    fi
    kill "$pid"; wait "$sampler" 2>/dev/null
    peak=$(sort -n /tmp/spike-image-vram-$seqs.txt | tail -1)
    echo -e "$seqs\t$ctx\t$st\t$idle\t$peak\t$ms" >> "$OUT"
    sleep 5
done
cat "$OUT"
