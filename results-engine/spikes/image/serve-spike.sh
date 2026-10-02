#!/usr/bin/env bash
# Throwaway spike server: the spike-image worktree's llama-server with Qwen3.8-27B + its mmproj on 127.0.0.1:8097.
# Usage (test machine): serve-spike.sh LOGFILE [extra llama-server args]; prints the PID. SEQS / CTX override defaults.
set -euo pipefail
LOG=$1; shift
BIN=/home/marduk/llama.cpp-decision/engine-spike-image/build/bin
D=/home/marduk/.lmstudio/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
# IMG_MIN: --image-min-tokens (default 1024, the user's launcher); IMG_MIN=0 leaves the projector's default
IMG_ARGS=()
if [ "${IMG_MIN:-1024}" != 0 ]; then IMG_ARGS=(--image-min-tokens "${IMG_MIN:-1024}"); fi
if pgrep -x llama-server >/dev/null; then echo "a llama-server runs already" >&2; pgrep -ax llama-server >&2; exit 1; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1500 ]; then echo "GPU busy: $used MiB" >&2; exit 1; fi
LLAMA_MEDIA_MARKER="<__media__>" nohup "$BIN/llama-server" -m "$D/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" --mmproj "$D/mmproj-Qwen3.8-27B-BF16.gguf" \
    ${IMG_ARGS[@]+"${IMG_ARGS[@]}"} -ngl 999 -fa on -ctk q8_0 -ctv q8_0 --jinja --reasoning off -np 1 \
    -c "${CTX:-16384}" --decide-seqs "${SEQS:-16}" --host 127.0.0.1 --port 8097 "$@" >"$LOG" 2>&1 &
echo $!
