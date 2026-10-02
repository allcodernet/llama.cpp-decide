#!/usr/bin/env bash
# Start our engine's llama-server with the presets file. Usage: ./serve-engine.sh
# The router loads a model on the first request that names it (autoload is the default);
# with --models-max 1 a request for the other preset unloads the current one.
# /v1/decide and /v1/decide/info are enabled by decide-seqs in the presets file.
# ENGINE_PRESETS: the presets file (default models-engine.ini); ENGINE_BIN: the directory of llama-server (default
# engine/build/bin, e.g. engine-63ea2c51a/build/bin); relative values from the project directory.
set -euo pipefail
cd "$(dirname "$0")"
PRESETS=${ENGINE_PRESETS:-models-engine.ini}
BIN=${ENGINE_BIN:-engine/build/bin}
[[ $BIN = /* ]] || BIN=./$BIN
echo "serving presets from $PRESETS on :8097"
exec "$BIN/llama-server" --models-preset "$PRESETS" --models-max 1 --port 8097
