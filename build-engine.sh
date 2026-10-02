#!/usr/bin/env bash
# Build our decide engine: upstream llama.cpp at the pinned commit plus the patch series in
# patches/decide/ (branch `decide` in engine/, which is git-ignored), then build it.
#
# engine/ missing: clone upstream (from LOCAL_SEED when it is set and holds the pinned commit, else
#   from GitHub), create branch decide at UPSTREAM_COMMIT and apply patches/decide/*.patch with
#   `git am` (author, committer and dates from the patches).
# engine/ present: left as it is (checked out on branch decide); the patches are not re-applied.
#
# Env overrides:
#   BUILD_CUDA=0     CPU-only build (-DGGML_CUDA=OFF), e.g. to check a fresh checkout quickly
#   TARGETS="a b"    build only these targets (default: server, CLIs, decide and tokenizer tests)
#   EXTRA_TARGETS    appended to the default targets
#   LOCAL_SEED=path  local llama.cpp clone to clone from instead of GitHub (optional, faster; unset: GitHub)
#   CUDA_BIN=dir     prepended to PATH (default /opt/cuda/bin)
#   CUDA_ARCH=list   -DCMAKE_CUDA_ARCHITECTURES (default 86)
#   CUDA_HOST_COMPILER=path  -DCMAKE_CUDA_HOST_COMPILER (default /usr/bin/g++-15)
#   (an empty CUDA_BIN, CUDA_ARCH or CUDA_HOST_COMPILER drops that setting)
set -euo pipefail
cd "$(dirname "$0")"
CUDA_BIN=${CUDA_BIN-/opt/cuda/bin}
CUDA_ARCH=${CUDA_ARCH-86}
CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER-/usr/bin/g++-15}
if [ -n "$CUDA_BIN" ]; then export PATH=$CUDA_BIN:$PATH; fi
UPSTREAM_URL=https://github.com/ggml-org/llama.cpp
UPSTREAM_COMMIT=60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f
LOCAL_SEED=${LOCAL_SEED:-}
PATCHES="$PWD/patches/decide"

if [ ! -d engine ]; then
    if [ -n "$LOCAL_SEED" ] && [ -d "$LOCAL_SEED/.git" ] && git -C "$LOCAL_SEED" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
        git clone --no-hardlinks --no-checkout "$LOCAL_SEED" engine
        git -C engine remote set-url origin "$UPSTREAM_URL"
    else
        git clone --no-checkout "$UPSTREAM_URL" engine
    fi
    git -C engine checkout -q -b decide "$UPSTREAM_COMMIT"
    for p in "$PATCHES"/*.patch; do
        from=$(sed -n 's/^From: //p' "$p" | head -1)
        GIT_COMMITTER_NAME="${from% <*}" GIT_COMMITTER_EMAIL=$(sed 's/.*<\(.*\)>.*/\1/' <<<"$from") \
            git -C engine am -q --committer-date-is-author-date "$p"
    done
fi
cd engine
git checkout -q decide

if [ "${BUILD_CUDA:-1}" = 0 ]; then
    BACKEND=(-DGGML_CUDA=OFF)
else
    BACKEND=(-DGGML_CUDA=ON)
    if [ -n "$CUDA_ARCH" ]; then BACKEND+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"); fi
    if [ -n "$CUDA_HOST_COMPILER" ]; then BACKEND+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER"); fi
fi
cmake -B build "${BACKEND[@]}" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
# llama-decide-cli builds bin/llama-decide; the library target is llama-decide. Every decide test is built (test-engine.sh
# runs them all).
TARGETS=${TARGETS:-"llama-server llama-cli llama-decide-cli test-tokenizer-0 test-decide-schema test-decide-score test-decide-compat test-decide-trie test-decide-cache test-decide-fullpath test-decide-debias test-decide-after test-decide-image ${EXTRA_TARGETS:-}"}
cmake --build build --config Release -j"$(nproc)" --target $TARGETS
echo "build ok: $(git rev-parse --short HEAD) on $(git branch --show-current), tree $(git rev-parse HEAD^{tree})"
