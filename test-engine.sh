#!/usr/bin/env bash
# Run every decide test through ctest; no model needed: test-decide-trie and test-decide-after read the vocab-only GGUFs
# of two families in engine/models (a trie run that skips fails). ctest prints the output of every failing test.
set -euo pipefail
cd "$(dirname "$0")/engine/build"
ctest -R '^test-decide-' --output-on-failure --no-tests=error
