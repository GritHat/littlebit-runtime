#!/usr/bin/env bash
# Build Ollama with LittleBit support. Ollama applies every patch in llama/compat/ to the llama.cpp version it
# pins, so installing the patch there is all it takes.
#
#   scripts/setup_ollama.sh [cmake preset = cpu]       needs Go and CMake
set -euo pipefail
source "$(dirname "$0")/common.sh"
PRESET=${1:-cpu}
DIR="${OLLAMA_DIR:-$TP/ollama}"
clone_at https://github.com/ollama/ollama "$DIR" "$OLLAMA_REF"
[ "$(tr -d '[:space:]' < "$DIR/LLAMA_CPP_VERSION")" = "$LLAMA_CPP_REF" ] \
    || log "warning: this Ollama does not pin llama.cpp $LLAMA_CPP_REF, the patch may not apply"
BUILD="$DIR/build/llama-server-$PRESET"
if [ -f "$DIR/llama/compat/100-littlebit.patch" ] && ! cmp -s "$PATCH" "$DIR/llama/compat/100-littlebit.patch"; then
    log "the patch changed: discarding the patched llama.cpp in $BUILD"
    rm -rf "$BUILD"
fi
cp "$PATCH" "$DIR/llama/compat/100-littlebit.patch"
cd "$DIR"
cmake -S llama/server --preset "$PRESET"
[ -f "$BUILD/_deps/llama_cpp-src/src/llama-littlebit.cpp" ] || die "the patch was not applied to $BUILD"
cmake --build "$BUILD" --parallel "$JOBS"
go build -o ollama .
log "run '$DIR/ollama serve', then 'scripts/ollama_create.sh model.gguf name'"
