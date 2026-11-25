# shellcheck shell=bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC2034
{
PATCH="$ROOT/patches/llama.cpp/0001-littlebit-factorized-binary-linear.patch"
TP="$ROOT/third_party"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
OLLAMA_REF="${OLLAMA_REF:-2bd4d6aabdf4caa39e7095e201dcda4cc1a3a406}"
LITTLEBIT_REF="${LITTLEBIT_REF:-42d658b0c79f76450b34b6a3547462c7cdc3e1a0}"
LLAMA_CPP_REF=b12257
}
log() { printf '[littlebit] %s\n' "$*"; }
die() { printf '[littlebit] error: %s\n' "$*" >&2; exit 1; }

# clone_at <url> <dir> <commit>
clone_at() {
    [ -d "$2/.git" ] && return 0
    git init -q "$2"
    git -C "$2" fetch -q --depth 1 "$1" "$3"
    git -C "$2" checkout -q --detach FETCH_HEAD
}
