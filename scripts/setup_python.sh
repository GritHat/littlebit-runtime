#!/usr/bin/env bash
# Python environment for conversion (CPU torch) in .venv. --reference also fetches SamsungLabs/LittleBit
# (CC BY-NC 4.0), which tools/ and the comparison tests use; the runtime and converter do not need it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LLAMA="${LLAMA_CPP_DIR:-$ROOT/build/_deps/llama_cpp-src}"
[ -f "$LLAMA/conversion/base.py" ] || die "llama.cpp sources not found; run 'cmake -B build' first or set LLAMA_CPP_DIR"
[ -d "$ROOT/.venv" ] || "${PYTHON:-python3}" -m venv "$ROOT/.venv"
PIP="$ROOT/.venv/bin/pip"
"$PIP" install -q --upgrade pip
"$PIP" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$PIP" install -q -e "${ROOT}[convert,test]"
if [ "${1:-}" = --reference ]; then
    "$PIP" install -q pyarrow
    clone_at https://github.com/SamsungLabs/LittleBit "$TP/LittleBit" "$LITTLEBIT_REF"
fi
log "ready: $ROOT/.venv/bin/littlebit-convert"
