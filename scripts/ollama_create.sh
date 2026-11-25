#!/usr/bin/env bash
# Register a LittleBit GGUF with a LittleBit-enabled Ollama.
#
#   scripts/ollama_create.sh model.gguf name [num_ctx=2048] [num_thread]
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: $0 model.gguf name [num_ctx] [num_thread]"
OLLAMA="${OLLAMA:-$TP/ollama/ollama}"
[ -x "$OLLAMA" ] || die "$OLLAMA not found; run scripts/setup_ollama.sh or set OLLAMA"
MF=$(mktemp)
trap 'rm -f "$MF"' EXIT
{
    printf 'FROM %s\nPARAMETER num_ctx %s\n' "$(realpath "$1")" "${3:-2048}"
    if [ -n "${4:-}" ]; then printf 'PARAMETER num_thread %s\n' "$4"; fi
} > "$MF"
"$OLLAMA" create "$2" -f "$MF"
