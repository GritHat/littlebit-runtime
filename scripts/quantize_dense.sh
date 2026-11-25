#!/usr/bin/env bash
# Re-quantize the dense tensors (token embedding, lm_head) of a LittleBit GGUF; the binary factors are copied.
#
#   scripts/quantize_dense.sh in.gguf out.gguf [embedding=q4_k] [lm_head=q6_k]
#
# Models with tied embeddings have no separate lm_head: the embedding type then applies to both.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: $0 in.gguf out.gguf [embedding type] [lm_head type]"
Q="${LLAMA_QUANTIZE:-$ROOT/build/bin/llama-quantize}"
[ -x "$Q" ] || die "$Q not found; build with -DLITTLEBIT_SERVER=ON or set LLAMA_QUANTIZE"
EMB=${3:-q4_k}; OUT=${4:-q6_k}
"$Q" --token-embedding-type "$EMB" --output-tensor-type "$OUT" "$1" "$2" "$(tr '[:lower:]' '[:upper:]' <<< "$OUT" | sed 's/^\(Q[345]_K\)$/\1_M/')"
