#!/usr/bin/env bash
# Re-quantize the dense tensors (token embedding, lm_head) of a LittleBit GGUF; the binary factors are copied.
#
#   scripts/quantize_dense.sh in.gguf out.gguf [embedding=q4_k] [lm_head=q6_k]
#
# K-quants need a hidden size that is a multiple of 256. For other models q3_k and q4_k become q4_0, and q6_k
# becomes q8_0. Models with tied embeddings have no separate lm_head: the embedding type then applies to both.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 2 ] || die "usage: $0 in.gguf out.gguf [embedding type] [lm_head type]"
IN=$1; OUT=$2; EMB=${3:-q4_k}; HEAD=${4:-q6_k}
[ "$IN" != "$OUT" ] || die "input and output are the same file"
Q="${LLAMA_QUANTIZE:-$ROOT/build/bin/llama-quantize}"
[ -x "$Q" ] || die "$Q not found; build with -DLITTLEBIT_SERVER=ON or set LLAMA_QUANTIZE"
TMP="$OUT.partial"
trap 'rm -f "$TMP" "$TMP.log"' EXIT

quantize() {
    local base=Q8_0 # of no effect: every dense 2D tensor is given its type explicitly
    "$Q" --token-embedding-type "$1" --output-tensor-type "$2" "$IN" "$TMP" "$base" > "$TMP.log" 2>&1
}
fallback() {
    case $1 in
        q2_k | q3_k | q4_k) echo q4_0 ;;
        q5_k) echo q5_0 ;;
        q6_k) echo q8_0 ;;
        *) echo "$1" ;;
    esac
}
if ! quantize "$EMB" "$HEAD"; then
    E2=$(fallback "$EMB"); H2=$(fallback "$HEAD")
    if [ "$E2 $H2" = "$EMB $HEAD" ] || ! quantize "$E2" "$H2"; then
        tail -n 5 "$TMP.log" >&2
        die "llama-quantize failed"
    fi
    log "hidden size not a multiple of 256: used $E2 for the embedding and $H2 for the lm_head"
fi
mv "$TMP" "$OUT"
log "wrote $OUT ($(du -h "$OUT" | cut -f1))"
