#!/usr/bin/env bash
# Validate the runtime on a real QAT-trained LittleBit-2 checkpoint (third-party upload, CC BY-NC 4.0):
#   LittleBitLLM/littlebit-qwen3-8b-0.55bpw  (Qwen3-8B, 0.55 bpw, residual, Joint-ITQ; ~3 GB download)
# Compares wikitext-2 perplexity / per-token KL of llama.cpp vs the official PyTorch implementation on
# identical windows, and greedy continuations. Needs ~20 GB RAM for the fp32 PyTorch reference.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
CK=models/lb/littlebit-qwen3-8b-0.55bpw
GG=models/gguf/qwen3-8b-lb0.55-bf16.gguf
THREADS=${THREADS:-4}
$PY -c "from huggingface_hub import snapshot_download as s; s('LittleBitLLM/littlebit-qwen3-8b-0.55bpw', local_dir='$CK', allow_patterns=['*.json','*.safetensors','*.txt'])"
[ -f "$GG" ] || $PY -m littlebit_gguf.convert "$CK" --outtype bf16 --outfile "$GG"
$PY tests/llm/ppl_compare.py "$CK" "$GG" --windows "${WINDOWS:-8}" --seqlen 512 --threads "$THREADS" | grep -E "PPL|KL"
echo "--- greedy: llama.cpp"
build/bin/littlebit run "$GG" -p "The capital of France is" -n 16 --temp 0 -t "$THREADS" 2>/dev/null
echo "--- greedy: official PyTorch"
$PY tools/llm/official_generate.py "$CK" "$CK" "The capital of France is" 16 2>/dev/null | tail -1
