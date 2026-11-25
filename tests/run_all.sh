#!/usr/bin/env bash
# Compare the runtime with Samsung's PyTorch code on untrained initializations of real models. Needs
# scripts/setup_python.sh --reference and the HF models in models/hf/. Slow: the SVD initialization runs on the CPU.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
fail=0
run() { "$PY" tests/test_e2e.py "$@" 2>&1 | grep -E "PASS|FAIL|kl_mean|Error" || true; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1; }
run models/hf/SmolLM2-135M-Instruct SmolLM2-135M-lb1.0r  --eff-bit 1.0 --residual --use-itq
run models/hf/SmolLM2-135M-Instruct SmolLM2-135M-lb0.1r  --eff-bit 0.1 --residual --use-itq
run models/hf/gemma-3-270m-it       gemma3-270m-lb1.0    --eff-bit 1.0
run models/hf/Qwen2.5-0.5B-Instruct qwen2.5-0.5b-lb0.55r --eff-bit 0.55 --residual
exit $fail
