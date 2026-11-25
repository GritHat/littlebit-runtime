#!/usr/bin/env python3
"""End-to-end: dense HF model -> official LittleBit checkpoint -> GGUF -> runtime, vs the official PyTorch code.

    python tests/llm/test_e2e.py <hf_dir> <name> [make_littlebit_checkpoint.py args...]

Pass criteria
  1. every LittleBit layer of the model equals numpy with Q8_0 activations (tests/llm/test_layer.py --all)
  2. with F32 factors the runtime agrees with the official PyTorch forward as well as ordinary dense F32 weights do
Q1_0 results are reported, not judged: the checkpoints are untrained initializations, whose sensitivity to the
8-bit activations of any quantized llama.cpp format ranges from negligible to total.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
BIN = ROOT / "build" / "bin"
sys.path.insert(0, str(ROOT / "tests" / "llm"))
from compare_logits import compare  # noqa: E402
from test_layer import check_gguf  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Germany is"


def run(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def logits(gguf: Path, toks: str, out: Path, rows: int) -> np.ndarray:
    run(BIN / "littlebit", "logits", gguf, "--tokens", toks, "--out", out, "--kv", "f32", stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32).reshape(rows, -1)


def main() -> int:
    hf, name, extra = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    from transformers import AutoTokenizer
    toks = ",".join(map(str, AutoTokenizer.from_pretrained(hf)(PROMPT)["input_ids"]))
    ck = ROOT / "models" / "lb" / name
    gg = ROOT / "models" / "gguf"
    gg.mkdir(parents=True, exist_ok=True)
    make = [PY, ROOT / "tools" / "llm" / "make_littlebit_checkpoint.py", hf, ck, "--no-tokenizer", "--ref-tokens", toks, *extra]
    if (ck / "model.safetensors").exists():
        make.append("--ref-only")
    run(*make)
    ref = np.load(ck / "ref_logits.npy")

    files = {"q1_0": gg / f"{name}-q1_0.gguf", "factors_f32": gg / f"{name}-f32.gguf", "dense_f32": gg / f"{name}-dense.gguf"}
    opts = {"q1_0": [], "factors_f32": ["--factor-type", "f32"], "dense_f32": ["--dense"]}
    for k, f in files.items():
        run(PY, "-m", "littlebit_gguf.convert", ck, "--outtype", "f32", "--tokenizer-dir", hf, "--outfile", f, *opts[k],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    layers_ok = check_gguf(files["q1_0"], all_layers=True)
    res = {k: compare(ref, logits(f, toks, f.with_suffix(".bin"), ref.shape[0])) for k, f in files.items()}
    for k, m in res.items():
        print(f"{name:28s} {k:11s} " + "  ".join(f"{a}={b:.4g}" for a, b in m.items()))
    for f in files.values():
        f.with_suffix(".bin").unlink()
    files["factors_f32"].unlink()
    files["dense_f32"].unlink()
    e2e_ok = res["factors_f32"]["kl_mean"] <= 2 * res["dense_f32"]["kl_mean"] + 1e-5
    ok = layers_ok and e2e_ok
    print(f"{name}: {'PASS' if ok else 'FAIL'} (layers={'ok' if layers_ok else 'FAIL'}, e2e={'ok' if e2e_ok else 'FAIL'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
