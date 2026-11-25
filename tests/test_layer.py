#!/usr/bin/env python3
"""Check llama_littlebit_mul_mat (ggml CPU) against the numpy reference for layers of a LittleBit GGUF.

    python tests/test_layer.py model.gguf [--all] [blk.0.attn_q ...]

Compares against (a) the exact float math and (b) the same math with GGML's Q8_0 activation quantization
simulated at both matmul inputs, which is what the CPU backend does for Q1_0 weights.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "build" / "_deps" / "llama_cpp-src")) / "gguf-py"))

from gguf import GGUFReader  # noqa: E402

from littlebit_gguf.q1 import EncodedLayer, _as_float  # noqa: E402

BIN = Path(os.environ.get("LITTLEBIT_BIN", ROOT / "build" / "bin")) / "test_layer"


def q8_0(x: np.ndarray) -> np.ndarray:
    """GGML Q8_0 round trip along the last axis: integers use the fp32 scale, the stored scale is fp16."""
    s = x.astype(np.float32).reshape(*x.shape[:-1], -1, 32)
    d = np.abs(s).max(-1, keepdims=True) / np.float32(127.0)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d > 0)
    q = np.rint(s * inv)
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape).astype(np.float64)


def load_layer(reader: GGUFReader, base: str) -> EncodedLayer:
    t = {x.name: x for x in reader.tensors}
    v2 = [np.array(t[f"{base}.lb_v2"].data, dtype=np.float32)]
    v = [np.array(t[f"{base}.lb_v"].data)]
    if f"{base}.lb_v_r" in t:
        v2.append(np.array(t[f"{base}.lb_v2_r"].data, dtype=np.float32))
        v.append(np.array(t[f"{base}.lb_v_r"].data))
    u = np.array(t[f"{base}.lb_u"].data)
    in_pad = int(t[f"{base}.lb_v"].shape[0])
    r = int(t[f"{base}.lb_v"].shape[1])
    r_pad = int(t[f"{base}.lb_u"].shape[0]) // len(v)
    return EncodedLayer(v2, v, u, n_in=v2[0].size, n_out=int(t[f"{base}.lb_u"].shape[1]),
                        in_pad=in_pad, r=r, r_pad=r_pad)


def forward(enc: EncodedLayer, x: np.ndarray, quant_act: bool) -> np.ndarray:
    def act(a, factor):  # ggml rounds activations to Q8_0 for Q1_0 factors and keeps F32 for float factors
        return q8_0(a) if quant_act and factor.dtype == np.uint8 else a

    hs = []
    for v2, v in zip(enc.v2, enc.v):
        h = np.pad(x * v2[None, :], ((0, 0), (0, enc.in_pad - enc.n_in)))
        h = act(h, v) @ _as_float(v, enc.in_pad).T.astype(np.float64)
        hs.append(np.pad(h, ((0, 0), (0, enc.r_pad - enc.r))))
    h = np.concatenate(hs, axis=1)
    return act(h, enc.u) @ _as_float(enc.u, h.shape[1]).T.astype(np.float64)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_gguf(path, binary=BIN, runner=(), names=None, all_layers=False) -> bool:
    reader = GGUFReader(path)
    names = names or sorted({x.name.rsplit(".", 1)[0] for x in reader.tensors
                             if x.name.endswith(".lb_u") and (all_layers or x.name.startswith("blk.0."))})
    n_tok = 7
    ok = bool(names)
    for base in names:
        enc = load_layer(reader, base)
        with tempfile.TemporaryDirectory() as d:
            xf, yf = Path(d) / "x.bin", Path(d) / "y.bin"
            subprocess.run([*runner, str(binary), str(path), base, str(n_tok), str(xf), str(yf)], check=True,
                           stdout=subprocess.DEVNULL)
            x = np.fromfile(xf, dtype=np.float32).reshape(n_tok, enc.n_in).astype(np.float64)
            y = np.fromfile(yf, dtype=np.float32).reshape(n_tok, enc.n_out)
        e_float = rel(y, forward(enc, x, False))
        e_act = rel(y, forward(enc, x, True))
        # 2e-3 leaves room for a few rounding ties; the typical value is 5e-8
        good = e_act < 2e-3
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {base:24s} rel.err vs float={e_float:.2e}  vs rounded activations={e_act:.2e}")
    return ok


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--all"]
    return 0 if check_gguf(args[0], names=args[1:], all_layers="--all" in sys.argv) else 1


if __name__ == "__main__":
    sys.exit(main())
