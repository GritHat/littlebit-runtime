"""Bit-level helpers: LittleBit packed signs <-> GGML Q1_0 blocks, and scale folding.

LittleBit (quantization/utils/binary_packer.py) packs a {-1,+1} matrix row-major into int32 words,
LSB first, with bit=1 meaning -1 and +1 used as padding.

GGML Q1_0 stores blocks of 128 values as {fp16 d; uint8 qs[16]}, LSB first, with bit=1 meaning +d
and bit=0 meaning -d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QK1_0 = 128
Q1_0_BLOCK_BYTES = 2 + QK1_0 // 8  # 18


def round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def unpack_littlebit_signs(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """LittleBit int32-packed signs -> bool matrix `positive` (True == +1) of `shape`."""
    rows, cols = int(shape[0]), int(shape[1])
    packed = np.ascontiguousarray(packed)
    words = (cols + 31) // 32
    if packed.shape != (rows, words):
        raise ValueError(f"packed shape {packed.shape} does not match {(rows, words)} for {shape}")
    b = packed.astype("<i4", copy=False).view(np.uint8).reshape(rows, words * 4)
    bits = np.unpackbits(b, axis=1, bitorder="little")[:, :cols]
    return bits == 0  # bit=1 -> -1


def signs_from_float(latent: np.ndarray) -> np.ndarray:
    """Legacy unpacked checkpoints keep the latent float U/V; LittleBit binarizes with sign(), sign(0)=+1."""
    return np.asarray(latent) >= 0


def pack_q1_0(positive: np.ndarray, d: np.ndarray, n_cols_pad: int) -> np.ndarray:
    """Build Q1_0 rows.

    positive: bool [rows, cols]           (padded columns are filled with +1; activations there are zero)
    d:        float [rows] or [rows, n_blocks] block scales
    returns uint8 [rows, n_blocks*18]
    """
    rows, cols = positive.shape
    assert n_cols_pad % QK1_0 == 0 and n_cols_pad >= cols
    if n_cols_pad > cols:
        positive = np.concatenate([positive, np.ones((rows, n_cols_pad - cols), dtype=bool)], axis=1)
    nb = n_cols_pad // QK1_0
    qs = np.packbits(positive.reshape(rows, nb, QK1_0), axis=2, bitorder="little")  # [rows, nb, 16]
    d = np.asarray(d, dtype=np.float32)
    if d.ndim == 1:
        d = np.repeat(d[:, None], nb, axis=1)
    assert d.shape == (rows, nb), (d.shape, (rows, nb))
    dh = d.astype("<f2").view(np.uint8).reshape(rows, nb, 2)
    return np.ascontiguousarray(np.concatenate([dh, qs], axis=2).reshape(rows, nb * Q1_0_BLOCK_BYTES))


def dequant_q1_0(data: np.ndarray, n_cols: int) -> np.ndarray:
    """Reference dequantization of Q1_0 rows (for tests)."""
    rows = data.shape[0]
    nb = n_cols // QK1_0
    blk = data.reshape(rows, nb, Q1_0_BLOCK_BYTES)
    d = blk[:, :, :2].copy().view("<f2").astype(np.float32)  # [rows, nb, 1]
    bits = np.unpackbits(blk[:, :, 2:], axis=2, bitorder="little").astype(np.float32)
    return ((2 * bits - 1) * d).reshape(rows, n_cols)


def _pow2_floor(x: float) -> float:
    if not np.isfinite(x):
        raise ValueError("LittleBit scales contain NaN or Inf")
    return float(2.0 ** np.floor(np.log2(x))) if x > 0 else 1.0


@dataclass
class Branch:
    """One LittleBit branch:  y = u1 * (U (mid * (V (v2 * x))))  with mid = v1*u2."""
    U: np.ndarray    # bool [out, r]   positive mask
    V: np.ndarray    # bool [r, in]
    u1: np.ndarray   # [out]
    mid: np.ndarray  # [r]
    v2: np.ndarray   # [in]

    def normalized(self) -> "Branch":
        """Rescale so that max |u1| and max |mid| lie in [1, 2); the slack goes to the fp32 v2.

        u1' = u1/a, mid' = mid*a/b, v2' = v2*b  leaves the layer function unchanged. a and b are powers of
        two, so the rescaling itself is exact and bf16 values of u1 stay exactly representable in fp16.
        """
        a = _pow2_floor(np.max(np.abs(self.u1), initial=0.0))
        mid = self.mid.astype(np.float64) * a
        b = _pow2_floor(np.max(np.abs(mid), initial=0.0))
        return Branch(self.U, self.V, (self.u1.astype(np.float64) / a).astype(np.float32),
                      (mid / b).astype(np.float32), (self.v2.astype(np.float64) * b).astype(np.float32))

    def permute_out(self, perm: np.ndarray) -> "Branch":
        return Branch(self.U[perm], self.V, self.u1[perm], self.mid, self.v2)

    def dense(self) -> np.ndarray:
        """Dense float64 weight [out, in] (for tests)."""
        U = np.where(self.U, 1.0, -1.0)
        V = np.where(self.V, 1.0, -1.0)
        return (self.u1[:, None] * U) @ (self.mid[:, None] * V * self.v2[None, :])


@dataclass
class EncodedLayer:
    v2: list[np.ndarray]   # per branch, f32 [in]
    v: list[np.ndarray]    # per branch: Q1_0 rows (uint8) [r, in_pad/128*18], or float [r, in_pad]
    u: np.ndarray          # Q1_0 rows (uint8) [out, n_br*r_pad/128*18], or float [out, n_br*r_pad]
    n_in: int
    n_out: int
    in_pad: int
    r: int
    r_pad: int
    fp16_underflow: int = 0  # number of nonzero scales flushed to zero by fp16


def encode_layer(branches: list[Branch], factor_type: str = "q1_0") -> EncodedLayer:
    """Encode 1 (plain) or 2 (residual) branches into the GGUF factor tensors (see llama-littlebit.h)."""
    if not 1 <= len(branches) <= 2:
        raise ValueError("a LittleBit layer has one or two branches")
    if factor_type not in ("q1_0", "f16", "f32"):
        raise ValueError(f"unsupported factor type {factor_type}")
    n_out, r = branches[0].U.shape
    n_in = branches[0].V.shape[1]
    for b in branches:
        if b.U.shape != (n_out, r) or b.V.shape != (r, n_in):
            raise ValueError("all LittleBit branches of a layer must share shapes")
    in_pad = round_up(n_in, QK1_0)
    r_pad = round_up(r, QK1_0)
    nbr = r_pad // QK1_0

    q1_0 = factor_type == "q1_0"
    ftype = np.float32 if factor_type == "f32" else np.float16

    def dense_rows(positive: np.ndarray, d: np.ndarray, n_cols_pad: int) -> np.ndarray:
        out = np.zeros((positive.shape[0], n_cols_pad), dtype=ftype)
        out[:, : positive.shape[1]] = np.where(positive, d[:, None], -d[:, None])
        return out

    v2s, vs, us, u_d = [], [], [], []
    underflow = 0
    for b in branches:
        b = b.normalized()
        if factor_type != "f32":
            for s in (b.mid, b.u1):
                underflow += int(np.count_nonzero((s != 0) & (s.astype(np.float16) == 0)))
        v2s.append(np.ascontiguousarray(b.v2))
        vs.append(pack_q1_0(b.V, b.mid, in_pad) if q1_0 else dense_rows(b.V, b.mid, in_pad))
        if q1_0:
            us.append(np.concatenate([b.U, np.ones((n_out, r_pad - r), dtype=bool)], axis=1))
            u_d.append(np.repeat(b.u1[:, None], nbr, axis=1))
        else:
            us.append(dense_rows(b.U, b.u1, r_pad))
    u = np.concatenate(us, axis=1)
    if q1_0:
        u = pack_q1_0(u, np.concatenate(u_d, axis=1), u.shape[1])
    return EncodedLayer(v2s, vs, u, n_in, n_out, in_pad, r, r_pad, underflow)


def _as_float(data: np.ndarray, n_cols: int) -> np.ndarray:
    return dequant_q1_0(data, n_cols) if data.dtype == np.uint8 else data.astype(np.float32)


def reference_forward(enc: EncodedLayer, x: np.ndarray) -> np.ndarray:
    """Exactly what llama_littlebit_mul_mat computes (in float, without activation quantization)."""
    hs = []
    for v2, v in zip(enc.v2, enc.v):
        h = x * v2[None, :]
        h = np.pad(h, ((0, 0), (0, enc.in_pad - enc.n_in)))
        h = h @ _as_float(v, enc.in_pad).T
        hs.append(np.pad(h, ((0, 0), (0, enc.r_pad - enc.r))))
    h = np.concatenate(hs, axis=1)
    return h @ _as_float(enc.u, h.shape[1]).T


def encoded_dense(enc: EncodedLayer) -> np.ndarray:
    """Dense float32 weight [out, in] equivalent to the encoded factors (debug / compatibility export)."""
    U = _as_float(enc.u, len(enc.v) * enc.r_pad).astype(np.float64)
    W = np.zeros((enc.n_out, enc.n_in), dtype=np.float64)
    for i, (v2, v) in enumerate(zip(enc.v2, enc.v)):
        Vd = _as_float(v, enc.in_pad)[:, : enc.n_in].astype(np.float64) * v2[None, :]
        Ub = U[:, i * enc.r_pad: i * enc.r_pad + enc.r]
        W += Ub @ Vd
    return W.astype(np.float32)
