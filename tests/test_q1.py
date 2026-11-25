"""Unit tests for the LittleBit -> Q1_0 encoding (fast, no model needed).  Run: pytest tests/test_q1.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from littlebit_gguf.ckpt import find_littlebit_groups  # noqa: E402
from littlebit_gguf.q1 import (QK1_0, Branch, dequant_q1_0, encode_layer, encoded_dense, pack_q1_0,  # noqa: E402
                               reference_forward, unpack_littlebit_signs)
from littlebit_gguf.sizing import gguf_linear_bytes, littlebit_split_dim  # noqa: E402

rng = np.random.default_rng(0)


def rand_branch(n_out, r, n_in):
    return Branch(U=rng.random((n_out, r)) < 0.5, V=rng.random((r, n_in)) < 0.5,
                  u1=rng.normal(size=n_out).astype(np.float32) * 0.1,
                  mid=rng.normal(size=r).astype(np.float32) * 3.0,
                  v2=rng.normal(size=n_in).astype(np.float32))


@pytest.mark.parametrize("rows,cols", [(8, 32), (5, 100), (3, 4096), (16, 1)])
def test_unpack_matches_official_packer(rows, cols):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "third_party" / "LittleBit"))
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")
    pytest.importorskip("quantization", reason="run scripts/setup_python.sh --reference")
    from quantization.utils.binary_packer import binary_packer, binary_unpacker
    signs = torch.where(torch.rand(rows, cols) < 0.5, -1, 1).to(torch.int8)
    packed = binary_packer(signs).numpy()
    pos = unpack_littlebit_signs(packed, (rows, cols))
    assert np.array_equal(pos, signs.numpy() == 1)
    assert np.array_equal(binary_unpacker(torch.from_numpy(packed), (rows, cols)).numpy() == 1, pos)


def test_q1_0_roundtrip_and_ggml_layout():
    pos = rng.random((4, 256)) < 0.5
    d = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    q = pack_q1_0(pos, d, 256)
    assert q.shape == (4, 2 * 18)
    deq = dequant_q1_0(q, 256)
    assert np.array_equal(deq, np.where(pos, 1.0, -1.0) * d[:, None])
    # cross-check with gguf-py's own Q1_0 dequantizer
    sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "build/_deps/llama_cpp-src")) / "gguf-py"))
    gguf = pytest.importorskip("gguf")
    ref = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q1_0)
    assert np.array_equal(ref.reshape(deq.shape), deq)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("n_out,r,n_in", [(64, 24, 576), (130, 128, 256), (96, 200, 100)])
def test_encode_equals_littlebit_forward(residual, n_out, r, n_in):
    brs = [rand_branch(n_out, r, n_in) for _ in range(2 if residual else 1)]
    enc = encode_layer(brs)
    assert enc.in_pad % QK1_0 == 0 and enc.r_pad % QK1_0 == 0
    x = rng.normal(size=(3, n_in))
    want = sum(x @ b.dense().T for b in brs)  # official formula, float64
    got = reference_forward(enc, x)
    assert np.allclose(got, want, rtol=2e-3, atol=1e-3 * np.abs(want).max())
    assert np.allclose(x @ encoded_dense(enc).T, got, rtol=1e-4, atol=1e-5 * np.abs(want).max())


def test_permutation_is_applied_to_rows_and_scales():
    b = rand_branch(16, 8, 128)
    perm = rng.permutation(16)
    assert np.allclose(b.permute_out(perm).dense(), b.dense()[perm])


def test_find_groups_packed_and_legacy():
    names = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.bias",
             *[f"model.layers.0.self_attn.q_proj.{k}" for k in
               ("U_packed", "U_shape", "V_packed", "V_shape", "u1", "u2", "v1", "v2", "_eff_bit_target",
                "_split_dim_final", "_eff_bit_actual")],
             *[f"model.layers.0.mlp.up_proj.{k}" for k in ("U", "V", "u1", "u2", "v1", "v2")]]
    groups, consumed = find_littlebit_groups(names)
    assert set(groups) == {"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"}
    assert groups["model.layers.0.self_attn.q_proj"].packed and not groups["model.layers.0.mlp.up_proj"].packed
    assert "model.layers.0.self_attn.q_proj.bias" not in consumed
    assert "model.embed_tokens.weight" not in consumed


def test_sizing_matches_encoder():
    for a, b, eb, res in [(576, 1536, 1.0, True), (4096, 4096, 0.1, True), (896, 4864, 0.55, False)]:
        r = littlebit_split_dim(a, b, eb, res)
        enc = encode_layer([rand_branch(b, r, a) for _ in range(2 if res else 1)])
        assert gguf_linear_bytes(a, b, r, res) == sum(x.nbytes for x in (*enc.v2, *enc.v, enc.u))


def test_unpack_known_bits():
    # LittleBit: LSB first, bit 1 = -1, padding bits 0
    packed = np.array([[0b101, 0], [-(2**31), 1]], dtype=np.int32)
    pos = unpack_littlebit_signs(packed, (2, 34))
    assert [list(np.flatnonzero(~r)) for r in pos] == [[0, 2], [31, 32]]


def test_split_dim_matches_official():
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "third_party" / "LittleBit"))
    pytest.importorskip("quantization", reason="run scripts/setup_python.sh --reference")
    from quantization.modules.littlebit import LittleBitLinear as L
    for a, b in ((576, 1536), (4096, 4096), (4096, 1024), (640, 256)):
        for eb in (1.0, 0.55, 0.1):
            for res in (False, True):
                want = L._finalize_split_dim(L._estimate_split_dim(a, b, eb, res), 1024, 8)
                assert littlebit_split_dim(a, b, eb, res) == want, (a, b, eb, res)


@pytest.mark.parametrize("ftype", ["f16", "f32"])
def test_float_factors(ftype):
    brs = [rand_branch(96, 200, 100) for _ in range(2)]
    x = rng.normal(size=(3, 100))
    want = sum(x @ b.dense().T for b in brs)
    assert np.allclose(reference_forward(encode_layer(brs, ftype), x), want, rtol=2e-3, atol=1e-3 * np.abs(want).max())


def test_normalization_keeps_bf16_scales_exact():
    b = rand_branch(64, 24, 128)
    b.u1 = (b.u1.view(np.uint32) & 0xFFFF0000).view(np.float32).copy()  # bf16 values
    n = b.normalized()
    assert 1 <= np.abs(n.u1).max() < 2 and 1 <= np.abs(n.mid).max() < 2
    assert np.array_equal(n.u1.astype(np.float16).astype(np.float32), n.u1)
    assert np.allclose(n.dense(), b.dense(), rtol=1e-6)


def test_bad_scales_are_rejected():
    b = rand_branch(8, 8, 128)
    b.u1[0] = np.nan
    with pytest.raises(ValueError):
        encode_layer([b])
