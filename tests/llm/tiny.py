"""Tiny random LittleBit GGUF models (no downloads) for the runtime tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "build" / "_deps" / "llama_cpp-src"))
sys.path[:0] = [str(ROOT / "python"), str(LLAMA / "gguf-py")]

import gguf  # noqa: E402

from littlebit_gguf.q1 import Branch, encode_layer, encoded_dense  # noqa: E402

ARCHS = ("llama", "qwen2", "qwen3", "gemma2", "gemma3")
HIDDEN, FFN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 96, 160, 2, 3, 1, 32
CHAT = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"


def branch(rng, n_out, r, n_in) -> Branch:
    # scales give W entries of roughly unit spectral size so that activations stay well conditioned
    s = (1.0 / np.sqrt(r * max(n_in, n_out))) ** (1 / 3)
    f = lambda n: (s * rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n)).astype(np.float32)  # noqa: E731
    return Branch(U=rng.random((n_out, r)) < 0.5, V=rng.random((r, n_in)) < 0.5, u1=f(n_out), mid=f(r), v2=f(n_in))


def write(path: Path, arch: str, kind: str, residual: bool = True, seed: int = 0, *, version: int | None = 1,
          mutate=None) -> Path:
    """kind: dense (materialized f32 weights) | q1_0 | f16 | f32 (factor type).  mutate(name, array, dtype) -> list."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_context_length(256)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(FFN)
    w.add_block_count(LAYERS)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-6)
    if arch.startswith("gemma"):
        w.add_sliding_window(64)
    if version is not None and kind != "dense":
        w.add_uint32("littlebit.version", version)

    toks = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)] + ["▁"]
    types = [2, 3, 3] + [6] * 256 + [1]
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores([0.0] * len(toks))
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_chat_template(CHAT)

    GT = gguf.GGMLQuantizationType

    def add(name, arr, dtype):
        for n, a, d in (mutate(name, arr, dtype) if mutate else [(name, arr, dtype)]):
            w.add_tensor(n, a, raw_dtype=d)

    def f32(name, *shape, scale=1.0):
        add(name, (rng.normal(size=shape) * scale).astype(np.float32), GT.F32)

    def linear(base, n_in, n_out, r):
        enc = encode_layer([branch(rng, n_out, r, n_in) for _ in range(2 if residual else 1)],
                           "f32" if kind == "dense" else kind)
        if kind == "dense":
            add(f"{base}.weight", encoded_dense(enc), GT.F32)
            return
        for v2, v, sfx in zip(enc.v2, enc.v, ("", "_r")):
            add(f"{base}.lb_v2{sfx}", v2, GT.F32)
            add(f"{base}.lb_v{sfx}", v, GT[kind.upper()])
        add(f"{base}.lb_u", enc.u, GT[kind.upper()])

    def norm(name, n):
        add(name, rng.uniform(0.8, 1.2, n).astype(np.float32), GT.F32)

    f32("token_embd.weight", len(toks), HIDDEN, scale=0.1 if arch.startswith("gemma") else 1.0)
    norm("output_norm.weight", HIDDEN)
    if not arch.startswith("gemma"):
        f32("output.weight", len(toks), HIDDEN)
    for i in range(LAYERS):
        b = f"blk.{i}"
        norms = ["attn_norm", "ffn_norm"]
        if arch.startswith("gemma"):
            norms += ["post_attention_norm", "post_ffw_norm"]
        for n in norms:
            norm(f"{b}.{n}.weight", HIDDEN)
        if arch in ("qwen3", "gemma3"):
            norm(f"{b}.attn_q_norm.weight", HEAD_DIM)
            norm(f"{b}.attn_k_norm.weight", HEAD_DIM)
        q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
        linear(f"{b}.attn_q", HIDDEN, q, 40)
        linear(f"{b}.attn_k", HIDDEN, kv, 24)
        linear(f"{b}.attn_v", HIDDEN, kv, 24)
        linear(f"{b}.attn_output", q, HIDDEN, 40)
        linear(f"{b}.ffn_gate", HIDDEN, FFN, 136)
        linear(f"{b}.ffn_up", HIDDEN, FFN, 128)
        linear(f"{b}.ffn_down", FFN, HIDDEN, 48)
        if arch == "qwen2":
            for n, d in (("attn_q", q), ("attn_k", kv), ("attn_v", kv)):
                f32(f"{b}.{n}.bias", d, scale=0.1)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
