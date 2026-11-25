#!/usr/bin/env python3
"""Estimate LittleBit GGUF size and runtime RAM for a model, to plan what fits on a phone / watch.

    python -m littlebit_gguf.sizing path/to/config.json --eff-bit 0.1 --residual --ram 2G
    python -m littlebit_gguf.sizing llama-3.1-8b --sweep --ram 2G --ctx 2048

Uses the rank LittleBit picks for a bit budget and the exact tensor layout of this
runtime (Q1_0 blocks, 128-padding, folded scales), plus the dense tensors (embeddings / lm_head) at a
chosen GGML type and an f16 or q8_0 KV cache. Built-in configs are from the public HF config.json files and
are given for convenience; pass a config.json for exact numbers.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .q1 import Q1_0_BLOCK_BYTES, QK1_0, round_up

# bytes per weight of GGML types (block bytes / block size)
TYPE_BPW = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q6_k": 210 / 256, "q5_k": 176 / 256,
            "q4_k": 144 / 256, "q4_0": 18 / 32, "q3_k": 110 / 256, "q2_k": 84 / 256}

# hidden, ffn, layers, heads, kv_heads, head_dim, vocab, tied
BUILTIN = {
    "llama-2-7b":   (4096, 11008, 32, 32, 32, 128, 32000, False),
    "llama-2-13b":  (5120, 13824, 40, 40, 40, 128, 32000, False),
    "llama-2-70b":  (8192, 28672, 80, 64, 8, 128, 32000, False),
    "llama-3.2-1b": (2048, 8192, 16, 32, 8, 64, 128256, True),
    "llama-3.2-3b": (3072, 8192, 28, 24, 8, 128, 128256, True),
    "llama-3.1-8b": (4096, 14336, 32, 32, 8, 128, 128256, False),
    "qwen2.5-0.5b": (896, 4864, 24, 14, 2, 64, 151936, True),
    "qwen2.5-7b":   (3584, 18944, 28, 28, 4, 128, 152064, False),
    "qwen2.5-14b":  (5120, 13824, 48, 40, 8, 128, 152064, False),
    "qwen2.5-32b":  (5120, 27648, 64, 40, 8, 128, 152064, False),
    "qwen3-0.6b":   (1024, 3072, 28, 16, 8, 128, 151936, True),
    "qwen3-8b":     (4096, 12288, 36, 32, 8, 128, 151936, False),
    "qwen3-14b":    (5120, 17408, 40, 40, 8, 128, 151936, False),
    "qwen3-32b":    (5120, 25600, 64, 64, 8, 128, 151936, False),
    "gemma-3-270m": (640, 2048, 18, 4, 1, 256, 262144, True),
    "gemma-3-4b":   (2560, 10240, 34, 8, 4, 256, 262208, True),
    "gemma-3-12b":  (3840, 15360, 48, 16, 8, 256, 262208, True),
    "gemma-3-27b":  (5376, 21504, 62, 32, 16, 128, 262208, True),
}


@dataclass
class Arch:
    hidden: int
    ffn: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool

    @classmethod
    def from_config(cls, cfg: dict) -> "Arch":
        if "text_config" in cfg and "hidden_size" not in cfg:
            cfg = {**cfg, **cfg["text_config"]}
        h = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        return cls(h, cfg["intermediate_size"], cfg["num_hidden_layers"], heads,
                   cfg.get("num_key_value_heads", heads), cfg.get("head_dim") or h // heads, cfg["vocab_size"],
                   bool(cfg.get("tie_word_embeddings", False)))

    def linears(self) -> list[tuple[str, int, int]]:
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return [("q", self.hidden, q), ("k", self.hidden, kv), ("v", self.hidden, kv), ("o", q, self.hidden),
                ("gate", self.hidden, self.ffn), ("up", self.hidden, self.ffn), ("down", self.ffn, self.hidden)]


def littlebit_split_dim(a: int, b: int, eff_bit: float, residual: bool, ratio: float = 1.0, min_split: int = 8) -> int:
    """Largest rank (a multiple of 8) whose factors and 16-bit scales fit in eff_bit * a * b bits.

    One branch of rank r needs r * (a + b) sign bits, 16 * r bits for the latent scale and 16 * (a + b) bits for
    the input and output scales.
    """
    branches = 2 if residual else 1
    r = (eff_bit * a * b / branches - 16 * (a + b)) / (a + b + 16) * ratio
    return max(int(r) // 8 * 8, min_split)


def gguf_linear_bytes(a: int, b: int, r: int, residual: bool) -> int:
    """Bytes of one LittleBit layer in this runtime's GGUF layout (see llama-littlebit.h)."""
    nbr = 2 if residual else 1
    in_pad, r_pad = round_up(a, QK1_0), round_up(r, QK1_0)
    v = r * (in_pad // QK1_0) * Q1_0_BLOCK_BYTES + a * 4
    u = b * (nbr * r_pad // QK1_0) * Q1_0_BLOCK_BYTES
    return nbr * v + u


def estimate(arch: Arch, eff_bit: float, residual: bool, emb_type: str, out_type: str, ctx: int,
             kv_type: str, kv_factor: float = 1.0) -> dict:
    lin = dense = 0
    for name, a, b in arch.linears():
        r = littlebit_split_dim(a, b, eff_bit, residual, kv_factor if name in ("k", "v") else 1.0)
        lin += gguf_linear_bytes(a, b, r, residual)
        dense += a * b
    lin *= arch.layers
    dense *= arch.layers
    emb = arch.vocab * arch.hidden * TYPE_BPW[emb_type]
    out = 0 if arch.tied else arch.vocab * arch.hidden * TYPE_BPW[out_type]
    norms = (2 * arch.layers + 1) * arch.hidden * 4
    kv = 2 * arch.layers * arch.kv_heads * arch.head_dim * ctx * TYPE_BPW[kv_type]
    # compute buffer: logits + a few activations of the widest layer (rough, matches llama.cpp order of magnitude)
    compute = arch.vocab * 4 * 2 + 8 * max(arch.ffn, arch.hidden) * 4 * 64
    weights = lin + emb + out + norms
    return dict(params_linear=dense, linear=lin, embed=emb, output=out, weights=weights, kv=kv, compute=compute,
                total=weights + kv + compute, bpw_linear=8 * lin / dense)


def parse_size(s: str) -> float:
    s = s.strip().upper()
    mult = {"K": 2**10, "M": 2**20, "G": 2**30}
    return float(s[:-1]) * mult[s[-1]] if s[-1] in mult else float(s)


def mib(x: float) -> str:
    return f"{x / 2**20:8.1f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="config.json path, HF model dir, or one of: " + ", ".join(BUILTIN))
    ap.add_argument("--eff-bit", type=float, default=0.1)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="table over eff_bit in {1.0, 0.7, 0.55, 0.3, 0.1}")
    ap.add_argument("--emb-type", default="q4_k", choices=TYPE_BPW)
    ap.add_argument("--out-type", default="q4_k", choices=TYPE_BPW)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--kv-type", default="q8_0", choices=["f16", "q8_0"])
    ap.add_argument("--ram", type=str, default=None, help="memory budget for the model process, e.g. 1.5G")
    args = ap.parse_args(argv)

    if args.model in BUILTIN:
        arch = Arch(*BUILTIN[args.model])
    else:
        p = Path(args.model)
        p = p / "config.json" if p.is_dir() else p
        arch = Arch.from_config(json.loads(p.read_text()))

    budget = parse_size(args.ram) if args.ram else None
    bits = [1.0, 0.7, 0.55, 0.3, 0.1] if args.sweep else [args.eff_bit]
    dense_f16 = (sum(a * b for _, a, b in arch.linears()) * arch.layers + arch.vocab * arch.hidden * (1 if arch.tied else 2)) * 2
    print(f"{args.model}: {arch}")
    print(f"dense f16 weights: {dense_f16 / 2**30:.2f} GiB; ctx={args.ctx} kv={args.kv_type} "
          f"emb={args.emb_type} out={args.out_type} residual={args.residual}")
    print(f"{'eff_bit':>7} {'bpw*':>6} {'linear':>8} {'embed':>8} {'output':>8} {'weights':>8} {'kv':>8} "
          f"{'total':>8}  MiB" + ("   fits" if budget else ""))
    for eb in bits:
        e = estimate(arch, eb, args.residual, args.emb_type, args.out_type, args.ctx, args.kv_type)
        fit = ""
        if budget:
            fit = "   yes" if e["total"] <= budget else "   no"
        print(f"{eb:7.2f} {e['bpw_linear']:6.3f} {mib(e['linear'])} {mib(e['embed'])} {mib(e['output'])} "
              f"{mib(e['weights'])} {mib(e['kv'])} {mib(e['total'])}{fit}")
    print("* bpw of the factorized linears as stored (includes fp16 block scales and 128-padding of the latent rank)")


if __name__ == "__main__":
    main()
