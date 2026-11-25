#!/usr/bin/env python3
"""Convert a Next-Concept Modeling (NCM) generator and tokenizer to GGUF for `littlebit ncm`.

    python -m littlebit_gguf.ncm --checkpoint-dir artifacts/sudoku/easy [--student runs/easy-0.55/student.pt] --outfile m.gguf

Reads the checkpoints of HazyResearch/ncm (MIT, arXiv 2610.14162) and, with --student, a generator whose linear layers were
trained as LittleBit layers (tools/ncm/train_sudoku.py). Supports the Sudoku and TinyStories models.
"""
from __future__ import annotations

import argparse
import io
import pickle
import re
from pathlib import Path

import numpy as np

from .ckpt import find_littlebit_groups
from .convert import default_llama_cpp_dir
from .q1 import encode_layer

FORMAT_VERSION = 1


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _require(cond, what):
    if not cond:
        raise SystemExit(f"unsupported NCM variant: {what}")


def check_config(cfg: dict) -> None:
    m, v = cfg["model"], cfg["tokenizer"]["vqvae"]
    q = v["vector_quantizer_config"]
    for key in ("predict_level0", "shared_output_head", "contextual_prefix", "l0_autoregressive", "use_swiglu", "use_geglu",
                "use_relu2", "parallel_block", "normformer", "use_layerscale", "use_rezero", "tie_heads_trainable",
                "tie_heads_bias"):
        _require(not m.get(key), key)
    for key in ("ctx_encoder", "post_upsample_conv"):
        _require(not (m.get(key) or {}).get("enabled"), key)
    _require(m.get("raw_token_prefix") and m.get("use_rope") and m.get("use_qk_norm") and m.get("use_level_encoding"),
             "needs raw_token_prefix, use_rope, use_qk_norm and use_level_encoding")
    _require(m.get("n_kv_heads", m["n_head"]) == m["n_head"] and float(m.get("mlp_ratio", 4)) == 4, "grouped heads or MLP ratio")
    _require(v.get("compression_factor", 1) == 1 and v.get("dec_layers", 1) == 1 and not v.get("dec_positional"), "decoder")
    r = q["quant_resi"]
    _require(r.get("enabled") and r.get("share_mode") == 0 and not r.get("learnable_ratio")
             and r.get("arch", "blend") == "blend" and np.isscalar(r["ratio"]), "quant_resi")
    _require(q.get("learned_l1_sampling") and not q.get("learned_all_sampling") and not q.get("learned_l1_cascade_norm"),
             "code upsampling")
    _require(q["levels"].get("use_manual_levels"), "level schedule")
    g = cfg["generation"]
    _require(not g.get("cfg_scale") and not g.get("frequency_penalty") and not g.get("adjacency_penalty"), "guided decoding")


def phi_index(level: int, n: int) -> int:
    """Which residual convolution a level uses (PhiNonShared.__getitem__)."""
    ticks = np.linspace(1 / 3 / n, 1 - 1 / 3 / n, n) if n == 4 else np.linspace(1 / 2 / n, 1 - 1 / 2 / n, n)
    return int(np.argmin(np.abs(ticks - level / max(n - 1, 1))))


def conv_matrix(w: np.ndarray) -> np.ndarray:
    """Conv1d weight [out, in, 3] -> matrix [out, 3*in] over the concatenation of x[t-1], x[t], x[t+1]."""
    return np.concatenate([w[:, :, j] for j in range(w.shape[2])], axis=1)


def load_checkpoints(d: Path, student: Path | None, config: Path | None = None):
    import torch
    import yaml
    cfg = yaml.safe_load((config or d / "config.yaml").read_text())
    if "soft_decisions" not in cfg["generation"]:
        print("note: the configuration does not set generation.soft_decisions; using hard decisions (the official evaluations "
              "use config/eval/*.yaml of the NCM repository, see --config)")
    check_config(cfg)
    gen = torch.load(d / "generator.pt", map_location="cpu", weights_only=True)
    tok = torch.load(d / cfg["checkpoints"]["tokenizer"], map_location="cpu", weights_only=True)["model"]
    extra = gen.get("inference") or {}
    sd = dict(gen["model"])
    if student is not None:
        sd = torch.load(student, map_location="cpu", weights_only=True)
    fl = lambda t: t.float().numpy() if t.is_floating_point() else t.numpy()  # noqa: E731
    return (cfg, {k: fl(v) for k, v in sd.items()}, {k: fl(v) for k, v in tok.items()},
            {k: fl(v) for k, v in extra.items() if hasattr(v, "numpy")})


def bpe_pieces(path: Path) -> list[bytes]:
    vocab = _SafeUnpickler(io.BytesIO(path.read_bytes())).load()
    out = [b""] * (max(vocab) + 1)
    for k, v in vocab.items():
        out[k] = v if isinstance(v, bytes) else str(v).encode()
    return out


def convert(cfg: dict, gen: dict, tok: dict, extra: dict, out: Path, *, factor_type: str = "q1_0", dense_type: str = "q8_0",
            pieces: list[bytes] | None = None, llama_cpp: Path | None = None) -> dict:
    import sys
    sys.path.insert(0, str((llama_cpp or default_llama_cpp_dir()) / "gguf-py"))
    import gguf
    GT = gguf.GGMLQuantizationType
    m, v, g = cfg["model"], cfg["tokenizer"]["vqvae"], cfg["generation"]
    levels = [int(x) for x in v["vector_quantizer_config"]["levels"]["manual_levels"]]
    n_lvl, L, D = len(levels), levels[-1], int(v["embed_dim"])
    query_l0, tied = bool(m.get("query_level0")), bool(m.get("tie_heads_to_codebook"))
    first = 0 if query_l0 else 1
    sudoku = cfg["tokenizer"]["pre_tokenizer"]["name"] == "sudoku"

    w = gguf.GGUFWriter(str(out), "ncm")
    w.add_name(str(cfg.get("task", "ncm")))
    w.add_uint32("ncm.version", FORMAT_VERSION)
    w.add_string("ncm.task", "sudoku" if sudoku else "text")
    for k, x in (("block_count", m["n_layer"]), ("embedding_length", m["n_embd"]), ("head_count", m["n_head"]),
                 ("prefix_length", m["prefix_len"]), ("code_length", D), ("decoder.head_count", v["n_layers"]),
                 ("vocab_size", v["in_vocab_size"])):
        w.add_uint32(f"ncm.{k}", int(x))
    w.add_float32("ncm.rope_base", float(m.get("rope_base", 10000)))
    w.add_array("ncm.levels", levels)
    w.add_bool("ncm.query_level0", query_l0)
    w.add_bool("ncm.tied_heads", tied)
    w.add_bool("ncm.soft_decisions", bool(g.get("soft_decisions", False)))
    w.add_float32("ncm.soft_temperature", float(g.get("soft_temperature", 1.0)))
    w.add_float32("ncm.temperature", float(g.get("temperature", 1.0)))
    if pieces is not None:
        w.add_array("ncm.tokenizer.pieces_hex", [p.hex() or " " for p in pieces])
        special = list((cfg["tokenizer"]["pre_tokenizer"].get("special_tokens") or {}).values())
        w.add_int32("ncm.tokenizer.eos", special.index("<|endoftext|>") if "<|endoftext|>" in special else -1)
        w.add_int32("ncm.tokenizer.pad", special.index("<|pad|>") if "<|pad|>" in special else -1)
    stats = dict(littlebit_layers=0, littlebit_bytes=0, dense_bytes=0)

    def add(name, a, big=False, table=False):
        """big: a weight matrix or table; table: looked up by the host, so never quantized"""
        a = np.ascontiguousarray(a, dtype=np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"{name}: not finite")
        t = GT.F32
        if big and dense_type == "q8_0" and not table and a.shape[-1] % 32 == 0:
            a, t = gguf.quants.quantize(a, GT.Q8_0), GT.Q8_0
        elif big and dense_type != "f32" and np.abs(a).max(initial=0) < 65504:
            a, t = a.astype(np.float16), GT.F16
        stats["dense_bytes"] += a.nbytes
        w.add_tensor(name, a, raw_dtype=t)

    groups, consumed = find_littlebit_groups(list(gen))
    groups = {k.replace("_orig_mod.", ""): x for k, x in groups.items()}
    if groups:
        w.add_uint32("littlebit.version", 1)

    def linear(name, src):
        if src in groups:
            enc = encode_layer(groups[src].load_branches(lambda k: gen[k]), factor_type)
            T = GT[factor_type.upper()]
            for v2, vv, s in zip(enc.v2, enc.v, ("", "_r")):
                w.add_tensor(f"{name}.lb_v2{s}", v2, raw_dtype=GT.F32)
                w.add_tensor(f"{name}.lb_v{s}", vv, raw_dtype=T)
            w.add_tensor(f"{name}.lb_u", enc.u, raw_dtype=T)
            stats["littlebit_layers"] += 1
            stats["littlebit_bytes"] += sum(x.nbytes for x in (*enc.v2, *enc.v, enc.u))
        else:
            add(f"{name}.weight", gen[src + ".weight"], big=True)
        if src + ".bias" in gen:
            add(f"{name}.bias", gen[src + ".bias"])

    T = "transformer."
    add("in.weight", gen[T + "wte.weight"], big=True)
    add("in.bias", gen[T + "wte.bias"])
    add("level_embd", gen[T + "wle.weight"])
    if query_l0:
        add("l0_query", gen["l0_queries"][0])
    for i in range(m["n_layer"]):
        s, d = f"{T}blocks.{i}.", f"blk.{i}."
        for a, b in (("ln_1", "ln1"), ("ln_2", "ln2")):
            add(f"{d}{b}.weight", gen[f"{s}{a}.weight"])
            add(f"{d}{b}.bias", gen[f"{s}{a}.bias"])
        add(d + "q_norm", gen[s + "attn.q_norm.weight"])
        add(d + "k_norm", gen[s + "attn.k_norm.weight"])
        for a, b in (("attn.c_attn", "qkv"), ("attn.c_proj", "proj"), ("mlp.c_fc", "up"), ("mlp.c_proj", "down")):
            linear(d + b, s + a)
    add("ln_f.weight", gen[T + "ln_f.weight"])
    add("ln_f.bias", gen[T + "ln_f.bias"])

    books = [tok[f"vq.embeddings.{i}.weight"] for i in range(n_lvl)]
    # the TinyStories tokenizer has level-0 codes of 1e10, which its upsampling scales down again: keep that path in F32
    wide = np.abs(books[0]).max() > 1000
    for i, b in enumerate(books):
        add(f"codebook.{i}", b, big=not (wide and i == 0), table=True)
    scales = []
    for h in range(n_lvl - first):
        if tied:
            E = books[h + first].astype(np.float64)
            sq = (E ** 2).sum(-1)
            scale = np.exp(float(gen[f"lm_heads.{h}.log_inv_T"])) / max(sq.mean(), 1e-12)
            scales.append(2 * scale)
            add(f"head.{h}.proj", gen[f"lm_heads.{h}.proj.weight"], big=True)
            add(f"head.{h}.bias", -sq * scale)
        else:
            add(f"head.{h}.weight", gen[f"lm_heads.{h}.weight"], big=True)
            if f"lm_heads.{h}.bias" in gen:
                add(f"head.{h}.bias", gen[f"lm_heads.{h}.bias"])
    if tied:
        w.add_array("ncm.head_scale", [float(x) for x in scales])
    if not query_l0:
        add("level0_unigram", extra["level0_unigram"])

    # level residual convolutions, with the identity part of the blend folded in
    ratio = abs(float(v["vector_quantizer_config"]["quant_resi"]["ratio"]))
    mats = []
    for i in range(n_lvl):
        j = phi_index(i, n_lvl)
        W = conv_matrix(tok[f"vq.quant_resi.{j}.weight"].astype(np.float64)) * ratio
        W[:, D:2 * D] += (1 - ratio) * np.eye(D)
        mats.append((W, tok[f"vq.quant_resi.{j}.bias"].astype(np.float64) * ratio))

    # level 0 goes through learned transposed convolutions; kernel == stride, so each is linear in one input position
    stages = sorted({int(x.group(1)) for k in tok if (x := re.match(r"vq\.learned_up_l1\.(\d+)\.weight", k))})
    ups = [(tok[f"vq.learned_up_l1.{j}.weight"], tok[f"vq.learned_up_l1.{j}.bias"]) for j in stages] if stages else \
        [(tok["vq.learned_up_l1.weight"], tok["vq.learned_up_l1.bias"])]
    _require(levels[0] * int(np.prod([a.shape[2] for a, _ in ups])) == L, "level 0 upsampling")
    K0 = books[0].shape[0]
    if levels[0] == 1 and K0 * L <= sum(a.shape[2] for a, _ in ups) * D:
        # cheaper as a table of finished level-0 contributions: affine, so it is exact for soft mixtures too
        z = books[0][:, :, None].astype(np.float64)  # [K, D, T]
        for a, b in ups:
            z = np.einsum("kdt,doj->kotj", z, a.astype(np.float64)).reshape(K0, D, -1) + b[None, :, None]
        pad = np.pad(z, ((0, 0), (0, 0), (1, 1)))
        W, b = mats[0]
        y = sum(np.einsum("od,kdt->kot", W[:, j * D:(j + 1) * D], pad[:, :, j:j + L]) for j in range(3)) + b[None, :, None]
        add("l0_table", y.transpose(0, 2, 1).reshape(K0, L * D), big=True, table=True)
        mats[0] = None
    else:
        w.add_array("ncm.l0_strides", [int(a.shape[2]) for a, _ in ups])
        for j, (a, b) in enumerate(ups):
            add(f"l0_up.{j}.weight", a.transpose(2, 1, 0).reshape(a.shape[2] * D, D), big=not wide)
            add(f"l0_up.{j}.bias", b)
    for i, x in enumerate(mats):
        if x is not None:
            add(f"phi.{i}.weight", x[0], big=not (wide and i == 0))
            add(f"phi.{i}.bias", x[1])

    add("token_embd", tok["encoder.wte.weight"], big=True, table=True)
    s = "decoder.middle_block."
    for a, b in (("ln_1", "ln1"), ("ln_2", "ln2")):
        add(f"dec.{b}.weight", tok[f"{s}{a}.weight"])
        add(f"dec.{b}.bias", tok[f"{s}{a}.bias"])
    for a, b in (("attn.c_attn", "qkv"), ("attn.c_proj", "proj"), ("mlp.c_fc", "up"), ("mlp.c_proj", "down")):
        add(f"dec.{b}.weight", tok[f"{s}{a}.weight"], big=True)
        add(f"dec.{b}.bias", tok[f"{s}{a}.bias"])
    add("dec.q_norm", tok[s + "attn.q_norm.weight"])
    add("dec.k_norm", tok[s + "attn.k_norm.weight"])
    add("dec.ln_f.weight", tok["decoder.ln_f.weight"])
    add("dec.output", tok["decoder.output_layer.weight"], big=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    stats["file"] = out.stat().st_size
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="default: config.yaml of the checkpoint")
    ap.add_argument("--student", type=Path, help="state dict of a LittleBit generator")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--factor-type", default="q1_0", choices=["q1_0", "f16", "f32"])
    ap.add_argument("--dense-type", default="q8_0", choices=["q8_0", "f16", "f32"])
    ap.add_argument("--llama-cpp", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg, gen, tok, extra = load_checkpoints(a.checkpoint_dir, a.student, a.config)
    pieces = None
    if cfg["tokenizer"]["pre_tokenizer"]["name"] == "bpe":
        pieces = bpe_pieces(a.checkpoint_dir / cfg["checkpoints"]["bpe_vocab"])
    s = convert(cfg, gen, tok, extra, a.outfile, factor_type=a.factor_type, dense_type=a.dense_type, pieces=pieces,
                llama_cpp=a.llama_cpp)
    print(f"{a.outfile}: {s['file'] / 1e6:.2f} MB, {s['littlebit_layers']} LittleBit layers {s['littlebit_bytes'] / 1e6:.2f} MB, "
          f"dense tensors {s['dense_bytes'] / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
