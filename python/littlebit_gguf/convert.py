#!/usr/bin/env python3
"""Convert a LittleBit / LittleBit-2 checkpoint (SamsungLabs/LittleBit) to GGUF for llama.cpp / Ollama.

The dense parts of the model (embeddings, norms, lm_head, biases) and all architecture metadata and
tokenizer handling are delegated to llama.cpp's own `convert_hf_to_gguf` machinery;
the runtime accepts llama, qwen2/2.5, qwen3, gemma2 and gemma3. Each LittleBit linear is
written as binary Q1_0 factors with folded scales (see patches/llama.cpp/*littlebit*.patch).

Trick used for architecture-specific row reorderings (e.g. llama q/k RoPE permutation): the model class'
own `modify_tensors` is fed an index column vector arange(out_features) in place of the dense weight; the
values coming back tell us which output rows of the factorization go where (and to which GGUF tensor).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .ckpt import LittleBitGroup, find_littlebit_groups
from .q1 import QK1_0, encode_layer, encoded_dense

logger = logging.getLogger("littlebit-convert")

GGUF_KV_VERSION = "littlebit.version"
LITTLEBIT_GGUF_VERSION = 1

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json", "chat_template.json",
    "chat_template.jinja", "preprocessor_config.json",
)
# keys LittleBit adds to config.json that must not leak into HF tooling
LITTLEBIT_CONFIG_KEYS = ("auto_map", "quant_func", "quant_mod", "eff_bit", "split_dim", "residual", "kv_factor",
                         "min_split_dim", "use_itq", "itq_n_iter", "is_po2", "num_expert")


def default_llama_cpp_dir() -> Path:
    if env := os.environ.get("LLAMA_CPP_DIR"):
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    return root / "build" / "_deps" / "llama_cpp-src"


def import_llama_cpp_converter(llama_cpp: Path):
    if not (llama_cpp / "conversion" / "base.py").is_file():
        raise SystemExit(f"llama.cpp converter not found under {llama_cpp} (set --llama-cpp or LLAMA_CPP_DIR)")
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    sys.path.insert(0, str(llama_cpp))
    import gguf  # noqa: F401
    import conversion  # noqa: F401
    return conversion


def stage_model_dir(src: Path, tokenizer_dir: Path | None, work: Path) -> tuple[Path, dict]:
    """Symlink the checkpoint into `work`, add missing tokenizer files, and write a clean config.json."""
    work.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "config.json":
            continue
        (work / f.name).symlink_to(f.resolve())
    if tokenizer_dir is not None:
        for name in TOKENIZER_FILES:
            if (tokenizer_dir / name).is_file() and not (work / name).exists():
                (work / name).symlink_to((tokenizer_dir / name).resolve())
    if not any((work / n).exists() for n in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise SystemExit("no tokenizer files in the checkpoint; pass --tokenizer-dir <base HF model dir>")

    cfg = json.loads((src / "config.json").read_text())
    lb_cfg = {}
    lb_json = src / "littlebit_config.json"
    if lb_json.is_file():
        lb_cfg.update(json.loads(lb_json.read_text()))
    for k in LITTLEBIT_CONFIG_KEYS:
        if k in cfg:
            v = cfg.pop(k)
            if k != "auto_map":
                lb_cfg.setdefault(k, v)
    (work / "config.json").write_text(json.dumps(cfg, indent=2))
    return work, lb_cfg


def make_littlebit_model_class(base_cls, *, lb_cfg: dict, stats: dict, dense: bool = False, factor_type: str = "q1_0"):
    import gguf
    import torch
    from conversion.base import LazyTorchTensor

    def eager_np(t) -> np.ndarray:
        t = LazyTorchTensor.to_eager(t)
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()

    class LittleBitModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch  # required to be in the class __dict__ by TextModel.__init_subclass__
        _lb_groups: dict[str, LittleBitGroup]
        _lb_loaders: dict

        def index_tensors(self, remote_hf_model_id=None):
            tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
            groups, consumed = find_littlebit_groups(list(tensors))
            if not groups:
                raise SystemExit("no LittleBit layers found - is this a LittleBit checkpoint?")
            self._lb_loaders = {n: tensors.pop(n) for n in consumed}
            self._lb_groups = {}
            for prefix, g in groups.items():
                wname = prefix + ".weight"
                if wname in tensors:
                    raise ValueError(f"{prefix}: has both a dense weight and LittleBit factors")
                self._lb_groups[wname] = g
                tensors[wname] = (lambda g=g: self._lb_index_marker(g))
            logger.info("found %d LittleBit layers (%s)", len(groups),
                        "residual" if any(g.residual for g in groups.values()) else "non-residual")
            return tensors

        def _lb_load(self, name: str) -> np.ndarray:
            return eager_np(self._lb_loaders[name]())

        def _lb_index_marker(self, g: LittleBitGroup):
            n_out = g.out_features(self._lb_load)
            return torch.arange(n_out, dtype=torch.float32).reshape(n_out, 1)

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            w = self.gguf_writer
            w.add_uint32(GGUF_KV_VERSION, LITTLEBIT_GGUF_VERSION)
            w.add_uint32("littlebit.block_size", QK1_0)
            if "eff_bit" in lb_cfg and lb_cfg["eff_bit"] is not None:
                w.add_float32("littlebit.eff_bit", float(lb_cfg["eff_bit"]))
            if "residual" in lb_cfg:
                w.add_bool("littlebit.residual", bool(lb_cfg["residual"]))
            if "use_itq" in lb_cfg:
                w.add_bool("littlebit.itq", bool(lb_cfg["use_itq"]))
            if "quant_func" in lb_cfg:
                w.add_string("littlebit.quant_func", str(lb_cfg["quant_func"]))

        def modify_tensors(self, data_torch, name, bid):
            g = self._lb_groups.get(name)
            if g is None:
                yield from super().modify_tensors(data_torch, name, bid)
                return
            marker = LazyTorchTensor.to_eager(data_torch)
            outs = list(super().modify_tensors(marker, name, bid))
            if not outs:
                raise NotImplementedError(f"{name}: the {base_cls.__name__} converter buffers this tensor "
                                          "(e.g. tensor fusion); not supported for LittleBit layers")
            branches = g.load_branches(self._lb_load)
            n_out = branches[0].U.shape[0]
            seen = np.zeros(n_out, dtype=bool)
            for new_name, t in outs:
                t = eager_np(t)
                if t.ndim != 2 or t.shape[1] != 1:
                    raise NotImplementedError(f"{name} -> {new_name}: converter applies a non row-wise transform "
                                              f"(shape {t.shape}); not supported for LittleBit layers")
                perm = np.rint(t[:, 0]).astype(np.int64)
                if not np.allclose(t[:, 0], perm) or perm.min() < 0 or perm.max() >= n_out:
                    raise NotImplementedError(f"{name} -> {new_name}: converter modifies weight values; "
                                              "not supported for LittleBit layers")
                seen[perm] = True
                if not new_name.endswith(".weight"):
                    raise ValueError(f"unexpected GGUF tensor name {new_name}")
                self._lb_write(new_name[: -len(".weight")], [b.permute_out(perm) for b in branches])
            if not seen.all():
                raise NotImplementedError(f"{name}: converter dropped output rows")
            return

        def _lb_write(self, base: str, branches):
            enc = encode_layer(branches, factor_type)
            W = gguf.GGMLQuantizationType
            FT = W[factor_type.upper()]
            add = self.gguf_writer.add_tensor
            if dense:  # debug / stock-runtime export: materialize the binary factorization as a dense weight
                add(f"{base}.weight", encoded_dense(enc), raw_dtype=W.F32)
                stats["layers"] += 1
                stats["dense_params"] += enc.n_in * enc.n_out
                stats["bytes"] += enc.n_in * enc.n_out * 4
                return
            sfx = ["", "_r"]
            for i, (v2, v) in enumerate(zip(enc.v2, enc.v)):
                add(f"{base}.lb_v2{sfx[i]}", v2, raw_dtype=W.F32)
                add(f"{base}.lb_v{sfx[i]}", v, raw_dtype=FT)
            add(f"{base}.lb_u", enc.u, raw_dtype=FT)

            nbytes = sum(a.nbytes for a in (*enc.v2, *enc.v, enc.u))
            stats["layers"] += 1
            stats["bytes"] += nbytes
            stats["dense_params"] += enc.n_in * enc.n_out
            stats["fp16_underflow"] += enc.fp16_underflow
            logger.info("%-28s littlebit in=%d out=%d r=%d (pad %d) x%d -> %.1f KiB, %.3f bpw",
                        base, enc.n_in, enc.n_out, enc.r, enc.r_pad, len(enc.v), nbytes / 1024,
                        8 * nbytes / (enc.n_in * enc.n_out))

    LittleBitModel.__name__ = f"LittleBit{base_cls.__name__}"
    return LittleBitModel


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="LittleBit checkpoint directory (config.json + model*.safetensors)")
    ap.add_argument("--outfile", type=Path, help="output .gguf path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"],
                    help="type for the dense tensors (embeddings, lm_head); re-quantize them afterwards with "
                         "`llama-quantize` (see scripts/quantize_dense.sh) for K-quants")
    ap.add_argument("--tokenizer-dir", type=Path, help="base model directory to take tokenizer files from")
    ap.add_argument("--llama-cpp", type=Path, default=default_llama_cpp_dir(), help="llama.cpp source tree")
    ap.add_argument("--factor-type", default="q1_0", choices=["q1_0", "f16", "f32"],
                    help="storage of the binary factors; f16/f32 are 14x/28x larger but avoid 8-bit activation "
                         "quantization (reference for testing)")
    ap.add_argument("--dense", action="store_true",
                    help="debug: write each LittleBit layer as its materialized dense F32 weight instead of "
                         "binary factors (runs on unpatched llama.cpp, but is not small)")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    import_llama_cpp_converter(args.llama_cpp)
    import gguf
    import torch
    from conversion import ModelBase, ModelType, get_model_architecture, get_model_class

    src = args.model.resolve()
    if not (src / "config.json").is_file():
        raise SystemExit(f"{src}/config.json not found")

    with tempfile.TemporaryDirectory(prefix="littlebit-stage-") as tmp:
        work, lb_cfg = stage_model_dir(src, args.tokenizer_dir, Path(tmp) / src.name)
        hparams = ModelBase.load_hparams(work, False)
        arch = get_model_architecture(hparams, ModelType.TEXT)
        base_cls = get_model_class(arch, mmproj=False)
        logger.info("architecture %s -> %s", arch, base_cls.__name__)

        stats = dict(layers=0, bytes=0, dense_params=0, fp16_underflow=0)
        cls = make_littlebit_model_class(base_cls, lb_cfg=lb_cfg, stats=stats, dense=args.dense,
                                         factor_type=args.factor_type)
        ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                 "bf16": gguf.LlamaFileType.MOSTLY_BF16, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.outtype]
        outfile = args.outfile or Path(f"{src.name}-littlebit-{args.outtype}.gguf")
        with torch.inference_mode():
            model = cls(work, ftype, outfile, model_name=args.model_name or src.name)
            model.write()

    lb_mib = stats["bytes"] / 2**20
    total_mib = outfile.stat().st_size / 2**20
    logger.info("LittleBit layers: %d, %.1f MiB for %.2fB dense params (%.3f bpw); file %.1f MiB",
                stats["layers"], lb_mib, stats["dense_params"] / 1e9,
                8 * stats["bytes"] / max(stats["dense_params"], 1), total_mib)
    if stats["fp16_underflow"]:
        logger.warning("%d nonzero LittleBit scales underflowed in fp16 (tiny channels, usually harmless)",
                       stats["fp16_underflow"])
    print(outfile)


if __name__ == "__main__":
    main()
