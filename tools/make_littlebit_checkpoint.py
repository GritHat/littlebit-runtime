#!/usr/bin/env python3
"""Produce a LittleBit checkpoint in the *official* on-disk format using SamsungLabs' own code.

It applies `apply_littlebit_patch(do_train=True)` (SVD / LittleBit-2 Joint-ITQ initialization) to a dense
HF model and saves the packed `state_dict()` exactly like `quantization/hub.py` does. No QAT is run, so the
result is the *initialization* only: useful to exercise the runtime pipeline end-to-end on a laptop, and as
the starting point of QAT on a GPU (`main.py` in the LittleBit repo). Real deployments should use a
QAT-trained checkpoint.

Optionally dumps reference logits from the official PyTorch LittleBit forward for a token sequence, so the
llama.cpp runtime can be checked against it (tests/test_e2e.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="dense HF model directory")
    ap.add_argument("out", type=Path, help="output LittleBit checkpoint directory")
    ap.add_argument("--eff-bit", type=float, default=1.0)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--use-itq", action="store_true", help="LittleBit-2 Joint-ITQ initialization")
    ap.add_argument("--itq-iters", type=int, default=50)
    ap.add_argument("--kv-factor", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="dtype of the saved scales / dense tensors (QAT checkpoints are bf16)")
    ap.add_argument("--no-tokenizer", action="store_true", help="do not copy tokenizer files (like hub.py)")
    ap.add_argument("--littlebit-repo", type=Path, default=ROOT / "third_party" / "LittleBit")
    ap.add_argument("--ref-only", action="store_true", help="only (re)compute reference logits for an existing <out>")
    ap.add_argument("--ref-tokens", type=str, default=None,
                    help="comma separated token ids; dump the official forward's logits to <out>/ref_logits.npy")
    args = ap.parse_args()

    sys.path.insert(0, str(args.littlebit_repo))
    qargs = argparse.Namespace(
        quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=args.residual, split_dim=1024,
        eff_bit=args.eff_bit, min_split_dim=8, use_itq=args.use_itq, itq_n_iter=args.itq_iters,
        kv_factor=args.kv_factor, model_id=str(args.base),
    )
    if not args.ref_only:
        write_checkpoint(args, qargs)
    if args.ref_tokens:
        dump_reference(args, qargs)


def write_checkpoint(args, qargs) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from quantization.utils.quant_util import apply_littlebit_patch

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        apply_littlebit_patch(model, qargs, do_train=True)
    dtype = getattr(torch, args.dtype)
    model.to(dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()  # LittleBitLinear.state_dict() -> packed binary format
    # break shared storage (tied embeddings) for safetensors
    sd = {k: v.detach().contiguous().clone() for k, v in sd.items()}
    if getattr(model.config, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    save_file(sd, str(args.out / "model.safetensors"), metadata={"format": "pt"})

    lb_cfg = dict(quant_func=qargs.quant_func, eff_bit=args.eff_bit, split_dim=1024, residual=args.residual,
                  kv_factor=args.kv_factor, min_split_dim=8, use_itq=args.use_itq, itq_n_iter=args.itq_iters)
    (args.out / "littlebit_config.json").write_text(json.dumps(lb_cfg, indent=2))
    cfg = model.config.to_dict()
    cfg["auto_map"] = {"AutoModelForCausalLM": "quantization.hub.LittleBitModel"}
    cfg.update({k: lb_cfg[k] for k in ("quant_func", "eff_bit", "split_dim", "residual", "use_itq", "itq_n_iter")})
    cfg["torch_dtype"] = args.dtype
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    if not args.no_tokenizer:
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
                  "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"):
            if (args.base / f).is_file():
                shutil.copy(args.base / f, args.out / f)

    n_lb = sum(1 for k in sd if k.endswith(".U_packed"))
    print(f"wrote {args.out} ({n_lb} LittleBit layers)")



def dump_reference(args, qargs) -> None:
    """Logits of the official LittleBit forward, after a save -> load_quantized_model round trip."""
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from official_loader import load_official

    ids = torch.tensor([[int(t) for t in args.ref_tokens.split(",")]])
    ref = load_official(args.out, args.littlebit_repo)
    with torch.no_grad():
        logits = ref(ids).logits[0].float().numpy()
    np.save(args.out / "ref_logits.npy", logits)
    (args.out / "ref_tokens.txt").write_text(args.ref_tokens)
    print(f"reference logits {logits.shape} -> {args.out / 'ref_logits.npy'}")

if __name__ == "__main__":
    main()
