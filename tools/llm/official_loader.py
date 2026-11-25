"""Load a LittleBit checkpoint with the *official* SamsungLabs loader, with two compatibility fixes for
current transformers versions (LittleBit pins transformers 4.51):

1. `config.json` written by LittleBit's hub.py has a 3-part `auto_map` that newer transformers rejects;
   we load from a staged copy without it.
2. `load_quantized_model` builds the model on `meta` and calls `to_empty()`, which leaves *non-persistent*
   buffers (not in the state dict) uninitialized. For Gemma 3 these are `embed_scale` and the rotary
   `inv_freq` tables -> NaN logits. We rebuild those buffers from a model constructed with meta
   parameters but real buffers. The packed-format path also leaves scales in bf16; we compute in fp32.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _meta_parameters():
    """Like accelerate.init_empty_weights(include_buffers=False): parameters on meta, buffers real."""
    import torch
    import torch.nn as nn
    old = nn.Module.register_parameter

    def register_parameter(module, name, param):
        old(module, name, param)
        if param is not None:
            p = module._parameters[name]
            module._parameters[name] = nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)

    nn.Module.register_parameter = register_parameter
    try:
        with torch.no_grad():
            yield
    finally:
        nn.Module.register_parameter = old


def load_official(ckpt: Path, littlebit_repo: Path | None = None):
    sys.path.insert(0, str(littlebit_repo or ROOT / "third_party" / "LittleBit"))
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from quantization.utils.quant_util import load_quantized_model

    lb = json.loads((ckpt / "littlebit_config.json").read_text())
    lb.setdefault("quant_mod", "LittleBitLinear")
    qargs = argparse.Namespace(model_id=str(ckpt), **lb)
    with tempfile.TemporaryDirectory() as tmp:
        for f in ckpt.iterdir():
            if f.name != "config.json":
                (Path(tmp) / f.name).symlink_to(f.resolve())
        cfg = json.loads((ckpt / "config.json").read_text())
        cfg.pop("auto_map", None)
        (Path(tmp) / "config.json").write_text(json.dumps(cfg))
        model = load_quantized_model(tmp, qargs, torch_dtype=torch.float32, device="cpu")
        config = AutoConfig.from_pretrained(tmp)

    persistent = set(model.state_dict().keys())
    with _meta_parameters():
        fresh = AutoModelForCausalLM.from_config(config)
    fresh_bufs = dict(fresh.named_buffers())
    fixed = []
    for name, buf in list(model.named_buffers()):
        if name not in persistent and name in fresh_bufs and fresh_bufs[name].device.type != "meta":
            mod_name, _, attr = name.rpartition(".")
            mod = model.get_submodule(mod_name) if mod_name else model
            mod._buffers[attr] = fresh_bufs[name].to(buf.dtype if buf.is_floating_point() else fresh_bufs[name].dtype)
            fixed.append(name)
    if fixed:
        print(f"official_loader: re-initialized {len(fixed)} non-persistent buffers ({fixed[0]}, ...)")
    return model.float().eval()
