#!/usr/bin/env python3
"""Post-training quantization baseline for the NCM Sudoku generator, for comparison with train_sudoku.py.

Quantizes exactly the linear layers that train_sudoku.py replaces (those of the transformer blocks), without training, and
scores the published test puzzles. Weights are rounded per block of --block values along each row with an fp16 scale (and,
with "a", an fp16 offset); scale and offset minimize the squared error of the block. bpw counts them.
"""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn


def quantize(W, bits, block, offset):
    """W [out, in] -> (rounded W, stored bits). Each block is alone: levels 0..2^bits-1 map to scale * level + lo."""
    out, n = W.shape
    pad = -n % block
    x = torch.cat([W, W.new_zeros(out, pad)], 1).reshape(-1, block)
    n_lev = 2 ** bits - 1
    best = best_err = None
    lo0, hi0 = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    for f in torch.linspace(0.3, 1.05, 31).tolist():
        if offset:
            mid, half = (hi0 + lo0) / 2, (hi0 - lo0) / 2 * f
            scale, lo = (2 * half / n_lev).clamp_min(1e-9), mid - half
        else:  # odd levels, symmetric about 0: -(2^b-1)..(2^b-1) in steps of 2
            scale = (2 * x.abs().amax(1, keepdim=True) * f / n_lev).clamp_min(1e-9)
            lo = -scale * n_lev / 2
        scale, lo = scale.half().to(W.dtype).clamp_min(6e-8), lo.half().to(W.dtype) if offset else lo
        q = ((x - lo) / scale).round().clamp(0, n_lev)
        y = q * scale + lo
        if offset:  # least-squares scale and offset for these levels
            qm, xm = q.mean(1, keepdim=True), x.mean(1, keepdim=True)
            var = ((q - qm) ** 2).sum(1, keepdim=True)
            s = torch.where(var > 0, ((q - qm) * (x - xm)).sum(1, keepdim=True) / var.clamp_min(1e-12), scale)
            o = xm - s * qm
            s, o = s.half().to(W.dtype), o.half().to(W.dtype)
            y = torch.where((s * q + o - x).pow(2).sum(1, keepdim=True) < (y - x).pow(2).sum(1, keepdim=True), s * q + o, y)
        err = (y - x).pow(2).sum(1, keepdim=True)
        if best is None:
            best, best_err = y, err
        else:
            best = torch.where(err < best_err, y, best)
            best_err = torch.minimum(err, best_err)
    n_blocks = x.shape[0]
    return best.reshape(out, n + pad)[:, :n], bits * out * n + n_blocks * (32 if offset else 16)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncm", required=True, help="HazyResearch/ncm checkout")
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--configs", default="4:32:s,4:32:a,3:32:a,2:32:a,2:16:a", help="bits:block:s|a, comma separated")
    ap.add_argument("--out", type=Path, help="write one json line per result here")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.ncm).resolve()))
    from omegaconf import OmegaConf
    from utils.generation import Generator
    from utils.inference import add_speed_args
    p = argparse.ArgumentParser()
    add_speed_args(p)
    opt = p.parse_args(["--no-compile", "--no-cuda-graphs", "--precision", "fp32"])
    cfg = OmegaConf.load(Path(a.ncm) / f"config/eval/sudoku-{a.difficulty}.yaml")
    rt = Generator(cfg, a.artifacts / "sudoku" / a.difficulty, opt)
    teacher = rt.model.eval().requires_grad_(False)
    puz, sol = (torch.tensor(SafeUnpickler(open(a.data / "sudoku" / a.difficulty / f, "rb")).load(), device="cuda")
                for f in ("puzzles.pickle", "solutions.pickle"))

    def score(model):
        rt.model = model
        ok = cells = 0
        for i in range(0, len(puz), 250):
            out = rt.generate_tokens(puz[i:i + 250])
            ok += (out == sol[i:i + 250]).all(1).sum().item()
            cells += (out == sol[i:i + 250]).float().mean(1).sum().item()
        rt.model = teacher
        return 100 * ok / len(puz), 100 * cells / len(puz)

    def report(**kw):
        print(json.dumps(kw), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(kw) + "\n")

    report(difficulty=a.difficulty, config="teacher fp32", solved=score(teacher))
    for c in a.configs.split(","):
        bits, block, kind = c.split(":")
        m = copy.deepcopy(teacher)
        n = tot = 0
        e = w2 = 0.0
        for mod in m.transformer.blocks.modules():
            if type(mod) is nn.Linear:
                W = mod.weight.data.float()
                y, b = quantize(W, int(bits), int(block), kind == "a")
                e += (y - W).pow(2).sum().item()
                w2 += W.pow(2).sum().item()
                mod.weight.data = y.to(mod.weight.dtype)
                n += W.numel()
                tot += b
        report(difficulty=a.difficulty, config=c, bpw=round(tot / n, 3), weights=n, rel_err=round((e / w2) ** 0.5, 4), solved=score(m))


if __name__ == "__main__":
    main()
