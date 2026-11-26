#!/usr/bin/env python3
"""LittleBit quantization-aware training of a Next-Concept Modeling (NCM) Sudoku generator.

The generator of HazyResearch/ncm (MIT, arXiv 2610.14162) is the teacher. The student is a copy whose transformer blocks
use SamsungLabs/LittleBit layers (CC BY-NC 4.0: results are non-commercial); input projection, embeddings and output heads
stay dense. Puzzles are generated here; the 2000 published test puzzles are only used for scoring.
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ncm", required=True, help="HazyResearch/ncm checkout")
ap.add_argument("--littlebit", required=True, help="SamsungLabs/LittleBit checkout")
ap.add_argument("--artifacts", type=Path, required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
ap.add_argument("--eff-bit", type=float, default=0.55, help="0: keep dense weights (control)")
ap.add_argument("--itq", type=int, default=1)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--mse", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--base-grids", type=int, default=20000)
ap.add_argument("--val", type=int, default=0, help="held-out puzzles, generated like the training ones, to tune on instead of the test set")
ap.add_argument("--resume", action="store_true", help="continue from state.pt in --out")
ap.add_argument("--state-every", type=int, default=0, help="steps between writes of state.pt")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
for k in ("ncm", "littlebit"):
    setattr(args, k, str(Path(getattr(args, k)).resolve()))
args.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, args.ncm)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

BLANKS = {"easy": 41, "medium": 46, "hard": 51}[args.difficulty]
DEV = "cuda"
T0 = time.time()
LOG = open(args.out / "log.jsonl", "a")


def log(**kw):
    kw = {"t": round(time.time() - T0, 1), **kw}
    LOG.write(json.dumps(kw) + "\n")
    LOG.flush()
    print(json.dumps(kw), flush=True)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def random_grid(seed):
    """A complete grid by randomized backtracking on the most constrained cell."""
    rng = random.Random(seed)
    g = [0] * 81
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9

    def fill():
        best, opts = -1, None
        for i in range(81):
            if g[i] == 0:
                used = rows[i // 9] | cols[i % 9] | boxes[i // 27 * 3 + i % 9 // 3]
                o = [d for d in range(9) if not used >> d & 1]
                if opts is None or len(o) < len(opts):
                    best, opts = i, o
                    if len(o) <= 1:
                        break
        if best < 0:
            return True
        rng.shuffle(opts)
        r, c, b = best // 9, best % 9, best // 27 * 3 + best % 9 // 3
        for d in opts:
            m = 1 << d
            g[best] = d + 1
            rows[r] |= m
            cols[c] |= m
            boxes[b] |= m
            if fill():
                return True
            g[best] = 0
            rows[r] ^= m
            cols[c] ^= m
            boxes[b] ^= m
        return False

    fill()
    return g


def base_grids():
    cache = args.out.parent / f"grids_{args.base_grids}.npy"
    if cache.exists():
        return np.load(cache)
    with mp.get_context("fork").Pool(6) as pool:
        g = np.array(pool.map(random_grid, range(1000, 1000 + args.base_grids), chunksize=200), dtype=np.int64)
    np.save(cache, g)
    return g


def augment(g):
    """Random Sudoku symmetries of grids [B, 81] on the GPU: digits, transpose, bands, rows, stacks, columns."""
    B = g.shape[0]
    g = g.view(B, 9, 9)
    t = torch.rand(B, device=g.device) < 0.5
    g = torch.where(t[:, None, None], g.transpose(1, 2), g)

    def perm():
        band = torch.rand(B, 3, device=g.device).argsort(1)
        inner = torch.rand(B, 3, 3, device=g.device).argsort(2)
        return (band[:, :, None] * 3 + inner).view(B, 9)

    g = g.gather(1, perm()[:, :, None].expand(B, 9, 9))
    g = g.gather(2, perm()[:, None, :].expand(B, 9, 9))
    digits = torch.rand(B, 9, device=g.device).argsort(1) + 1
    return digits.gather(1, (g - 1).reshape(B, 81))


def tokens(g, blank):
    """[B, 81] -> (prefix [B, 91], solution [B, 89]) with row separators (10) and markers (11)."""
    def lay(x):
        B = x.shape[0]
        x = x.view(B, 9, 9)
        sep = torch.full((B, 9, 1), 10, device=x.device)
        return torch.cat([x, sep], 2).view(B, 90)[:, :89]

    p = lay(g.masked_fill(blank, 0))
    m = torch.full((g.shape[0], 1), 11, device=g.device)
    return torch.cat([m, p, m], 1), lay(g)


def make_batch(grids):
    g = augment(grids[torch.randint(len(grids), (args.batch,), device=DEV)])
    blank = torch.zeros_like(g, dtype=torch.bool)
    blank.scatter_(1, torch.rand(g.shape, device=DEV).argsort(1)[:, :BLANKS], True)
    return tokens(g, blank)


def load_teacher():
    from utils.generation import Generator
    from utils.inference import add_speed_args
    p = argparse.ArgumentParser()
    add_speed_args(p)
    opt = p.parse_args(["--no-compile", "--no-cuda-graphs", "--precision", "fp32"])
    cfg = OmegaConf.load(Path(args.ncm) / f"config/eval/sudoku-{args.difficulty}.yaml")
    return Generator(cfg, args.artifacts / "sudoku" / args.difficulty, opt)


def solved(runtime, model, puzzles, sols):
    old, runtime.model = runtime.model, model
    was = model.training
    model.eval()
    ok = cells = 0
    for i in range(0, len(puzzles), 250):
        out = runtime.generate_tokens(puzzles[i:i + 250])
        ok += (out == sols[i:i + 250]).all(1).sum().item()
        cells += (out == sols[i:i + 250]).float().mean(1).sum().item()
    model.train(was)
    runtime.model = old
    return 100 * ok / len(puzzles), 100 * cells / len(puzzles)


def littlebit_student(teacher):
    sys.path.insert(1, args.littlebit)
    os.environ.setdefault("LOCAL_RANK", "0")
    import quantization.utils  # noqa: F401  (breaks a circular import in quantization.modules)
    from quantization.functions import SmoothSign
    from quantization.modules import LittleBitLinear
    student = copy.deepcopy(teacher)
    dense = bits = n = 0
    for mod in student.transformer.blocks.modules():
        if type(mod) is nn.Linear:
            mod.__class__ = LittleBitLinear
            mod.__quant_convert__(do_train=True, quant_func=SmoothSign, split_dim=1024, eff_bit=args.eff_bit, residual=True,
                                  ratio_factor=1.0, min_split_dim=8, use_itq=bool(args.itq), itq_n_iter=50)
            dense += mod.in_features * mod.out_features
            bits += mod.total_bit_usage
            n += 1
    student.to(DEV)
    log(event="student", layers=n, quantized_weights=dense, actual_bpw=round(bits / dense, 4),
        total_params=sum(p.numel() for p in teacher.parameters()))
    return student


torch.manual_seed(args.seed)
runtime = load_teacher()
teacher = runtime.model.eval().requires_grad_(False)
puzzles, sols = [torch.tensor(SafeUnpickler(open(args.data / "sudoku" / args.difficulty / f, "rb")).load(), device=DEV)
                 for f in ("puzzles.pickle", "solutions.pickle")]
log(event="teacher", difficulty=args.difficulty, solved=solved(runtime, teacher, puzzles, sols))
grids = torch.tensor(base_grids(), device=DEV)
test_set = {tuple(r) for r in sols.tolist()}
assert not any(tuple(r) in test_set for r in tokens(grids, torch.zeros_like(grids, dtype=torch.bool))[1].tolist())
val = None
if args.val:
    held, grids = grids[:args.val], grids[args.val:]
    g = torch.Generator(device=DEV).manual_seed(12345)
    blank = torch.zeros_like(held, dtype=torch.bool)
    blank.scatter_(1, torch.rand(held.shape, device=DEV, generator=g).argsort(1)[:, :BLANKS], True)
    val = tokens(held, blank)
log(event="grids", n=len(grids), val=args.val)

student = littlebit_student(teacher) if args.eff_bit > 0 else copy.deepcopy(teacher)
student.requires_grad_(True)
log(event="init", solved=solved(runtime, student, puzzles, sols))
codec = runtime.tokenizer.vqvae
trunk = {"t": [], "s": []}
recording = False
for key, m in (("t", teacher), ("s", student)):
    for b in m.transformer.blocks:
        b.register_forward_hook(lambda _m, _i, o, key=key: trunk[key].append(o) if recording else None)


class Checkpointed(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, **kw):
        if torch.is_grad_enabled() and not kw.get("return_kv") and kw.get("past_kv") is None:
            return torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=False, **kw)
        return self.block(x, **kw)


blocks = student.transformer.blocks
params = [p for p in student.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
student.eval()  # no dropout: the targets are the teacher's deterministic outputs
best = (-1.0, 0)
start = 0
state_file = args.out / "state.pt"
if args.resume and state_file.exists():
    st = torch.load(state_file, map_location=DEV, weights_only=True)
    with torch.no_grad():
        for n, q in student.named_parameters():
            q.copy_(st["params"][n])
    opt.load_state_dict(st["opt"])
    start = st["step"]
    torch.set_rng_state(st["rng"].cpu())
    torch.cuda.set_rng_state(st["cuda_rng"].cpu())
    log(event="resumed", step=start)
for step in range(start + 1, args.steps + 1):
    f = step / args.steps
    lr = args.lr * (f / 0.02 if f < 0.02 else 0.5 * (1 + math.cos(math.pi * (f - 0.02) / 0.98)))
    for g in opt.param_groups:
        g["lr"] = lr
    prefix, sol = make_batch(grids)
    with torch.no_grad():
        idx = codec.encode(sol)
        xin = torch.cat([teacher.build_l0_query_input(len(sol)), codec.vq.idx_to_ncm_input(idx)], 1)
        zp = codec.encoder.wte(prefix)[:, : teacher.prefix_len]
    trunk["t"].clear()
    trunk["s"].clear()
    student.transformer.blocks = nn.ModuleList(Checkpointed(b) for b in blocks)
    recording = True
    with torch.autocast(DEV, dtype=torch.float16, enabled=False):
        with torch.no_grad():
            lt = teacher(torch.cat([zp, xin], 1))
        ls = student(torch.cat([zp, torch.cat([student.build_l0_query_input(len(sol)), xin[:, teacher.levels[0]:]], 1)], 1))
    recording = False  # not during recomputation in backward
    kl = F.kl_div(F.log_softmax(ls, -1), F.log_softmax(lt, -1), log_target=True, reduction="sum") / lt.shape[:2].numel()
    ce = F.cross_entropy(ls.flatten(0, 1), idx.flatten())
    assert len(trunk["s"]) == len(trunk["t"]) == len(blocks)
    mse = sum(F.mse_loss(a, b) for a, b in zip(trunk["s"], trunk["t"])) / len(trunk["t"])
    loss = kl + ce + args.mse * mse
    if not torch.isfinite(loss):
        raise SystemExit(f"loss is not finite at step {step}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    student.transformer.blocks = blocks
    trunk["t"].clear()
    trunk["s"].clear()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step in (1, 20) or step % 100 == 0:
        log(event="step", step=step, kl=round(kl.item(), 4), ce=round(ce.item(), 4), mse=round(mse.item(), 3), lr=lr,
            sec=round((time.time() - T0), 1), vram=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    if step % args.eval_every == 0 or step == args.steps:
        s = solved(runtime, student, puzzles, sols)
        log(event="eval", step=step, solved=s, **({"val": solved(runtime, student, *val)} if val else {}))
        best = max(best, (s[0], step))
    if args.state_every and step % args.state_every == 0:
        tmp = args.out / "state.tmp"
        torch.save({"params": {n: q.detach() for n, q in student.named_parameters()}, "opt": opt.state_dict(), "step": step,
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp)
        os.replace(tmp, state_file)
log(event="done", best=best)
trunk["s"].clear()
trunk["t"].clear()
for b in student.transformer.blocks:
    b._forward_hooks.clear()
torch.save({k: v.cpu() for k, v in student.state_dict().items()}, args.out / "student.pt")
