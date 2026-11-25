#!/usr/bin/env python3
"""LittleBit-2 QAT on one GPU, with a time or step limit, perplexity during training and resumable state.

Same recipe as SamsungLabs/LittleBit main.py (logit KL + layer-wise MSE distillation, AdamW, cosine schedule, frozen
embeddings and lm_head), on their own layers and losses. The learning rate follows --steps, or --minutes without it.
Checkpoints are in the official format; state.pt keeps the latent weights and optimizer for --resume.

Needs transformers 4.51 and a checkout of SamsungLabs/LittleBit (CC BY-NC 4.0: results are non-commercial).
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--littlebit", default="/root/LittleBit")
ap.add_argument("--out", type=Path, default=Path("/root/out"))
ap.add_argument("--eff-bit", type=float, default=1.0)
ap.add_argument("--minutes", type=float, default=80)
ap.add_argument("--steps", type=int, default=0, help="total optimizer steps of the schedule (0: use --minutes)")
ap.add_argument("--resume", action="store_true", help="continue from <out>/state.pt")
ap.add_argument("--state-minutes", type=float, default=30)
ap.add_argument("--deadline", type=float, default=0, help="unix time after which training must be over")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--seqs-per-step", type=int, default=4)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--docs", type=int, default=100000)
ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--eval-minutes", type=float, default=8)
ap.add_argument("--eval-windows", type=int, default=40)
ap.add_argument("--synthetic-data", action="store_true", help="random tokens instead of C4 + wikitext (smoke test)")
args = ap.parse_args()

sys.path.insert(0, args.littlebit)
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.utils.quant_util import apply_littlebit_patch
from utils.kd_utils import KDTrainer
from utils.utils import prepare_model_for_training

T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
args.out.mkdir(parents=True, exist_ok=True)
LOG = open(args.out / "log.jsonl", "a")
os.environ.setdefault("LOCAL_RANK", "0")
torch.manual_seed(42)


def log(**kw):
    kw = {"t": round(time.time() - T0, 1), **kw}
    LOG.write(json.dumps(kw) + "\n")
    LOG.flush()
    print(json.dumps(kw), flush=True)


def load_data(tok):
    if args.synthetic_data:
        r = np.random.default_rng(0)
        ids = r.integers(0, 5000, (64, args.seqlen)).astype(np.int32)
        return ids, r.integers(0, 5000, (args.eval_windows, args.seqlen)).astype(np.int64)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("allenai/c4", "en/c4-train.00000-of-01024.json.gz", repo_type="dataset")
    docs = []
    with gzip.open(f, "rt", encoding="utf-8") as h:
        for line in h:
            docs.append(json.loads(line)["text"])
            if len(docs) >= args.docs:
                break
    stream = []
    for i in range(0, len(docs), 2000):
        for ids in tok(docs[i:i + 2000])["input_ids"]:
            stream.append(np.array(ids, dtype=np.int32))
    wt = lambda s: "\n\n".join(pq.read_table(hf_hub_download(  # noqa: E731
        "Salesforce/wikitext", f"wikitext-2-raw-v1/{s}-00000-of-00001.parquet", repo_type="dataset")).column("text").to_pylist())
    stream.append(np.array(tok(wt("train"))["input_ids"], dtype=np.int32))
    flat = np.concatenate(stream)
    n = len(flat) // args.seqlen
    data = flat[: n * args.seqlen].reshape(n, args.seqlen)
    data = data[np.random.default_rng(42).permutation(n)]
    test = np.array(tok(wt("test"))["input_ids"], dtype=np.int64)
    m = len(test) // args.seqlen
    return data, test[: m * args.seqlen].reshape(m, args.seqlen)


@torch.no_grad()
def ppl(model, windows):
    was = model.training
    model.eval()
    tot = 0.0
    for w in windows:
        ids = torch.from_numpy(w)[None].to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            lg = model(ids).logits[0, :-1]
        tot += F.cross_entropy(lg.float(), ids[0, 1:]).item()
    model.train(was)
    return math.exp(tot / len(windows))


def save(model, tok, name):
    d = args.out / name
    d.mkdir(exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        if v.dtype == torch.float32 and "packed" not in k and "shape" not in k:
            v = v.to(torch.bfloat16)
        sd[k] = v.contiguous().clone()
    if model.config.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
    lb = dict(quant_func="SmoothSign", eff_bit=args.eff_bit, split_dim=1024, residual=True, kv_factor=1.0, min_split_dim=8,
              quant_mod="LittleBitLinear", use_itq=True, itq_n_iter=50)
    (d / "littlebit_config.json").write_text(json.dumps(lb, indent=2))
    cfg = model.config.to_dict()
    cfg.update({k: lb[k] for k in lb}, use_cache=True, torch_dtype="bfloat16")
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    tok.save_pretrained(d)


tok = AutoTokenizer.from_pretrained(args.model)
data, test = load_data(tok)
log(event="data", train_seqs=len(data), test_windows=len(test))

teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(DEV).eval()
teacher.config.use_cache = False
teacher.requires_grad_(False)
log(event="teacher", ppl_full=ppl(teacher, test), ppl=ppl(teacher, test[: args.eval_windows]))

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.config.use_cache = False
prepare_model_for_training(model)
t = time.time()
apply_littlebit_patch(model, argparse.Namespace(
    quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=True, split_dim=1024, eff_bit=args.eff_bit,
    min_split_dim=8, use_itq=True, itq_n_iter=50, kv_factor=1.0, model_id=args.model), do_train=True)
model.to(DEV).train()
params = [p for p in model.parameters() if p.requires_grad]
log(event="init", seconds=time.time() - t, trainable=sum(p.numel() for p in params), ppl_full=ppl(model, test),
    ppl=ppl(model, test[: args.eval_windows]))

opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
kd = KDTrainer.__new__(KDTrainer)  # only its loss functions are used
start = time.time()
end = start + (args.minutes * 60 if not args.steps else 1e9)
if args.deadline:
    end = min(end, args.deadline)
batch, pos, step, window = args.batch, 0, 0, []
last_eval, last_save = start, start
if args.resume:
    st = torch.load(args.out / "state.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p, v in zip(params, st["params"], strict=True):
            p.copy_(v)
    opt.load_state_dict(st["opt"])
    step, pos = st["step"], st["pos"]
    del st
    log(event="resumed", step=step, ppl=ppl(model, test[: args.eval_windows]))


def save_state():
    tmp = args.out / "state.tmp"
    torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(), "step": step, "pos": pos}, tmp)
    tmp.replace(args.out / "state.pt")


def loss_on(ids):
    with torch.autocast(DEV, dtype=torch.bfloat16):
        with torch.no_grad():
            to = teacher(ids, output_hidden_states=True)
        so = model(ids, output_hidden_states=True)
        k = kd.ce_loss(so.logits, to.logits)
        m = 10.0 * sum(kd.mse_loss(s, u) for s, u in zip(so.hidden_states[1:], to.hidden_states[1:]))
    return k + m, k.detach(), m.detach()


while time.time() < end and not (args.steps and step >= args.steps):
    frac = step / args.steps if args.steps else min(1.0, (time.time() - start) / (end - start))
    lr = args.lr * (frac / 0.02 if frac < 0.02 else 0.5 * (1 + math.cos(math.pi * (frac - 0.02) / 0.98)))
    for g in opt.param_groups:
        g["lr"] = lr
    s0 = time.time()
    ids = torch.from_numpy(data[np.arange(pos, pos + args.seqs_per_step) % len(data)].astype(np.int64)).to(DEV)
    try:
        n = math.ceil(args.seqs_per_step / batch)
        kl = ml = 0.0
        for mb in ids.split(batch):
            l, k, m = loss_on(mb)
            (l / n).backward()
            kl += k.item() / n
            ml += m.item() / n
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if batch == 1:
            raise
        batch = max(1, batch // 2)
        log(event="oom", new_batch=batch)
        continue
    if not math.isfinite(kl + ml):
        log(event="nonfinite", step=step, kl=kl, mse=ml)
        raise SystemExit("loss is not finite")
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    pos += args.seqs_per_step
    step += 1
    window.append(time.time() - s0)
    if step in (1, 5, 20) or step % 50 == 0:
        log(event="step", step=step, kl_per_token=round(kl / args.seqlen, 4), mse=round(ml, 3), gnorm=round(gn, 2), lr=lr,
            sec_per_step=round(float(np.mean(window[-50:])), 3), batch=batch,
            vram=round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEV == "cuda" else 0)
    if time.time() - last_eval > args.eval_minutes * 60:
        log(event="eval", step=step, ppl=ppl(model, test[: args.eval_windows]))
        last_eval = time.time()
    if time.time() - last_save > args.state_minutes * 60:
        save(model, tok, "ckpt")
        save_state()
        last_save = time.time()
        log(event="saved", step=step)

log(event="done", step=step, seqs=pos, ppl_full=ppl(model, test), ppl=ppl(model, test[: args.eval_windows]),
    mean_sec_per_step=float(np.mean(window)) if window else None)
save(model, tok, "final")
save_state()
log(event="saved_final")
