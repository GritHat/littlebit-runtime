#!/usr/bin/env python3
"""Perplexity + per-token agreement of the llama.cpp LittleBit runtime vs the official PyTorch implementation,
on identical wikitext-2 test windows (LittleBit eval.py methodology: non-overlapping windows over
"\\n\\n".join(test), loss on all seqlen-1 positions).

    python tests/ppl_compare.py <littlebit ckpt> <model.gguf> [--windows 4] [--seqlen 512] [--skip-official]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def wikitext2_test_ids(tok) -> np.ndarray:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pq.read_table(f).column("text").to_pylist())
    return np.array(tok(text)["input_ids"], dtype=np.int64)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def official_logits(ck: Path, windows: list[np.ndarray]) -> list[np.ndarray]:
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    from official_loader import load_official
    model = load_official(ck)
    out = []
    with torch.no_grad():
        for w in windows:
            out.append(model(torch.from_numpy(w)[None]).logits[0].float().numpy())
            print(f"  official window {len(out)}/{len(windows)}", flush=True)
    return out


def runtime_logits(gguf: Path, windows: list[np.ndarray], threads: int) -> list[np.ndarray]:
    out = []
    with tempfile.TemporaryDirectory() as d:
        for i, w in enumerate(windows):
            f = Path(d) / "l.bin"
            subprocess.run([str(ROOT / "build" / "bin" / "littlebit"), "logits", str(gguf), "--tokens", ",".join(map(str, w)),
                            "--out", str(f), "-t", str(threads)],
                           check=True, stderr=subprocess.DEVNULL)
            out.append(np.fromfile(f, dtype=np.float32).reshape(len(w), -1))
            print(f"  runtime window {i + 1}/{len(windows)}", flush=True)
    return out


def nll(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    lp = log_softmax(logits[:-1].astype(np.float64))
    return -lp[np.arange(len(ids) - 1), ids[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-official", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.ckpt)
    ids = wikitext2_test_ids(tok)
    windows = [ids[i * a.seqlen:(i + 1) * a.seqlen] for i in range(a.windows)]
    print(f"wikitext2 test: {len(ids)} tokens, {a.windows} windows x {a.seqlen}")

    rt = runtime_logits(a.gguf, windows, a.threads)
    rt_nll = np.concatenate([nll(l, w) for l, w in zip(rt, windows)])
    print(f"runtime  (llama.cpp) PPL = {np.exp(rt_nll.mean()):.4f}")
    if a.skip_official:
        return 0
    of = official_logits(a.ckpt, windows)
    of_nll = np.concatenate([nll(l, w) for l, w in zip(of, windows)])
    kl = np.concatenate([(np.exp(log_softmax(o.astype(np.float64))) *
                          (log_softmax(o.astype(np.float64)) - log_softmax(r.astype(np.float64)))).sum(-1)
                         for o, r in zip(of, rt)])
    top1 = np.mean(np.concatenate([o.argmax(-1) == r.argmax(-1) for o, r in zip(of, rt)]))
    print(f"official (PyTorch)   PPL = {np.exp(of_nll.mean()):.4f}")
    print(f"KL(official||runtime) mean={kl.mean():.5f} p99={np.quantile(kl, 0.99):.5f}  top1 agreement={top1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
