#!/usr/bin/env python3
"""Compare llama.cpp logits (littlebit logits) with reference logits (.npy). Exit 1 if they disagree."""
import argparse
import sys

import numpy as np


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    def logsoftmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    lr, lg = logsoftmax(ref.astype(np.float64)), logsoftmax(got.astype(np.float64))
    pr = np.exp(lr)
    kl = (pr * (lr - lg)).sum(-1)
    corr = [np.corrcoef(a, b)[0, 1] for a, b in zip(ref, got)]
    top1 = (ref.argmax(-1) == got.argmax(-1)).mean()
    top5 = np.mean([len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5])) / 5 for a, b in zip(ref, got)])
    return dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), corr_min=float(np.min(corr)),
                top1=float(top1), top5_overlap=float(top5),
                max_abs=float(np.abs(ref - got).max()), ref_std=float(ref.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_npy")
    ap.add_argument("got_bin")
    ap.add_argument("--max-kl", type=float, default=0.05)
    ap.add_argument("--min-top1", type=float, default=0.9)
    a = ap.parse_args()
    ref = np.load(a.ref_npy)
    got = np.fromfile(a.got_bin, dtype=np.float32).reshape(ref.shape)
    m = compare(ref, got)
    print("  ".join(f"{k}={v:.4g}" for k, v in m.items()))
    ok = m["kl_mean"] <= a.max_kl and m["top1"] >= a.min_top1
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
