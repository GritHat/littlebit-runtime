#!/usr/bin/env python3
"""Greedy generation with the *official* LittleBit PyTorch implementation (reference for the runtime).

    python tools/llm/official_generate.py <littlebit ckpt dir> <tokenizer dir> "prompt" [n_tokens]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from official_loader import load_official  # noqa: E402

ck, tok_dir, prompt = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
model = load_official(ck)
tok = AutoTokenizer.from_pretrained(tok_dir)
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
print(repr(tok.decode(out[0, ids.shape[1]:])))
