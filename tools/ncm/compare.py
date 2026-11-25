#!/usr/bin/env python3
"""Compare `littlebit ncm` with the official PyTorch code on the CPU: logits of every level and of the output.

With --student, the generator's linear layers are replaced by the dense matrices that the LittleBit factors multiply out to.
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("model", type=Path)
ap.add_argument("--ncm", type=Path, required=True)
ap.add_argument("--artifacts", type=Path, required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--difficulty", default="easy")
ap.add_argument("--student", type=Path)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--bin", type=Path, default=Path(__file__).resolve().parents[2] / "build/bin/littlebit")
a = ap.parse_args()
sys.path[:0] = [str(a.ncm.resolve()), str(Path(__file__).resolve().parents[2])]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402

torch.set_num_threads(a.threads)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


cfg = OmegaConf.load(a.ncm / f"config/eval/sudoku-{a.difficulty}.yaml")
d = a.artifacts / "sudoku" / a.difficulty
vq = VQVAE(**cfg.tokenizer.vqvae)
vq.load_state_dict(torch.load(d / "tokenizer.pt", weights_only=True, map_location="cpu")["model"])
vq.eval()
m = NextConceptTransformer(vqvae_local=vq, config=cfg.model)
sd = torch.load(d / "generator.pt", weights_only=True, map_location="cpu")["model"]
if a.student:
    from littlebit_gguf.ckpt import find_littlebit_groups
    ck = torch.load(a.student, weights_only=True, map_location="cpu")
    fl = {k: v.float().numpy() if v.is_floating_point() else v.numpy() for k, v in ck.items()}
    groups, _ = find_littlebit_groups(list(fl))
    for k in sd:
        if k in ck:
            sd[k] = ck[k]
    for k, g in groups.items():
        W = sum(b.dense() for b in g.load_branches(lambda x: fl[x]))
        sd[k.replace("_orig_mod.", "") + ".weight"] = torch.from_numpy(np.asarray(W, dtype=np.float32))
print(m.load_state_dict(sd, strict=False))
m.eval()
rec = []
orig = m._sample_level
m._sample_level = lambda lg, *x, **k: (rec.append(lg[0].float().numpy().copy()), orig(lg, *x, **k))[1]
puz, sol = (SafeUnpickler(open(a.data / "sudoku" / a.difficulty / f, "rb")).load() for f in ("puzzles.pickle", "solutions.pickle"))
g = cfg.generation
K = 16
for i in range(a.n):
    rec.clear()
    with torch.no_grad():
        _, z = m.generate(None, torch.tensor([list(map(int, puz[i]))]), temperature=g.temperature, top_k=g.top_k, top_p=g.top_p,
                          soft_decisions=True, soft_temperature=g.soft_temperature, return_latent=True)
        rec.append(vq.decoder(z)[0].numpy())
    with tempfile.NamedTemporaryFile() as f:
        out = subprocess.run([a.bin, "ncm", a.model, "-t", str(a.threads), "--tokens", ",".join(map(str, map(int, puz[i]))), "--out", f.name],
                             check=True, capture_output=True, text=True).stdout
        mine = np.fromfile(f.name, dtype=np.float32)
    ref = np.concatenate([r.reshape(-1) for r in rec])
    assert ref.size == mine.size, (ref.size, mine.size)
    toks = np.array([int(x) for x in out.split(",")])
    pos, errs = 0, []
    for r in rec:
        e = mine[pos:pos + r.size] - r.reshape(-1)
        errs.append(float(np.sqrt((e ** 2).mean() / (r ** 2).mean())))
        pos += r.size
    print(f"puzzle {i}: rel. error per level " + " ".join(f"{e:.1e}" for e in errs[:-1]) + f", output {errs[-1]:.1e}, tokens equal "
          f"{(toks == rec[-1].argmax(-1)).mean():.3f}, solved here {bool((toks == np.array(sol[i])).all())} "
          f"there {bool((rec[-1].argmax(-1) == np.array(sol[i])).all())}")
