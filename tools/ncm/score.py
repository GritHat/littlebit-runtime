#!/usr/bin/env python3
"""Score `littlebit ncm` on the official sudoku test set: score.py MODEL DATA_DIR [--n 2000] [--threads 2]"""
import argparse
import pickle
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("model", type=Path)
ap.add_argument("data", type=Path)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--bin", type=Path, default=Path(__file__).resolve().parents[2] / "build/bin/littlebit")
a = ap.parse_args()


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def grid(toks):
    return "".join(str(int(t)) for t in toks if int(t) <= 9)


puz, sol = (SafeUnpickler(open(a.data / f, "rb")).load()[:a.n] for f in ("puzzles.pickle", "solutions.pickle"))
r = subprocess.run([a.bin, "ncm", a.model, "-t", str(a.threads)], input="\n".join(map(grid, puz)) + "\n", check=True,
                   capture_output=True, text=True)
out = r.stdout.split()
assert len(out) == len(puz)
ok = sum(o == grid(s) for o, s in zip(out, sol))
print(f"{a.model.name}: {ok}/{len(puz)} solved ({100 * ok / len(puz):.2f} %) {r.stderr.strip().splitlines()[-1]}")
