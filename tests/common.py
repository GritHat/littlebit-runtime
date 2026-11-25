"""Paths and the command line tool, for the tests of both engines."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "build" / "_deps" / "llama_cpp-src"))
sys.path[:0] = [str(ROOT / "python"), str(LLAMA / "gguf-py")]

BIN = Path(os.environ.get("LITTLEBIT_BIN", ROOT / "build" / "bin"))
RUNNER = os.environ.get("LITTLEBIT_RUNNER", "").split()  # e.g. an emulator for cross-compiled binaries


def run(*cmd, check=True, **kw):
    return subprocess.run([*RUNNER, *map(str, cmd)], check=check, capture_output=True, text=True, errors="replace", timeout=600, **kw)


def cli(*args, **kw):
    return run(BIN / "littlebit", *args, **kw)


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
