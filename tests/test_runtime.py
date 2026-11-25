"""Runtime tests on tiny random models: graph correctness, loader validation, C API, CLI. No downloads needed.

    LITTLEBIT_BIN=build/bin pytest tests/test_runtime.py      (or: ctest --test-dir build)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import tiny  # noqa: E402
from test_layer import check_gguf  # noqa: E402

BIN = Path(os.environ.get("LITTLEBIT_BIN", ROOT / "build" / "bin"))
RUNNER = os.environ.get("LITTLEBIT_RUNNER", "").split()  # e.g. an emulator for cross-compiled binaries
TOKENS = [1, 10, 50, 90, 7, 33, 201, 150, 60, 11, 12, 13]


def run(*cmd, check=True, **kw):
    return subprocess.run([*RUNNER, *map(str, cmd)], check=check, capture_output=True, text=True, errors="replace", timeout=600, **kw)


def cli(*args, **kw):
    return run(BIN / "littlebit", *args, **kw)


def logits(path: Path, *args) -> np.ndarray:
    out = path.with_suffix(".bin")
    cli("logits", path, "--tokens", ",".join(map(str, TOKENS)), "--out", out, "-t", 2, "--kv", "f32", *args)
    return np.fromfile(out, dtype=np.float32).reshape(len(TOKENS), -1)


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("arch", tiny.ARCHS)
def test_graph_matches_materialized_weights(tmp_path, arch, residual):
    """Same weights as dense matrices (unpatched code path), as float factors, and as Q1_0 factors."""
    res = {k: logits(tiny.write(tmp_path / f"{k}.gguf", arch, k, residual)) for k in ("dense", "f32", "f16", "q1_0")}
    assert np.isfinite(res["dense"]).all() and res["dense"].std() > 0.1
    err = {k: rel(v, res["dense"]) for k, v in res.items()}
    print(arch, residual, err)
    # unpatched llama.cpp differs from itself by up to 2e-4 here when only the batch size changes
    assert err["f32"] < 5e-4
    assert err["f16"] < 5e-3   # + fp16 scales
    assert err["q1_0"] < 1e-1  # + 8-bit activations; exactness of the kernels is checked by test_kernels


def test_batch_splitting_does_not_change_results(tmp_path):
    p = tiny.write(tmp_path / "m.gguf", "llama", "q1_0")
    assert rel(logits(p), logits(p, "-b", 1)) < 5e-4


@pytest.mark.parametrize("batch", [1, 256])
def test_repacked_kernels(tmp_path, batch):
    """Repacked weights (ARM) give the same result as the plain kernels; a no-op where nothing is repacked."""
    p = tiny.write(tmp_path / "m.gguf", "llama", "q1_0")
    assert rel(logits(p, "-b", batch), logits(p, "-b", batch, "--no-repack")) < 1e-5


@pytest.mark.parametrize("kind", ["q1_0", "f16", "f32"])
@pytest.mark.parametrize("repack", [False, True])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])
def test_kernels(tmp_path, arch, repack, kind):
    """Every layer equals numpy with the activation rounding ggml applies (Q8_0 or none)."""
    assert check_gguf(tiny.write(tmp_path / "m.gguf", arch, kind), BIN / "test_layer", runner=RUNNER, all_layers=True,
                      repack=repack)


def bad(name_part, fn):
    def mutate(name, arr, dtype):
        return fn(name, arr, dtype) if name.startswith("blk.0.ffn_up.") and name.endswith(name_part) else [(name, arr, dtype)]
    return mutate


Q8 = tiny.gguf.GGMLQuantizationType.Q8_0
F32 = tiny.gguf.GGMLQuantizationType.F32
INVALID = {
    "missing lb_v": (bad(".lb_v", lambda n, a, d: []), "lb_v"),
    "missing lb_v2": (bad(".lb_v2", lambda n, a, d: []), "lb_v2"),
    "missing lb_v2_r": (bad(".lb_v2_r", lambda n, a, d: []), "lb_v2_r"),
    "rank mismatch": (bad(".lb_v_r", lambda n, a, d: [(n, a[:-8], d)]), "lb_v_r"),
    "wrong outputs": (bad(".lb_u", lambda n, a, d: [(n, a[:-1], d)]), "inconsistent factor shapes"),
    "rank larger than lb_u": (bad(".lb_v", lambda n, a, d: [(n, np.repeat(a, 4, 0), d)]), "inconsistent factor shapes"),
    "wrong type": (bad(".lb_v", lambda n, a, d: [(n, np.zeros((a.shape[0], 4 * 34), np.uint8), Q8)]), "unsupported type"),
    "dense and factors": (bad(".lb_u", lambda n, a, d: [(n, a, d), ("blk.0.ffn_up.weight",
                          np.zeros((tiny.FFN, tiny.HIDDEN), np.float32), F32)]), "both dense and factorized"),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_files_are_rejected(tmp_path, case):
    mutate, message = INVALID[case]
    r = cli("info", tiny.write(tmp_path / "m.gguf", "llama", "q1_0", mutate=mutate), check=False)
    assert r.returncode == 1 and message in r.stderr, r.stderr


@pytest.mark.parametrize("version", [None, 0, 2])
def test_unknown_format_version_is_rejected(tmp_path, version):
    r = cli("info", tiny.write(tmp_path / "m.gguf", "llama", "q1_0", version=version), check=False)
    assert r.returncode == 1 and "version" in r.stderr, r.stderr


def test_unsupported_architecture_is_rejected(tmp_path):
    r = cli("info", tiny.write(tmp_path / "m.gguf", "smollm3", "q1_0"), check=False)
    assert r.returncode == 1 and "architecture 'smollm3' is not supported" in r.stderr, r.stderr


def test_truncated_file_is_rejected(tmp_path):
    p = tiny.write(tmp_path / "m.gguf", "llama", "q1_0")
    p.write_bytes(p.read_bytes()[:-4096])
    r = cli("info", p, check=False)
    assert r.returncode == 1, r.stderr


def test_c_api(tmp_path):
    for arch in ("llama", "gemma3"):
        assert run(BIN / "test_api", tiny.write(tmp_path / f"{arch}.gguf", arch, "q1_0")).stdout.strip() == "ok"


def test_cli(tmp_path):
    p = tiny.write(tmp_path / "m.gguf", "llama", "q1_0")
    assert "littlebit format 1" in cli("info", p).stdout
    a = run(BIN / "littlebit", "run", p, "-p", "hi", "-n", 8, "--temp", 0, "--kv", "q8_0").stdout
    b = run(BIN / "littlebit", "run", p, "-n", 8, "--temp", 0, "--kv", "q8_0", input="hi").stdout
    assert a == b
    assert "tok/s" in cli("bench", p, "-t", 2).stderr
    cli("chat", p, "-n", 4, input="hello\nagain\n/reset\nhello\n/quit\n")
    assert cli("version").stdout.startswith("littlebit ")
    for args in (["nope", p], ["run"], ["run", p, "--bogus"], ["run", p, "-n", "x"], ["run", p, "-c", "0"],
                 ["info", tmp_path / "missing.gguf"]):
        assert cli(*args, check=False, stdin=subprocess.DEVNULL).returncode == 1, args
