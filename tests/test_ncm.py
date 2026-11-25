"""NCM engine tests on tiny random models against a numpy implementation of the published model. No downloads, no torch.

    LITTLEBIT_BIN=build/bin pytest tests/test_ncm.py      (or: ctest --test-dir build)
"""
from __future__ import annotations

import os
import sys
from math import erf
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from littlebit_gguf import ncm as N  # noqa: E402
from test_runtime import cli, rel  # noqa: E402

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", ROOT / "build/_deps/llama_cpp-src"))
VERF = np.vectorize(erf)

VARIANTS = {
    # one code at level 0, predicted from a learned query, soft decisions (as in the sudoku models)
    "query": dict(levels=[1, 2, 3, 6], prefix=5, query=True, tied=False, soft=True, strides=[6]),
    # same, with several level-0 codes and stacked upsampling
    "cascade": dict(levels=[2, 3, 5, 8], prefix=3, query=True, tied=False, soft=False, strides=[2, 2]),
    # level 0 drawn from a unigram, heads tied to the codebooks (as in the TinyStories model)
    "text": dict(levels=[1, 3, 4, 8, 8], prefix=1, query=False, tied=True, soft=False, strides=[2, 4]),
    # several soft level-0 codes through the upsampling
    "cascade-soft": dict(levels=[2, 4, 8], prefix=2, query=True, tied=False, soft=True, strides=[4]),
}
E, H, D, DH, K, VOC, LAYERS = 128, 4, 32, 2, 7, 12, 2


def make(variant: dict, littlebit: bool, seed=0):
    rng = np.random.default_rng(seed)
    r = lambda *s, sd=1.0: (rng.standard_normal(s) * sd).astype(np.float32)  # noqa: E731
    lv, n, first = variant["levels"], len(variant["levels"]), 0 if variant["query"] else 1
    cfg = {
        "task": "test",
        "model": dict(n_layer=LAYERS, n_head=H, n_embd=E, prefix_len=variant["prefix"], use_rope=True, rope_base=1000, use_qk_norm=True,
                      use_level_encoding=True, raw_token_prefix=True, query_level0=variant["query"], tie_heads_to_codebook=variant["tied"]),
        "tokenizer": dict(pre_tokenizer=dict(name="other"), vqvae=dict(
            n_layers=DH, embed_dim=D, in_vocab_size=VOC, compression_factor=1, dec_layers=1,
            vector_quantizer_config=dict(learned_l1_sampling=True, quant_resi=dict(enabled=True, ratio=0.3, share_mode=0),
                                         levels=dict(use_manual_levels=True, manual_levels=lv)))),
        "generation": dict(temperature=0.0, soft_decisions=variant["soft"], soft_temperature=1.7),
    }
    gen, tok = {}, {}

    def block(sd, p, d, hd, lb):
        for ln in ("ln_1", "ln_2"):
            sd[f"{p}{ln}.weight"], sd[f"{p}{ln}.bias"] = 1 + r(d, sd=.1), r(d, sd=.1)
        sd[p + "attn.q_norm.weight"], sd[p + "attn.k_norm.weight"] = 1 + r(hd, sd=.1), 1 + r(hd, sd=.1)
        for name, i, o in (("attn.c_attn", d, 3 * d), ("attn.c_proj", d, d), ("mlp.c_fc", d, 4 * d), ("mlp.c_proj", 4 * d, d)):
            sd[f"{p}{name}.bias"] = r(o, sd=.1)
            if not lb:
                sd[f"{p}{name}.weight"] = r(o, i, sd=.08)
                continue
            rank = 20
            for s in ("", "_R"):
                sd[f"{p}{name}.U{s}"], sd[f"{p}{name}.V{s}"] = r(o, rank), r(rank, i)
                sd[f"{p}{name}.u1{s}"], sd[f"{p}{name}.u2{s}"] = .1 * (1 + r(o, sd=.2)), 1 + r(rank, sd=.2)
                sd[f"{p}{name}.v1{s}"], sd[f"{p}{name}.v2{s}"] = .1 * (1 + r(rank, sd=.2)), 1 + r(i, sd=.2)

    gen["transformer.wte.weight"], gen["transformer.wte.bias"] = r(E, D, sd=.2), r(E, sd=.1)
    gen["transformer.wle.weight"] = r(n - first, E, sd=.3)
    gen["transformer.ln_f.weight"], gen["transformer.ln_f.bias"] = 1 + r(E, sd=.1), r(E, sd=.1)
    if variant["query"]:
        gen["l0_queries"] = r(1, lv[0], D)
    for i in range(LAYERS):
        block(gen, f"transformer.blocks.{i}.", E, E // H, littlebit)
    for h in range(n - first):
        if variant["tied"]:
            gen[f"lm_heads.{h}.proj.weight"], gen[f"lm_heads.{h}.log_inv_T"] = r(D, E, sd=.2), np.float32(.3 + .1 * h)
        else:
            gen[f"lm_heads.{h}.weight"], gen[f"lm_heads.{h}.bias"] = r(K, E, sd=.2), r(K)
    tok["encoder.wte.weight"] = r(VOC, D)
    for i in range(n):
        tok[f"vq.embeddings.{i}.weight"] = r(K, D)
        tok[f"vq.quant_resi.{i}.weight"], tok[f"vq.quant_resi.{i}.bias"] = r(D, D, 3, sd=.15), r(D, sd=.1)
    for j, s in enumerate(variant["strides"]):
        tok[f"vq.learned_up_l1.{j}.weight"], tok[f"vq.learned_up_l1.{j}.bias"] = r(D, D, s, sd=.2), r(D, sd=.1)
    block(tok, "decoder.middle_block.", D, D // DH, False)
    tok["decoder.ln_f.weight"], tok["decoder.output_layer.weight"] = 1 + r(D, sd=.1), r(VOC, D, sd=.3)
    uni = np.zeros(K, np.float32)
    uni[3] = 1
    return cfg, gen, tok, {"level0_unigram": uni}


class Reference:
    """The model as published, in float64 and without caches."""

    def __init__(self, variant, gen, tok):
        self.v, self.gen, self.tok = variant, gen, tok
        groups, _ = N.find_littlebit_groups(list(gen))
        self.dense = {k: sum(b.dense() for b in g.load_branches(lambda x: gen[x])) for k, g in groups.items()}

    def lin(self, sd, p, x):
        return x @ (self.dense[p] if p in self.dense else sd[p + ".weight"]).T + sd.get(p + ".bias", 0)

    @staticmethod
    def norm(sd, p, x):
        x = x - x.mean(-1, keepdims=True)
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * sd[p + ".weight"] + sd.get(p + ".bias", 0)

    def block(self, sd, p, x, heads, pos, mask):
        T, C = x.shape
        hd = C // heads
        q, k, v = (a.reshape(T, heads, hd).transpose(1, 0, 2)
                   for a in np.split(self.lin(sd, p + "attn.c_attn", self.norm(sd, p + "ln_1", x)), 3, axis=1))

        def rms(a, w):
            return a / np.sqrt((a ** 2).mean(-1, keepdims=True) + 1e-6) * w

        def rot(a):
            if pos is None:
                return a
            h = hd // 2
            ang = np.array(pos)[:, None] * 1000.0 ** -(np.arange(h) / h)
            c, s = np.cos(np.concatenate([ang, ang], 1)), np.sin(np.concatenate([ang, ang], 1))
            return a * c + np.concatenate([-a[..., h:], a[..., :h]], -1) * s

        q, k = rot(rms(q, sd[p + "attn.q_norm.weight"])), rot(rms(k, sd[p + "attn.k_norm.weight"]))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        y = (s / s.sum(-1, keepdims=True) @ v).transpose(1, 0, 2).reshape(T, C)
        x = x + self.lin(sd, p + "attn.c_proj", y)
        u = self.lin(sd, p + "mlp.c_fc", self.norm(sd, p + "ln_2", x))
        return x + self.lin(sd, p + "mlp.c_proj", 0.5 * u * (1 + VERF(u / np.sqrt(2))))

    def run(self, prefix):
        gen, tok, lv, n = self.gen, self.tok, self.v["levels"], len(self.v["levels"])
        L, P, first, logs = lv[-1], len(prefix), 0 if self.v["query"] else 1, []
        pre = self.lin(gen, "transformer.wte", tok["encoder.wte.weight"][prefix].astype(np.float64))
        z, x = np.zeros((L, D)), gen["l0_queries"][0] if self.v["query"] else None
        for k, l in enumerate(lv):
            E_k = tok[f"vq.embeddings.{k}.weight"].astype(np.float64)
            if k < first:
                e = E_k[[3] * l]
            else:
                h = np.concatenate([pre, self.lin(gen, "transformer.wte", x) + gen["transformer.wle.weight"][k - first]])
                mask = np.zeros((P + l, P + l))
                mask[:P, P:] = -np.inf
                pos = [*range(P), *range(l)]
                for i in range(LAYERS):
                    h = self.block(gen, f"transformer.blocks.{i}.", h, H, pos, mask)
                h = self.norm(gen, "transformer.ln_f", h)[P:]
                p = f"lm_heads.{k - first}"
                if self.v["tied"]:
                    sq = (E_k ** 2).sum(-1)
                    lg = (2 * (h @ gen[p + ".proj.weight"].T) @ E_k.T - sq) / sq.mean() * np.exp(gen[p + ".log_inv_T"])
                else:
                    lg = self.lin(gen, p, h)
                logs.append(lg)
                if self.v["soft"]:
                    s = np.exp((lg - lg.max(-1, keepdims=True)) / 1.7)
                    e = s / s.sum(-1, keepdims=True) @ E_k
                else:
                    e = E_k[lg.argmax(-1)]
            if k == 0:
                for j, s in enumerate(self.v["strides"]):
                    w, o = tok[f"vq.learned_up_l1.{j}.weight"], np.zeros((len(e) * s, D))
                    for t in range(len(e)):
                        for a in range(s):
                            o[t * s + a] = e[t] @ w[:, :, a] + tok[f"vq.learned_up_l1.{j}.bias"]
                    e = o
            elif l != L:
                c = np.clip((np.arange(L) + 0.5) * l / L - 0.5, 0, l - 1)
                e = np.stack([np.interp(c, np.arange(l), e[:, d]) for d in range(D)], 1)
            j = N.phi_index(k, n)
            w, pad = tok[f"vq.quant_resi.{j}.weight"], np.pad(e, ((1, 1), (0, 0)))
            conv = sum(pad[a:a + L] @ w[:, :, a].T for a in range(3)) + tok[f"vq.quant_resi.{j}.bias"]
            z = z + 0.7 * e + 0.3 * conv
            if k + 1 < n:
                m = lv[k + 1]
                x = np.stack([z[t * L // m:-(-(t + 1) * L // m)].mean(0) for t in range(m)])
        y = self.block(tok, "decoder.middle_block.", z, DH, None, np.zeros((L, L)))
        logs.append(self.lin(tok, "decoder.output_layer", self.norm(tok, "decoder.ln_f", y)))
        return logs


def engine(path, prefix, *args):
    out = path.with_suffix(".bin")
    r = cli("ncm", path, "--tokens", ",".join(map(str, prefix)), "--out", out, "-t", 2, *args)
    return np.fromfile(out, dtype=np.float32), [int(x) for x in r.stdout.split(",")]


def write(path, variant, littlebit, **kw):
    cfg, gen, tok, extra = make(variant, littlebit)
    N.check_config(cfg)
    N.convert(cfg, gen, tok, extra, path, llama_cpp=LLAMA, **kw)
    return gen, tok


@pytest.mark.parametrize("littlebit", [False, True])
@pytest.mark.parametrize("name", VARIANTS)
def test_matches_reference(tmp_path, name, littlebit):
    v = VARIANTS[name]
    prefix = [(3 * i + 1) % VOC for i in range(v["prefix"])]
    gen, tok = write(tmp_path / "m.gguf", v, littlebit, factor_type="f32", dense_type="f32")
    ref = Reference(v, gen, tok).run(prefix)
    flat = np.concatenate([x.reshape(-1) for x in ref])
    got, toks = engine(tmp_path / "m.gguf", prefix)
    assert got.shape == flat.shape
    assert flat.std() > 0.1
    assert rel(got, flat) < 2e-5
    assert toks == ref[-1].argmax(-1).tolist()


@pytest.mark.parametrize("repack", [True, False])
def test_quantized(tmp_path, repack):
    """Q1_0 factors and Q8_0 dense tensors stay close to the reference; soft decisions, so errors cannot flip a code."""
    v = VARIANTS["query"]
    prefix = [1, 4, 7, 10, 1]
    gen, tok = write(tmp_path / "m.gguf", v, True)
    flat = np.concatenate([x.reshape(-1) for x in Reference(v, gen, tok).run(prefix)])
    got, _ = engine(tmp_path / "m.gguf", prefix, *([] if repack else ["--no-repack"]))
    assert rel(got, flat) < 5e-2


@pytest.mark.parametrize("dense_type", ["f16", "q8_0"])
def test_huge_level0_codes(tmp_path, dense_type):
    """As in the released TinyStories tokenizer: level-0 codes far beyond fp16, scaled down by the upsampling."""
    v = VARIANTS["text"]
    cfg, gen, tok, extra = make(v, False)
    tok["vq.embeddings.0.weight"] *= 1e10
    tok["vq.learned_up_l1.0.weight"] *= 1e-10
    N.convert(cfg, gen, tok, extra, tmp_path / "m.gguf", dense_type=dense_type, llama_cpp=LLAMA)
    ref = Reference(v, gen, tok).run([1])
    got, _ = engine(tmp_path / "m.gguf", [1])
    assert np.isfinite(got).all()
    n = ref[0].size  # the first predicted level only: later ones depend on hard decisions
    assert rel(got[:n], ref[0].reshape(-1)) < 5e-2


def test_sampling(tmp_path):
    write(tmp_path / "m.gguf", VARIANTS["text"], False)
    run = lambda *a: engine(tmp_path / "m.gguf", [1], *a)[0].tobytes()  # noqa: E731
    assert run("--temp", "1", "--seed", "5") == run("--temp", "1", "--seed", "5")
    assert len({run("--temp", "1", "--seed", str(s)) for s in range(4)}) > 1
    assert run("--temp", "1", "--top-k", "1", "--seed", "1") == run("--temp", "0")
    assert run("--temp", "1", "--top-p", "0", "--seed", "1") == run("--temp", "0")


def test_bad_input(tmp_path):
    p = tmp_path / "m.gguf"
    write(p, VARIANTS["query"], True)
    for toks in ("1,2,3", "1,2,3,4,12"):
        r = cli("ncm", p, "--tokens", toks, check=False)
        assert r.returncode == 1 and r.stderr.strip()
    data = p.read_bytes()
    for n in (0, 10, 1000, len(data) // 2, len(data) - 100):
        (tmp_path / "t.gguf").write_bytes(data[:n])
        r = cli("ncm", tmp_path / "t.gguf", "--tokens", "1,2,3,4,5", check=False)
        assert r.returncode == 1, n
    assert cli("run", p, "-p", "x", check=False).returncode == 1


@pytest.mark.parametrize("name,patch", [
    ("level", lambda s: s.replace(b"ncm.version", b"ncm.versiom")),
    ("block", lambda s: s.replace(b"blk.1.qkv.lb_v_r", b"blk.1.qkv.lb_v_x")),
    ("unknown", lambda s: s.replace(b"dec.output", b"dec.outpuu")),
])
def test_malformed(tmp_path, name, patch):
    write(tmp_path / "m.gguf", VARIANTS["query"], True)
    data = (tmp_path / "m.gguf").read_bytes()
    bad = patch(data)
    assert bad != data
    (tmp_path / "bad.gguf").write_bytes(bad)
    r = cli("ncm", tmp_path / "bad.gguf", "--tokens", "1,2,3,4,5", check=False)
    assert r.returncode == 1 and "cannot load model" in r.stderr


def test_unsupported_configs_are_rejected():
    cfg = make(VARIANTS["query"], False)[0]
    cfg["model"]["predict_level0"] = True
    with pytest.raises(SystemExit):
        N.check_config(cfg)
