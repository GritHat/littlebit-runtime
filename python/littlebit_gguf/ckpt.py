"""Recognize LittleBit layers in a checkpoint's tensor list (packed or legacy/unpacked format)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .q1 import Branch, signs_from_float, unpack_littlebit_signs

_SCALES = ("u1", "u2", "v1", "v2")
_RE_PACKED = re.compile(r"^(?P<p>.+)\.(?P<k>U|V|U_R|V_R)_(?P<s>packed|shape)$")
_RE_LEGACY = re.compile(r"^(?P<p>.+)\.(?P<k>U|V|U_R|V_R)$")
_RE_SCALE = re.compile(r"^(?P<p>.+)\.(?P<k>u1|u2|v1|v2|u1_R|u2_R|v1_R|v2_R)$")
_RE_BUFFER = re.compile(r"^(?P<p>.+)\.(?P<k>_eff_bit_target|_split_dim_final|_eff_bit_actual|_split_dim)$")


@dataclass
class LittleBitGroup:
    prefix: str                                   # e.g. model.layers.0.self_attn.q_proj
    names: dict[str, str] = field(default_factory=dict)  # role (U_packed, u1, V, ...) -> tensor name

    @property
    def residual(self) -> bool:
        return "u1_R" in self.names

    @property
    def packed(self) -> bool:
        return "U_packed" in self.names

    def validate(self) -> None:
        sfx = ["", "_R"] if self.residual else [""]
        for s in sfx:
            for k in _SCALES:
                if k + s not in self.names:
                    raise ValueError(f"{self.prefix}: missing LittleBit scale '{k + s}'")
            for m in ("U", "V"):
                if self.packed:
                    if f"{m}{s}_packed" not in self.names or f"{m}{s}_shape" not in self.names:
                        raise ValueError(f"{self.prefix}: missing '{m}{s}_packed'/'{m}{s}_shape'")
                elif f"{m}{s}" not in self.names:
                    raise ValueError(f"{self.prefix}: missing latent '{m}{s}'")

    def load_branches(self, load: Callable[[str], np.ndarray]) -> list[Branch]:
        out = []
        for s in (["", "_R"] if self.residual else [""]):
            if self.packed:
                U = unpack_littlebit_signs(load(self.names[f"U{s}_packed"]),
                                           tuple(np.asarray(load(self.names[f"U{s}_shape"])).tolist()))
                V = unpack_littlebit_signs(load(self.names[f"V{s}_packed"]),
                                           tuple(np.asarray(load(self.names[f"V{s}_shape"])).tolist()))
            else:
                U = signs_from_float(load(self.names[f"U{s}"]))
                V = signs_from_float(load(self.names[f"V{s}"]))
            u1, u2, v1, v2 = (np.asarray(load(self.names[k + s]), dtype=np.float32).reshape(-1) for k in _SCALES)
            n_out, r = U.shape
            if V.shape[0] != r or u1.size != n_out or u2.size != r or v1.size != r or v2.size != V.shape[1]:
                raise ValueError(f"{self.prefix}{s}: inconsistent LittleBit shapes U={U.shape} V={V.shape} "
                                 f"u1={u1.size} u2={u2.size} v1={v1.size} v2={v2.size}")
            out.append(Branch(U=U, V=V, u1=u1, mid=(v1.astype(np.float64) * u2).astype(np.float32), v2=v2))
        return out

    def out_features(self, load: Callable[[str], np.ndarray]) -> int:
        return int(np.asarray(load(self.names["u1"])).size)


def find_littlebit_groups(names: list[str]) -> tuple[dict[str, LittleBitGroup], set[str]]:
    """Return (prefix -> group, all tensor names that belong to LittleBit layers incl. dropped buffers)."""
    groups: dict[str, LittleBitGroup] = {}
    consumed: set[str] = set()
    scale_prefixes = {m["p"] for n in names if (m := _RE_SCALE.match(n)) and m["k"] == "u1"}

    for n in names:
        m = _RE_PACKED.match(n)
        if m:
            groups.setdefault(m["p"], LittleBitGroup(m["p"])).names[f"{m['k']}_{m['s']}"] = n
            consumed.add(n)
            continue
        m = _RE_SCALE.match(n)
        if m and m["p"] in scale_prefixes:
            groups.setdefault(m["p"], LittleBitGroup(m["p"])).names[m["k"]] = n
            consumed.add(n)
            continue
        m = _RE_LEGACY.match(n)
        if m and m["p"] in scale_prefixes:
            groups.setdefault(m["p"], LittleBitGroup(m["p"])).names[m["k"]] = n
            consumed.add(n)
            continue
        m = _RE_BUFFER.match(n)
        if m:
            consumed.add(n)  # bookkeeping buffers, not needed at inference

    for g in groups.values():
        g.validate()
    return groups, consumed
