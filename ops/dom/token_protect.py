#!/usr/bin/env python3
"""Token-path protection for an anchored domain mask (E2 follow-up, quantum: "qiskit" misspelling).

E2 finding (docs/domain/E2_RESULTS.md, paper sec:eresults): quantum-anc_k4 and A192p fail most qiskit_he items with
ModuleNotFoundError on a misspelled package ('qit', 'qisskit', 'qqiskit'); quantum-easyep never does. "qiskit"
tokenises as q + isk + it, so the loss is in the experts that copy a rare identifier, not quantum knowledge.
EASY-EP's per-token output-aware score keeps them; our bank-level REAP mass does not (203 'isk' tokens are noise in
a 400k-token bucket).

Fix, EASY-EP method restricted to the identifier path: from the EASY-EP recorder pass (s03_easyep: per-token
routing idx, g*||e|| and cos), score every expert over the PREDICTING positions of the identifier only
(t-1 -> 'q', t -> 'isk', t+1 -> 'it'):

    S_tok(e) = sum_{t in P} sum_slots [idx_t == e] * g*||e||_t * max(1 - cos_t, 0)

then extend the anchored mask with up to Q extra swaps per layer, using anchored.py's own rule:
  out: next-lowest mixed-score A192 experts still in the mask (never a guard / protected general slice, never one of
       the mask's own domain swap-ins);
  in:  highest S_tok experts not in the mask, only while S_tok(in) > S_tok(out).
CPU only.
  systemd-run --user --scope -q -p MemoryMax=6G nice -n 19 .venv/bin/python ops/dom/token_protect.py \
      --bucket quantum --base anc_k4 --q 4
-> $W/dom/d2/anchored/quantum/anc_k4_tokq4.json (keepset format) + .info.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import d2lib as D  # noqa: E402
from ops.dom import select_dom as S  # noqa: E402
from ops.dom.anchored import MIX_W, OUT  # noqa: E402
from ops.dom.easyep import COS_COL, easyep_scores  # noqa: E402
from ops.lib import env, keepset  # noqa: E402

STAGE = "token_protect"
# DeepSeek-V4.1 tokenizer: 'qiskit' = q|isk|it ; ' qiskit' = Ġq|isk|it ; Qiskit variants likewise
IDENT = {"qiskit": {"first": (83, 5618, 51, 1646), "mid": (8089,), "last": (279,)}}


def ident_positions(tok: np.ndarray, region: np.ndarray, first, mid, last) -> np.ndarray:
    """Positions whose NEXT token is a piece of the identifier (the router state that predicts it)."""
    t = np.asarray(tok)
    n = len(t)
    a = np.flatnonzero(np.isin(t[:-2], first) & (t[1:-1] == mid[0]) & (t[2:] == last[0]))  # 'q' at a
    pos = np.unique(np.concatenate([a - 1, a, a + 1]))
    pos = pos[(pos >= 0) & (pos < n)]
    return pos[region[pos]]


def extend(keep: list[int], a192: list[int], mixed: np.ndarray, s: np.ndarray, fixed: np.ndarray, q: int):
    K = np.zeros(D.E, bool); K[keep] = True
    A = np.zeros(D.E, bool); A[a192] = True
    fx = np.zeros(D.E, bool); fx[fixed] = True
    outs = [int(e) for e in np.argsort(mixed, kind="stable") if A[e] and K[e] and not fx[e]]
    ins = [int(e) for e in np.argsort(-s, kind="stable") if not K[e]]
    new, swaps = set(keep), []
    for o, i in zip(outs, ins):
        if len(swaps) >= q or s[i] <= s[o]:
            break
        new.discard(o); new.add(i); swaps.append((o, i))
    assert len(new) == len(keep)
    return sorted(new), swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="quantum")
    ap.add_argument("--base", default="anc_k4")
    ap.add_argument("--ident", default="qiskit")
    ap.add_argument("--q", type=int, default=4)
    a = ap.parse_args()
    C, R = D.DOM / "corpus_easyep", D.DOM / "s03_easyep"
    tok = np.load(C / "tokens.npy", mmap_mode="r")
    valid = np.load(C / "valid.npy", mmap_mode="r").astype(bool)
    region = np.zeros(len(tok), bool)
    for r in map(json.loads, open(C / "rows.jsonl")):
        if r["bucket"] == a.bucket:
            region[r["offset"]:r["offset"] + r["length"]] = True
    region &= valid
    P = ident_positions(tok, region, **IDENT[a.ident])
    env.log(f"{a.bucket}/{a.ident}: {len(P)} predicting positions", STAGE)
    base = keepset.load(OUT / a.bucket / f"{a.base}.json")[0]
    a192m = keepset.load(D.S04 / "keepset.json")[0]
    av = D.available_buckets()
    avail = {b: av[b] for b in D.TIER1 if b in av}
    sup, protect, _ = S._guards_cached(D.DOM / "d2", avail)
    mask, info = {}, {"positions": int(len(P)), "layers": {}}
    for L in range(D.N_LAYERS):
        idx = np.asarray(np.load(R / "routing" / f"L{L}_idx.npy", mmap_mode="r")[P])
        gn = np.asarray(np.load(R / "easyep" / f"L{L}_gn.npy", mmap_mode="r")[P])
        cos = np.asarray(np.load(R / "easyep" / f"L{L}_cos.npy", mmap_mode="r")[P])[:, COS_COL["full"]]
        s = easyep_scores(idx, gn, cos, np.ones(len(P), bool))
        li = D.LayerInputs(L, {a.bucket: avail[a.bucket]}, D.guards(L, sup, protect), buckets=[a.bucket], halves=False)
        fixed = np.union1d(li.guard, li.gen_slice(D.DEFAULT["pf"])).astype(int)
        mixed = li.mixed(a.bucket, MIX_W, 1, dom=li.domain(a.bucket, 1))["reap"]
        keep, swaps = extend(base[str(L)], a192m[str(L)], mixed, s, fixed, a.q)
        mask[str(L)] = keep
        tot = s.sum()
        info["layers"][L] = {"swaps": swaps, "tok_mass_base": float(s[base[str(L)]].sum() / tot) if tot else None,
                             "tok_mass_new": float(s[keep].sum() / tot) if tot else None}
        env.log(f"L{L}: {len(swaps)} swaps, token mass {info['layers'][L]['tok_mass_base']:.3f} -> "
                f"{info['layers'][L]['tok_mass_new']:.3f}", STAGE)
        del li
    name = f"{a.base}_tok{a.ident[0]}{a.q}"
    sha = keepset.save(OUT / a.bucket / f"{name}.json", mask, arm=f"{a.bucket}:{name}", anchored_on="A192",
                       mix_w=MIX_W, base=a.base, ident=a.ident, q=a.q)
    info.update(sha256=sha, n_swaps=sum(len(v["swaps"]) for v in info["layers"].values()))
    env.atomic_json(OUT / a.bucket / f"{name}.info.json", info)
    env.log(f"wrote {name} sha {sha[:8]}: {info['n_swaps']} swaps", STAGE)


if __name__ == "__main__":
    main()
