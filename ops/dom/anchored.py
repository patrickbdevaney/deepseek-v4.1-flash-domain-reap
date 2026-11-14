#!/usr/bin/env python3
"""A192-anchored swap-budget domain masks (D2 re-selection round, docs/domain/DOMAINS.md status 2026-11-04).

Why: the D2 replay showed every from-scratch domain mask (~158/192 experts per layer shared with A192) costs
+0.3..0.55 general kl32 vs A192 (gate: +0.1). General quality is very sensitive to the ~34 experts the masks drop,
so instead of re-solving the whole keep-set we start FROM A192 and swap a small budget k per layer:

  out: the k A192 experts with the lowest mixed score (w=0.45 domain/general mix, REAP criterion), never a guard
       (super-experts) nor the protected general slice (pf=0.04);
  in:  the k non-A192 experts with the highest domain score (unit domain REAP), only while
       domain(in) > domain(out) for the paired expert (a swap that does not gain domain mass is not made).

k in K_GRID traces the domain-vs-general curve; the replay picks per bucket the largest k that passes the gate.
Polymath uses the sum of the ten unit domain scores. CPU only; reads the D1 banks and A192 bank like select_dom.

RUN (CPU, capped):
  systemd-run --user --scope -q -p MemoryMax=6G nice -n 19 python3 ops/dom/anchored.py [--procs 2]
Writes $V41_WORK/dom/d2/anchored/<bucket>/anc_k<k>.json (keepset.save format) + _layers/ per-layer json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import d2lib as D  # noqa: E402
from ops.dom import select_dom as S  # noqa: E402
from ops.lib import env, keepset  # noqa: E402

STAGE = "d2_anchored"
K_GRID = (4, 8, 16, 32)
MIX_W = 0.45
OUT = D.DOM / "d2" / "anchored"
_G: dict = {}


def swap(a192: list[int], mixed: np.ndarray, dom: np.ndarray, fixed: np.ndarray, k: int) -> tuple[list[int], int]:
    A = np.zeros(D.E, bool); A[a192] = True
    fx = np.zeros(D.E, bool); fx[fixed] = True
    outs = [int(e) for e in np.argsort(mixed, kind="stable") if A[e] and not fx[e]]
    ins = [int(e) for e in np.argsort(-dom, kind="stable") if not A[e]]
    keep = set(a192)
    n = 0
    for o, i in zip(outs, ins):
        if n >= k or dom[i] <= dom[o]:
            break
        keep.discard(o); keep.add(i); n += 1
    assert len(keep) == len(a192)
    return sorted(keep), n


def _job(job):
    L, buckets, avail = job
    t0 = time.time()
    sup, protect = _G["sup"], _G["protect"]
    li = D.LayerInputs(L, avail, D.guards(L, sup, protect), buckets=list(avail), halves=False)
    a192 = keepset.load(D.S04 / "keepset.json")[0][str(L)]
    fixed = np.union1d(li.guard, li.gen_slice(D.DEFAULT["pf"])).astype(int)
    doms = {b: li.domain(b, 1) for b in avail}
    res = {"L": L, "fixed": fixed.tolist(), "cands": {}, "n_swaps": {}, "ret": {}}
    for b in buckets:
        if b == "polymath":
            dom = {k: sum(D.unit(d[k]) for d in doms.values()) for k in ("reap", "mass", "F")}
            mixed = li.mixed(next(iter(avail)), MIX_W, 1, dom=dom)
        else:
            dom = doms[b]
            mixed = li.mixed(b, MIX_W, 1, dom=dom)
        d = D.unit(dom["reap"])
        for k in K_GRID:
            keep, n = swap(a192, mixed["reap"], d, fixed, k)
            res["cands"][f"{b}:anc_k{k}"] = keep
            res["n_swaps"][f"{b}:anc_k{k}"] = n
            if b != "polymath":
                r = li.retained(keep, [b])
                res["ret"][f"{b}:anc_k{k}"] = {"dom_ho": float(np.mean([v for kk, v in r["ho"].items()
                                                                        if kk.startswith(b + "/") and v is not None])),
                                               "gen_ho": r["ho"].get("A192/general")}
        if b != "polymath":
            r = li.retained(a192, [b])
            res["ret"][f"{b}:A192"] = {"dom_ho": float(np.mean([v for kk, v in r["ho"].items()
                                                                if kk.startswith(b + "/") and v is not None])),
                                       "gen_ho": r["ho"].get("A192/general")}
    env.atomic_json(OUT / "_layers" / f"L{L}.json", res)
    return L, time.time() - t0


def _init(sup, protect):
    _G["sup"], _G["protect"] = sup, protect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    av = D.available_buckets()
    avail = {b: av[b] for b in D.TIER1 if b in av}
    sup, protect, ginfo = S._guards_cached(D.DOM / "d2", avail)
    buckets = sorted(avail) + ["polymath"]
    jobs = [(L, buckets, avail) for L in range(D.N_LAYERS)
            if a.force or not (OUT / "_layers" / f"L{L}.json").exists()]
    env.log(f"anchored: {len(jobs)} layers, buckets {buckets}, k {K_GRID}", STAGE)
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(sup, protect), maxtasksperchild=4) as pool:
        for L, dt in pool.imap_unordered(_job, jobs):
            env.log(f"anchored L{L} in {dt:.0f}s", STAGE)
    # assemble keep-sets
    per = [json.load(open(OUT / "_layers" / f"L{L}.json")) for L in range(D.N_LAYERS)]
    summary = {}
    for name in per[0]["cands"]:
        b, k = name.split(":")
        mask = {str(L): per[L]["cands"][name] for L in range(D.N_LAYERS)}
        sha = keepset.save(OUT / b / f"{k}.json", mask, arm=name, anchored_on="A192", mix_w=MIX_W, k=int(k[5:]))
        swaps = [per[L]["n_swaps"][name] for L in range(D.N_LAYERS)]
        summary[name] = {"sha256": sha, "swaps_mean": float(np.mean(swaps)), "swaps_min": int(min(swaps))}
        if name in per[0]["ret"]:
            for key in ("dom_ho", "gen_ho"):
                summary[name][key] = float(np.mean([per[L]["ret"][name][key] for L in range(D.N_LAYERS)]))
    for b in sorted(avail):
        summary[f"{b}:A192"] = {key: float(np.mean([per[L]["ret"][f"{b}:A192"][key] for L in range(D.N_LAYERS)]))
                                for key in ("dom_ho", "gen_ho")}
    env.atomic_json(OUT / "summary.json", summary)
    env.log(f"anchored: {len(summary)} entries -> {OUT / 'summary.json'}", STAGE)


if __name__ == "__main__":
    main()
