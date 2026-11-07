#!/usr/bin/env python3
"""D2: CPU domain-mask selection (docs/domain/D2_METHOD.md items 1-7 and 10). numpy only, no GPU, no model.

Reads the A192 9-bucket bank + routing ($V41_WORK/s03) and every COMPLETED domain pass bank
($V41_WORK/dom/s03_<pass>/bank, routing alongside) directly; it never writes $V41_WORK/dom/bank.
Writes everything under $V41_WORK/dom/d2 (or --out):

  <bucket>/<candidate>.json   keep-sets, s04/keepset.json format (ops/lib/keepset.save), 192 x 40 layers
  polymath/<candidate>.json   polymath rr4 / rr6 / control, and mix_* (same with the domain-mask general mix)
  manifest.json               every candidate: params, sha256, inputs
  diag/*.json                 self-trace bias, HOPE stability (+ shrinkage), router entropy, domain overlap,
                              retained mass, F-reconstruction validation
  _layers/L<L>/<bucket>.json  per-layer work units (resumable; assemble reads them)

Commands:
  status                  passes, completion, buckets available
  validate [--layers]     routing->stat-row mapping (gat) and reconstructed-vs-banked pooled F (Frobenius + HOPE Jaccard)
  entropy                 router entropy per layer (pooled, per bucket, A192) vs 7.30 bits
  layers  [--buckets]     per-layer candidate solves + stability halves (multiprocess, resumable)
  shrink                  only if a bucket's mean half-Jaccard < 0.9: shrunk-F stability + shrink candidates
  polymath                round-robin top-k per domain + HOPE fill, and the plain normalised-sum control
  assemble                write candidate JSONs, manifest, diagnostics
  all                     entropy -> validate -> layers -> shrink -> polymath -> assemble (what each new pass re-runs)

Run (always): systemd-run --user --scope -p MemoryMax=6G nice -n 19 .venv/bin/python ops/dom/select_dom.py all
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# one BLAS thread per worker: a 384x384 HOPE solve is 0.6 s single-threaded and ~6 s with every core's threads
# fighting over a matvec this small (measured on Thor, 2026-11-02)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import d2lib as D  # noqa: E402
from ops.lib import env, keepset  # noqa: E402
from ops.lib import select as SEL  # noqa: E402

STAGE = "d2_select"
OUT = D.D2


def _avail(a) -> dict:
    av = D.available_buckets(require_complete=not a.allow_incomplete)
    want = [b for b in (a.buckets or D.TIER1) if b in av]
    return {b: av[b] for b in want}


def _guards_cached(out: Path, avail: dict):
    """Super-expert union over A192 fmax and every available pass fmax; cached per pass set."""
    f = out / "diag" / "guards.json"
    passes = sorted({v["pass"] for v in avail.values()})
    if f.exists():
        g = json.load(open(f))
        if g["passes"] == passes:
            sup = np.zeros((D.N_LAYERS, D.E), bool)
            for L, ids in g["super"].items():
                sup[int(L), ids] = True
            return sup, json.load(open(D.S04 / "position_protect.json"))["protect"], g["info"]
    fm = None
    for p in passes:
        info = next(v for v in avail.values() if v["pass"] == p)
        # an unfinished pass (--allow-incomplete, testing only) contributes zeros for layers not banked yet
        x = np.stack([np.load(info["bank"] / f"L{L}.npz")["fmax"] if (info["bank"] / f"L{L}.npz").exists()
                      else np.zeros(D.E) for L in range(D.N_LAYERS)])
        if not info["complete"]:
            passes = passes + [f"{p}:incomplete"]
        fm = x if fm is None else np.maximum(fm, x)
    sup, protect, info = D.load_guards(fm)
    env.atomic_json(f, {"passes": passes, "info": info,
                        "super": {str(L): np.flatnonzero(sup[L]).tolist() for L in range(D.N_LAYERS)}})
    return sup, protect, info


# ----------------------------------------------------------------------------- status / validate / entropy
def status(a):
    for p, info in D.pass_dirs().items():
        n = sum((info["bank"] / f"L{L}.npz").exists() for L in range(D.N_LAYERS))
        print(f"{p}: buckets {info['buckets']} complete={info['complete']} layers banked {n}/{D.N_LAYERS}")
    av = D.available_buckets(require_complete=True)
    print("tier-1 available:", [b for b in D.TIER1 if b in av], " missing:", [b for b in D.TIER1 if b not in av])


def check_pass_bank(info: dict, L: int = 0) -> dict:
    """The pass bank must carry all spec bucket names and all spec sub-area rows, non-zero only for its own buckets."""
    from ops.dom import corpus_spec_dom as spec
    z = np.load(info["bank"] / f"L{L}.npz")
    names, sa = [str(x) for x in z["buckets"]], [str(x) for x in z["subareas"]]
    live = [sa[i] for i in np.flatnonzero(z["cnt"].sum(1) > 0)]
    out = {"pass": info["pass"], "buckets_18": names == spec.BUCKETS, "subareas_177": sa == spec.SUBAREAS,
           "rows_shape": list(z["sum"].shape), "live_subareas": len(live),
           "live_outside_pass": sorted({s.split("/")[0] for s in live} - set(info["buckets"])),
           "pass_buckets_without_live": sorted(set(info["buckets"]) - {s.split("/")[0] for s in live})}
    out["ok"] = (out["buckets_18"] and out["subareas_177"] and not out["live_outside_pass"]
                 and not out["pass_buckets_without_live"])
    return out


def validate(a):
    avail = _avail(a)
    passes = {v["pass"]: v for v in avail.values()}
    res = {"bank_format": [check_pass_bank(v) for v in passes.values()], "layers": {}}
    C9 = D.corpus_cached(D.A192_CORPUS, "bucket_id")
    for L in a.layers:
        r = {}
        b9 = SEL.load_bank(D.A192_BANK, L)
        r["A192_gat_relerr"] = D.routing_gat_check(b9, C9, D.A192_BANK / "routing", L, range(9))
        s9 = D.layer_sources(b9, C9, D.A192_BANK / "routing", L, range(9))
        r["A192_F"] = _cmp_F(b9["F_sum"], sum(s9["F"].values()))
        for p, info in passes.items():
            bk = SEL.load_bank(info["s03"], L)
            C = D.corpus_cached(info["corpus"], "sub_id")
            ids = [int(i) for i in np.flatnonzero(bk["cnt"].sum(1) > 0)]
            r[f"{p}_gat_relerr"] = D.routing_gat_check(bk, C, info["routing"], L, ids)
            s = D.layer_sources(bk, C, info["routing"], L, ids)
            r[f"{p}_F"] = _cmp_F(bk["F_sum"], sum(s["F"].values()))
        res["layers"][str(L)] = r
        env.log(f"validate L{L}: {json.dumps(r)}", STAGE)
    env.atomic_json(a.out / "diag" / "validate.json", res)
    print(json.dumps(res["bank_format"], indent=1))


def _cmp_F(Fb: np.ndarray, Fr: np.ndarray) -> dict:
    off = ~np.eye(D.E, dtype=bool)
    kb = SEL.keep_from_pruned(SEL.prune_set("hope", n_keep=D.K_KEEP, protect=np.zeros(0, int), F=Fb), D.E)
    kr = SEL.keep_from_pruned(SEL.prune_set("hope", n_keep=D.K_KEEP, protect=np.zeros(0, int), F=Fr), D.E)
    return {"rel_frob": float(np.linalg.norm(Fb - Fr) / np.linalg.norm(Fb)),
            "rel_frob_offdiag": float(np.linalg.norm((Fb - Fr)[off]) / np.linalg.norm(Fb[off])),
            "corr_offdiag": float(np.corrcoef(Fb[off], Fr[off])[0, 1]),
            "hope_keep_jaccard": D.jaccard(kb, kr)}


def entropy(a):
    avail = _avail(a)
    passes = sorted({v["pass"] for v in avail.values()})
    res = {"ref_bits": D.ENTROPY_REF, "max_bits": float(np.log2(D.E)), "layers": {}}
    for L in range(D.N_LAYERS):
        r = {"A192": D.entropy_bits(SEL.load_bank(D.A192_BANK, L)["cnt"].sum(0))}
        z9 = SEL.load_bank(D.A192_BANK, L)
        for i, b in enumerate(D.ORIG9):
            r[f"A192/{b}"] = D.entropy_bits(z9["cnt"][i])
        pooled = np.zeros(D.E)
        for p in passes:
            info = next(v for v in avail.values() if v["pass"] == p)
            z = np.load(info["bank"] / f"L{L}.npz")
            cnt, sa = z["cnt"], [str(x) for x in z["subareas"]]
            pooled += cnt.sum(0)
            for b in info["buckets"]:
                if b in avail:
                    r[b] = D.entropy_bits(cnt[[i for i, n in enumerate(sa) if n.split("/")[0] == b]].sum(0))
        r["domain_pooled"] = D.entropy_bits(pooled)
        res["layers"][str(L)] = r
    keys = list(res["layers"]["0"])
    res["summary"] = {k: {"mean": float(np.mean([res["layers"][str(L)][k] for L in range(D.N_LAYERS)])),
                          "min": float(np.min([res["layers"][str(L)][k] for L in range(D.N_LAYERS)])),
                          "max": float(np.max([res["layers"][str(L)][k] for L in range(D.N_LAYERS)])),
                          "layers_above_ref": int(sum(res["layers"][str(L)][k] >= D.ENTROPY_REF
                                                      for L in range(D.N_LAYERS)))} for k in keys}
    s = res["summary"]["domain_pooled"]
    res["near_flat"] = bool(s["layers_above_ref"] > D.N_LAYERS // 2)
    res["rule"] = ("near_flat = more than half the layers have pooled domain selection entropy >= 0.85*log2(384); "
                   "if so, the fair-allocation candidate is added (D2_METHOD item 6)")
    env.atomic_json(a.out / "diag" / "entropy.json", res)
    env.log(f"entropy: domain pooled mean {s['mean']:.3f} bits (min {s['min']:.3f}, max {s['max']:.3f}, "
            f"{s['layers_above_ref']}/{D.N_LAYERS} layers >= {D.ENTROPY_REF:.2f}); "
            f"A192 mean {res['summary']['A192']['mean']:.3f}; near_flat={res['near_flat']}", STAGE)
    return res


# ----------------------------------------------------------------------------- layers
def _grid(a):
    for crit in a.crit:
        for w in a.w:
            for pf in a.pf:
                for st in a.st:
                    yield crit, w, pf, st


_G = {}


def _job_layer(job):
    L, buckets, avail, opts = job          # buckets = the ones to (re)compute; avail = all (stability pool)
    out = Path(opts["out"])
    sup, protect = _G["sup"], _G["protect"]
    t0 = time.time()
    li = D.LayerInputs(L, avail, D.guards(L, sup, protect), buckets=list(avail), halves=opts["stability"])
    a192 = keepset.load(D.S04 / "keepset.json")[0][str(L)]
    for b in buckets:
        f = out / "_layers" / f"L{L}" / f"{b}.json"
        res = {"L": L, "bucket": b, "guard": li.guard.tolist(), "cands": {}, "ret": {}, "stab": {}}
        if f.exists() and not opts["force"]:          # extend an existing unit with the missing grid points only
            old = json.load(open(f))
            if old["guard"] == res["guard"]:
                res = old
                if opts["stability"] and res.get("stab", {}).get("pool_buckets") != sorted(set(avail) - {b}):
                    res["stab"] = {}                          # stability pool changed (new pass): recompute
        res["ret"]["A192"] = li.retained(a192, [b])
        doms = {st: li.domain(b, st) for st in (0, 1)}
        mixes = {}
        for crit, w, pf, st in opts["grid"]:
            name = D.cand_name(crit, w, pf, st)
            if name in res["cands"]:
                continue
            if (w, st) not in mixes:
                mixes[(w, st)] = li.mixed(b, w, st, dom=doms[st])
            keep = li.solve(crit, mixes[(w, st)], li.gen_slice(pf))
            res["cands"][name] = keep
            res["ret"][name] = li.retained(keep, [b])
        dn = D.cand_name("hope", D.DIAG_W, D.DEFAULT["pf"], 1, "_diag")
        if b in D.DIAG_BUCKETS and dn not in res["cands"]:
            res["cands"][dn] = li.solve("hope", li.mixed(b, D.DIAG_W, 1, dom=doms[1], diag=True), li.gen_slice(D.DEFAULT["pf"]))
            res["ret"][dn] = li.retained(res["cands"][dn], [b])
        if opts["fair"]:
            for w in opts["w"]:
                name = D.cand_name("fair", w, D.DEFAULT["pf"], 1)
                res["cands"][name] = li.fair(b, w, 1, D.DEFAULT["pf"])
                res["ret"][name] = li.retained(res["cands"][name], [b])
        if opts["stability"] and not res.get("stab"):
            res["stab"] = _stability(li, b, avail, buckets)
        env.atomic_json(f, res)
    return L, buckets, time.time() - t0


def _halves(li: D.LayerInputs, b: str) -> tuple[dict, list]:
    topical = [n for n in li.sub[b] if n not in D.SHARED_SA]
    th = {n: ("A" if i % 2 == 0 else "B") for i, n in enumerate(topical)}
    return th, topical


def _stability(li: D.LayerInputs, b: str, avail, buckets) -> dict:
    th, _ = _halves(li, b)
    hA, hB = li.domain(b, 1, "A", th), li.domain(b, 1, "B", th)
    st = {"topical_half": th, "shared_split": "row parity (even rows A, odd rows B) for " + ", ".join(D.SHARED_SA)}
    kA = li.solve("hope", {"F": D.unit(hA["F"])}, np.zeros(0, int))
    kB = li.solve("hope", {"F": D.unit(hB["F"])}, np.zeros(0, int))
    st["domain_only"] = D.jaccard(kA, kB)
    st["domain_only_nonguard"] = _jac_ng(kA, kB, li.guard)
    w, pf = D.DEFAULT["w"], D.DEFAULT["pf"]
    # the default mix needs the half's REAP/mass too only for non-HOPE criteria; HOPE uses F alone
    mA = li.mixed(b, w, 1, dom={"reap": np.ones(D.E), "mass": np.ones(D.E), "F": hA["F"]})
    mB = li.mixed(b, w, 1, dom={"reap": np.ones(D.E), "mass": np.ones(D.E), "F": hB["F"]})
    kmA, kmB = li.solve("hope", mA, li.gen_slice(pf)), li.solve("hope", mB, li.gen_slice(pf))
    st["default_mix"] = D.jaccard(kmA, kmB)
    st["default_mix_nonguard"] = _jac_ng(kmA, kmB, np.union1d(li.guard, li.gen_slice(pf)))
    others = [o for o in li.sub if o != b]
    if others:
        pool = sum(D.unit(li.domain(o, 1)["F"]) for o in others)
        st["shrink"] = {}
        for lam in D.SHRINK_GRID:
            sA = (1 - lam) * D.unit(hA["F"]) + lam * D.unit(pool)
            sB = (1 - lam) * D.unit(hB["F"]) + lam * D.unit(pool)
            ka, kb = li.solve("hope", {"F": sA}, np.zeros(0, int)), li.solve("hope", {"F": sB}, np.zeros(0, int))
            st["shrink"][str(lam)] = {"jaccard": D.jaccard(ka, kb), "nonguard": _jac_ng(ka, kb, li.guard)}
        st["shrink_target"] = f"sum of normalised full F of the other available domains: {others}"
    st["pool_buckets"] = sorted(others)
    return st


def _jac_ng(a, b, g) -> float:
    g = set(map(int, g))
    return D.jaccard(set(a) - g, set(b) - g)


def _init(sup, protect):
    _G["sup"], _G["protect"] = sup, protect


def layers(a):
    avail = _avail(a)
    if not avail:
        raise SystemExit("no tier-1 bucket has a completed pass bank (use --allow-incomplete only for testing)")
    sup, protect, ginfo = _guards_cached(a.out, avail)
    ent = a.out / "diag" / "entropy.json"
    fair = bool(json.load(open(ent))["near_flat"]) if ent.exists() else False
    opts = {"out": str(a.out), "grid": list(_grid(a)), "fair": fair, "w": list(a.w), "stability": not a.no_stability,
            "force": a.force}
    want = {D.cand_name(*g) for g in opts["grid"]} | ({D.cand_name("fair", w, D.DEFAULT["pf"], 1) for w in a.w} if fair else set())

    def missing(L, b):
        f = a.out / "_layers" / f"L{L}" / f"{b}.json"
        if a.force or not f.exists():
            return True
        r = json.load(open(f))
        if r["guard"] != D.guards(L, sup, protect).tolist():      # a new pass grew the super-expert union: redo
            return True
        stab_stale = opts["stability"] and (r.get("stab") or {}).get("pool_buckets") != sorted(set(avail) - {b})
        diag = {D.cand_name("hope", D.DIAG_W, D.DEFAULT["pf"], 1, "_diag")} if b in D.DIAG_BUCKETS else set()
        return bool((want | diag) - set(r["cands"])) or stab_stale

    jobs = []
    for L in a.layer_list:
        todo = [b for b in avail if missing(L, b)]
        if todo:
            jobs.append((L, todo, avail, opts))
    env.log(f"layers: {len(jobs)} layer jobs, buckets {list(avail)}, {len(opts['grid'])} grid points/bucket, "
            f"fair={fair}, guards {ginfo}", STAGE)
    if not jobs:
        return
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(sup, protect), maxtasksperchild=4) as pool:
        for L, bs, dt in pool.imap_unordered(_job_layer, jobs):
            env.log(f"L{L} done ({','.join(bs)}) in {dt:.0f}s", STAGE)


# ----------------------------------------------------------------------------- shrink candidates
def shrink(a):
    """If a bucket's mean domain-only half-Jaccard < 0.9 (item 5), pick the smallest lambda in SHRINK_GRID whose
    mean half-Jaccard reaches 0.9 (else the largest) and add hope_w*_pf<default>_st1_shr<lam> candidates, the
    domain F shrunk toward the pooled all-domain F (all available tier-1 domains, normalised, the bucket included)."""
    avail = _avail(a)
    stab = _collect_stability(a.out, list(avail))
    need = {b: s["lambda"] for b, s in stab.items() if s.get("lambda") is not None}
    env.log(f"shrink: buckets needing shrinkage {need or 'none'}", STAGE)
    if not need:
        return
    sup, protect, _ = _guards_cached(a.out, avail)
    def stale(L):
        for b, lam in need.items():
            f = a.out / "_layers" / f"L{L}" / f"{b}.shrink.json"
            if a.force or not f.exists():
                return True
            r = json.load(open(f))
            if r["lambda"] != lam or r.get("pool") != sorted(avail):
                return True
        return False

    jobs = [(L, need, avail, str(a.out), list(a.w)) for L in a.layer_list if stale(L)]
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(sup, protect), maxtasksperchild=4) as pool:
        for L, dt in pool.imap_unordered(_job_shrink, jobs):
            env.log(f"shrink L{L} done in {dt:.0f}s", STAGE)


def _job_shrink(job):
    L, need, avail, out, ws = job
    t0 = time.time()
    li = D.LayerInputs(L, avail, D.guards(L, _G["sup"], _G["protect"]), buckets=list(avail), halves=False)
    pool = sum(D.unit(li.domain(o, 1)["F"]) for o in li.sub)
    a192 = keepset.load(D.S04 / "keepset.json")[0][str(L)]
    for b, lam in need.items():
        res = {"L": L, "bucket": b, "lambda": lam, "pool": sorted(avail), "cands": {}, "ret": {"A192": li.retained(a192, [b])}}
        dom = li.domain(b, 1)
        dom_s = {**dom, "F": (1 - lam) * D.unit(dom["F"]) + lam * D.unit(pool)}
        for w in ws:
            name = D.cand_name("hope", w, D.DEFAULT["pf"], 1, f"_shr{round(lam * 100)}")
            keep = li.solve("hope", li.mixed(b, w, 1, dom=dom_s), li.gen_slice(D.DEFAULT["pf"]))
            res["cands"][name] = keep
            res["ret"][name] = li.retained(keep, [b])
        env.atomic_json(Path(out) / "_layers" / f"L{L}" / f"{b}.shrink.json", res)
    return L, time.time() - t0


def _collect_stability(out: Path, buckets) -> dict:
    res = {}
    for b in buckets:
        per = {}
        for L in range(D.N_LAYERS):
            f = out / "_layers" / f"L{L}" / f"{b}.json"
            if f.exists():
                s = json.load(open(f)).get("stab")
                if s:
                    per[L] = s
        if len(per) < D.N_LAYERS:
            continue
        m = lambda key: float(np.mean([per[L][key] for L in per]))
        r = {"domain_only_mean": m("domain_only"), "domain_only_min": float(min(per[L]["domain_only"] for L in per)),
             "domain_only_nonguard_mean": m("domain_only_nonguard"), "default_mix_mean": m("default_mix"),
             "default_mix_nonguard_mean": m("default_mix_nonguard"),
             "per_layer_domain_only": [per[L]["domain_only"] for L in sorted(per)],
             "per_layer_default_mix": [per[L]["default_mix"] for L in sorted(per)],
             "topical_half": per[0]["topical_half"], "shared_split": per[0]["shared_split"]}
        if "shrink" in per[0]:
            r["shrink"] = {lam: {"mean": float(np.mean([per[L]["shrink"][lam]["jaccard"] for L in per])),
                                 "nonguard_mean": float(np.mean([per[L]["shrink"][lam]["nonguard"] for L in per]))}
                           for lam in per[0]["shrink"]}
            r["shrink_target"] = per[0]["shrink_target"]
        r["below_floor"] = r["domain_only_mean"] < D.STABILITY_FLOOR
        r["lambda"] = None
        if r["below_floor"] and "shrink" in r:
            ok = [float(l) for l, v in r["shrink"].items() if v["mean"] >= D.STABILITY_FLOOR]
            r["lambda"] = min(ok) if ok else max(float(l) for l in r["shrink"])
        res[b] = r
    return res


# ----------------------------------------------------------------------------- polymath
def polymath(a):
    avail = _avail(a)
    sup, protect, _ = _guards_cached(a.out, avail)
    def stale(L):
        f = a.out / "_layers" / f"L{L}" / "polymath.json"
        if a.force or not f.exists():
            return True
        r = json.load(open(f))
        return (r["domains"] != list(avail) or r["guard"] != D.guards(L, sup, protect).tolist()
                or "mix_control" not in r["cands"])

    jobs = [(L, avail, str(a.out)) for L in a.layer_list if stale(L)]
    if not jobs:
        return
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(sup, protect), maxtasksperchild=4) as pool:
        for L, dt in pool.imap_unordered(_job_poly, jobs):
            env.log(f"polymath L{L} done in {dt:.0f}s", STAGE)


def _job_poly(job):
    L, avail, out = job
    t0 = time.time()
    bs = list(avail)
    li = D.LayerInputs(L, avail, D.guards(L, _G["sup"], _G["protect"]), buckets=bs, halves=False)
    doms = {b: li.domain(b, 1) for b in bs}
    Fsum = sum(D.unit(d["F"]) for d in doms.values())
    res = {"L": L, "domains": bs, "guard": li.guard.tolist(), "cands": {}, "rr": {}, "ret": {}}
    res["cands"]["control"] = li.solve("hope", {"F": Fsum}, np.zeros(0, int))
    ranks = {b: [int(e) for e in np.argsort(-D.unit(doms[b]["reap"]), kind="stable")] for b in bs}
    for k in D.POLY_K:
        rr = poly_round_robin(ranks, k, set(li.guard.tolist()))
        res["rr"][f"rr{k}"] = sorted(rr)
        res["cands"][f"rr{k}"] = li.solve("hope", {"F": Fsum}, np.array(sorted(rr), int))
    # mix_*: the domain masks' recipe with the summed domains as DOMAIN (default w, general slice protected).
    # The pure candidates above have no general/code/agentic share and drop general mass ~0.15 vs A192.
    pdom = {k: sum(D.unit(d[k]) for d in doms.values()) for k in ("reap", "mass", "F")}
    pmix = li.mixed(bs[0], D.DEFAULT["w"], 1, dom=pdom)
    gs = li.gen_slice(D.DEFAULT["pf"])
    res["cands"]["mix_control"] = li.solve("hope", pmix, gs)
    for k in D.POLY_K:
        res["cands"][f"mix_rr{k}"] = li.solve("hope", pmix, np.union1d(gs, res["rr"][f"rr{k}"]).astype(int))
    a192 = keepset.load(D.S04 / "keepset.json")[0][str(L)]
    res["ret"]["A192"] = li.retained(a192, bs)
    for n, keep in res["cands"].items():
        res["ret"][n] = li.retained(keep, bs)
    env.atomic_json(Path(out) / "_layers" / f"L{L}" / "polymath.json", res)
    return L, time.time() - t0


def poly_round_robin(ranks: dict[str, list[int]], k: int, guard: set) -> set:
    """Each domain's top-k by its own normalised REAP saliency, interleaved by rank (rank 0 of every domain,
    then rank 1, ...). An expert already kept (guard or another domain) counts toward the domain's k, so the
    set is exactly the union of the per-domain top-k: overlaps are shared, not re-filled."""
    out = set()
    for r in range(k):
        for b in ranks:
            e = ranks[b][r]
            if e not in guard:
                out.add(e)
    return out


# ----------------------------------------------------------------------------- assemble + diagnostics
def assemble(a):
    avail = _avail(a)
    bs = list(avail)
    out = a.out
    man = {"ts": env.now(), "inputs": _inputs(avail), "default": D.DEFAULT_NAME,
           "grid": {"criteria": list(a.crit), "w": list(a.w), "pf": list(a.pf), "st": list(a.st)},
           "mix": {str(w): D.mix_weights(w) for w in a.w}, "candidates": []}
    guards_info = json.load(open(out / "diag" / "guards.json"))["info"] if (out / "diag" / "guards.json").exists() else {}
    man["guards"] = guards_info
    ret_all, keeps = {}, {}
    for b in bs:
        per = [json.load(open(out / "_layers" / f"L{L}" / f"{b}.json")) for L in range(D.N_LAYERS)]
        sh = [out / "_layers" / f"L{L}" / f"{b}.shrink.json" for L in range(D.N_LAYERS)]
        if all(f.exists() for f in sh):
            sp = [json.load(open(f)) for f in sh]
            for L in range(D.N_LAYERS):
                per[L]["cands"].update(sp[L]["cands"]); per[L]["ret"].update(sp[L]["ret"])
        names = list(per[0]["cands"])
        keeps[b] = {}
        for n in names:
            keep = {str(L): per[L]["cands"][n] for L in range(D.N_LAYERS)}
            ret = _mean_ret([per[L]["ret"][n] for L in range(D.N_LAYERS)])
            meta = _parse(n)
            fp = keepset.save(out / b / f"{n}.json", keep, bucket=b, candidate=n, **meta,
                              mix=D.mix_weights(meta["w"], meta.get("diagnostic", False)) if meta.get("w") else None,
                              default=(n == D.DEFAULT_NAME), retained_mean_over_layers=ret, guards=guards_info,
                              inputs=man["inputs"]["passes"].get(avail[b]["pass"]))
            keeps[b][n] = keep
            man["candidates"].append({"bucket": b, "name": n, "file": f"{b}/{n}.json", "sha256": fp, **meta})
        ret_all[b] = {n: _mean_ret([per[L]["ret"][n] for L in range(D.N_LAYERS)]) for n in names + ["A192"]}
    poly = [out / "_layers" / f"L{L}" / "polymath.json" for L in range(D.N_LAYERS)]
    if all(f.exists() for f in poly):
        pp = [json.load(open(f)) for f in poly]
        for n in pp[0]["cands"]:
            keep = {str(L): pp[L]["cands"][n] for L in range(D.N_LAYERS)}
            fp = keepset.save(out / "polymath" / f"{n}.json", keep, bucket="polymath", candidate=n, domains=pp[0]["domains"],
                              rr_k=int(n[2:]) if n.startswith("rr") else 0,
                              rr_size_per_layer=[len(pp[L]["rr"].get(n, [])) for L in range(D.N_LAYERS)],
                              retained_mean_over_layers=_mean_ret([pp[L]["ret"][n] for L in range(D.N_LAYERS)]))
            man["candidates"].append({"bucket": "polymath", "name": n, "file": f"polymath/{n}.json", "sha256": fp,
                                      "domains": pp[0]["domains"]})
        ret_all["polymath"] = {n: _mean_ret([pp[L]["ret"][n] for L in range(D.N_LAYERS)]) for n in list(pp[0]["cands"]) + ["A192"]}
        man["polymath_rr_size"] = {n: [len(pp[L]["rr"][n]) for L in range(D.N_LAYERS)] for n in pp[0]["rr"]}
    listed = {out / c["file"] for c in man["candidates"]}
    for d in [out / b for b in bs] + [out / "polymath"]:       # stale names (e.g. an old shrink lambda) leave d2/
        for f in d.glob("*.json") if d.is_dir() else []:
            if f not in listed:
                f.unlink()
    env.atomic_json(out / "manifest.json", man)
    env.atomic_json(out / "diag" / "retained_mass.json", _ret_diag(ret_all))
    env.atomic_json(out / "diag" / "selftrace_bias.json", _st_bias(keeps))
    env.atomic_json(out / "diag" / "overlap.json", _overlap(keeps, out))
    env.atomic_json(out / "diag" / "stability.json", _collect_stability(out, bs))
    env.log(f"assembled {len(man['candidates'])} candidates for {bs}" + (" + polymath" if "polymath" in ret_all else ""),
            STAGE)


def _inputs(avail) -> dict:
    p = {}
    for b, i in avail.items():
        p.setdefault(i["pass"], {"bank": str(i["bank"]), "buckets": i["buckets"],
                                 "bank_mtime": max(os.path.getmtime(i["bank"] / f"L{L}.npz") for L in range(D.N_LAYERS))})
    return {"passes": p, "a192_bank": str(D.A192_BANK / "bank"), "a192_keepset": str(D.S04 / "keepset.json"),
            "a192_keepset_sha": json.load(open(D.S04 / "keepset.json"))["sha256"]}


def _parse(n: str) -> dict:
    p = n.split("_")
    out = {"criterion": p[0], "w": int(p[1][1:]) / 100, "pf": int(p[2][2:]) / 100, "selftraces": int(p[3][2:])}
    if len(p) > 4 and p[4] == "diag":
        out["diagnostic"] = True                  # never shipped; replay_filter.decide excludes it from the gate
    elif len(p) > 4:
        out["shrink_lambda"] = int(p[4][3:]) / 100
    return out


def _mean_ret(rs: list[dict]) -> dict:
    out = {}
    for part in ("cal", "ho"):
        out[part] = {}
        for k in rs[0][part]:
            v = [r[part][k] for r in rs if r[part][k] is not None]
            out[part][k] = float(np.mean(v)) if v else None
    return out


def _ret_diag(ret_all: dict) -> dict:
    out = {"note": "fraction of each sub-area's saliency mass kept, mean over 40 layers. cal = banked REAP sum on "
                   "calibration rows (exact). ho = held-out rows, sum over routed slots of g * mean||f|| "
                   "(approximation: the bank excludes held-out rows; routing covers them). A192/<bucket> = the "
                   "original 9-bucket corpus; 'A192' entry = s04/keepset.json for reference.", "buckets": {}}
    for b, r in ret_all.items():
        a = r["A192"]
        rows = {}
        for n, v in r.items():
            rows[n] = {"cal": v["cal"], "ho": v["ho"],
                       "general_cal_minus_A192": v["cal"]["A192/general"] - a["cal"]["A192/general"],
                       "general_ho_minus_A192": (v["ho"]["A192/general"] - a["ho"]["A192/general"])
                       if v["ho"]["A192/general"] is not None else None}
            dk = [k for k in v["cal"] if not k.startswith("A192/")]
            rows[n]["domain_cal_mean"] = float(np.mean([v["cal"][k] for k in dk])) if dk else None
            rows[n]["domain_cal_worst"] = min(((v["cal"][k], k) for k in dk), default=(None, None))
            hk = [k for k in dk if v["ho"][k] is not None]
            rows[n]["domain_ho_mean"] = float(np.mean([v["ho"][k] for k in hk])) if hk else None
        out["buckets"][b] = rows
    return out


def _st_bias(keeps: dict) -> dict:
    out = {}
    for b, ks in keeps.items():
        pairs = {}
        for n in ks:
            if n.endswith("_st1") and n[:-1] + "0" in ks:
                j = [D.jaccard(ks[n][str(L)], ks[n[:-1] + "0"][str(L)]) for L in range(D.N_LAYERS)]
                pairs[n[:-4]] = {"mean": float(np.mean(j)), "min": float(np.min(j)), "per_layer": j,
                                 "changed_experts_per_layer_mean": float(np.mean(
                                     [len(set(ks[n][str(L)]) - set(ks[n[:-1] + "0"][str(L)])) for L in range(D.N_LAYERS)]))}
        d = D.DEFAULT_NAME[:-4]
        out[b] = {"default": pairs.get(d), "all": {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"}
                                                   for k, v in pairs.items()}}
    return out


def _overlap(keeps: dict, out: Path) -> dict:
    bs = [b for b in keeps if D.DEFAULT_NAME in keeps[b]]
    res = {"buckets": bs, "layers": {}}
    for L in range(D.N_LAYERS):
        guard = set(json.load(open(out / "_layers" / f"L{L}" / f"{bs[0]}.json"))["guard"]) if bs else set()
        sets = [set(keeps[b][D.DEFAULT_NAME][str(L)]) for b in bs]
        u, i = set().union(*sets), set.intersection(*sets)
        cnt = np.zeros(D.E, int)
        for s in sets:
            cnt[list(s)] += 1
        res["layers"][str(L)] = {"union": len(u), "intersection": len(i), "conflict": len(u) - len(i),
                                 "guard": len(guard), "union_nonguard": len(u - guard),
                                 "kept_by_exactly_one": int((cnt == 1).sum()), "kept_by_all": int((cnt == len(bs)).sum())}
    pj = {}
    for x in bs:
        for y in bs:
            if x < y:
                pj[f"{x}|{y}"] = float(np.mean([D.jaccard(keeps[x][D.DEFAULT_NAME][str(L)], keeps[y][D.DEFAULT_NAME][str(L)])
                                                for L in range(D.N_LAYERS)]))
    res["pairwise_jaccard_mean"] = pj
    for k in ("union", "intersection", "conflict", "kept_by_exactly_one"):
        res[f"{k}_mean"] = float(np.mean([res["layers"][str(L)][k] for L in range(D.N_LAYERS)])) if bs else None
    res["definition"] = ("per layer over the domain DEFAULT candidates: union = experts kept by any domain; "
                         "intersection = kept by all; conflict = union - intersection (kept by some domain, dropped "
                         "by another); guards count in all of them (listed separately)")
    return res


def report(a):
    """docs/domain/D2_DIAG.md from diag/*.json (headline numbers; the JSON files hold everything)."""
    dg = a.out / "diag"
    J = lambda n: json.load(open(dg / f"{n}.json")) if (dg / f"{n}.json").exists() else None
    man, ent, val, stab, stb, ov, ret, gu = (J("../manifest") if (a.out / "manifest.json").exists() else None,
                                             J("entropy"), J("validate"), J("stability"), J("selftrace_bias"),
                                             J("overlap"), J("retained_mass"), J("guards"))
    L = ["# D2 diagnostics (generated by `ops/dom/select_dom.py report`)", "",
         f"Generated {env.now()} from `{a.out}`. Buckets: {sorted(stab or {})}. "
         f"Candidates: {len(man['candidates']) if man else 0}. Default: `{D.DEFAULT_NAME}`. "
         "Per-layer numbers are in `$V41_WORK/dom/d2/diag/*.json`; this page is the summary.", ""]
    if gu:
        L += ["## Guards (item 1)", "", f"{gu['info']}: super-expert candidates are s04's set union "
              "`drift.super_expert_candidates` over max(A192 fmax, pass fmax); position-protect from "
              "`s04/position_protect.json`. Hard keep in every candidate (asserted per layer).", ""]
    if val:
        L += ["## F reconstruction (per-source F rebuilt from routing)", "",
              "| layer | source | gate-mass map err | rel Frobenius | off-diag corr | HOPE keep Jaccard (banked vs rebuilt) |",
              "|---|---|---|---|---|---|"]
        for l, r in val["layers"].items():
            for k in [x[:-2] for x in r if x.endswith("_F")]:
                f = r[f"{k}_F"]
                L.append(f"| {l} | {k} | {r[f'{k}_gat_relerr']:.1e} | {f['rel_frob']:.4f} | {f['corr_offdiag']:.3f} | "
                         f"{f['hope_keep_jaccard']:.3f} |")
        L += ["", "Bank format: " + "; ".join(f"{x['pass']}: 18 buckets {x['buckets_18']}, 177 sub-area rows "
                                               f"{x['subareas_177']}, {x['live_subareas']} live, live outside pass "
                                               f"{x['live_outside_pass'] or 'none'}" for x in val["bank_format"]), ""]
    if ent:
        s = ent["summary"]
        L += ["## Router entropy (item 6)", "", f"Selection distribution (banked `cnt`, biased top-6) per layer, "
              f"bits; reference 0.85 x log2 384 = {ent['ref_bits']:.2f}, max {ent['max_bits']:.2f}.", "",
              "| source | mean | min | max | layers >= 7.30 |", "|---|---|---|---|---|"]
        for k in ["domain_pooled", "A192"] + [b for b in D.TIER1 if b in s] + [f"A192/{b}" for b in D.ORIG9]:
            if k in s:
                L.append(f"| {k} | {s[k]['mean']:.3f} | {s[k]['min']:.3f} | {s[k]['max']:.3f} | {s[k]['layers_above_ref']} |")
        L += ["", f"near_flat = {ent['near_flat']} ({ent['rule']}).", ""]
    if stab:
        L += ["## HOPE stability (item 5)", "", "Two disjoint halves per bucket: topical sub-areas alternate A/B; "
              "selftraces, tool-use and agentic-chat are split by row parity so both halves carry them. "
              "Keep-set Jaccard per layer (192 kept; guards included, `nonguard` excludes them).", "",
              "| bucket | domain-only mean (min) | non-guard | default mix | shrink 0.25 / 0.5 / 0.75 | lambda chosen |",
              "|---|---|---|---|---|---|"]
        for b, r in stab.items():
            sh = " / ".join(f"{r['shrink'][k]['mean']:.3f}" for k in sorted(r.get("shrink", {}), key=float)) or "-"
            L.append(f"| {b} | {r['domain_only_mean']:.3f} ({r['domain_only_min']:.3f}) | "
                     f"{r['domain_only_nonguard_mean']:.3f} | {r['default_mix_mean']:.3f} | {sh} | {r['lambda']} |")
        L += ["", "Shrink target in the stability test: the other available domains' normalised F (so the halves "
              "share no data of their own bucket); the shrink CANDIDATES use the all-domain pool including the bucket. "
              "Half-data Jaccard understates full-data stability.", ""]
    if stb:
        L += ["## Self-trace bias (item 3)", "", "Jaccard of selftraces-off vs on, per layer, mean (min); "
              "`changed` = experts per layer that differ.", "", "| bucket | default | changed/layer | all 27 pairs: mean range |",
              "|---|---|---|---|"]
        for b, r in stb.items():
            d = r["default"]
            al = [v["mean"] for v in r["all"].values()]
            L.append(f"| {b} | {d['mean']:.3f} ({d['min']:.3f}) | {d['changed_experts_per_layer_mean']:.1f} | "
                     f"{min(al):.3f}-{max(al):.3f} |")
        L.append("")
    if ov and ov.get("buckets"):
        L += ["## Domain overlap (item 10)", "", f"Default candidates of {ov['buckets']}: per-layer union mean "
              f"{ov['union_mean']:.1f}, intersection {ov['intersection_mean']:.1f}, conflict (union - intersection) "
              f"{ov['conflict_mean']:.1f}, kept by exactly one domain {ov['kept_by_exactly_one_mean']:.1f}.", "",
              "Pairwise mean Jaccard: " + ", ".join(f"{k} {v:.3f}" for k, v in ov["pairwise_jaccard_mean"].items()), ""]
    if ret:
        L += ["## Retained mass", "", "Mean over layers of the fraction of saliency mass kept. cal = banked calibration "
              "rows (exact); ho = held-out rows via routing x mean ||f|| (approximation). A192 row = s04/keepset.json.", "",
              "| bucket | candidate | domain cal mean | worst sub-area (cal) | domain ho mean | general cal (vs A192) | general ho (vs A192) |",
              "|---|---|---|---|---|---|---|"]
        for b, rows in ret["buckets"].items():
            show = [n for n in ("A192", D.DEFAULT_NAME, "hope_w45_pf4_st0", "mass_w45_pf4_st1", "reap_w45_pf4_st1",
                                "hope_w60_pf4_st1", "hope_w35_pf4_st1", "control", "rr4", "rr6",
                                "mix_control", "mix_rr4", "mix_rr6", "hope_w80_pf4_st1_diag") if n in rows]
            show += [n for n in rows if "_shr" in n]
            for n in show:
                r = rows[n]
                wc = r["domain_cal_worst"]
                gh = r["ho"]["A192/general"]
                L.append(f"| {b} | {n} | {_f(r['domain_cal_mean'])} | {_f(wc[0])} {wc[1] or ''} | {_f(r['domain_ho_mean'])} | "
                         f"{r['cal']['A192/general']:.4f} ({r['general_cal_minus_A192']:+.4f}) | "
                         f"{_f(gh)} ({_f(r['general_ho_minus_A192'], '+.4f')}) |")
        L.append("")
    if man and man.get("polymath_rr_size"):
        L += ["## Polymath", "", "rr-k = union of each domain's top-k (own normalised REAP) after the guards, then HOPE "
              "fill on the sum of normalised domain F; control = that HOPE with guards only. Round-robin set size "
              "per layer: " + ", ".join(f"{k} mean {np.mean(v):.1f} (max {max(v)})" for k, v in man["polymath_rr_size"].items()),
              ""]
    dst = a.doc or (env.REPO / "docs" / "domain" / "D2_DIAG.md" if a.out.resolve() == OUT.resolve() else a.out / "D2_DIAG.md")
    dst.write_text("\n".join(L) + "\n")
    env.log(f"wrote {dst} ({len(L)} lines)", STAGE)


def _f(x, fmt=".4f"):
    return "-" if x is None else format(x, fmt)


def all_(a):
    entropy(a)
    validate(a)
    layers(a)
    shrink(a)
    polymath(a)
    assemble(a)
    report(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "validate", "entropy", "layers", "shrink", "polymath", "assemble", "report", "all"])
    ap.add_argument("--buckets", nargs="*", help="default: every tier-1 bucket with a completed pass")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--doc", type=Path, help="report: where to write D2_DIAG.md (default docs/domain/ for the default --out)")
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 20, 39], help="validate only")
    ap.add_argument("--only-layers", type=int, nargs="*", help="layers/shrink/polymath: restrict (testing)")
    ap.add_argument("--crit", nargs="*", default=list(D.CRITERIA))
    ap.add_argument("--w", type=float, nargs="*", default=list(D.W_GRID))
    ap.add_argument("--pf", type=float, nargs="*", default=list(D.PF_GRID))
    ap.add_argument("--st", type=int, nargs="*", default=list(D.ST_GRID))
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute per-layer units that exist")
    ap.add_argument("--allow-incomplete", action="store_true", help="TESTING ONLY: use pass banks whose s03 is unfinished")
    a = ap.parse_args()
    assert all(w <= 0.6 + 1e-9 for w in a.w), "D2_METHOD item 4: domain weight never above 0.6"
    a.layer_list = a.only_layers if a.only_layers is not None else list(range(D.N_LAYERS))
    a.out.mkdir(parents=True, exist_ok=True)
    {"status": status, "validate": validate, "entropy": entropy, "layers": layers, "shrink": shrink,
     "polymath": polymath, "assemble": assemble, "report": report, "all": all_}[a.cmd](a)


if __name__ == "__main__":
    main()
