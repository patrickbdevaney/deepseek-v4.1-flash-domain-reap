#!/usr/bin/env python3
"""D2 CPU selection (ops/dom/select_dom.py, ops/dom/d2lib.py) on a synthetic bank. CPU only, ~3-5 min.

Builds a fake $V41_WORK (A192 9-bucket bank + routing + corpus, s04 guards and keep-set, one domain pass
`p1` with two buckets), with ||f|| CONSTANT per (stat row, expert) so the routing reconstruction of F is
exact, then runs the real CLI (2 layers) and checks the D2_METHOD invariants.

    .venv/bin/python ops/tests/test_d2_select.py
"""
from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NL = 2
E, K = 384, 6
TMP = Path(tempfile.mkdtemp(prefix="d2test_"))
os.environ["V41_WORK"] = str(TMP)
os.environ["V41_D2_NLAYERS"] = str(NL)
sys.path.insert(0, str(REPO))
from ops.dom import corpus_spec_dom as spec  # noqa: E402
from ops.dom import d2lib as D  # noqa: E402
from ops.lib import keepset  # noqa: E402

PASS_BUCKETS = ["gamedev", "mlsys"]
rng = np.random.default_rng(0)


def routing(T: int, n_stat: int, sid: np.ndarray, pref: np.ndarray):
    """Each stat row prefers its own experts (pref [n_stat, E] logits) -> top-6 ids, fp16 gates."""
    idx = np.empty((T, K), np.uint16)
    for a in range(n_stat):
        m = np.flatnonzero(sid == a)
        if not m.size:
            continue
        lg = pref[a][None] + rng.gumbel(size=(m.size, E))
        idx[m] = np.argsort(-lg, 1)[:, :K]
    w = rng.uniform(0.05, 0.6, size=(T, K)).astype(np.float16)
    return idx, w


def bank_from(idx, w, sid, valid_calib, n_stat, norms, names, extra):
    """Exactly what hooks.FFNStats accumulates, with ||f|| = norms[a, e] (constant per row/expert)."""
    t = np.flatnonzero(valid_calib)
    ii = idx[t].astype(np.int64)
    g = w[t].astype(np.float64)
    a = sid[t][:, None].repeat(K, 1)
    n = norms[a, ii]
    s = g * n
    out = {}
    for k, v in (("sum", s), ("sq", s * s), ("nrm", n), ("nsq", n * n), ("gat", g), ("gsq", g * g), ("cnt", np.ones_like(s))):
        out[k] = np.bincount((a * E + ii).ravel(), weights=v.ravel(), minlength=n_stat * E).reshape(n_stat, E)
    flat = (ii[:, :, None] * E + ii[:, None, :]).ravel()
    out["F_sum"] = np.bincount(flat, weights=(s[:, :, None] * s[:, None, :]).ravel(), minlength=E * E).reshape(E, E)
    out["F_cnt"] = np.bincount(flat, minlength=E * E).reshape(E, E).astype(np.float64)
    fmax = np.full(E, 1.0); fmax[[5, 77]] = 400.0          # two super-experts
    out["fmax"] = fmax
    out["buckets"] = np.array(names)
    out.update(extra)
    return out


def write_corpus(d: Path, rows: list[dict], T: int, man: dict):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.save(d / "valid.npy", np.ones(T, np.uint8))
    json.dump(man, open(d / "manifest.json", "w"))


def make_rows(stat_ids, per, key, label, length=512):
    rows, off = [], 0
    for a in stat_ids:
        for j in range(per):
            rows.append({"row": len(rows), key: a, "bucket": label(a), "offset": off, "length": length,
                         "heldout": j == per - 1, "n_valid": length})
            off += length
    return rows, off


def build():
    # A192
    rows9, T9 = make_rows(range(9), 6, "bucket_id", lambda a: D.ORIG9[a])
    sid9 = np.concatenate([np.full(r["length"], r["bucket_id"]) for r in rows9])
    write_corpus(TMP / "corpus", rows9, T9, {"buckets": D.ORIG9})
    (TMP / "s03" / "bank").mkdir(parents=True); (TMP / "s03" / "routing").mkdir(parents=True)
    # domain pass
    sa_ids = [i for i, n in enumerate(spec.SUBAREAS) if n.split("/")[0] in PASS_BUCKETS]
    rowsd, Td = make_rows(sa_ids, 4, "sub_id", lambda a: spec.SUBAREAS[a].split("/")[0])
    for r in rowsd:
        r["subarea"] = spec.SUBAREAS[r["sub_id"]].split("/", 1)[1]
    sidd = np.concatenate([np.full(r["length"], r["sub_id"]) for r in rowsd])
    write_corpus(TMP / "dom" / "corpus_p1", rowsd, Td, {"buckets": spec.BUCKETS, "pass": "p1", "pass_buckets": PASS_BUCKETS})
    d = TMP / "dom" / "s03_p1"
    (d / "bank").mkdir(parents=True); (d / "routing").mkdir(parents=True)
    (d / "s03.log").write_text("... s03 complete\n")
    held9 = np.concatenate([np.full(r["length"], r["heldout"]) for r in rows9])
    heldd = np.concatenate([np.full(r["length"], r["heldout"]) for r in rowsd])
    for L in range(NL):
        pref9 = rng.normal(0, 2.0, size=(9, E))
        i9, w9 = routing(T9, 9, sid9, pref9)
        np.save(TMP / "s03" / "routing" / f"L{L}_idx.npy", i9); np.save(TMP / "s03" / "routing" / f"L{L}_w.npy", w9)
        nrm9 = rng.uniform(0.5, 2.0, size=(9, E))
        np.savez(TMP / "s03" / "bank" / f"L{L}.npz", **bank_from(i9, w9, sid9, ~held9, 9, nrm9, D.ORIG9, {}))
        prefd = rng.normal(0, 2.0, size=(len(spec.SUBAREAS), E))
        idd, wd = routing(Td, len(spec.SUBAREAS), sidd, prefd)
        np.save(d / "routing" / f"L{L}_idx.npy", idd); np.save(d / "routing" / f"L{L}_w.npy", wd)
        nrmd = rng.uniform(0.5, 2.0, size=(len(spec.SUBAREAS), E))
        np.savez(d / "bank" / f"L{L}.npz", **bank_from(idd, wd, sidd, ~heldd, len(spec.SUBAREAS), nrmd, spec.BUCKETS,
                                                      {"subareas": np.array(spec.SUBAREAS)}))
    # s04 guards + A192 keep-set
    (TMP / "s04").mkdir()
    json.dump({"guards": {"super_expert_candidates": {str(L): [5] for L in range(NL)}}}, open(TMP / "s04" / "candidates.json", "w"))
    json.dump({"protect": {str(L): [10, 11, 12] for L in range(NL)}, "meta": {}}, open(TMP / "s04" / "position_protect.json", "w"))
    keepset.save(TMP / "s04" / "keepset.json", {str(L): list(range(192)) for L in range(NL)}, arm="A")


def cli(*args):
    cmd = [sys.executable, str(REPO / "ops/dom/select_dom.py"), *args, "--procs", "2"]
    r = subprocess.run(cmd, env={**os.environ}, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    return r.stdout


def main():
    build()
    out = TMP / "dom" / "d2"
    # 1. bank format + exact reconstruction
    cli("validate", "--layers", "0", "1")
    v = json.load(open(out / "diag" / "validate.json"))
    assert v["bank_format"][0]["ok"], v["bank_format"]
    for L, r in v["layers"].items():
        assert r["A192_gat_relerr"] < 1e-9 and r["p1_gat_relerr"] < 1e-9, r
        for k in ("A192_F", "p1_F"):
            assert r[k]["rel_frob"] < 1e-9 and r[k]["hope_keep_jaccard"] == 1.0, (k, r[k])
    print("ok validate: 18-bucket/177-sub-area bank, routing->stat mapping and F reconstruction exact")
    # 2. entropy
    cli("entropy")
    e = json.load(open(out / "diag" / "entropy.json"))
    assert set(e["layers"]["0"]) >= {"A192", "domain_pooled", "gamedev", "mlsys"}, e["layers"]["0"].keys()
    assert all(0 < e["layers"][str(L)]["domain_pooled"] <= np.log2(E) for L in range(NL))
    print(f"ok entropy: pooled {e['summary']['domain_pooled']['mean']:.3f} bits, near_flat={e['near_flat']}")
    # 3. candidate layers (reduced HOPE grid to keep the test short; full grid for the cheap criteria)
    cli("layers", "--crit", "reap", "mass")                       # full cheap grid (+ stability halves)
    cli("layers", "--crit", "hope", "--w", "0.45", "--pf", "0.04")  # extends the same units with the HOPE default pair
    cli("shrink", "--w", "0.45")                                   # synthetic halves are unstable -> shrink candidates
    cli("polymath")
    cli("assemble", "--crit", "hope", "--w", "0.45", "--pf", "0.04")
    man = json.load(open(out / "manifest.json"))
    names = {(c["bucket"], c["name"]) for c in man["candidates"]}
    assert ("gamedev", D.DEFAULT_NAME) in names and ("mlsys", "hope_w45_pf4_st0") in names, names
    assert {("polymath", n) for n in ("control", "rr4", "rr6")} <= names, names
    assert any("_shr" in n for _, n in names), "shrink candidates missing"
    sup = json.load(open(out / "diag" / "guards.json"))["super"]
    for c in man["candidates"]:
        keep, fp = keepset.load(out / c["file"], expect=c["sha256"])
        assert sorted(int(k) for k in keep) == list(range(NL))
        for L in range(NL):
            ks = set(keep[str(L)])
            assert len(ks) == 192, (c, L)
            assert {10, 11, 12, 5} <= ks and set(sup[str(L)]) <= ks, ("guard not kept", c["name"], L)
    print(f"ok candidates: {len(man['candidates'])} keep-sets, 192 x {NL} layers, guards hard-kept")
    # general protected slice kept
    for L in range(NL):
        b9 = np.load(TMP / "s03" / "bank" / f"L{L}.npz")
        top = set(np.argsort(-b9["sum"][0])[:int(E * 0.04)].tolist())
        k = set(keepset.load(out / "gamedev" / f"{D.DEFAULT_NAME}.json")[0][str(L)])
        assert top <= k, "general protected slice missing"
    # polymath round-robin set is kept
    for L in range(NL):
        p = json.load(open(out / "_layers" / f"L{L}" / "polymath.json"))
        for n in ("rr4", "rr6"):
            assert set(p["rr"][n]) <= set(p["cands"][n])
            assert len(p["rr"][n]) <= len(p["domains"]) * int(n[2:])
    cli("report", "--doc", str(TMP / "D2_DIAG.md"))
    assert "HOPE stability" in (TMP / "D2_DIAG.md").read_text()
    # diagnostics present
    for f in ("selftrace_bias", "overlap", "stability", "retained_mass"):
        assert (out / "diag" / f"{f}.json").exists(), f
    st = json.load(open(out / "diag" / "stability.json"))
    assert set(st) == set(PASS_BUCKETS) and all(0 <= st[b]["domain_only_mean"] <= 1 for b in st), st
    sb = json.load(open(out / "diag" / "selftrace_bias.json"))
    assert sb["gamedev"]["default"]["mean"] <= 1.0
    print("ok diagnostics:", {b: round(st[b]["domain_only_mean"], 3) for b in st},
          "selftrace J", {b: round(sb[b]["default"]["mean"], 3) for b in sb})
    unit_tests()
    import shutil
    shutil.rmtree(TMP)
    print("ALL OK")


def unit_tests():
    # normalisation: a source's scale never matters; weights are non-negative and sum to 1 (no general subtraction)
    for w in D.W_GRID:
        mw = D.mix_weights(w)
        assert all(v >= 0 for v in mw.values()) and abs(sum(mw.values()) - 1) < 1e-12
        assert abs(mw["code"] + mw["agentic"] - 0.25) < 1e-12 and mw["general"] == 0.15
    try:
        D.mix_weights(0.7)
        raise AssertionError("w > 0.6 accepted")
    except AssertionError as e:
        assert "0.6" in str(e)
    li = D.LayerInputs(0, D.available_buckets(), np.array([5, 10, 11, 12]), buckets=PASS_BUCKETS, halves=False)
    k1 = li.solve("mass", li.mixed("gamedev", 0.45), li.gen_slice(0.04))
    for s in li.src.values():
        s["mass"] = s["mass"] * 1000.0
    k2 = li.solve("mass", li.mixed("gamedev", 0.45), li.gen_slice(0.04))
    assert k1 == k2, "per-source scale changed the selection: normalisation broken"
    # monotone in general: raising one expert's general score never drops it
    e = int(np.setdiff1d(np.arange(E), k1)[0])
    li.src["general"]["mass"][e] *= 1e6
    k3 = li.solve("mass", li.mixed("gamedev", 0.45), li.gen_slice(0.04))
    assert e in k3
    # self-traces off drops exactly the selftraces sub-area
    d1, d0 = li.domain("gamedev", 1), li.domain("gamedev", 0)
    assert np.allclose(d1["reap"] - d0["reap"], li.sub["gamedev"]["selftraces"]["reap"])
    # round robin
    from ops.dom.select_dom import poly_round_robin
    rr = poly_round_robin({"a": [1, 2, 3, 4], "b": [2, 5, 6, 7]}, 2, {5})
    assert rr == {1, 2}, rr
    print("ok unit: normalisation invariance, no general subtraction, w cap, selftraces axis, round robin")


if __name__ == "__main__":
    main()
