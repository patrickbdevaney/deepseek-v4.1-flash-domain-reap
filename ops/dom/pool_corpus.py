#!/usr/bin/env python3
"""P: the GPTQ calibration corpus of the 384-expert pool (docs/domain/P_POOL.md), CPU only.

The pool's Hessians come from ONE replay stream through all 384 experts, so the token mix has to cover
every domain a mask may later keep. This writes a merged, s03-layout corpus plus its Engram rows, so
`s07 pool` replays it with the stock replay machinery (V41_DOM=1 V41_DOM_PASS=pool):

  $V41_WORK/dom/corpus_pool/{tokens,types,valid}.npy, rows.jsonl, manifest.json
  $V41_WORK/dom/s03_pool/engram/L{1,14}_{vals,scales}.npy      rows at the merged token offsets

Mix (409,600 tokens; median routed tokens per expert ~ A192's 6,460 at 192 experts):
  domain   the domain buckets x 3 calibration rows (8,192 tokens each, never held-out rows) from the D1
           pass corpora corpus_<pass>: tool-use first, then two more sub-areas in a fixed hash order;
           `selftraces` excluded (A192's own outputs: a mask-specific bias the pool must not carry)
  general  A192's own s07 calibration rows (s07/thor_replay/run.json), 2 per bucket, the shortest
           longctx row once: 163,840 tokens of the original 9-bucket corpus (vision rows keep their images)

Engram rows are copied byte for byte from each source pass's s03 engram files (the D1 rows on the cold store);
every copied slice is re-read and compared. Sources must be COMPLETE s03 passes (ledger `head` done).

  pool_corpus.py build [--out DIR --engram-out DIR] [--domain-rows 3] [--general-per-bucket 2] [--buckets ...]
  pool_corpus.py plan                      print the selection only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import corpus_spec_dom as dspec  # noqa: E402
from ops.lib import env, rowsets  # noqa: E402

STAGE = "pool_corpus"
DOM = env.WORK / "dom"
TIER1 = ["gamedev", "mlsys", "robotics", "quantum"]      # original run: ten tier-1 buckets
PASSES = tuple(dspec.PASSES)                          # original run: p1, t2, t3
ENGRAM_LAYERS = (1, 14)
N_COLS = 24
EXCLUDE_SUB = {"selftraces"}
FIRST_SUB = "tool-use"


def _h(*xs) -> str:
    return hashlib.sha256(":".join(map(str, xs)).encode()).hexdigest()


def _source(name: str) -> dict:
    """corpus dir, s03 dir (engram), ledger of one source."""
    if name == "general":
        return {"name": name, "corpus": env.WORK / "corpus", "s03": env.WORK / "s03", "ledger": env.LOGS / "s03_ledger.json"}
    return {"name": name, "corpus": DOM / f"corpus_{name}", "s03": DOM / f"s03_{name}", "ledger": DOM / f"s03_{name}" / "s03_ledger.json"}


def _require_complete(src: dict) -> None:
    led = json.load(open(src["ledger"])) if Path(src["ledger"]).exists() else {}
    if not led.get("head", {}).get("done"):
        raise SystemExit(f"{src['name']}: s03 pass not complete ({src['ledger']} has no finished `head`); its Engram rows "
                         f"may still be written. Run after D1.")
    for L in ENGRAM_LAYERS:
        for k in ("vals", "scales"):
            if not led.get(f"engram/L{L}/{k}", {}).get("done"):
                raise SystemExit(f"{src['name']}: engram/L{L}/{k} not done in {src['ledger']}")


def _rows_of(mm: np.ndarray, start: int, n: int) -> np.ndarray:
    """rows [start, start+n) of a .npy opened with mmap_mode='r', read with plain file reads: the cold store is NTFS
    over FUSE, where mmap page faults crawl."""
    row = int(np.prod(mm.shape[1:])) * mm.dtype.itemsize
    out = np.empty((n, *mm.shape[1:]), mm.dtype)
    with open(mm.filename, "rb") as fh:
        fh.seek(mm.offset + start * row)
        if fh.readinto(memoryview(out).cast("B")) != n * row:
            raise SystemExit(f"short read from {mm.filename}")
    return out


def select(a) -> list[dict]:
    """-> [{src, row (source row dict)}] in pool order (domain buckets in TIER1 order, then general)."""
    picks = []
    by_pass = {}
    for p in PASSES:
        c = DOM / f"corpus_{p}"
        if not (c / "manifest.json").exists():
            continue
        man = json.load(open(c / "manifest.json"))
        for b in man["pass_buckets"]:
            by_pass[b] = p
    for b in a.buckets:
        if b not in by_pass:
            raise SystemExit(f"bucket {b} is in no D1 pass corpus ({sorted(by_pass)})")
        p = by_pass[b]
        rows = [json.loads(l) for l in open(DOM / f"corpus_{p}" / "rows.jsonl")]
        cand = [r for r in rows if r["bucket"] == b and not r["heldout"] and r["subarea"] not in EXCLUDE_SUB and not r["images"]]
        subs = sorted({r["subarea"] for r in cand}, key=lambda s: (s != FIRST_SUB, _h("pool-sub", b, s)))
        got = []
        for s in subs:
            if len(got) == a.domain_rows:
                break
            rs = sorted((r for r in cand if r["subarea"] == s), key=lambda r: _h("pool-row", b, r["row"]))
            got.append(rs[0])
        if len(got) < a.domain_rows:          # fewer sub-areas than rows: second rows by hash, any sub-area
            rest = sorted((r for r in cand if r not in got), key=lambda r: _h("pool-row", b, r["row"]))
            got += rest[: a.domain_rows - len(got)]
        picks += [{"src": p, "row": r} for r in got]
    if a.general_per_bucket:
        run = json.load(open(env.WORK / "s07" / "thor_replay" / "run.json"))
        grows = [json.loads(l) for l in open(env.WORK / "corpus" / "rows.jsonl")]
        by_b = {}
        for i in run["rows"]:
            by_b.setdefault(grows[i]["bucket"], []).append(grows[i])
        for b in sorted(by_b):
            rs = sorted(by_b[b], key=lambda r: (r["length"], r["row"]))
            n = 1 if b == "longctx" else a.general_per_bucket
            picks += [{"src": "general", "row": r} for r in rs[:n]]
    return picks


def plan(a) -> None:
    picks = select(a)
    T = sum(p["row"]["length"] for p in picks)
    for p in picks:
        r = p["row"]
        print(f"{p['src']:8s} row {r['row']:5d} {r['bucket']:12s} {r.get('subarea', '-'):18s} {r['length']:6d} tok"
              f"{'  images ' + str(len(r['images'])) if r['images'] else ''}")
    print(f"{len(picks)} rows, {T:,} tokens; replay scratch ~{T * 4 * 5120 * 2 * 3.25 / 2**30:.0f} GiB; "
          f"Engram rows {T * N_COLS * 264 * 2 / 2**30:.1f} GiB")


def build(a) -> None:
    picks = select(a)
    srcs = {s: _source(s) for s in sorted({p["src"] for p in picks})}
    for s in srcs.values():
        _require_complete(s)
    out, eout = Path(a.out), Path(a.engram_out)
    tmp, etmp = out.with_name(out.name + ".partial"), eout.with_name(eout.name + ".partial")
    for d in (tmp, etmp):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    T = sum(p["row"]["length"] for p in picks)
    arr = {s: {k: np.load(srcs[s]["corpus"] / f"{k}.npy", mmap_mode="r") for k in ("tokens", "types", "valid")} for s in srcs}
    eg = {s: {(L, k): np.load(srcs[s]["s03"] / "engram" / f"L{L}_{k}.npy", mmap_mode="r") for L in ENGRAM_LAYERS
              for k in ("vals", "scales")} for s in srcs}
    k0 = next(iter(arr.values()))
    mm = {k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=k0[k].dtype, shape=(T,)) for k in k0}
    e0 = next(iter(eg.values()))
    em = {key: np.lib.format.open_memmap(etmp / f"L{key[0]}_{key[1]}.npy", mode="w+", dtype=v.dtype,
                                         shape=(T * N_COLS, v.shape[1])) for key, v in e0.items()}
    rows, off, digest = [], 0, hashlib.sha256()
    for i, p in enumerate(picks):
        r, s = p["row"], p["src"]
        n, o = r["length"], r["offset"]
        for k in mm:
            mm[k][off:off + n] = arr[s][k][o:o + n]
        for key in em:
            em[key][off * N_COLS:(off + n) * N_COLS] = _rows_of(eg[s][key], o * N_COLS, n * N_COLS)
        dom_b = r["bucket"] if s != "general" else None
        rows.append({"row": i, "bucket": r["bucket"] if dom_b else f"general:{r['bucket']}",
                     "bucket_id": r["bucket_id"] if dom_b else -1, "sub_id": r.get("sub_id", -1) if dom_b else -1,
                     "subarea": r.get("subarea", r["bucket"]), "offset": off, "length": n, "heldout": False,
                     "n_valid": r["n_valid"], "docs": r["docs"], "images": r["images"],
                     "src": s, "src_row": r["row"], "src_offset": o})
        digest.update(f"{s}:{r['row']}:{o}:{n};".encode())
        off += n
    for m in list(mm.values()) + list(em.values()):
        m.flush()
    # read back and compare every copied slice (tokens + Engram rows) against its source
    for r in rows:
        s, n, o = r["src"], r["length"], r["src_offset"]
        for k in mm:
            if not np.array_equal(mm[k][r["offset"]:r["offset"] + n], arr[s][k][o:o + n]):
                raise SystemExit(f"copy check failed: {k} row {r['row']}")
        for key in em:
            a_, b_ = em[key][r["offset"] * N_COLS:(r["offset"] + n) * N_COLS], _rows_of(eg[s][key], o * N_COLS, n * N_COLS)
            if hashlib.sha256(np.ascontiguousarray(a_)).digest() != hashlib.sha256(np.ascontiguousarray(b_)).digest():
                raise SystemExit(f"copy check failed: engram {key} row {r['row']}")
    del mm, em
    with open(tmp / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    src_specs = {s: json.load(open(srcs[s]["corpus"] / "manifest.json"))["spec_hash"] for s in srcs}
    spec_hash = hashlib.sha256((json.dumps(src_specs, sort_keys=True) + digest.hexdigest()).encode()).hexdigest()[:16]
    comp = {}
    for r in rows:
        comp[r["bucket"]] = comp.get(r["bucket"], 0) + r["length"]
    env.atomic_json(tmp / "manifest.json", {
        "spec_hash": spec_hash, "pool": True, "pass": "pool", "buckets": dspec.BUCKETS, "subareas": dspec.SUBAREAS,
        "rows": len(rows), "tokens_total": T, "tokens_valid": int(sum(r["n_valid"] for r in rows)),
        "max_row_len": max(r["length"] for r in rows), "longctx_lengths": [r["length"] for r in rows if r["length"] > 8192],
        "composition_tokens": comp, "sources": {s: {"corpus": str(srcs[s]["corpus"]), "engram": str((srcs[s]["s03"] / "engram").resolve()),
                                                     "spec_hash": src_specs[s]} for s in srcs},
        "rule": "tier-1 domains x domain_rows calib rows (tool-use first, no selftraces, no held-out) + A192 s07 general rows",
        "domain_rows": a.domain_rows, "general_per_bucket": a.general_per_bucket, "buckets_used": a.buckets,
        "rows_hash": rowsets.rows_hash([r["row"] for r in rows], spec_hash), "ts": env.now()})
    for d, t in ((out, tmp), (eout, etmp)):
        if d.exists():
            shutil.rmtree(d)
        os.replace(t, d)
    env.log(f"pool corpus: {len(rows)} rows, {T:,} tokens -> {out}; Engram rows -> {eout} (spec {spec_hash})", STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "plan"])
    ap.add_argument("--out", default=str(DOM / "corpus_pool"))
    ap.add_argument("--engram-out", default=str(DOM / "s03_pool" / "engram"))
    ap.add_argument("--domain-rows", type=int, default=3)
    ap.add_argument("--general-per-bucket", type=int, default=2)
    ap.add_argument("--buckets", nargs="+", default=TIER1)
    a = ap.parse_args()
    {"build": build, "plan": plan}[a.cmd](a)


if __name__ == "__main__":
    main()
