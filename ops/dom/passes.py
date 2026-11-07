#!/usr/bin/env python3
"""D1 in passes: run the domain saliency pass (s03) over bucket subsets that fit the NVMe, then merge.

The full domain corpus ($V41_WORK/dom/corpus, every bucket of corpus_spec_dom.BUCKETS) is too large
for one s03 pass on the NVMe (~55 KB/token of scratch). `split` cuts it into per-pass corpora
$V41_WORK/dom/corpus_<pass>/ holding only that pass's buckets. Rows keep their GLOBAL bucket_id, held-out
flag and relative visit order, and the manifest keeps the full `buckets` list, so s03 (V41_DOM=1
V41_DOM_PASS=<pass>) allocates accumulators for every bucket and only the pass's buckets become
non-zero. Pass banks therefore merge exactly: sums add, maxima take the max.

  split [--passes p1=b1,b2 p2=...]   write corpus_<pass>/ (default grouping: corpus_spec_dom.PASSES)
  plan                                tokens and NVMe scratch per pass
  retire <pass>                       after that pass's s03 completes: delete its stream, Engram rows,
                                      journals and state; keep bank/, teacher/, routing/, taps/, ledger
  merge                               $V41_WORK/dom/s03_<pass>/bank/L*.npz -> $V41_WORK/dom/bank/L*.npz
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.dom import corpus_spec_dom as spec  # noqa: E402
from ops.lib import env  # noqa: E402

STAGE = "dom_passes"
DOM = env.WORK / "dom"
FULL = DOM / "corpus"          # --draw 2: DOM / "corpus_draw2", passes named draw2_<p>
BYTES_PER_TOKEN = 40_976 + 12_864 + 960 + 264        # stream+pre_mix, Engram rows+hash, routing, teacher
MAX_KEYS = {"fmax", "gfmax", "amax_in", "amax_mid", "amax_in_sh", "amax_mid_sh"}
PARK = os.path.join(os.environ.get("V41_COLD", "/mnt/v41-cold"), "v41_1017_parked", "dom")   # cold store
RETIRE = ("h.npy", "pre_mix.npy", "engram", "journal", "state")


def passes_arg(a) -> dict[str, list[str]]:
    if not a.passes:
        return spec.PASSES
    out = {}
    for p in a.passes:
        name, bs = p.split("=")
        out[name] = bs.split(",")
    return out


def split(a) -> None:
    FULL = DOM / ("corpus_draw2" if a.draw == "2" else "corpus")
    pre = "draw2_" if a.draw == "2" else ""
    man = json.load(open(FULL / "manifest.json"))
    assert man["buckets"] == spec.BUCKETS, (man["buckets"], spec.BUCKETS)
    rows = [json.loads(l) for l in open(FULL / "rows.jsonl")]
    arrs = {k: np.load(FULL / f"{k}.npy", mmap_mode="r") for k in ("tokens", "types", "valid")}
    P = passes_arg(a)
    seen = [b for bs in P.values() for b in bs]
    assert sorted(seen) == sorted(set(seen)) and set(seen) <= set(spec.BUCKETS), P
    missing = set(spec.BUCKETS) - set(seen)
    if missing:
        env.log(f"buckets in no pass: {sorted(missing)}", STAGE, "WARN")
    for name, bs in P.items():
        out = DOM / f"corpus_{pre}{name}"
        out.mkdir(parents=True, exist_ok=True)
        sub = [r for r in rows if r["bucket"] in bs]
        T = sum(r["length"] for r in sub)
        mm = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=arrs[k].dtype, shape=(T,)) for k in arrs}
        off = 0
        with open(out / "rows.jsonl.tmp", "w") as fh:
            for i, r in enumerate(sub):
                n = r["length"]
                for k in arrs:
                    mm[k][off:off + n] = arrs[k][r["offset"]:r["offset"] + n]
                fh.write(json.dumps({**r, "row": i, "offset": off, "full_row": r["row"]}) + "\n")
                off += n
        for m in mm.values():
            m.flush()
        os.replace(out / "rows.jsonl.tmp", out / "rows.jsonl")
        env.atomic_json(out / "manifest.json", {
            **{k: man[k] for k in ("spec_hash", "generator_policy", "revision", "draw")},
            "buckets": spec.BUCKETS, "subareas": spec.SUBAREAS, "pass": pre + name, "pass_buckets": bs, "rows": len(sub), "tokens_total": T,
            "tokens_valid": int(sum(r["n_valid"] for r in sub)), "order": [r["bucket"] for r in sub],
            "max_row_len": max(r["length"] for r in sub), "longctx_lengths": [], "source_manifest_ts": man["ts"],
            "ts": env.now()})
        env.log(f"pass {name}: {bs} -> {len(sub)} rows, {T:,} tokens, scratch ~{T * BYTES_PER_TOKEN / 1e9:.0f} GB", STAGE)


def plan(a) -> None:
    rows = [json.loads(l) for l in open(FULL / "rows.jsonl")] if (FULL / "rows.jsonl").exists() else None
    for name, bs in passes_arg(a).items():
        T = sum(r["length"] for r in rows if r["bucket"] in bs) if rows else len(bs) * spec.ROWS_PER_DOMAIN * spec.SEQ_LEN
        print(f"{name}: {','.join(bs):50s} {T:>10,} tokens  scratch ~{T * BYTES_PER_TOKEN / 1e9:.0f} GB persistent "
              f"(+~10 GB transient){'' if rows else '  [planned]'}")


def retire(a) -> None:
    d = DOM / f"s03_{a.name}"
    led = json.load(open(d / "s03_ledger.json"))
    if not led.get("head", {}).get("done"):
        raise SystemExit(f"{d}: s03 not complete (head unit not done); refusing to delete its stream")
    # Engram rows are needed again by teacher/candidate replay on held-out rows (D2s ranking, E tier 1): park them on
    # the cold store (copy, verify size, symlink) instead of deleting them
    eg = d / "engram"
    if eg.is_dir() and not eg.is_symlink():
        park = Path(PARK) / f"s03_{a.name}_engram"
        if park.exists():
            shutil.rmtree(park)
        shutil.copytree(eg, park)
        n_src = sum(f.stat().st_size for f in eg.rglob("*") if f.is_file())
        n_dst = sum(f.stat().st_size for f in park.rglob("*") if f.is_file())
        if n_src != n_dst:
            raise SystemExit(f"{eg}: parked copy is {n_dst} B, source {n_src} B; not retiring")
        shutil.rmtree(eg)
        eg.symlink_to(park)
        env.log(f"parked {eg} -> {park} ({n_src / 1e9:.1f} GB)", STAGE)
    for x in RETIRE:
        p = d / x
        if x == "engram":
            continue
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    env.log(f"retired {d}: removed {RETIRE[:2] + RETIRE[3:]}, engram parked; kept bank/ teacher/ routing/ taps/ ledger", STAGE)


def merge(a) -> None:
    pre = "draw2_" if a.draw == "2" else ""
    dirs = sorted(d for d in DOM.glob(f"s03_{pre}*/bank") if a.draw == "2" or not d.parent.name.startswith("s03_draw2_"))
    if not dirs:
        raise SystemExit("no pass banks")
    out = DOM / ("bank_draw2" if a.draw == "2" else "bank"); out.mkdir(exist_ok=True)
    layers = sorted({p.name for d in dirs for p in d.glob("L*.npz")}, key=lambda n: int(n[1:-4]))
    for ln in layers:
        have = [d / ln for d in dirs if (d / ln).exists()]
        if len(have) != len(dirs):
            raise SystemExit(f"{ln} missing in {[str(d) for d in dirs if not (d / ln).exists()]}")
        acc = None
        for p in have:
            with np.load(p) as z:
                cur = {k: z[k] for k in z.files}
            assert list(cur["buckets"]) == spec.BUCKETS, (p, cur["buckets"])
            assert list(cur["subareas"]) == spec.SUBAREAS, (p, "sub-area order")
            if acc is None:
                acc = cur
                continue
            for k, v in cur.items():
                if k in ("buckets", "subareas"):
                    continue
                acc[k] = np.maximum(acc[k], v) if k in MAX_KEYS else acc[k] + v
        np.savez(out / ln, **acc, passes=np.array([d.parent.name for d in dirs]))
    env.log(f"merged {len(layers)} layers from {[d.parent.name for d in dirs]} -> {out}", STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "plan", "retire", "merge"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--passes", nargs="*")
    ap.add_argument("--draw", choices=["1", "2"], default="1", help="split the dose-response draw (corpus_draw2)")
    a = ap.parse_args()
    {"split": split, "plan": plan, "retire": retire, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
