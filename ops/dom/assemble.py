#!/usr/bin/env python3
"""Cut a 192-expert thor-v1 build out of the 384-expert pool (s07 pool) for one keep-set. CPU only.

The CUDA server loads the pool plus a keep-set directly (pool-keepset branch), so serving never needs this.
It exists for gate A(b) (pool + mask must run bit-identically to a build of that mask cut from the pool) and as
a fallback for runtimes that only take a plain thor-v1 build.

What a cut build holds, per layer file, in the pool file's own tensor order and metadata:
  * routed experts (`layers.L.ffn.experts.*`, every per-expert stacked tensor): the kept experts' slabs,
    ascending original id = renumbered 0..n-1 (thor_format.md), bytes copied verbatim
  * `ffn.gate.weight` rows, `ffn.gate.bias`, `ffn.gate.bias_vl`: the same rows
  * every other tensor byte for byte; metadata `keepset_sha256` = the mask, `calib_tokens_per_expert` sliced
globals / vision / mtp files are symlinked to the pool's (identical bytes; --copy-globals copies them).

The writer streams one tensor at a time with plain reads (the pool lives on the cold store, NTFS over FUSE,
where mmap faults crawl), so RSS stays at one expert slab plus buffers. Its byte layout is st_io.write's
exactly (256-byte alignment, `__pad.<n>` tensors, space-padded header): cutting a build with its own
identity keep-set reproduces the file byte for byte (ops/tests/test_pool.py).

  assemble.py <keepset.json> --out DIR [--pool DIR] [--layers 0,1,...] [--copy-globals]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ops.lib import env, keepset, st_io  # noqa: E402
from ops.lib import thor_format as TF  # noqa: E402

STAGE = "assemble"
POOL = env.WORK / "thor" / "POOL384"
GLOBAL_FILES = ("globals.safetensors", "vision.safetensors", "mtp-0.safetensors", "mtp-1.safetensors", "mtp-2.safetensors")
CHUNK = 64 << 20


def per_expert(name: str, L: int) -> bool:
    """tensors indexed by routed expert along dim 0 (sliced to the keep-set)."""
    p = f"layers.{L}."
    if name in (p + "ffn.gate.weight", p + "ffn.gate.bias", p + "ffn.gate.bias_vl"):
        return True
    return name.startswith(p + "ffn.experts.") and not name.endswith(".rht_sign")


def _layout(entries: list[tuple[str, str, list[int], int]], metadata: dict | None, align: int = st_io.ALIGN):
    """st_io.write's header + data layout for (name, dtype, shape, nbytes) -> (header bytes, [(name|None pad, nbytes)])."""
    hdr, plan, off, pad_i = {}, [], 0, 0
    for name, dt, shape, nb in entries:
        if align > 1 and off % align:
            k = align - off % align
            pn = f"__pad.{pad_i}"; pad_i += 1
            hdr[pn] = {"dtype": "U8", "shape": [k], "data_offsets": [off, off + k]}
            plan.append((None, k))
            off += k
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        plan.append((name, nb))
        off += nb
    if metadata:
        hdr["__metadata__"] = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((-(8 + len(hj))) % max(align, 8))
    return struct.pack("<Q", len(hj)) + hj, plan


def cut_layer_file(src: Path, dst: Path, L: int, ids: list[int], meta: dict | None = None) -> dict:
    """Write dst = src with every per-expert tensor of layer L reduced to `ids` (ascending). -> {sha256, bytes,
    tensors: {name: (dtype, shape)}}. meta: metadata overrides (others are copied from src)."""
    ids = sorted(int(i) for i in ids)
    h, base = st_io.header(src)
    smeta = h.pop("__metadata__", {})
    names = sorted((n for n in h if not n.startswith("__pad.")), key=lambda n: h[n]["data_offsets"][0])
    n_src = None
    entries, slabs = [], {}
    for n in names:
        e = h[n]
        nb = e["data_offsets"][1] - e["data_offsets"][0]
        shape = list(e["shape"])
        if per_expert(n, L):
            if n_src is None:
                n_src = shape[0]
            if shape[0] != n_src or nb % shape[0]:
                raise SystemExit(f"{src}: {n} leading dim {shape[0]} != {n_src} experts")
            slab = nb // shape[0]
            slabs[n] = slab
            shape[0] = len(ids)
            nb = slab * len(ids)
        entries.append((n, e["dtype"], shape, nb))
    if n_src is None:
        raise SystemExit(f"{src}: no per-expert tensors for layer {L}")
    if ids and (ids[0] < 0 or ids[-1] >= n_src):
        raise SystemExit(f"keep ids outside [0, {n_src})")
    md = dict(smeta)
    if "calib_tokens_per_expert" in md:
        c = json.loads(md["calib_tokens_per_expert"])
        md["calib_tokens_per_expert"] = json.dumps([c[i] for i in ids])
    md.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (meta or {}).items()})
    head, plan = _layout(entries, md)
    sha = hashlib.sha256()
    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    zero = bytes(st_io.ALIGN)

    def out(fo, b):
        fo.write(b); sha.update(b)

    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        out(fo, head)
        for name, nb in plan:
            if name is None:
                out(fo, zero[:nb])
                continue
            a0 = base + h[name]["data_offsets"][0]
            runs = [(a0 + i * slabs[name], slabs[name]) for i in ids] if name in slabs else [(a0, nb)]
            # coalesce consecutive slabs into one read
            merged = []
            for o, k in runs:
                if merged and merged[-1][0] + merged[-1][1] == o:
                    merged[-1] = (merged[-1][0], merged[-1][1] + k)
                else:
                    merged.append((o, k))
            for o, k in merged:
                fi.seek(o)
                while k:
                    b = fi.read(min(k, CHUNK))
                    if not b:
                        raise SystemExit(f"short read in {src}")
                    out(fo, b)
                    k -= len(b)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return {"sha256": sha.hexdigest(), "bytes": dst.stat().st_size,
            "tensors": {n: (dt, sh) for n, dt, sh, _ in entries}}


def assemble(keep_path: Path, out: Path, pool: Path, layers: list[int] | None, copy_globals: bool) -> None:
    pm = json.load(open(pool / "thor.json"))
    if not pm.get("pool") or not pm.get("complete"):
        raise SystemExit(f"{pool}: not a complete pool build (thor.json pool/complete)")
    keep, fp = keepset.load(keep_path)
    n_pool = pm["n_routed_experts"]
    ks = {len(v) for v in keep.values()}
    if len(ks) != 1 or sorted(int(k) for k in keep) != list(range(len(pm["alloc"]))):
        raise SystemExit(f"{keep_path}: must keep the same count in every one of the {len(pm['alloc'])} layers")
    n = ks.pop()
    layers = list(range(len(pm["alloc"]))) if layers is None else layers
    if (out / "thor.json").exists():
        old = json.load(open(out / "thor.json"))
        if old.get("keepset_sha256") != fp:
            raise keepset.KeepsetMismatch(f"{out}: existing build is for {old.get('keepset_sha256')}, asked {fp}")
    meta = {k: pm[k] for k in ("format", "source_revision", "heal", "alloc", "protected", "engram_tables") if k in pm}
    b = TF.Build(out, create=True, keepset_sha256=fp, n_routed_experts=n, arm="A",
                 cut_from={"pool": str(pool), "pool_keepset_sha256": pm["keepset_sha256"], "pool_calib": pm.get("calib")},
                 **{k: v for k, v in meta.items() if k != "format"})
    for L in layers:
        f = TF.layer_file(L)
        if b.has(f):
            continue
        r = cut_layer_file(pool / f, out / f, L, keep[str(L)], {"keepset_sha256": fp})
        for name in [t for t, v in b.m["tensors"].items() if v["file"] == f]:
            del b.m["tensors"][name]
        for name, (dt, sh) in r["tensors"].items():
            src_t = pm["tensors"][name]
            shape = list(src_t["shape"])
            if per_expert(name, L):
                shape[0] = len(keep[str(L)])
            b.m["tensors"][name] = {"file": f, "dtype": dt, "qtype": src_t.get("qtype"), "stored_shape": sh, "shape": shape}
        b.m["files"][f] = {"bytes": r["bytes"], "sha256": r["sha256"]}
        b.save()
        env.log(f"L{L}: {r['bytes'] / 2**30:.2f} GiB", STAGE)
    for f in GLOBAL_FILES:
        if b.has(f) or f not in pm["files"]:
            continue
        dst = out / f
        dst.unlink(missing_ok=True)
        if copy_globals:
            shutil.copyfile(pool / f, dst)
        else:
            os.symlink((pool / f).resolve(), dst)
        for name, t in pm["tensors"].items():
            if t["file"] == f:
                b.m["tensors"][name] = dict(t)
        b.m["files"][f] = dict(pm["files"][f])
        b.save()
    if all(b.has(TF.layer_file(L)) for L in range(len(pm["alloc"]))) and all(b.has(f) for f in GLOBAL_FILES if f in pm["files"]):
        b.m["expert_bytes"] = int(b.expert_bytes())
        b.m["expert_bpw"] = b.m["expert_bytes"] * 8 / (n * 3 * 5120 * 2304 * len(pm["alloc"]))
        b.m["complete"] = True
        b.save()
        env.log(f"cut build {out}: {n} experts/layer, {b.m['expert_bytes'] / 2**30:.2f} GiB experts, keep-set {fp[:16]}", STAGE)
    else:
        env.log(f"cut build {out}: partial ({len(layers)} layers written), complete=false", STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keepset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--layers", help="comma list (default all)")
    ap.add_argument("--copy-globals", action="store_true")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    assemble(Path(a.keepset), Path(a.out), Path(a.pool), layers, a.copy_globals)


if __name__ == "__main__":
    main()
