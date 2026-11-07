#!/usr/bin/env python3
"""P pool (s07 pool, ops/dom/pool_corpus.py, ops/dom/assemble.py), CPU only, no GPU, no model load.

  1. assemble.cut_layer_file on a synthetic pool layer == st_io.write of the numpy-sliced tensors, byte for byte
     (experts, router rows, bias, bias_vl, kt_scale sliced; rht_sign and dense tensors untouched; metadata).
  2. s07 pool inputs in a throwaway $V41_WORK: identity keep-set, A192's allocation copied verbatim
     (bytes per 192-mask == A192's expert_bytes, pool == 2x), migrate_file (verified copy + symlink),
     `s07 pool --dry-run` end to end (corpus, Build manifest, disk preflight).
  3. With --corpus DIR (a pool_corpus.py build, e.g. the small CPU slice): s03.Corpus and replay.Run open it under
     V41_DOM=1 V41_DOM_PASS=pool, and the Engram files have one 24-column row block per token.
  4. With --real-layer L: cutting thor/A192/layer-L with its own identity keep-set reproduces the file exactly
     (sha256 == thor.json): the streaming writer's layout is st_io.write's. Reads ~2.4 GiB.

    .venv/bin/python ops/tests/test_pool.py [--corpus DIR] [--real-layer 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REAL_WORK = Path(os.environ.get("V41_WORK", Path.home() / "v41-work"))
TMP = Path(tempfile.mkdtemp(prefix="pooltest_"))
os.environ["V41_WORK"] = str(TMP / "work")
sys.path.insert(0, str(REPO))
from ops.dom import assemble  # noqa: E402
from ops.lib import keepset, st_io  # noqa: E402
from ops.lib import thor_format as TF  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_cut_synthetic():
    rng = np.random.default_rng(0)
    L, n = 3, 8
    T = [("layers.3.attn.wq_a.weight", "F8_E4M3", rng.integers(0, 255, (16, 32), dtype=np.uint8)),
         ("layers.3.attn_norm.weight", "BF16", rng.integers(0, 65535, (32,), dtype=np.uint16)),
         ("layers.3.ffn.gate.weight", "BF16", rng.integers(0, 65535, (n, 32), dtype=np.uint16)),
         ("layers.3.ffn.gate.bias", "F32", rng.standard_normal(n).astype(np.float32)),
         ("layers.3.ffn.gate.bias_vl", "F32", rng.standard_normal(n).astype(np.float32)),
         ("layers.3.ffn.experts.w1.qweight", "U8", rng.integers(0, 255, (n, 4, 84), dtype=np.uint8)),
         ("layers.3.ffn.experts.w3.qweight", "U8", rng.integers(0, 255, (n, 4, 84), dtype=np.uint8)),
         ("layers.3.ffn.experts.w2.qweight", "U8", rng.integers(0, 255, (n, 3, 110), dtype=np.uint8)),
         ("layers.3.ffn.experts.w2.kt_scale", "F16", rng.standard_normal((n, 3, 2)).astype(np.float16)),
         ("layers.3.ffn.experts.w13.rht_sign", "U8", rng.integers(0, 255, (5,), dtype=np.uint8))]
    meta = {"format": "thor-v1", "layer": "3", "keepset_sha256": "a" * 64, "calib_tokens_per_expert": json.dumps(list(range(100, 100 + n)))}
    src = TMP / "pool_layer.safetensors"
    st_io.write(src, T, metadata=meta)
    ids = [1, 3, 4, 6]
    dst = TMP / "cut.safetensors"
    r = assemble.cut_layer_file(src, dst, L, ids, {"keepset_sha256": "b" * 64})
    ref = TMP / "ref.safetensors"
    exp = [(nm, dt, (a[ids] if assemble.per_expert(nm, L) else a)) for nm, dt, a in T]
    st_io.write(ref, exp, metadata={**meta, "keepset_sha256": "b" * 64, "calib_tokens_per_expert": json.dumps([101, 103, 104, 106])})
    assert sha(dst) == sha(ref) == r["sha256"], "cut layer differs from st_io.write of the sliced tensors"
    rd = st_io.Reader(dst)
    assert rd.get("layers.3.ffn.experts.w1.qweight").shape == (4, 4, 84)
    assert np.array_equal(rd.get("layers.3.ffn.gate.bias"), T[3][2][ids])
    assert np.array_equal(rd.get("layers.3.ffn.experts.w13.rht_sign"), T[9][2])
    # identity keep reproduces the source exactly
    r2 = assemble.cut_layer_file(src, TMP / "ident.safetensors", L, list(range(n)), {"keepset_sha256": "a" * 64})
    assert r2["sha256"] == sha(src)
    print("ok 1 cut_layer_file synthetic: sliced == st_io.write(sliced), identity == source")


def test_s07_pool(corpus: Path | None):
    W = TMP / "work"
    (W / "s07").mkdir(parents=True)
    (W / "thor" / "A192").mkdir(parents=True)
    shutil.copy(REAL_WORK / "s07" / "alloc.json", W / "s07" / "alloc.json")
    shutil.copy(REAL_WORK / "thor" / "A192" / "thor.json", W / "thor" / "A192" / "thor.json")
    sys.path.insert(0, str(REPO / "ops"))
    import ops.s07_quant as S7
    keep, fp, kp = S7.pool_keepset()
    assert len(keep) == S7.N_LAYERS and all(v == list(range(384)) for v in keep.values())
    assert keepset.load(kp)[1] == fp == keepset.fingerprint({str(L): list(range(384)) for L in range(S7.N_LAYERS)})
    al = S7.pool_alloc(fp)
    a192 = json.load(open(W / "thor" / "A192" / "thor.json"))
    assert al["alloc"] == a192["alloc"]
    assert al["bytes_per_mask"] == a192["expert_bytes"], (al["bytes_per_mask"], a192["expert_bytes"])
    assert al["bytes"] == 2 * a192["expert_bytes"]
    assert S7.pool_alloc(fp) == al                                  # idempotent
    est = S7.kq_build_estimate(al["alloc"], 384)
    print(f"ok 2a pool keep-set {fp[:16]}, alloc = A192's ({al['avg_bpw']:.4f} bpw), experts {al['bytes'] / 2**30:.2f} GiB "
          f"(per mask {al['bytes_per_mask'] / 2**30:.2f} GiB), build estimate {est / 2**30:.1f} GiB")
    # migrate_file: verified copy, symlink, idempotent
    b = TF.Build(W / "thor" / "MIGTEST", create=True, keepset_sha256=fp, n_routed_experts=384)
    b.write_file("layer-00.safetensors", [{"name": "x", "dtype": "U8", "data": np.arange(1000, dtype=np.uint8)}])
    store = TMP / "store"
    S7.migrate_file(b, "layer-00.safetensors", store)
    p = b.root / "layer-00.safetensors"
    assert p.is_symlink() and p.resolve() == (store / "layer-00.safetensors").resolve()
    assert b.has("layer-00.safetensors") and b.verify() == []
    S7.migrate_file(b, "layer-00.safetensors", store)                 # already a symlink: no-op
    print("ok 2b migrate_file: verified copy on the store, symlink back, Build.has/verify follow it")
    if corpus is None:
        return
    (W / "dom").mkdir(exist_ok=True)
    os.symlink(corpus / "corpus_pool", W / "dom" / "corpus_pool")
    (W / "dom" / "s03_pool").mkdir()
    os.symlink(corpus / "s03_pool" / "engram", W / "dom" / "s03_pool" / "engram")
    env = {**os.environ, "V41_DOM": "1", "V41_DOM_PASS": "pool", "V41_WORK": str(W)}
    out = subprocess.run([sys.executable, str(REPO / "ops" / "s07_quant.py"), "pool", "--dry-run"], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    m = json.load(open(W / "thor" / "POOL384" / "thor.json"))
    assert m["pool"] is True and m["n_routed_experts"] == 384 and m["keepset_sha256"] == fp and m["alloc"] == a192["alloc"]
    assert set(m["heal"]["gains"].values()) == {1.0} and m["complete"] is False
    print("ok 2c s07 pool --dry-run:", [l.split("s07", 1)[-1].strip() for l in out.stdout.splitlines() if "pool:" in l][0])
    bad = subprocess.run([sys.executable, str(REPO / "ops" / "s07_quant.py"), "pool", "--dry-run"],
                         env={**env, "V41_DOM_PASS": "t3"}, capture_output=True, text=True)
    assert bad.returncode != 0 and "V41_DOM_PASS=pool" in bad.stderr
    print("ok 2d s07 pool refuses to run without V41_DOM_PASS=pool")
    code = f"""
import sys, json, numpy as np
sys.path.insert(0, {str(REPO)!r})
from ops import s03_saliency as s03
from ops.lib import replay
C = s03.Corpus()
assert str(s03.CORPUS).endswith('dom/corpus_pool') and str(s03.D).endswith('dom/s03_pool')
for L in (1, 14):
    v = np.load(s03.D / 'engram' / f'L{{L}}_vals.npy', mmap_mode='r'); s = np.load(s03.D / 'engram' / f'L{{L}}_scales.npy', mmap_mode='r')
    assert v.shape == (C.T * s03.N_COLS, s03.EHD) and s.shape[0] == C.T * s03.N_COLS, (v.shape, s.shape, C.T)
rows = [r['row'] for r in C.rows]
run = replay.prepare({str(TMP / 'replay')!r}, rows, [{{'name': 'pool', 'keep': {{str(L): list(range(384)) for L in range(40)}}}}],
                     json.load(open(s03.CORPUS / 'manifest.json'))['spec_hash'])
assert run.T == C.T and run.gsl(rows[-1]).stop == C.T and run.arm('pool')['keep_sha256'] == {fp!r}
print(C.T, len(rows))
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    print(f"ok 3 s03.Corpus + replay.Run open the pool corpus ({r.stdout.split()[0]} tokens, {r.stdout.split()[1]} rows), "
          f"Engram files sized T x 24")


def test_stage_shard():
    """the pool's NVMe staging of a layer's source shard, against a fake $V41_SRC (subprocess: env.SRC is import-time)."""
    src = TMP / "src"
    src.mkdir()
    real = Path(os.environ.get("V41_SRC_REAL", Path.home() / "DeepSeek-V4.1-Flash-1017"))
    os.symlink(real / "inference", src / "inference")
    shutil.copy(real / "config.json", src / "config.json")
    blob = np.random.default_rng(1).integers(0, 255, 3 << 20, dtype=np.uint8).tobytes()
    (TMP / "far").mkdir()
    (TMP / "far" / "model-00008-of-00048.safetensors").write_bytes(blob)       # layer 5's shard, "on the cold store"
    os.symlink(TMP / "far" / "model-00008-of-00048.safetensors", src / "model-00008-of-00048.safetensors")
    json.dump([{"type": "file", "path": "model-00008-of-00048.safetensors",
                "lfs": {"oid": hashlib.sha256(blob).hexdigest(), "size": len(blob)}}], open(TMP / "v41_1017_tree.json", "w"))
    code = f"""
import sys, os
sys.path.insert(0, {str(REPO)!r})
import ops.s07_quant as S7
m = S7._pool_src_mirror()
d = m / 'model-00008-of-00048.safetensors'
assert d.is_symlink() and (m / 'config.json').is_symlink() and (m / 'inference').is_symlink()
S7._stage_shard(5); assert not d.is_symlink() and d.read_bytes() == open({str(TMP / 'far' / 'model-00008-of-00048.safetensors')!r}, 'rb').read()
S7._stage_shard(5)
S7._unstage_shard(5); assert d.is_symlink() and d.resolve() == ({str(TMP / 'far' / 'model-00008-of-00048.safetensors')!r} and __import__('pathlib').Path({str(TMP / 'far' / 'model-00008-of-00048.safetensors')!r}).resolve())
open({str(TMP / 'far' / 'model-00008-of-00048.safetensors')!r}, 'r+b').write(b'X')   # corrupt the source
try:
    S7._stage_shard(5); raise AssertionError('corrupt shard staged')
except SystemExit as e:
    assert 'pinned sha256' in str(e)
print('staged')
"""
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "V41_SRC": str(src), "V41_WORK": str(TMP / "work2")},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "staged" in r.stdout, r.stdout + r.stderr
    print("ok 2e source-shard staging: mirror of symlinks, sha256-verified NVMe copy, unstage back to the symlink, corrupt copy refused")


def test_real_layer(L: int):
    src = REAL_WORK / "thor" / "A192" / TF.layer_file(L)
    m = json.load(open(REAL_WORK / "thor" / "A192" / "thor.json"))
    dst = TMP / "real_cut.safetensors"
    r = assemble.cut_layer_file(src, dst, L, list(range(m["n_routed_experts"])), {"keepset_sha256": m["keepset_sha256"]})
    ok = r["sha256"] == m["files"][TF.layer_file(L)]["sha256"]
    dst.unlink()
    assert ok, f"identity cut of A192 layer {L} differs from the file ({r['sha256']} vs {m['files'][TF.layer_file(L)]['sha256']})"
    print(f"ok 4 identity cut of thor/A192/{TF.layer_file(L)} == the file (sha256 {r['sha256'][:16]}, {r['bytes'] / 2**30:.2f} GiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="dir holding corpus_pool/ and s03_pool/engram/ from pool_corpus.py build")
    ap.add_argument("--real-layer", type=int)
    a = ap.parse_args()
    try:
        test_cut_synthetic()
        test_s07_pool(Path(a.corpus) if a.corpus else None)
        test_stage_shard()
        if a.real_layer is not None:
            test_real_layer(a.real_layer)
        print("ALL OK")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
