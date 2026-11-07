# P: the 384-expert K-quant pool (decision record, 2026-11-02)

The pool `$V41_WORK/thor/POOL384` holds all 384 routed experts of every layer, GPTQ-rounded once into ggml
K-quant blocks. A 192-expert domain mask then costs nothing to build. The CUDA server loads the pool plus the
mask's keep-set JSON and reads only the kept experts (node A, branch `pool-keepset`). The MXFP4 master side
stays the bit-exact s05 surgery. This file records the build decisions and why each was made.

## How it is built

`ops/s07_quant.py pool` runs the same two-phase per-layer units as `s07 thor` (the build that produced A192),
with keep = all 384:

1. Phase A replays the attention half and caches the FFN inputs.
2. The w1|w3 Hessians are accumulated and the experts GPTQ-rounded, then the w2 Hessians on the rounded
   w1|w3 activations and w2 GPTQ-rounded.
3. Phase B runs the quantized layer and advances the stream.

Existing modes are unchanged: the new code paths take effect only with `pool` or `unit --pool`
(`ops/tests/test_s07_format.py` still passes 9/9).

Inputs:
- `ops/dom/pool_corpus.py build` writes the calibration corpus to `dom/corpus_pool` and its Engram rows to
  `dom/s03_pool/engram`, in s03 layout.
- The replay finds both through s03's own paths, which `V41_DOM=1 V41_DOM_PASS=pool` select. The command
  refuses to run without them.

## Decisions

### 1. Calibration stream and routing

**Routing.** The stream is the unpruned teacher's: the reference router selects over all 384 experts, with
no bias masking.

**Stream.** As in A192's build, the stream advances through the already-quantized pool layers. Each expert's
Hessian therefore sees the inputs a quantized model feeds it. This is the "unpruned, teacher-routed,
quantized-stream" pass PLAN.md specifies.

**Token mix.** 409,600 tokens in 47 rows:

| part | rows | tokens | source |
|---|---:|---:|---|
| domain | 30 | 245,760 (60%) | the original run's 10 tier-1 domain buckets × 3 calibration rows of 8,192 tokens from the D1 pass corpora (`corpus_<pass>`; training split only, never held-out rows). This release's `pool_corpus.py` defaults to the four published buckets |
| general | 17 | 163,840 (40%) | A192's own s07 calibration rows (`s07/thor_replay/run.json`): 2 per original bucket, the shortest longctx row (32,768 tokens) once, and 2 vision rows with their images |

Why these rows:
- **Tool-use first.** Each domain bucket's three rows cover three sub-areas, tool-use first. A192's
  calibration stripped tool calls, and the spine's degraded DSML parameters trace back to that
  (A192 tool-call calibration gap). The pool must give the tool-emitting experts real Hessians.
- **No self-traces.** The `selftraces` sub-area is excluded. It holds A192's own outputs, a bias tied to
  one mask, and the pool has to serve every mask.
- **General rows reused from A192.** The general part overlaps A192's GPTQ calibration, which makes gate
  P1 (pool + A192 keep-set vs A192) a cleaner comparison.

Every Engram row slice is copied byte for byte from the source pass's s03 files and re-read to check it.

**Size.** 409,600 tokens × 6 routed slots / 384 experts gives a median of about 6,400 routed tokens per
expert. A192's build had a median of 6,460 per kept expert (from `calib_tokens_per_expert`, layer 10)
using 237,568 tokens over 192 experts. So the pool keeps A192's per-expert Hessian sample size, at 1.72×
the tokens.

**Fallback for thin experts.** An expert with fewer than `--min-tokens 256` routed tokens gets a diagonal
(imatrix) Hessian. A192 had none below 256 at layer 10, with a minimum of 362. The diagonal comes from the
teacher's second moments summed over the original s03 bank and the D1 pass banks (`pool_bank`; `V41_POOL_PASSES`), not
the general bank alone, because domain-specialist experts are rarely routed by general text.

### 2. Heal gain = 1.0

The masks ship unhealed (D2_METHOD.md), and a per-layer routed gain depends on which experts survive, so
the pool has no gain that is independent of the mask.

This also matches A192 exactly. Its s10 heal panel did not adopt the s06 gains (d_kl32 CI hi +0.0056), so
`thor/A192/thor.json` records every gain as 1.0. The gain therefore adds nothing to the pool-vs-A192
difference: only the GPTQ stream differs.

### 3. Per-layer K-quant types = A192's, verbatim

`s07/pool/alloc_pool.json` is `s07/alloc.json` copied as-is. `pool_alloc` refuses to copy it unless it equals
`thor/A192/thor.json["alloc"]` and has the same keep-set:

| layers | w13 / w2 | count |
|---|---|---:|
| ordinary | Q2_K / Q2_K | 28 |
| upgraded | Q2_K / Q3_K | 3 |
| protected | Q3_K / Q3_K | 9 (0, 1, 2, 8, 14, 20, 37, 38, 39) |

That is 2.828 bpw. Allocation is per layer, not per expert, so every 192-mask cut from the pool has
exactly A192's expert bytes: 89.48 GiB, checked as `bytes_per_mask == thor/A192 expert_bytes` in
`ops/tests/test_pool.py`. Every mask therefore fits the same ~113 GiB serving envelope and runs at the
same speed.

### 4. Size and location

**Size.**

| part | size |
|---|---:|
| experts | 178.97 GiB |
| non-expert (A192's 17.39 GiB; the 384-row router and bias add 0.07 GiB) | 17.46 GiB |
| **total** | **196.4 GiB (210.9 GB)** |

The largest layer file is about 5.3 GiB.

**Where it lives.** The pool's files live on the cold store (`v41_staging/thor/POOL384`). The NVMe directory
`thor/POOL384` keeps `thor.json` and one symlink per file. The NVMe has ~125 GB free and must also hold
the 51 GiB replay scratch, so it cannot take the pool.

Each finished file is written on the NVMe first, then moved:
1. its phase B reads it back from the NVMe;
2. it is copied to the cold store with fsync;
3. the copy is sha256-checked against `thor.json`;
4. the NVMe file is replaced by a symlink to the copy (`migrate_file`).

`Build.has` and `verify`, assemble.py and the CUDA loader (pread) all follow the symlinks. `--pool-store none`
keeps everything on the NVMe if space ever allows.

**Source-shard staging.** The master shards now also sit on the cold store, and mmap reads over its FUSE mount
measured about 20 MB/s, against 65-75 MB/s for plain reads (2026-11-02, while D1 was also reading it). A layer
unit mmaps its ~7.4 GB shard, so each unit instead runs with `V41_SRC=$V41_WORK/s07/pool_src`. That is a
symlink mirror of the master in which the current layer's shard is a sha256-verified NVMe copy. The next
layer's shard is copied in the background while the current unit runs.

**NVMe use during the build.**

| item | size |
|---|---:|
| replay scratch | 51 GiB |
| ≤ 2 staged layer files | ~11 GiB |
| ≤ 2 staged source shards | 15 GiB |
| corpus and Engram rows | 5 GiB |

The preflight requires these plus a 20 GiB margin.

**Server load-time cost from the cold store.** The server reads only the kept slabs: 89.5 GiB of experts in
coalesced runs of ~3.9 MB each, plus the dense and global tensors. The dense NVFP4 overlay comes from
`thor/overlay_nvfp4` on the NVMe.

| source | measured throughput | ~92.5 GiB load |
|---|---|---|
| A192 on NVMe | — | 61 s |
| pool on the cold store, sequential reads (contended) | 74 MB/s | ~22 min |
| pool on the cold store, half-selection strided reads (contended) | 65 MB/s | ~25 min |

Both cold-store measurements were taken while D1 was also reading the disk. An idle cold store should be faster
(roughly 12-15 min, not measured). That cost is paid once per server start, so once per mask switch.

**Fallback for a mask served all the time** (the default harness mask): cut it to the NVMe with
`ops/dom/assemble.py` (107 GiB, the same bytes) once space allows.

### 5. Expected GPU hours (from the A192 s07 log, `results/logs/s07_thor.log`)

The A192 build took 18.3 h in total.

Per-layer timings measured from the A192 log:

| step | time |
|---|---|
| phase A + load | ~108 s |
| one GPTQ batch of 32 experts | 222 s (Q2_K layers), 259 s (Q3_K layers) |
| phase B | ~147 s |
| batches per layer | 6 |

All measured at 237,568 tokens.

The pool has 12 batches per layer at 1.72× the tokens. The token-proportional share of a batch (re-routing
and Hessian accumulation) cannot be separated from the fixed share (the GPTQ solve plus the numpy K-quant
fit) in the log, so the estimate brackets it:

| token-proportional share of a batch | GPU hours |
|---|---:|
| 0% | 35.6 |
| 40% | 44.5 |

Add about 1.5 h of wall time for:
- the per-layer copy to the cold store (~5 GiB a layer);
- the globals;
- the final full `verify()`, which reads all 211 GB from the cold store.

**Plan on ~40 GPU-h and 37-47 h of wall time.** `--stop-after-layer N` bounds a window, and a re-run resumes
at the first unfinished layer.

**Dependencies.** P needs only the D1 pass corpora, their Engram rows and their banks: D1 t3 must be complete.
It does not need D2. Whether it runs before or after the ~30 h D2 replay is a GPU-queue choice; both are on
the Nov 10-11 critical path.

## Run (not started)

```
setsid nohup ops/dom/run_pool.sh > ops/logs/dom_pool.log 2>&1 < /dev/null &
# chained after the D2 replay:
setsid nohup ops/dom/run_pool.sh --after-log ops/logs/dom_after_d1.log REPLAY_DONE > ops/logs/dom_pool.log 2>&1 < /dev/null &
# bounded window: ... run_pool.sh -- --stop-after-layer 9
```

`run_pool.sh` waits for the same guard as `run_d1_passes.sh`:
- memguard is active;
- no inference server is running;
- no `GPU_WINDOW.lock` is held;
- no other process is over 4 GiB RSS;
- the NVMe has ≥ 90 GB free and the cold store is mounted;
- MemAvailable is ≥ 108 GiB after drop_caches.

It then takes the lock, builds the pool corpus once, runs `s07 pool` (logging to `ops/logs/s07_pool.log`),
releases the lock, and logs `POOL_DONE`.

## Gates

**P1 = gate A(c)** (`cuda/scripts/gate_pool.sh` on branch `pool-keepset`). The pool plus A192's keep-set is
compared with the A192 build on the golden prompts against the teacher (`golden/src`, 2,062 saved rows).
Thresholds, registered before the pool exists:
- **primary:** |Δ top-1 vs teacher| ≤ 0.005 (the PLAN rule);
- **secondary:** mean full-vocab KL(A192 ‖ pool) ≤ 0.10 nats and top-1 agreement(A192, pool) ≥ 0.92. The
  anchor is s10, where the K-quant step itself cost +0.079 kl32 and −2.2 top-1 points (thorA vs mxfp4_A);
- **sanity:** KL > 0.

**Gate A(a) and A(b).** These are the loader's bit-exactness checks:
- (a) the A192 build through the new binaries must equal the base binaries;
- (b) pool + mask must equal a build of that mask cut from the pool.

**P2 is unchanged from PLAN.md.** It compares one domain mask assembled from the pool against a mask-specific
s07 build (paired kl32 CI within ±0.02), at a cost of ~20 GPU-h.

## Files

| file | role |
|---|---|
| `ops/s07_quant.py` | `pool`, `unit --pool`; `migrate_file`, `pool_bank`, shard staging |
| `ops/lib/replay.py` | `thor_experts` serves any keep-set from a pool build (for Python replays of masks) |
| `ops/dom/pool_corpus.py` | the calibration corpus |
| `ops/dom/assemble.py` | cuts a plain thor-v1 192 build from the pool. Byte layout identical to `st_io.write`: an identity cut of `thor/A192/layer-01` and `layer-03` reproduced the files' sha256 |
| `ops/dom/run_pool.sh` | detached driver |
| `ops/tests/test_pool.py` | CPU tests |
