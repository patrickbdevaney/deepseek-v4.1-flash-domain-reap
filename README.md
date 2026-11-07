# DeepSeek-V4.1-Flash domain REAP masks (gamedev, mlsys, robotics, quantum)

Domain-specific expert keep-sets for **DeepSeek-V4.1-Flash-1017** and the pipeline that selects them.
Each mask keeps **192 of the 384 routed experts in each of the 40 MoE layers**. Experts are scored on
the **unpruned teacher's** saliency (REAP gate x norm, HOPE expert-interaction matrix, routing mass)
for a domain, mixed with the saliency of the original general-purpose calibration buckets (general,
code, agentic and six others) so the mask stays usable outside its domain.

This repository is the domain track of
[deepseek-v4.1-flash-reap-thor](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor)
(the "parent repo"), extracted and restricted to four public buckets:

| bucket | scope |
|---|---|
| `gamedev` | Rust / Bevy game development |
| `mlsys` | AI/ML research engineering, CUDA kernels, inference serving |
| `robotics` | simulation, control, planning, manipulation/locomotion, VLA policies, ROS 2 |
| `quantum` | quantum computing: Qiskit 2.x, Cirq, PennyLane, CUDA-Q, OpenQASM 3, Q#, quantum error correction (Stim, PyMatching) |

Licence: MIT (same as the parent repo), see `LICENSE`.

## What a mask is

A keep-set JSON, in the format of `ops/lib/keepset.py`:

```json
{"sha256": "<fingerprint of the canonical keep-set>", "n_keep": [192], "bucket": "mlsys",
 "candidate": "hope_w45_pf4_st1", "keep": {"0": [3, 7, ...], "1": [...], ..., "39": [...]}}
```

`keep` maps every decoder layer (0..39) to its 192 kept routed-expert ids (ascending). The fingerprint is
the sha256 of the canonical JSON; every consumer re-derives it and refuses a mismatch.

## How the masks are made

Hardware assumption for every GPU stage: **one NVIDIA Jetson AGX Thor** (sm_110, ~117 GiB usable unified
memory, NVMe scratch plus a large cold-store disk). Every GPU stage runs alone: no inference server or
other model-loading process beside it, and it refuses to start below ~100-108 GiB `MemAvailable`. CPU
stages are capped (`systemd-run --scope -p MemoryMax=6G nice -n 19`). Long stages run detached
(`setsid nohup ... &`).

| stage | what | code | cost |
|---|---|---|---|
| D0 | domain calibration corpora: 50 rows x 8,192 tokens per bucket, packed per sub-area, 10% held out, >= 10% tool-call tokens, licence recorded per source | `ops/dom/prep.py` (git clones, GitHub threads), `ops/dom/corpus_spec_dom.py` (sources and sub-areas), `ops/s02_calib_corpus.py` with `V41_DOM=1`, `ops/dom/run_d0.sh` | CPU + network |
| D0t | self-traces: the 192-expert A192 spine answers held-out domain prompts; the answers become a 15% calibration slot | `ops/dom/selftraces.py`, `ops/dom/render.py`, `ops/dom/server_window.sh`, `ops/dom/selftrace_all.sh` | GPU (server window) |
| D1 | teacher saliency passes: the unpruned MXFP4 teacher over the domain corpus, banking REAP/HOPE statistics per sub-area plus teacher top-32 logits and routing | `ops/s03_saliency.py` with `V41_DOM=1 V41_DOM_PASS=<p>`, `ops/dom/passes.py` (split / retire / merge), `ops/dom/run_d1.sh`, `ops/dom/run_d1_passes.sh` | GPU, ~15-20 h per 4-bucket pass |
| D2 | selection: per bucket, criteria {hope, reap, mass} x domain weight {0.35, 0.45, 0.6} x general-protect slice {0.02, 0.04, 0.06} x self-traces {on, off}; super experts and position-protected experts are a hard keep | `ops/dom/select_dom.py`, `ops/dom/d2lib.py`, `ops/dom/run_d2.sh` | CPU, minutes |
| D2 replay | teacher-routed held-out replay of the shortlisted candidates: kl32 / top-1 with row-cluster bootstrap CIs; the gate (domain d_kl32 CI hi < 0 vs A192 and general kl32 <= A192 + 0.10) | `ops/dom/d2_gpu/replay_filter.py` (`plan`, `run`, `decide`); `ops/dom/d2_gpu/freerun_gate.py` (free-running gate) | GPU, ~15 min per arm |
| P | the 384-expert K-quant pool (`POOL384`): every expert GPTQ-rounded once with A192's per-layer types, so any 192-mask is a pure selection with A192's bytes | `ops/dom/pool_corpus.py`, `ops/s07_quant.py pool`, `ops/dom/run_pool.sh` | GPU, ~40 h once |

Method and decision records: `docs/domain/PLAN.md`, `docs/domain/D2_METHOD.md`, `docs/domain/P_POOL.md`,
`docs/domain/DOMAINS.md`; diagnostics of the original selection run: `docs/domain/D2_DIAG.md`; per-bucket
source catalogues with verified licences: `docs/domain/sources/{gamedev,mlsys,robotics,quantum}.md`;
operational detail: `ops/dom/README.md`.

### Running it

```
export V41_WORK=/path/to/v41-work                 # every output (default $HOME/v41-work)
export V41_SRC=/path/to/DeepSeek-V4.1-Flash-1017   # the teacher checkpoint (default $HOME/DeepSeek-V4.1-Flash-1017)
export V41_ENGRAM_DIR=/path/to/engram-shards       # D1 / free-run gate: the teacher's Engram shard directory
export V41_COLD=/mnt/v41-cold                      # cold store for parked rows, the pool and the MXFP4 master
export V41_DOM_SRC=$V41_WORK/dom_src               # bare git clones (optional)

# D0: sources, then the corpus (CPU)
setsid nohup ops/dom/run_d0.sh > ops/logs/dom_run_d0.log 2>&1 < /dev/null &
# D0t: self-traces through the A192 server (GPU window), then rebuild the pools whose files changed
setsid nohup ops/dom/selftrace_all.sh > ops/logs/dom_selftrace_all.log 2>&1 < /dev/null &
# D1: split into passes and run s03 per pass, then merge (GPU, one pass at a time)
setsid nohup ops/dom/run_d1.sh > ops/logs/dom_run_d1.log 2>&1 < /dev/null &
# D2: CPU selection
ops/dom/run_d2.sh
# D2 replay filter (GPU window)
.venv/bin/python ops/dom/d2_gpu/replay_filter.py plan && \
  .venv/bin/python ops/dom/d2_gpu/replay_filter.py run && \
  .venv/bin/python ops/dom/d2_gpu/replay_filter.py decide
# P: the 384-expert pool (GPU, ~40 h, resumable per layer)
setsid nohup ops/dom/run_pool.sh > ops/logs/dom_pool.log 2>&1 < /dev/null &
```

Outputs: candidate keep-sets in `$V41_WORK/dom/d2/<bucket>/<candidate>.json`, the replay verdicts in
`$V41_WORK/dom/d2/replay/filter.json`, the pool in `$V41_WORK/thor/POOL384`.

The D1 and P drivers use the parent repo's memory guard (`dsv41-memguard` user unit from
`ops/memguard.sh`) and a `$V41_WORK/GPU_WINDOW.lock` convention shared by every GPU job.

Python: numpy, torch, tilelang (D1 kernels), safetensors, tokenizers, datasets, huggingface_hub,
pyarrow. The checkpoint's own `encoding/` and `inference/` directories are imported from `$V41_SRC`.

## How a mask is applied

- **MXFP4 master (HF format, vLLM/transformers):** `ops/s05_surgery.py run --keepset <mask.json> --no-upload`
  keeps the chosen experts' packed E2M1 bytes and E8M0 scales byte for byte, slices the router rows and
  bias to match and renumbers the kept experts 0..191. Bit-exact, verified tensor by tensor against the
  source. (Same stage as the parent repo's A192 master; it writes ~152 GiB to `$V41_COLD`.)
- **Thor CUDA build:** the parent repo's `v41_server` loads `POOL384` plus a keep-set JSON and reads only
  the kept experts, so switching domains is a restart with a different JSON at the same bytes, kernels
  and speed as A192. `ops/dom/assemble.py <mask.json> --out <dir>` alternatively cuts a plain 192-expert
  Thor build from the pool (107 GiB, identical bytes).

## Results

TODO: filled in after the D2 replay gate and the E evals have run on the Thor build.

| bucket | shipped candidate | keep-set sha256 | domain d_kl32 vs A192 (CI) | general kl32 vs A192 | domain task suite vs A192 |
|---|---|---|---|---|---|
| gamedev | TODO | TODO | TODO | TODO | TODO |
| mlsys | TODO | TODO | TODO | TODO | TODO |
| robotics | TODO | TODO | TODO | TODO | TODO |
| quantum | TODO | TODO | TODO | TODO | TODO |

## Dependencies on the parent repo

This repo carries the Python pipeline code the domain track imports (`ops/lib/*`, `ops/corpus_spec.py`,
`ops/s02_calib_corpus.py`, `ops/s03_saliency.py`, `ops/s05_surgery.py`, `ops/s07_quant.py`, at the
versions the domain track used). It does **not** copy:

- `cuda/`: the Thor CUDA inference server (`v41_server`), its build scripts and gates (`gate_pool.sh`),
  the K-quant kernels. Needed for self-traces (D0t), the pool's P1 gate and serving masks.
- The A192 artifacts the selection builds on: the 9-bucket teacher bank (`$V41_WORK/s03`), the A192
  keep-set and guards (`$V41_WORK/s04/keepset.json`, `position_protect.json`, `candidates.json`), the
  A192 s07 calibration rows (`$V41_WORK/s07/thor_replay/run.json`) and the A192 Thor build
  (`$V41_WORK/thor/A192`). These come from running the parent pipeline (s00-s07).
- `ops/memguard.sh`, the parent's stage-chaining shell scripts, `llamacpp/` (GGUF conversion).

Paths such as `cuda/...`, `golden/...` or `results/logs/...` in the docs refer to the parent repo.

## What is not included

- **Private owner transcripts.** The original run calibrated `gamedev` and `mlsys` partly on private
  owner agent transcripts. They are not released, and their rows and converter are removed from this
  repo's spec and code.
- **Private repositories.** The original `gamedev` corpus included the owner's private game
  repositories and their commit history (with a hold-out cutoff for evaluation). Not released;
  `OWNER_HOLDOUT["repos"]` is empty here.
- **GitHub issue/discussion threads.** `prep.py threads` can still build them, but no bucket of this
  release lists them, and no thread data is included.
- **Withheld buckets.** The original run used a larger frozen bucket list; the other buckets' source
  lists, catalogues, diagnostics and masks are deliberately unpublished. As a consequence, bucket ids,
  sub-area ids and the D1 pass grouping produced by this repo's spec differ from the banks behind the
  original masks, and the original `polymath` candidates (composed over all ten original tier-1
  buckets) are not reproduced here.
- **No data.** No corpora, banks, routing, logs or keep-sets are committed (`.jsonl`, `.npy`, `.npz`,
  logs are git-ignored).

Because private sources were part of the original gamedev and mlsys calibration, rebuilding from this
repo produces close but not identical masks for those two buckets.
