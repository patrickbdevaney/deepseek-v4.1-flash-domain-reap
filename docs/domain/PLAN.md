# Domain REAPs — plan (method record)

Reduced for the public release (2026-11): the method, phases and pre-registered gates of the
domain-REAP track, with only the four published buckets (gamedev, mlsys, robotics, quantum). The
original plan covered a larger domain set and an agent-harness track; those parts, and every
reference to private data, are omitted. The original REAPs of the parent repo (A192 keep-set
`1ace9b33…`, its Thor build, the MXFP4 master, GGUFs) are kept untouched by this track.

Provenance decision (owner, 2026-11-01, this track only): Claude-written text may be used for
calibration. Every such source is tagged `generator: claude` and can be switched off per build
(`--generator-policy no-claude`), so any mask can be rebuilt without it. Private owner transcripts
were used in the original run's calibration; they are not released.

## Domains (published subset)

| id | domain | primary public data | verifier for eval |
|---|---|---|---|
| gamedev | Rust / Bevy game development | Bevy + ecosystem repos, Rust book/reference/std | `cargo check/test`, Rust-SWE-bench Bevy slice |
| mlsys | AI/ML research engineering + CUDA kernels | CUTLASS/flashinfer/llama.cpp/vLLM/Triton sources, Nemotron-SFT-CUDA, ML arXiv | nvcc + numerics vs oracle, KernelBench-style |
| robotics | sim, control, planning, manipulation/locomotion, VLA policies, ROS 2 | MuJoCo/Newton/Drake/OMPL/MoveIt/ROS 2 docs and code, RAX robot sessions, cs.RO arXiv | MuJoCo-based task suites (sources/robotics.md C) |
| quantum | quantum computing (Qiskit 2.x, Cirq, PennyLane, CUDA-Q, OpenQASM 3, Q#, QEC) | Qiskit/Cirq/Qualtran/OpenFermion/PennyLane/CUDA-Q sources and docs, Stim/PyMatching/cudaq-qec, OpenQASM, Q# katas, quant-ph arXiv | Qiskit HumanEval Hard, QuanBench+, QCircuitBench verifiers, Stim/PyMatching QEC tasks (sources/quantum.md C) |

Per-domain source catalogues (calibration / fine-tune / eval, licences verified) live in
`sources/<bucket>.md` and are reconciled into the domain corpus spec before each pool is finalized.

## Architecture decision: one expert POOL, masks assembled at zero GPU cost

The A192 build allocates bits per LAYER (same K-quant type for every expert of a layer; `alloc` in
`thor/A192/*.json`). So a pool of all 384 experts GPTQ-rounded with the same per-layer types makes
every 192-mask a pure selection: same bytes as A192, no per-mask s07 (which costs ~20 GPU-h).
Pool Hessians come from the unpruned (teacher-routed, quantized-stream) pass instead of each
mask's pruned stream — a small approximation, measured once (gate P2) against a mask-specific s07
build of one domain mask.

## Phases and gates (pre-registered)

**D0 corpora (CPU/network).** Per domain ≥400k routed tokens of 8k rows, s02-compatible (`local:`
jsonl of OpenAI-format messages rendered by the checkpoint encoding, `git:` whole-repo rows, HF
datasets). 10% held-out per domain. Licence recorded per source.

**D1 domain saliency pass (GPU, ~15-20 h per 4-bucket pass).** s03 over the domain buckets only,
same teacher, same statistics; accumulators bank in `$V41_WORK/dom/s03_<pass>/bank` and are
additive with the original 9-bucket bank (same teacher, same expert ids). Teacher top-32 logits
for the domain held-out rows give each domain its own eval split. D1 runs as passes over bucket
subsets (banks merged) because the s03 scratch (~55 KB/token) exceeds the NVMe otherwise.

**D2 masks (CPU).** Per domain d: score = HOPE/mass objective over a weighted bucket mix
w_d = {d: 0.45, code+agentic: 0.25, general: 0.15, rest of the original 9: 0.15} with the
general-protect slice. Plus one `polymath` mask over all domains + ballast. Gate D2: on the
domain's held-out split (teacher-routed replay, s04 machinery) the domain mask beats A192 on kl32
with CI hi < 0, and general kl32 is no worse than A192 + 0.10. Revised in `D2_METHOD.md`.

**P pool (GPU, ~35-40 h).** s07 `thor` units with keep = all 384, per-layer alloc copied from
A192. Output `$V41_WORK/thor/POOL384` (~200 GiB; cold store if the NVMe cannot hold it).
Gate P1: pool expert tensors for A192's keep-set reproduce A192 serving top-1 vs teacher within
0.005. Gate P2: one domain mask assembled from the pool vs the same mask through a full s07 —
paired kl32 difference CI within ±0.02, else per-mask s07 for the winners. Details: `P_POOL.md`.

**A assembly + server (CPU + small CUDA change).** `ops/dom/assemble.py <keepset>` writes a
thor-v1 build (copies of pool tensors + sliced router rows and bias). Server: accept a build whose
keep-set differs (n_routed 192 unchanged — expected zero kernel changes). DSpark drafter: native
head first; per-mask acceptance measured, refit only if acceptance drops >10%.

**E eval (GPU).** Per mask: teacher-agreement on its own and every other domain split + general;
domain task suites with verifiers (KernelBench-style for mlsys; Rust/Bevy SWE tasks for gamedev;
MuJoCo task suites for robotics; Qiskit HumanEval Hard / QuanBench+ / QEC tasks for quantum). Publish nothing that has not
passed its gate.

**F fine-tune (optional, after E).** LoRA on attention + shared expert, routers trained, routed
experts frozen. Masks ship without it (D2_METHOD.md "Healing").

## What makes each mask as good as it can be — added gates

Exhaustive current catalogues are necessary, not sufficient. A mask is bounded by five things:
1. Sub-area stratified calibration. Every catalogue's sub-areas (e.g. robotics: locomotion/WBC,
   sim and scene description, ROS 2, motion planning, control/MPC, manipulation/VLA, RL rewards,
   task planning, sim-to-real) get a token floor inside the bucket (>= 8% of the bucket each, rest
   by value). An expert survives only if its sub-area is in the sample. Dose-response check: a
   second draw of the same bucket at 2x tokens must change < 5% of the mask, else the bucket gets 2x.
2. Self-traces. Each bucket adds ~15% rows of the A192 spine's OWN agent traces on held-out domain
   prompts (served on Thor before D1, greedy, verifier-filtered where possible); literature: +3-4
   points from own-output calibration.
3. Objective sweep per domain, not one fixed mix: domain weight in {0.35, 0.45, 0.6}, ballast
   (code+agentic+general) protect slice in {0.02, 0.04, 0.06}, several criteria — all CPU-cheap
   from the banks.
4. Selection on function, not proxies: candidates are ranked on per-sub-area held-out teacher
   kl32/top-1 (teacher replay, s04 machinery); the top 3 per domain are assembled from the pool and
   scored on the domain's executable eval suite (sources/<bucket>.md table C). Winner = best domain
   task score subject to general kl32 <= A192 + 0.10 and IFEval/tool-call parse rate not worse.
5. Stated ceilings: 192/384 uniform per layer (runtime constraint, ~5% vs adaptive), and a mask
   cannot add knowledge — it chooses what survives; out-of-domain loss is measured and reported.

Licence rule refinement (2026-11-01): NC/ND benchmarks MAY be used for private, non-commercial
EVALUATION only; they never enter calibration or training data, and published cards cite them
only as eval results.

The polymath mask's gate = mean per-bucket kl32 better than A192 (breadth), not depth in any one
bucket. The Thor server loads POOL384 + a keep-set json directly, so a mask costs a few KB.
