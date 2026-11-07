# Domain masks: the published buckets

Reduced for the public release (2026-11): method text and the rows of the four published buckets
(gamedev, mlsys, robotics, quantum). The original run used a larger frozen domain set whose other
buckets, and their source lists, are not part of this release. Source of truth for bucket order and
sub-areas in this repo: `ops/dom/corpus_spec_dom.py`; full design: `PLAN.md`.

## What a shipped mask is

A mask is a **keep-set**: for each of the 40 MoE layers, the ids of the 192 routed experts (of 384) the
model keeps. It is a few KB of JSON, in the same format as `s04/keepset.json` (the A192 mask). The same
file drops into both deployments:

| deployment | how the mask is applied | what is preserved |
|---|---|---|
| **MXFP4 master** (HF format, vLLM/transformers) | `ops/s05_surgery.py` with the keep-set: keeps the chosen experts' packed E2M1 bytes and E8M0 scales **byte for byte**, slices the router rows and bias to match, copies everything else as is | master truth: no re-quantization, verified tensor-by-tensor against the source bytes |
| **Thor CUDA build** (`v41_server`, parent repo) | the server loads one 384-expert K-quant **pool** plus the keep-set, so switching domains is a restart with a different JSON | same bytes per mask as A192 (~113 GiB envelope), same kernels, same speed |

Every mask is gated on both: the D2 selection gate measures the mask on the master-derived reference
(teacher-routed replay), and the E evals run it through the Thor build.

## How a domain mask is made (it is not a LoRA on A192)

1. **Saliency on the unpruned teacher.** D1 runs the full 384-expert MXFP4 master over each bucket's
   409,600-token calibration set and records, per expert and per bucket/sub-area, how much it
   contributes (REAP saliency, the HOPE 384x384 interaction matrix, heal gain), plus the teacher's
   top-32 logits and routing. A192's earlier choices play no part: an expert A192 dropped can come back.
2. **Selection per domain (CPU, minutes).** Score every expert on a weighted mix of buckets and keep
   the best 192 per layer. Baseline mix: the domain 0.45, code+agentic 0.25, general text 0.15, the
   rest of the original 9 buckets 0.15. A protected slice of general-text experts is always kept, so
   ballast cannot collapse. The domain weight is swept over {0.35, 0.45, 0.6} and the protect slice
   over {0.02, 0.04, 0.06}, with several criteria (D2_METHOD.md).
3. **Pick on function, not proxies.** Candidates are ranked on held-out teacher kl32/top-1 per sub-area;
   the top 3 are scored on the domain's executable evals. The winner must keep general kl32 within
   A192 + 0.10 and must not lose IFEval or tool-call parsing.
4. **Tool calls** are not a separate re-REAP: every bucket's calibration text carries >= 10% tool-call
   tokens plus `tool-use`, `agentic-chat` and self-trace sub-areas, so the experts that emit tool calls
   score inside every domain.
5. **LoRA (stage F) is optional and later:** per-mask adapters, trained on top of a shipped mask, to heal
   residual pruning damage. Masks ship without it.

## The published buckets

| bucket | scope | sub-areas (this release) |
|---|---|---|
| **gamedev** | Rust / Bevy game development | bevy-engine, bevy-ecosystem, rust-language (+ tool-use, agentic-chat, selftraces) |
| **mlsys** | AI/ML research engineering + CUDA kernels | cuda-kernels, compilers-dsl, inference-serving, training-dist, quantization, papers (+ tool-use, agentic-chat, selftraces) |
| **robotics** | sim, control, planning, manipulation/locomotion, VLA, ROS 2 | locomotion-wbc, sim-scene, ros2-middleware, motion-planning, control-mpc, manipulation-vla, rl-rewards, task-planning, sim2real-est (+ tool-use, selftraces) |
| **quantum** | quantum computing: SDKs, circuits, variational/QML, algorithms, error correction | sdk-docs, variational-qml, algorithms, qec, papers (+ tool-use, agentic-chat, selftraces) |

In the original run, gamedev also had two owner sub-areas (private repositories and their commit
history) and gamedev/mlsys had private agent transcripts in their `tool-use` / `agentic-chat`
sub-areas. Those sources are not released and are removed from this repo's spec.

Every bucket also carries `selftraces` (the A192 spine's own answers), `tool-use` and, in most,
`agentic-chat` sub-areas, so tool calling survives the prune. Each bucket is 409,600 calibration
tokens with ≥10% tool-call tokens.

A `polymath` mask (one mask composed from all bucket banks, no extra data, optimised for breadth) is
implemented in `ops/dom/select_dom.py`; the original polymath candidates were composed over buckets
that include withheld ones and are not part of this release.

## Gates a mask must pass before it ships (pre-registered, PLAN.md)

- **D2 (selection):** on the bucket's held-out split, the domain mask beats A192 on kl32 with the
  CI upper bound < 0. General kl32 is no worse than A192 + 0.10. IFEval and tool-call parse are not worse.
- **P1/P2 (pool):** the 384-pool K-quant matches the per-mask build closely enough that selection from
  the pool is a valid approximation.
- **E (evals, Thor build):** the bucket's suites against A192 on the same server.
- **"SOTA on a domain"** is only claimed where the mask beats the best other model that fits one Thor
  on that domain's suites, measured on our hardware, and the model card says exactly that.
