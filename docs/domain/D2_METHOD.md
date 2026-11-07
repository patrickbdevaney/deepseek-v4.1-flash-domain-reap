# D2 domain-mask method, revised after the 2026-11-02 literature audit

Evidence: a literature audit (selection objectives, calibration and eval, healing and merging; the
audit notes are not part of this release, arXiv ids are cited inline). This file is the
decision record: what we keep, change and add, and why. It supersedes PLAN.md §"What makes each mask as
good as it can be" items 3-4 where they differ.

## Verdict on the method as a whole

The audit supports the core:
- saliency on the unpruned teacher (REAP gate x norm + HOPE interactions);
- domain calibration with model-generated answers, 8k rows, chat template, a thinking + chat mix and
  >=10% tool tokens;
- an additive domain/general mix with a protected general slice;
- uniform 192 per layer;
- shipping the masks unhealed.

Domain masks at 50% have matched or beaten their parent in-domain:

| result | domain mask | parent |
|---|---|---|
| PreMoE, DeepSeek-R1 128/256 | 76.52 | 75.51 |
| EASY-EP, DeepSeek-V3, multi-domain mask | 54.92 | 54.08 |
| REAP, Kimi-K2, SWE-bench Verified | 0.576 | 0.554 |

Pure single-domain masks lose 9-26 points out of domain, which is why the general share and the gates
stay.

## Changes for D2 (all CPU or within the existing GPU plan; fit the 10-day window)

| # | change | why (evidence) | cost |
|---|---|---|---|
| 1 | **Super experts are a hard keep in every mask** (s04's `super_expert_candidates` guard plus the drift_guard position-protect list, reused) | removing 10 of 15,677 experts took DeepSeek-R1 from 75.64 to 1.81 (2507.23279); set is calibration-independent | 0 |
| 2 | **Normalise each source's per-layer scores before weighting** (each bucket's saliency and HOPE F scaled to unit mass per layer, then mixed) | EASY-EP / PreMoE mix normalised scores; without it the weights act on token counts and score scale (our GLM corpus-imbalance finding) | 0 |
| 3 | **Candidate grid adds a routing-mass criterion and a self-trace on/off axis.** No candidate may subtract or divide by a general score. | best criterion differs by family (routing mass beat REAP by 12.8 on Gemma; a domain/general ratio score collapsed HumanEval+ to 0.20, 2607.16721). Self-traces come from the pruned A192, while every primary source calibrates on the UNPRUNED model's answers (EASY-EP: own answer 61.1 vs gold 53.9 vs question-only 50.9 on LCB). The unpruned 763B cannot be served on Thor to regenerate them, so the bias is measured: a candidate with the selftraces sub-area weighted 0, and Jaccard vs the default | CPU minutes |
| 4 | **Domain weight stays in {0.35, 0.45, 0.6}; nothing above 0.6** | at 50% the general share costs ~0.5-1 pt in-domain, pure specialisation costs 9-26 out of domain; the specialist edge only grows past ~60% pruning | 0 |
| 5 | **Stability check before trusting HOPE at 384 experts.** Solve each domain on two disjoint sub-area halves, compare per-layer Jaccard. If below ~0.9, shrink the domain F toward the pooled all-domain F | HOPE needed ~10k prompts for stable masks at 256 experts; cross-calibration-set Jaccard was only ~0.6 | CPU minutes |
| 6 | **Router-entropy check** from banked routing (biased selection distribution) vs 7.30 bits (0.85 x log2 384). If routing is near-flat, add a domain-fair allocation candidate | flat-routing models invert perplexity-based selection (gpt-oss-20B); peaked ones don't | CPU minutes |
| 7 | **Ranking: kl32/top-1 is a FILTER, not the ranker.** Report kl32 with a row-grouped bootstrap CI; candidates whose CIs overlap are ties settled by task evals | among near-baseline variants, KL/top-1 vs benchmark correlation was ~0 in all 14 metric variants (2606.19558); matches our GLM v2 result | 0 |
| 8 | **Task evals are paired and pooled.** Every finalist runs on the same items, same settings, same seqmax; per-item differences; a domain's suites pooled to >= 300 items, preferring suites where A192 passes 30-70%; saturated suites (VerilogEval v2, RTLLM) are regression gates only | a 100-300-item suite resolves only 5-8 pt gaps; 5 pt needs ~300-600 paired items | GPU, inside E |
| 9 | **Free-running gate for the top 3 per domain:** each mask generates ~200 domain answers; one teacher-replay pass scores teacher NLL and the rate of tokens outside the teacher's top-32 | replay on teacher text cannot see generation drift, loops or DSML breakage | GPU, ~1 h per mask |
| 10 | **Polymath:** hard-keep super experts, then round-robin the top 4-6 experts per domain per layer (~50-70 of 192), fill the rest with HOPE on the sum of normalised domain F; control = plain normalised-sum HOPE. First count per-layer union and conflicts of the domain masks | PreMoE multi-domain 76.02 vs per-domain 76.52; TB-Coverage +4.6-6.4 generic at 50% (rule not isolated, hence the control) | CPU |
| 11 | **Hallucination/omission items in the evals of fact-heavy buckets** | pruned models' hallucination rose while ROUGE-style scores stayed flat | eval authoring |

## Not adopted (with reason)

- **Expert merging instead of dropping:** loses to dropping on generative tasks at 50% (REAP 0.780 vs
  M-SMoE 0.590 vs HC-SMoE 0.543 on Eval+; REAM's own paper REAP 62.8 > REAM 60.7). It also breaks the
  bit-exact MXFP4 master.
- **Layer-adaptive allocation:** ~0 with REAP-type scores (code 0.585 to 0.595; it cut EASY-EP's LCB
  61.1 to 44.3). The uniform 192 runtime constraint costs little.
- **Reconstruction (OptEP), Fisher, gradient and causal scores:** they need new hidden-state or backward
  passes on the 763B teacher, the gains come at retention below 40%, and the one causal estimate had a
  split-half r of 0.05.
- **Answer-token weighting (EASY-EP):** needs a per-token saliency breakdown the bank does not keep.
  Revisit for a later pass.
- **Slot redirection (ConMoE, MoNE):** keeps expert bytes but needs a server dispatch change, and the
  gains are small (71.19 vs 70.43, MC only). Post-ship option.

## Healing (stage F), revised

- **Ship the masks unhealed.** At 50%, small-budget healing recovers a minority of the gap: ~25-50% at
  ~3M tokens (extrapolated from LoRA r64 attention + shared + routers, 2607.16721). Near-parity has only
  been shown at ~100M tokens (SlimWise: BFCL 0.759 to 0.802 vs parent 0.811), which is 8-16 Thor-days
  per mask.
- **Router-only training is not a stage on its own.** It gains about 0 (+0.17 pp, worse at 50%;
  2603.02217, 2609.06076), and our selection bias gets no gradient anyway. Routers train only inside
  the LoRA scope, at 0.1x the learning rate.
- **KD targets:** top-32 plus a residual-mass bucket. Top-k-only KD is biased (Sparse Logit Sampling,
  ACL 2025).
- **Data:** on-policy rollouts of the mask model, scored by teacher logits, beat SFT/KD/SeqKD 1.6-4.4x
  (ShortOPD). Keep CE + KD combined, and a 70/30 domain/general mix (general ppl 30.9 back to 9.4).
- **Train through the exact serving weights** (Thor K-quant build), adapter unmerged. Every healed
  number is gated by an adapter-loaded check plus a byte diff of the frozen experts. Merging into
  FP4 loses up to 39 pp of the adaptation, and adapters do not transfer across quant drift.
- **Adapter layout:** one shared adapter across masks first (10-15M tokens, 1-2.5 Thor-days); per-mask
  adapters only where a gate fails. This is an extrapolation, so test it on one mask first.
- **Inside the 10 days:** at most one pilot heal (~3M tokens, 6-12 h) on one priority mask, and only if
  the GPU is free after E.
