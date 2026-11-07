# ops/dom: domain-REAP track (docs/domain/PLAN.md)

Reduced for the public release: the published buckets are gamedev, mlsys, robotics and quantum. The
original run used a larger frozen bucket list, private owner transcripts and private repositories;
none of those are in this repo (see the top-level README.md, "What is not included").

`V41_DOM=1` switches `ops/s02_calib_corpus.py` and `ops/s03_saliency.py` to the domain track.
Without it, both stages behave exactly as in the parent repo: same spec, same paths, same `--dry-run`
and `status` output, same module constants. Every domain-only branch is guarded by `DOM`.

| | original (unset) | `V41_DOM=1` |
|---|---|---|
| spec | `ops/corpus_spec.py`, 9 buckets | `ops/dom/corpus_spec_dom.py` (this release: 4 buckets) |
| s02 output | `$V41_WORK/corpus` | `$V41_WORK/dom/corpus` (`V41_DOM_DRAW=2`: `dom/corpus_draw2`) |
| s02 ledger / probe | `ops/logs/s02_ledger.json`, `s02_probe.json` | `ops/logs/dom_s02_ledger.json` (`dom_s02_draw2_ledger.json`), `dom_s02_probe.json` |
| s02 pool rebuild | one spec hash for the whole corpus | one hash per pool; `--pools b1 b2` builds only those pools and skips assembly |
| rows.jsonl | `bucket`, `bucket_id`, `heldout` | also `subarea`, `sub_id`; `heldout` is set per sub-area; tool-calling docs carry `"tool": true` |
| extra s02 output | none | `$V41_WORK/dom/prompts/<bucket>.jsonl` (self-trace prompts, below) |
| s03 corpus / work dir | `$V41_WORK/corpus`, `$V41_WORK/s03` | `V41_DOM_PASS=<p>`: `dom/corpus_<p>`, `dom/s03_<p>` (unset: `dom/corpus`, `dom/s03`) |
| s03 ledger | `ops/logs/s03_ledger.json` | `<work dir>/s03_ledger.json` |
| s03 accumulator rows | one per bucket | one per **sub-area** (`spec.SUBAREAS`), and the bank stores `buckets` + `subareas` |
| generator policy default | `no-claude` | `with-claude` (`--generator-policy no-claude` turns off every Claude source) |

## Buckets
`gamedev, mlsys, robotics, quantum` (this release's order; bucket and sub-area ids differ from the original frozen list).

- **Reconciled.** Every bucket is reconciled against its own `docs/domain/sources/<bucket>.md`
  (2026-11-01). The comment above each pool lists what was added, skipped and rejected. Every pool is
  hashed on its own, so a later change rebuilds only that pool.
- **Owner repos (optional, none in this release).** `OWNER_HOLDOUT["repos"]` in the spec can name git
  checkouts that `prep.py localrepos` turns into source-file rows (the tree of the last commit before the
  cutoff) and commit-diff rows. **Every commit dated on or after the cutoff, on any ref, is reserved for
  evaluation and is never read:** only commits whose author AND committer dates precede it are kept. The
  cutoff and the tree commit used are recorded in `prep_manifest.json` and in the corpus manifest
  (`owner_repo_holdout`). The original run used private repositories here; they are not released.
- **Eval sets never feed calibration.** No source named in table C (evaluation) of a bucket's catalogue
  is in that bucket's plan (dataset ids, repos, benchmark names); removed: KernelBench (mlsys), cap-x,
  CaPtain, CaPtain-Harness and robocode (robotics). The list is in `EXCLUDED["gated_or_eval_only"]`.
- **Stack Exchange, one rule for all buckets.** Only the pre-2024 `HuggingFaceH4/stack-exchange-preferences`
  dump (CC BY-SA), as calibration only, never fine-tune data. Its docs carry `"sa": true,
  "calibration_only": true` in `rows.jsonl`, the manifest reports `calibration_only_tokens` per bucket,
  and it is never a self-trace prompt source.

Each bucket has 50 rows of 8,192 tokens (409,600 tokens), packed per sub-area:
- **Sub-areas.** Every source maps to exactly one sub-area of its bucket (`_SA` in the spec, matched
  on repo / hf_id / `:config`). Chat material that spans topics (GitHub threads, SE Q&A, chat SFT sets)
  forms an `agentic-chat` sub-area where a bucket has such material.
- **Row budget.** Each topical sub-area gets max(8% of the bucket, its value share) in whole rows,
  where value is the summed source weight. `tool-use` gets at least 12% (6 rows). Rows never mix
  sub-areas.
- **Held-out split.** Each sub-area holds out its own last 10% of rows (at least 1). Rows are
  interleaved smoothly across sub-areas, so any prefix keeps the same mix.
- **Self-traces.** 15% of each bucket (8 rows) is reserved for `local:dom/selftraces/<bucket>.jsonl`.
  While that file is missing, its rows flow to the later sub-areas in proportion to their planned
  rows. Rebuild a pool after its file lands (`--pools <bucket>`).
- **Dose-response draw.** `V41_DOM_DRAW=2` builds `dom/corpus_draw2` with 2x rows. Every source
  resumes after the rows that the first draw read.

## Tool-calling DSML rows in every bucket (≥ 10% of tokens)
The original A192 calibration rendered chat through `corpus_spec._messages`. That function drops
`tool_calls`, `tools` and `reasoning_content`, so REAP never routed a DSML tool-call token. The
domain track fixes this in two ways:

1. **Every chat source renders through `corpus_spec_dom._chat`.** That covers `_auto` on chat
   columns, `_tooluse`, and threads / self-traces via `L(..., _chat)`. `_chat` keeps `tool_calls`,
   tool results (`role: tool`), `reasoning_content` and `tools`, and moves a row-level tool schema onto
   the leading system turn, which is the only place `encoding.py` renders tools. No domain source uses
   `corpus_spec._messages`.
2. **Each bucket has a `tool-use` sub-area of at least 12% of its rows.** Its sources use
   `_tooluse`, which keeps only rows with structured `tool_calls`, or they are tool-calling sets
   by construction. s02 tags every tool-calling document `"tool": true` and **refuses to finish a
   pool whose tool-calling tokens are under 10%** of its tokens. The achieved share per bucket is
   in `manifest.json` → `tool_token_share`.

| bucket | tool-use sources (in addition to the generic pair) |
|---|---|
| gamedev | none public (the original run used private agent transcripts here) |
| mlsys | `nvidia/Nemotron-SFT-CUDA-v1` (every row calls tools); the original run also used private agent transcripts |
| robotics | `robot-agent-exploration/robot-session-logs` transcripts |
| all | **generic pair.** `nvidia/Nemotron-SFT-Agentic-v2` `tool_calling` (CC-BY-4.0, read line by line from the hub jsonl), filtered to the bucket by `TOOL_RX[bucket]` on its tool schema and first user turn (weight 0.10; 0.03 where the bucket has its own set). Then `nvidia/Nemotron-Agentic-v1` `tool_calling` (CC-BY-4.0) unfiltered, which takes any shortfall. |

quantum has no permissive bucket-specific trajectory set with structured `tool_calls`, so it relies on
the generic pair. Bucket-specific tool trajectories arrive with the A192 self-traces.

## Self-trace prompt pool (`$V41_WORK/dom/prompts/<bucket>.jsonl`)
While building a pool (draw 1 only), s02 writes about 300 user-turn task prompts per bucket.
`selftraces.py` (via `server_window.sh selftraces`) runs the A192 spine greedily on them to produce
`dom/selftraces/<bucket>.jsonl`.
- **Fields.** `{"bucket", "subarea", "source", "prompt", "heldout", "pool_row", "in_corpus", "needs_tools", "tools"}`.
- **Origin.** Each prompt is the first user turn of a chat/instruction/agentic document that went
  into the pool: issue descriptions, SE questions, task statements, agent tasks.
- **Stratified.** Prompts are spread over sub-areas by planned rows, round-robin over the sources
  inside a sub-area. A thin sub-area's unused quota flows to the next one.
- **Self-contained first.** Prompts that presume a live session (open the repo, run this, look at
  my screen, ...) are dropped. At least 80% of each bucket's prompts are self-contained tasks answerable
  without tools; at most 20% (`needs_tools: true`) come from tool-calling documents and carry their tool
  schema in `tools`. The generic tool sets are never used as prompt sources (off-domain). When the pool's
  own documents give too few prompts, s02 tops up from the same chat sources past the rows the pool read
  (`in_corpus: false`, up to 3,000 rows per source). Stack Exchange (calibration only) is never a
  prompt source.
- **Filtered.** Prompts are deduplicated (normalised-text sha1), at least 40 characters and at
  most 2,048 tokens. Self-trace sources are never used, and neither is any source named in table C
  (evaluation) of `docs/domain/sources/<bucket>.md`.
- **`heldout: true`** marks a prompt whose document sits in a held-out corpus row. Do not reuse
  those prompts for training traces.

**Self-trace row schema** (one JSON object per line, `local:` chat rows): `id`, `sort` (sha1 of the id,
a deterministic shuffle), `source_file`, `split`, `split_key`, `domain`, `subarea`, `generator`,
`thinking`, `n_tokens` (exact: the checkpoint encoding + tokenizer, `render.py`) and `messages`
(OpenAI format; tool schemas only on a leading system message).

## Files
- `corpus_spec_dom.py`: sources, sub-areas, licence and generator policy, tool-use, and `PASSES`.
- `prep.py`: fetches the non-HF sources.
  - `clone`: bare shallow clones into `$V41_DOM_SRC` (default `$V41_WORK/dom_src`). Most use
    `--filter=blob:limit=512k`. Huge trees are `lazy`: cloned `blob:none`, with only the first
    `max_files` selected blobs prefetched.
  - `threads`: GitHub issues and discussions turned into chat rows (needs an authenticated `gh`;
    no bucket of this release lists the resulting `local:dom/threads` rows).
  - `localrepos`: `OWNER_HOLDOUT` repos from git (empty in this release).
  - Each step writes `$V41_WORK/dom/prep_manifest.json`.
- `passes.py`: `plan`, `split`, `retire`, `merge` for D1 in passes (below).
- `selftraces.py`, `render.py`, `server_window.sh`, `selftrace_all.sh`: self-trace generation (D0t).
- `select_dom.py`, `d2lib.py`, `run_d2.sh`: D2 CPU selection.
- `d2_gpu/replay_filter.py`, `d2_gpu/freerun_gate.py`: D2 GPU gates.
- `pool_corpus.py`, `run_pool.sh`, `assemble.py`: the 384-expert pool and per-mask builds
  (docs/domain/P_POOL.md).
- **Git sources.** Each file of the upstream ref becomes one document (`git cat-file`). Files are
  visited in hashed path order and filtered by include prefix and extension. CUTLASS `python/CuTeDSL`
  and `docs/pythonDSL` are under an NVIDIA EULA and are excluded.

## D0 run (CPU + network)
`ops/dom/run_d0.sh [bucket ...]` is the memory-guarded driver. Before every heavy step it waits
(polling every 60 s) until `$V41_WORK/GPU_WINDOW.lock` is absent, no `v41_server` / `llama-server`
runs, and MemAvailable is at least 40 GiB. It then runs clone, localrepos, threads for
`$D0_THREADS`, each pool (`--pools <b>`), and assembly when no buckets are named. The manual steps are:
```
cd <repo root>
export V41_WORK=/path/to/v41-work V41_SRC=/path/to/DeepSeek-V4.1-Flash-1017
export HF_DATASETS_CACHE=$V41_WORK/dom/hf_datasets_cache
setsid nohup .venv/bin/python ops/dom/prep.py clone > ops/logs/dom_prep.log 2>&1 < /dev/null &   # clones skip existing
V41_DOM=1 .venv/bin/python ops/s02_calib_corpus.py --dry-run
V41_DOM=1 .venv/bin/python ops/s02_calib_corpus.py --probe          # -> ops/logs/dom_s02_probe.json
V41_DOM=1 setsid nohup .venv/bin/python ops/s02_calib_corpus.py > ops/logs/dom_s02.log 2>&1 < /dev/null &   # stale pools, then assembly
V41_DOM=1 V41_DOM_DRAW=2 setsid nohup .venv/bin/python ops/s02_calib_corpus.py > ops/logs/dom_s02_draw2.log 2>&1 < /dev/null &  # optional
```
The pool hash covers a pool's sources, weights, filters, sub-area plan, the size of every `local:` jsonl
and per-source licence overrides, but not the body of a row function. So `--pools <all>` after a trace
window rebuilds exactly the pools whose files changed. After changing a `fn`, delete that pool's entry
from `ops/logs/dom_s02_ledger.json`.

**Live licence of git sources** (`git_licence`): every top-level LICENSE*/COPYING* file is classified.
GPL/LGPL/AGPL/GFDL are copyleft, allowed and flagged, never NC: the GPL object-code clauses
("occasionally and noncommercially", "only for noncommercial distribution") and the GFDL grant
"commercially or noncommercially" are stripped before the NC rule. s02 exits with `os._exit(0)` after a
successful DOM run, because leftover HF download threads aborted the interpreter at shutdown.

## D1 run: passes over bucket subsets
One pass needs ~55 KB of NVMe scratch per token. `PASSES` in the spec cuts the bucket order into
contiguous groups of at most 4 (`PASS_MAX`); with the four published buckets that is one pass, `p1`:

| pass | buckets | tokens | persistent scratch | + transient |
|---|---|---|---|---|
| p1 | gamedev, mlsys, robotics, quantum | 1,638,400 | 90.2 GB (84 GiB) | ~10 GB |

`passes.py plan` prints the real numbers from the built corpus.

**How the passes merge.**
- `split` copies each pass's rows out of the full corpus into `dom/corpus_<p>`, in global visit
  order. Every row keeps its global `bucket_id`, `sub_id` and held-out flag, and records `full_row`.
- The pass manifest keeps the full `buckets` / `subareas` lists. s03 therefore allocates
  accumulators for all sub-areas in every pass, and only that pass's sub-areas become non-zero.
- `merge` sums the pass banks layer by layer and takes the maximum for `fmax`, `gfmax`, `amax_*`.
  The result is exactly the bank one big pass would have produced.
- Routing, teacher top-32 and taps stay per pass.

**Running it.** D1 is GPU work, one process at a time, never beside any vLLM, llama-server or CUDA
server. `run_d1.sh` / `run_d1_passes.sh` wrap the steps below with a memory guard
(`$V41_ENGRAM_DIR` = the directory holding the teacher's Engram shards):
```
V41_DOM=1 .venv/bin/python ops/dom/passes.py split            # after the full s02 assembly
V41_DOM=1 .venv/bin/python ops/dom/passes.py plan
# per pass P, strictly one at a time:
mkdir -p $V41_WORK/dom/s03_$P
V41_DOM=1 V41_DOM_PASS=$P setsid nohup .venv/bin/python ops/s03_saliency.py run --kernels tilelang \
    --engram-dir "$V41_ENGRAM_DIR" > $V41_WORK/dom/s03_$P/s03.log 2>&1 < /dev/null &
# wait for "s03 complete" in that log, then:
V41_DOM=1 V41_DOM_PASS=$P .venv/bin/python ops/s03_saliency.py status
V41_DOM=1 .venv/bin/python ops/dom/passes.py retire $P        # frees ~70-90 GB; parks Engram rows on $V41_COLD
# after all passes:
V41_DOM=1 .venv/bin/python ops/dom/passes.py merge             # -> $V41_WORK/dom/bank/L*.npz
```
- **Memory.** `run` reclaims memory between GPU units and refuses to continue below 100 GiB
  MemAvailable (`--min-avail-gib`). Child units inherit `V41_DOM` and `V41_DOM_PASS`.
- **Never** use `--drop-shards` when teacher shards are symlinks: it unlinks them.
- **Draw 2.** The draw-2 corpus is split with `passes.py split --draw 2` (passes `draw2_<p>`, twice
  the tokens, so check `plan`) and merged with `merge --draw 2` into `dom/bank_draw2`.

### NVMe scratch per token (rates from the original A192 pass)
| item | bytes/token |
|---|---|
| residual stream `h.npy` (4 hc × 5120 × bf16) + `pre_mix` | 40,976 |
| Engram rows L1+L14 (fp8 [24,256] + scales) + hash | 12,864 |
| routing (40 layers × top-6 idx+w) | 960 |
| teacher top-32 logprobs/ids + lse + nll | 264 |
| **persistent** | **55,064** |

Transient use (journals for 8 rows, cross-layer state, held-out taps L37-39, bank) is about 8-12 GB.
After `retire`, a pass keeps only bank, teacher, routing and taps (about 2-3 GB).

## D2: CPU domain-mask selection (`select_dom.py`, `d2lib.py`, `run_d2.sh`)
Implements `docs/domain/D2_METHOD.md` items 1-7 and 10 on the CPU; diagnostics of the original run in
`docs/domain/D2_DIAG.md`. Reads the A192 bank (`$V41_WORK/s03`, from the parent pipeline) and every
**completed** pass bank (`dom/s03_<pass>/bank` + `routing/`) directly. It never writes `dom/bank`.

- **Candidates** per bucket: criteria {hope, reap, mass} x domain weight {0.35, 0.45, 0.6} x protected
  general slice {0.02, 0.04, 0.06} x self-traces {on, off}, named `<crit>_w<35|45|60>_pf<2|4|6>_st<1|0>`;
  default `hope_w45_pf4_st1`. Mix: domain w, code 0.125 + agentic 0.125, general 0.15, the six other original
  buckets share 1 - w - 0.40. Every source (each domain bucket, each original bucket) is normalised to unit mass
  per layer first. Hard keep: s04's super-experts UNION `drift.super_expert_candidates` on the A192 fmax maxed
  with every pass fmax, plus `s04/position_protect.json`. Nothing subtracts or divides by a general score.
  Conditional extras: `*_shr<lam>` (HOPE stability below 0.9, shrink toward the pooled domain F) and
  `fair_*` (router entropy near flat).
- **Per-source F** is rebuilt from the banked routing (s03 banks one pooled F per pass): s = g x the bank's
  mean ||f|| of that expert on that sub-area, diagonal replaced by the exact banked second moment.
  `validate` reports the error against the banked pooled F.
- **Output** `$V41_WORK/dom/d2/<bucket>/<candidate>.json` (s04 keep-set format), `polymath/{control,rr4,rr6}.json`,
  `manifest.json`, `diag/*.json`. Per-layer units in `d2/_layers/` make it resumable and incremental.
- **Run** (CPU, ~0.3 GiB per worker, 6 workers): `ops/dom/run_d2.sh` (or `--wait <pass>` to start when
  that pass completes). Test: `.venv/bin/python ops/tests/test_d2_select.py`.
- **GPU scripts** (`d2_gpu/`): `replay_filter.py` (teacher-routed held-out replay, kl32/top-1 with
  row-cluster CIs, the D2 gate, tie groups; ~15 min per arm at 40,960 tokens) and `freerun_gate.py`
  (item 9: ~200 generated answers per finalist, then one s03 teacher pass over them). Cost and memory
  notes in each docstring.
