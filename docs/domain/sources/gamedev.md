# gamedev: public source catalogue (Rust + Bevy game development, general Rust)

Domain id `gamedev` (PLAN.md). Private owner transcripts were used in the original run and are not released;
they are NOT listed. This file lists everything else.

**Compiled:** 2026-11-01 from live queries only (HF Hub API, GitHub REST API, arXiv export API + abs pages, web search).
Licence fields marked *verified 2026-11-01* were read live from HF `cardData.license` / card text or from the
GitHub licence API / LICENSE file on that date. **UNVERIFIED** marks anything not confirmed from a primary source.
Bevy is at **v0.20.1** (released 2026-10-28, v0.20.0 on 2026-10-08), per
https://api.github.com/repos/bevyengine/bevy/releases. Pin every git source to a commit when it is pulled.

## Search log (auditable)

All searches were run on 2026-11-01.

- HF datasets, newest first: `https://huggingface.co/api/datasets?search=<q>&sort=createdAt&direction=-1&limit=100&full=true`,
  filtered to `createdAt >= 2026-01-01`. Queries: `rust, bevy, wgsl, wgpu, gamedev, "game dev", "game development", godot,
  shader, rustc, clippy, cargo, "swe rust"`. That returned 168 rows from 2026. Also searched by downloads:
  `rust, bevy, wgsl, rust-sft, rust-code, swe-smith, multi-swe, swe-synth, swe-rebench, rust-trajector, RustEvo, CRUST, rust-swe`.
- HF models, newest first (`/api/models?...sort=createdAt`): `rust, bevy`. Only small fine-tunes (for example
  `nmuendler/DeepSeek-R1-Distill-Qwen-7B-rust-rlvr-*`, 2026-09). None is a useful teacher. No Bevy models.
- arXiv export API (`sortBy=submittedDate`, 2026 only): `abs:Rust AND abs:LLM AND abs:benchmark`,
  `abs:Rust AND abs:agent AND abs:repository`, `abs:Bevy`, `abs:"game development" AND abs:agent`,
  `abs:"game engine" AND abs:LLM`, `abs:WGSL OR abs:WebGPU`, `abs:shader AND abs:LLM`.
- GitHub: `topic:bevy-game` (stars), `bevy game language:rust stars:>300`,
  `bevy game language:rust created:>2025-01-01 stars:>40` (stars). Licence and branch details come from `gh api repos/<r>`.
- Web search: GameDevBench, RustEvo, CRUST-bench, Rust-SWE-bench, Bevy LLM benchmark, game-dev code benchmarks 2026,
  Multi-SWE-bench / SWE-bench Multilingual / Aider polyglot leaderboards.

**Headline finding:** there is **no Bevy or Rust game-dev benchmark** in any of these sources. All 2026 game-dev agent
benchmarks target Godot, Unity, Unreal, Roblox or browser games (Section C2). The only executable Bevy tasks in public
data are **46 Bevy issues in Rust-SWE-bench** and a **161-task Bevy candidate pool** (`r1v3r/bevyengine_bevy`), plus 5
`bevy_asset_loader` tasks in SWE-rebench-V2. Every other Bevy verifier has to be built (Section D).

---

## A. Calibration sources (routing statistics, ~400k tokens)

Shares are of the **public** ~400k-token block. If private transcripts share the same 400k budget, scale these
proportionally. The priority is diversity across sub-areas, with about 25% in chat/agentic format.

| # | id / URL | type | approx size | licence (verified 2026-11-01) | created / last updated | generator | sub-areas | share |
|---|---|---|---|---|---|---|---|---|
| 1 | https://github.com/bevyengine/bevy @ `main` (tag v0.20.1): `crates/`, `examples/`, `release-content/` | git repo | 176 MB repo (assets included); about 25-35 MB of .rs (estimate) | MIT OR Apache-2.0 (GH API reports Apache-2.0, repo is dual) | 2014 / pushed 2026-10-30 | human | ECS, render graph, PBR, UI, assets, audio, input, picking, animation, WASM | **20%** |
| 2 | https://github.com/bevyengine/bevy-website @ `main`: `content/learn/migration-guides/`, quick-start, release news | git/docs | 812 MB repo; text about 5 MB | MIT (GH API); site states MIT/Apache | pushed 2026-11-01 | human | version migrations (0.7 to 0.20), API idioms | **6%** |
| 3 | Ecosystem crates (one row per crate): [avianphysics/avian](https://github.com/avianphysics/avian) (Apache-2.0, 2026-10-20), [dimforge/rapier](https://github.com/dimforge/rapier) (Apache-2.0; bevy_rapier now lives in this monorepo, the old repo is archived), [emilk/egui](https://github.com/emilk/egui) + [vladbat00/bevy_egui](https://github.com/vladbat00/bevy_egui) (MIT), [Leafwing-Studios/leafwing-input-manager](https://github.com/Leafwing-Studios/leafwing-input-manager) / [simgine/bevy_enhanced_input](https://github.com/simgine/bevy_enhanced_input), [cBournhonesque/lightyear](https://github.com/cBournhonesque/lightyear) + [simgine/bevy_replicon](https://github.com/simgine/bevy_replicon) + [lucaspoffo/renet](https://github.com/lucaspoffo/renet), [NiklasEi/bevy_asset_loader](https://github.com/NiklasEi/bevy_asset_loader), [tesselode/kira](https://github.com/tesselode/kira) + [NiklasEi/bevy_kira_audio](https://github.com/NiklasEi/bevy_kira_audio) + [CorvusPrudens/bevy_seedling](https://github.com/CorvusPrudens/bevy_seedling), [djeedai/bevy_hanabi](https://github.com/djeedai/bevy_hanabi), [StarArawn/bevy_ecs_tilemap](https://github.com/StarArawn/bevy_ecs_tilemap) (MIT), [Trouv/bevy_ecs_ldtk](https://github.com/Trouv/bevy_ecs_ldtk), [Razaekel/noise-rs](https://github.com/Razaekel/noise-rs), [aevyrie/big_space](https://github.com/aevyrie/big_space), [jakobhellermann/bevy-inspector-egui](https://github.com/jakobhellermann/bevy-inspector-egui), [janhohenheim/rerecast](https://github.com/janhohenheim/rerecast) (navmesh, created 2025-06) | git repos | about 15-25 MB .rs total (estimate) | all Apache-2.0 or MIT per GH API except bevy_ecs_ldtk (NOASSERTION; dual MIT/Apache per README, **UNVERIFIED**) | all pushed 2026-09/10 except noise-rs (2025-02) | human | physics, UI, input, networking, assets, audio, VFX, tilemaps, procgen, navmesh | **14%** |
| 4 | Rendering stack: [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) @ `trunk` (includes naga and WGSL front end), [sotrh/learn-wgpu](https://github.com/sotrh/learn-wgpu) (MIT), WGSL + WebGPU specs [gpuweb/gpuweb](https://github.com/gpuweb/gpuweb) (W3C Software and Document License), [bevyengine/naga_oil](https://github.com/bevyengine/naga_oil) (archived, Apache-2.0), Bevy `crates/bevy_pbr/src/**/*.wgsl` | git/docs | wgpu 86 MB; learn-wgpu text about 3 MB | wgpu MIT OR Apache-2.0; gpuweb W3C (LICENSE.md read) | wgpu 2026-10-31; gpuweb 2026-10-30 | human | wgpu, WGSL shaders, render pipelines, compute | **10%** |
| 5 | Open-source Bevy games and templates: [TheBevyFlock/bevy_new_2d](https://github.com/TheBevyFlock/bevy_new_2d) (MIT), [janhohenheim/foxtrot](https://github.com/janhohenheim/foxtrot) (MIT), [NiklasEi/bevy_game_template](https://github.com/NiklasEi/bevy_game_template) (CC0, includes WASM/CI), [fishfolk/jumpy](https://github.com/fishfolk/jumpy) (MIT OR Apache-2.0 per README badge and LICENSE), [thetawavegame/thetawave](https://github.com/thetawavegame/thetawave) (MIT), [OutThisLife/azeroth-survivors-re](https://github.com/OutThisLife/azeroth-survivors-re) (Apache-2.0, created 2026-10-13), [renzora/engine](https://github.com/renzora/engine) (Apache-2.0, an editor on Bevy 0.20), [tnrjns/faith-runner](https://github.com/tnrjns/faith-runner) (MIT, 2026-10), [jamesfebin/ImpatientProgrammerBevyRust](https://github.com/jamesfebin/ImpatientProgrammerBevyRust) (MIT tutorial book, 2025-10) | git repos | about 10 MB .rs (estimate) | as listed (GH API) | most pushed 2026 | human | whole-game architecture, states, save/load, WASM deploy, CI | **10%** |
| 6 | Rust docs: [rust-lang/book](https://github.com/rust-lang/book) (MIT/Apache, GH NOASSERTION), [reference](https://github.com/rust-lang/reference), [nomicon](https://github.com/rust-lang/nomicon), [rust-by-example](https://github.com/rust-lang/rust-by-example), [async-book](https://github.com/rust-lang/async-book) (MIT), [edition-guide](https://github.com/rust-lang/edition-guide), [api-guidelines](https://github.com/rust-lang/api-guidelines), [google/comprehensive-rust](https://github.com/google/comprehensive-rust) (Apache-2.0), [rust-unofficial/too-many-lists](https://github.com/rust-unofficial/too-many-lists) (MIT), [tokio-rs/website](https://github.com/tokio-rs/website) (MIT) | git/docs | about 15 MB markdown | Apache-2.0/MIT (GH API) | all 2026 | human | ownership, traits, unsafe, async, editions, API design | **12%** |
| 7 | [rust-lang/rust](https://github.com/rust-lang/rust) `library/` (std/core/alloc) + [rust-lang/rust-clippy](https://github.com/rust-lang/rust-clippy) `clippy_lints/` + lint docs | git | std about 20 MB (estimate) | MIT OR Apache-2.0 | 2026-11-01 | human | std internals, clippy lints | **5%** |
| 8 | Agentic Rust trajectories: [AlienKevin/Multi-SWE-smith-Rust-GLM-4.6-trajectories](https://huggingface.co/datasets/AlienKevin/Multi-SWE-smith-Rust-GLM-4.6-trajectories) (142 trajectories, `resolved` flag) | hf dataset | 5.5 MB | **NONE on card**, see red flags | 2025-12-12 | GLM-4.6 | tool-use bug fixing in Rust repos | **5%** |
| 9 | [ByteDance-Seed/Multi-SWE-bench_trajs](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_trajs), Rust subset only | hf dataset | 6.3 GB total; Rust share **UNVERIFIED** | CC0-1.0 | 2025-12-19 | many agent+model submissions | agentic Rust (ripgrep, clap, tokio, serde...) | **5%**, calibration only. These are solutions to MSB eval tasks, so never use them for FT if MSB-Rust is an eval |
| 10 | [Fortytwo-Network/Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) | hf dataset | 191,008 examples, 0.44 GB | Apache-2.0 | 2026-01-05 | swarm of SLMs (Fortytwo network), rustc-checked (94.3% compile) | 15 task types: gen, explain, refactor, tests, docs | **4%** |
| 11 | [hongliu9903/stack_edu_rust](https://huggingface.co/datasets/hongliu9903/stack_edu_rust) (text-included Rust slice of [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)); keep `int_score>=4` and `license_type=permissive` | hf dataset | 1.14M files, 2.0 GB | no card licence; per-file `detected_licenses` field; parent stack-edu has none on card (**UNVERIFIED**; derived from The Stack v2 / SWH terms) | 2025-07-31 | human code, educational-score filter | broad idiomatic Rust | **4%** |
| 12 | [saurabh5/rlvr-code-data-Rust](https://huggingface.co/datasets/saurabh5/rlvr-code-data-Rust) (`messages` field) | hf dataset | 133,296 rows, 0.24 GB | **none on card** | 2025-06-26 | LLM-translated Tulu-3 code problems (model **UNVERIFIED**) | function-level Rust with tests | **3%** |
| 13 | [gpuweb/cts](https://github.com/gpuweb/cts) WGSL/shader test subtree | git | 532 MB repo; WGSL tests a few MB | BSD-3-Clause | 2026-10-24 | human | WGSL semantics and validation | **2%** |

Total 100%. Agentic/chat-format rows are #8, #9, #10 and #12 (about 17%). Add agentic transcripts to bring that to 25% or more.

---

## B. Fine-tuning sources (SFT, agent trajectories, RL environments)

| # | id / URL | type | size | licence (verified 2026-11-01) | created / updated | generator | verifier | quality notes | contamination |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) ([arXiv 2602.23866](https://arxiv.org/abs/2602.23866)), **3,123 Rust tasks** of 32,079 (language frequencies read live from datasets-server statistics) | RL env (Docker images, F2P/P2P) | 1.25 GB | CC-BY-4.0 (repo licence per instance) | 2026-10-01 | human PRs; LLM quality annotations | `cargo test` FAIL_TO_PASS | Newest and largest Rust SWE environment set; **supersedes SWE-rebench v1 (Python-only) and SWE-Gym**. A partial repo probe found `NiklasEi/bevy_asset_loader` (5), `gfx-rs/naga` (25), `servo/html5ever` (2). Full Rust repo list **UNVERIFIED** (datasets-server filter timed out) | Card warns of overlap with multilingual SWE benchmarks. Exclude IDs using https://github.com/raimondasl/agentleak before training |
| 2 | [AlienKevin/Multi-SWE-smith-tasks](https://huggingface.co/datasets/AlienKevin/Multi-SWE-smith-tasks) split `rust` | RL env (synthetic bug injection, SWE-smith method) | 11,299 Rust tasks | **NONE on card** | 2025-12-12 | LLM-injected bugs | tests | Largest Rust task pool, but official SWE-smith has no Rust split ([SWE-smith-{py,java,go,js,ts,cpp}](https://huggingface.co/datasets?search=swe-smith) only) | low (synthetic) |
| 3 | [ByteDance-Seed/Multi-SWE-RL](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-RL) Rust part (alacritty, ripgrep, clap, fish-shell, helix, nushell, rusqlite, mdBook, serde, bat, fd, bytes, tokio, tracing) | RL env | 28.7 GB total | card text: CC0 (tag "other") | 2025-07-23 | human PRs | Docker tests via [multi-swe-bench](https://github.com/multi-swe-bench/multi-swe-bench) (Apache-2.0) | Includes "discarded instances" from MSB | **Same repos as Multi-SWE-bench, so dedupe against MSB-Rust eval IDs** |
| 4 | [r1v3r/bevyengine_bevy](https://huggingface.co/datasets/r1v3r/bevyengine_bevy) | RL env candidates (SWE-bench schema) | **161 Bevy tasks**, 2020-08 to 2024-12 | **none on card** | 2025-07-03 | human PRs | F2P/P2P lists | **The only Bevy-specific executable task pool found.** Appears to be the candidate pool behind Rust-SWE-bench | **Superset of the 46 Bevy tasks in Rust-SWE-bench (C1 #1). Hold those 46 out** |
| 5 | [AlienKevin/Multi-SWE-smith-Rust-GLM-4.6-trajectories](https://huggingface.co/datasets/AlienKevin/Multi-SWE-smith-Rust-GLM-4.6-trajectories) | SFT trajectories | 142 | **none** | 2025-12-12 | GLM-4.6 | `resolved` flag | small; use resolved=true only | low |
| 6 | [Fortytwo-Network/Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) | SFT | 191k | Apache-2.0 | 2026-01-05 | SLM swarm | rustc compile (94.3%) | **Newest large Rust SFT set; supersedes Tesslate/Rust_Dataset and Magicoder-Rust slices.** Short single-function items | unknown |
| 7 | [saurabh5/rlvr-code-data-Rust](https://huggingface.co/datasets/saurabh5/rlvr-code-data-Rust) | RLVR (problem + translated tests) | 133k | **none on card** | 2025-06-26 | LLM-translated | `translated_test_cases` executable | good cheap RLVR warm-up; algorithmic, not game-dev | possible MBPP/HumanEval paraphrases, **UNVERIFIED** |
| 8 | [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) Rust rows / cleaned subset [WrittenWithRust/Magicoder-OSS-Instruct-Rust-cleaned-3.9K](https://huggingface.co/datasets/WrittenWithRust/Magicoder-OSS-Instruct-Rust-cleaned-3.9K) | SFT | about 3.9k Rust | MIT | 2023-12 / 2026-08-30 | gpt-3.5-turbo | none | older; superseded by #6 for volume | low |
| 9 | [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) `rust` config | SFT (commit message to diff) | Rust config about 10^4 (**UNVERIFIED**) | MIT | 2023-08-20 | human commits | none | real edit-style data | low |
| 10 | [dtolnay/proc-macro-workshop](https://github.com/dtolnay/proc-macro-workshop) | RL env | 5 projects, about 50 test steps | Apache-2.0/MIT | 2026-08-06 | human | `cargo test` + trybuild | excellent macro curriculum | public, so assume contaminated |
| 11 | [rust-lang/rustlings](https://github.com/rust-lang/rustlings) (MIT) + [exercism/rust](https://github.com/exercism/rust) (MIT) | RL env | about 94 + about 100 exercises | MIT | 2026-10 | human | `cargo test` | **Exercism Rust is the source of Aider-polyglot's 30 Rust items. Hold out the polyglot list** | yes, vs Aider polyglot |
| 12 | Bevy example migration env (to build): `bevy/examples` at tag N together with its migration guide (bevy-website) gives the task "port example to N+1" | RL env (to build) | about 300 examples x 13 version steps | MIT/Apache | n/a | human | `cargo check`/`cargo run --example` headless + screenshot diff (Bevy CI already does headless example screenshots) | Highest-value Bevy-specific environment; see D | none if built from post-cutoff versions (0.19 to 0.20) |
| 13 | [nvidia/Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) / [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | SFT trajectories | 40 GB / 5.8 GB | CC-BY-4.0 | 2026-01-12 / 2025-12-15 | NVIDIA pipelines (models per card) | – | general agent format; Rust fraction **UNVERIFIED** (likely small) | check vs SWE-bench |

Not useful, or rejected after inspection: `Tesslate/Rust_Dataset` (first row is a React prompt, so mislabelled);
`jedisct1/enriched-rust-finetune-dataset` (licence "unknown"); `PrometheanStudio/promethean-rust-corpus` (licence "other",
2026-10); Godot datasets from 2026 (`0xRavenBlack/godot-gamedev-instruct`, `icici121/godot-gdscript-dataset`; GDScript,
wrong engine). No 2026 Bevy or WGSL SFT dataset of usable size exists. The only WGSL hit is
`abdullahashraf122/aegis-wgsl-webgpu-shaders` (MIT, fewer than 1k rows, 2026-10-01).

---

## C. Evaluation (executable verifiers, runnable on Linux aarch64 / Jetson Thor)

Note for aarch64: SWE-bench-style harnesses publish **x86_64** images. Rust images rebuild natively on arm64 from each
instance's Dockerfile / `install_config` (no binary toolchains to emulate). The Rust-SWE-bench README recommends x86_64,
so the arm64 rebuild is **UNVERIFIED**; budget a gold-patch validation pass first.

### C1. Rust (and Bevy) benchmarks

| # | benchmark | tasks | metric | best published (source) | harness | licence | created / updated | aarch64 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Rust-SWE-bench** ([arXiv 2602.22764](https://arxiv.org/abs/2602.22764), ICSE 2026); data [user2f86/rustbench](https://huggingface.co/datasets/user2f86/rustbench) (500 rows, mirror [r1v3r/rustbench](https://huggingface.co/datasets/r1v3r/rustbench) 502) | 500 from 34 repos, **including 46 bevyengine/bevy and 49 boa-dev/boa** (repo counts read live from datasets-server) | % resolved | RustForger + Claude-3.7-Sonnet 28.6%; OpenHands + Claude-3.7 21.2% (paper; old models, newer scores **UNVERIFIED**) | [GhabiX/Rust-SWE-Bench](https://github.com/GhabiX/Rust-SWE-Bench) (SWE-bench fork) | **no licence on repo or HF**, see red flags | 2025-07 / paper 2026-02-26 | rebuild images (UNVERIFIED) |
| 2 | Multi-SWE-bench Rust ([ByteDance-Seed/Multi-SWE-bench](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench); Rust-only mirror [r1v3r/multi_SWE_Bench_Rust](https://huggingface.co/datasets/r1v3r/multi_SWE_Bench_Rust)) | 239 Rust of 1,632 | % resolved | overall top 21.62% (Gemini-2.5-Pro + MopenHands, per https://benchmarklist.com/benchmarks/multi_swe_bench/, **secondary source**); per-language Rust **UNVERIFIED** | [multi-swe-bench](https://github.com/multi-swe-bench/multi-swe-bench) Apache-2.0 | data CC0 (card text) | 2026-07-08 | rebuild |
| 3 | [SWE-bench Multilingual](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual) Rust slice | 43 Rust of 300 (tokio 9, bat 8, axum 7, ruff 7, coreutils 5, nushell 5, ripgrep 2; counted live) | % resolved | full set: Claude Opus 5 89.5%, open-weights Ornith-1.0-397B 78.9% (https://benchlm.ai/benchmarks/swe-bench-multilingual, aggregator, 2026-07-29, **secondary**) | [SWE-bench](https://github.com/SWE-bench/SWE-bench) MIT | MIT | 2026-08-17 | rebuild |
| 4 | [Aider polyglot](https://github.com/Aider-AI/polyglot-benchmark) Rust slice | 30 Rust of 225 | pass@2 (with edit format) | GPT-5 (high) 88.0% overall (https://www.developersdigest.tech/benchmarks/aider-polyglot citing aider.chat; board frozen since 2025-11) | [aider/benchmark](https://github.com/Aider-AI/aider/tree/main/benchmark) Apache-2.0 | exercises (c) Exercism, MIT via [exercism/rust](https://github.com/exercism/rust); polyglot repo has no LICENSE | 2024-12 | native `cargo test` |
| 5 | [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) `humaneval-rs`, `mbpp-rs` | 156 + 354 (datasets-server, 2026-11-01) | pass@1 | saturated (more than 90% for frontier models, **UNVERIFIED** current) | [nuprl/MultiPL-E](https://github.com/nuprl/MultiPL-E) / [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) | MIT (HF) | 2025-07-15 | native |
| 6 | [bigcode/humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack) Rust (synthesize/fix/explain) | 164 x 3 | pass@1 | saturated | bigcode-evaluation-harness | MIT | 2025-08-19 | native |
| 7 | **RustEvo²** ([arXiv 2503.16922](https://arxiv.org/abs/2503.16922)); data [SYSUSELab/RustEvo2](https://huggingface.co/datasets/SYSUSELab/RustEvo2) | 588 API-change tasks (380 std, 208 from 15 crates) | Pass@1 | 65.8% avg on stabilizations, 38.0% on behavioural changes (paper) | [SYSUSELab/RustEvo](https://github.com/SYSUSELab/RustEvo) | HF Apache-2.0; **GH repo has no licence** | 2025-05-12 | native |
| 8 | **CRUST-bench** ([arXiv 2504.15254](https://arxiv.org/abs/2504.15254)); [eth-sri/CRUST-bench](https://huggingface.co/datasets/eth-sri/CRUST-bench) | 100 C repos to safe Rust | % compiling + tests passing | o1 solved 15/100 single-shot (paper; newer **UNVERIFIED**) | [anirudhkhatry/CRUST-bench](https://github.com/anirudhkhatry/CRUST-bench) | **GPL-3.0** repo (copyleft, eval use only); HF "other" | 2026-09-17 / 2026-07-16 | native |
| 9 | RustRepoTrans ([SYSUSELab/RustRepoTrans](https://huggingface.co/datasets/SYSUSELab/RustRepoTrans)) | repo-level translation to Rust (count **UNVERIFIED**) | pass@1 | per paper | [SYSUSELab/RustRepoTrans](https://github.com/SYSUSELab/RustRepoTrans) | **no licence (GH + HF)** | 2025-07-28 | native |
| 10 | RustMizan ([arXiv 2607.04729](https://arxiv.org/abs/2607.04729), 2026-07); eval logs at [rustmizan-org/rustmizan-eval-logs](https://huggingface.co/datasets/rustmizan-org/rustmizan-eval-logs) (CC-BY-4.0) | compilable Rust vulnerability variants (crate/file/function) | detection/CWE/localisation F1 | binary 56-65% (paper) | per paper (repo **UNVERIFIED**) | logs CC-BY-4.0 | 2026-10-24 | native |
| 11 | Multi-SWT-Bench ([arXiv 2609.34752](https://arxiv.org/abs/2609.34752), 2026-09) | multilingual reproduction-test generation (Rust included, **UNVERIFIED** count) | F2P test rate | per paper | **UNVERIFIED** | **UNVERIFIED** | 2026-09-28 | rebuild |

### C2. Game-dev agent benchmarks (2026). None uses Bevy or Rust.

| # | benchmark | engine | tasks | metric | best published | harness / licence | date | relevance |
|---|---|---|---|---|---|---|---|---|
| 1 | GameDevBench ([arXiv 2602.11103](https://arxiv.org/abs/2602.11103), ICML 2026) | Godot | 333 | % solved | 53.8% (ICML abstract); GPT-5.4 41.1% to 52.0% with visual feedback | [waynchi/gamedevbench](https://github.com/waynchi/gamedevbench) Apache-2.0 (1.19 GB) | 2026-10-19 | method reference only; Godot 4 has linux arm64 builds |
| 2 | GameLogicBench ([arXiv 2609.21562](https://arxiv.org/abs/2609.21562)) | Godot | 72 tasks, 1,451 tick-level test cases | % pass | per paper | [NJU-LINK/GameLogicBench](https://github.com/NJU-LINK/GameLogicBench), **no licence** | 2026-10-15 | **best methodology to port to Bevy**: rules asserted at every tick, which maps directly onto Bevy `FixedUpdate` + `App::update()` in headless tests |
| 3 | SWE-Game ([arXiv 2609.33678](https://arxiv.org/abs/2609.33678)) | Godot (and porting to Unity) | 247 tasks over 41 reference games, 5 task types incl. 83 injected-fault repairs | composite | "Opus5 achieves the highest overall score in all five task types" (abstract) | repo **UNVERIFIED** | 2026-09-27 | task taxonomy to copy |
| 4 | OpenGameEval ([arXiv 2610.02563](https://arxiv.org/abs/2610.02563)) | Roblox Studio | 84 | pass@1 / pass^5 | 51.7% / 39.4% | [Roblox/open-game-eval](https://github.com/Roblox/open-game-eval) MIT | 2026-10-05 | needs Roblox Studio (Windows/macOS), **not runnable on Thor** |
| 5 | GameASG-Bench ([arXiv 2609.21293](https://arxiv.org/abs/2609.21293)) | browser games | 47 | L1 static + L2 browser checks | per paper | [areal-project/GameASG-Bench](https://github.com/areal-project/GameASG-Bench) Apache-2.0 | 2026-09-18 | runnable (headless Chromium); the "declare eval interface before generation" protocol is reusable for WASM Bevy builds |
| 6 | UnityBench-Mini3D ([2610.35980](https://arxiv.org/abs/2610.35980)), CraftBench-UE ([2609.23142](https://arxiv.org/abs/2609.23142)), GameEngineBench UE5 C++ ([2607.03525](https://arxiv.org/abs/2607.03525)), JAMER/JamBench Godot game jams ([2606.19830](https://arxiv.org/abs/2606.19830)), A2Z GameSpec-Bench ([2609.39564](https://arxiv.org/abs/2609.39564)), GameXpert-Bench ([2608.21833](https://arxiv.org/abs/2608.21833)), OpenGame-Bench ([2604.18394](https://arxiv.org/abs/2604.18394)), V-GameGym ([2509.20136](https://arxiv.org/abs/2509.20136)) | Unity / UE / Godot / web | 70-300 | various | per papers | mostly **UNVERIFIED** repos | 2026 | **not runnable on Thor** (Unity/UE editors), or a different engine. JamSet's idea (mine game-jam repos with headless execution) transfers to **Bevy Jam** entries |

Related: "The Essence of Entity Component System" ([arXiv 2606.14919](https://arxiv.org/abs/2606.14919), 2026-06)
formalises archetype ECS with Bevy as a reference. It is a useful doc for calibration, not an eval.

---

## D. Gaps, licence red flags, top-5 additions

### Gaps
1. **No Bevy benchmark exists** (searches above, 2026-11-01). The executable Bevy signal available today is the 46
   Rust-SWE-bench Bevy tasks (eval), the 161-task `r1v3r/bevyengine_bevy` pool (the 115 that remain after holding out
   those 46 go to training), and 5 SWE-rebench-V2 `bevy_asset_loader` tasks.
2. No Bevy, WGSL or wgpu SFT/chat data of usable size. Rendering/shader knowledge must come from code and docs (A4) plus
   self-generated trajectories.
3. No visual or playability verifier for Rust games. Bevy's CI screenshot/headless example runner is the natural base.
4. Full per-repo Rust list of SWE-rebench-V2 not obtained (the datasets-server filter API timed out). Pull the parquet
   `repo` column locally before building the FT mix.

### Licence red flags (NC/ND excluded; SA/copyleft flagged)
- **EXCLUDED (NC-SA):** Bevy Cheatbook **prose** is CC BY-NC-SA 4.0 (README of
  https://github.com/bevy-cheatbook/bevy-cheatbook). Its **code** is MIT-0/MIT and may be used. Same exclusion for
  `xCodeEval` (CC-BY-NC) and Shadertoy-derived sets (Shadertoy default licence is BY-NC-SA, **UNVERIFIED** per set).
- **Copyleft (flag):** CRUST-bench repo GPL-3.0 (eval-only use); `DigitalExtinction/Game` AGPL-3.0 (archived, do not
  include in training); `rust-unofficial/patterns` MPL-2.0 (weak copyleft, docs); `Ethertum`, `kestrel`, `skate-3-rust-engine`,
  `nv-rs` GPL-3.0 Bevy games (omitted from A5); McEval / McEval-Instruct CC-BY-SA-4.0.
- **No licence at all (all rights reserved by default):** Rust-SWE-bench repo + HF data, `r1v3r/*`, `AlienKevin/*`,
  RustEvo GH repo, RustRepoTrans, GameLogicBench repo, Aider polyglot repo (the underlying Exercism content is MIT),
  `saurabh5/rlvr-code-data-Rust`. Use for private eval/training only after the owner rules on it. The underlying task
  repos are permissive, but the curation is unlicensed.
- **The Stack v2 / stack-edu:** custom "other" terms plus SWH download, gated (`bigcode/the-stack-v2`, 427 GB). Use the
  text-included Rust slice (A11) and keep only `license_type=permissive`.
- **ODC-By / attribution:** OpenCoder `opc-annealing-corpus` (ODC-By), CC-BY sets (SWE-rebench-V2, Nemotron) need
  attribution in any release.

### Top-5 highest-value additions (prioritised)
1. **Build "BevyMigrate-Bench"** from bevy `examples/` and the bevy-website migration guides, using the 0.19 to 0.20
   step (released 2026-10, after any plausible model cutoff), so it is contamination-free. Verifier:
   `cargo check --examples` + headless `cargo run --example` with screenshot diff. It is both an RL environment and an
   eval, and it targets the domain directly.
2. **Rust-SWE-bench Bevy+Boa slices as the domain eval** (46 + 49 tasks), with the remaining `r1v3r/bevyengine_bevy`
   tasks (161 minus the 46 held out) as an RL environment. These are the only executable public Bevy issues.
3. **SWE-rebench-V2 Rust (3,123 tasks)** as the main Rust RL pool. It is the newest (2026-10) and supersedes SWE-Gym and
   SWE-rebench v1 for Rust. Dedupe against MSB/Multilingual with agentleak.
4. **Port GameLogicBench's tick-level assertion method to Bevy.** Write 30-50 small gameplay-rule tasks on
   `bevy_new_2d` / `foxtrot` with `App::update()`-driven property tests (deterministic, no GPU). That fills the
   "playability" gap that the Godot benchmarks cover.
5. **Mine Bevy Jam open-source entries** (itch.io Bevy Jam 1-6 with linked GitHub repos; JAMER's method) to make a
   licence-filtered whole-game corpus for calibration and SFT. Not yet catalogued, so **UNVERIFIED** yield.
