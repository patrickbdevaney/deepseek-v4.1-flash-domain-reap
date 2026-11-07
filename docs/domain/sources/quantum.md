# quantum — quantum computing: source catalogue

Compiled 2026-11-01 for the domain-REAP track (`docs/domain/PLAN.md`). Everything comes from **live queries run 2026-11-01** (search log at the end).
Licences were checked against the HF API `cardData.license` and GitHub LICENSE files on 2026-11-01.
Dates are `created / last-updated` (HF `createdAt` / `lastModified`; GitHub `created_at` / `pushed_at`).
**[UNVERIFIED]** marks unchecked claims. Every scheduled source is tagged `generator:`.

Thor note: Qiskit, Cirq, PennyLane, stim and PyMatching are CPU Python/C++ with aarch64 wheels [UNVERIFIED per package]. CUDA-Q has ARM builds [UNVERIFIED on sm_110]. All verifiers below are CPU statevector or stabilizer simulators, so none needs x86.

The public quantum corpus is **small and mostly docs/code**. HF date-sorted search finds almost no strong 2026 quantum *training* sets; most new 2026 activity is benchmarks.

---

## A. Calibration sources (~400k tokens)

| # | id / URL | type | created / updated | approx size | licence (verified 2026-11-01) | generator | sub-areas | share |
|---|---|---|---|---|---|---|---|---|
| 1 | [Qiskit/documentation@main](https://github.com/Qiskit/documentation): `docs/guides`, `docs/tutorials`, `learning/courses/*` (Basics of QI, Fundamentals of Quantum Algorithms, Foundations of QEC, Variational Algorithm Design, Utility-scale QC, Quantum Diagonalization Algorithms, …) | git (MDX + notebooks) | 2023-09-06 / **2026-11-01** | large | code Apache-2.0 ([LICENSE](https://github.com/Qiskit/documentation/blob/main/LICENSE)); **prose CC-BY-SA-4.0 ([LICENSE-DOCS](https://github.com/Qiskit/documentation/blob/main/LICENSE-DOCS)), SA flagged** | human (IBM, Watrous courses) | theory, algorithms, QEC, Qiskit 2.x primitives | 70k |
| 2 | [Qiskit/qiskit](https://github.com/Qiskit/qiskit), [qiskit-aer](https://github.com/Qiskit/qiskit-aer), [qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime), [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) | git | pushed 2026-11-01 / 10-30 / 10-30 / 10-30 | large | Apache-2.0 ×4 | human | SDK internals, transpiler, release notes (API drift) | 35k |
| 3 | [PennyLaneAI/demos@master](https://github.com/PennyLaneAI/demos) + [pennylane](https://github.com/PennyLaneAI/pennylane) | git | 2019-10-09 / 2026-10-30; 2026-11-01 | ~250 demos [UNVERIFIED] | Apache-2.0 ×2 | human | QML, variational methods, chemistry | 40k |
| 4 | [Cirq](https://github.com/quantumlib/Cirq), [Qualtran](https://github.com/quantumlib/Qualtran), [OpenFermion](https://github.com/quantumlib/OpenFermion), [ReCirq](https://github.com/quantumlib/ReCirq) | git | pushed 2026-10-30 / 10-29 / 10-28 / 10-15 | — | Apache-2.0 ×4 | human | Cirq, FT resource estimation, fermions | 35k |
| 5 | [quantumlib/Stim](https://github.com/quantumlib/Stim), [PyMatching@master](https://github.com/oscarhiggott/PyMatching), [NVIDIA/cudaq-qec](https://github.com/NVIDIA/cudaq-qec) | git | 2021-02-19 / 2026-10-29; 2019-11-08 / 2026-05-22; 2024-10-29 / 2026-10-31 | — | Apache-2.0 ×3 | human | **QEC**: stabilizer simulation, DEMs, decoding | 40k |
| 6 | [NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum) docs + examples | git | 2023-03-14 / 2026-11-01 | — | Apache-2.0 | human | CUDA-Q | 25k |
| 7 | [openqasm/openqasm](https://github.com/openqasm/openqasm) + [munich-quantum-toolkit/bench](https://github.com/munich-quantum-toolkit/bench) | git | 2017-03-03 / 2026-09-02; 2021-11-05 / 2026-10-31 | — | Apache-2.0 / MIT | human | OpenQASM 3, benchmark circuits | 10k |
| 8 | [microsoft/qdk](https://github.com/microsoft/qdk) katas + archived [QuantumKatas](https://github.com/microsoft/QuantumKatas) | git | 2023-01-26 / 2026-10-31; archived 2024-08 | 350+ exercises | MIT ×2 | human | Q#, algorithm exercises | 20k |
| 9 | [common-pile/arxiv_papers_filtered](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) ∩ `quant-ph` (join to CC0 [arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot), updated 2026-10-27). For 2026 papers: [secemp9/arxiv-complete](https://huggingface.co/datasets/secemp9/arxiv-complete) (2026-09-10 / 09-19) filtered to CC-BY/CC0 | hf | Common Pile 2025-05-23 / 2025-06-06 (**content mid-2025**) | 304k docs overall; quant-ph slice unmeasured | per-row CC-BY / CC-BY-SA / CC0; arxiv-complete is `other` (mixed author licences) [per-row filter UNVERIFIED] | human | research frontier | 60k |
| 10 | [errorcorrectionzoo/eczoo_data](https://github.com/errorcorrectionzoo/eczoo_data) | git (YAML) | 2021-10-29 / 2026-10-30 | ~1k codes [UNVERIFIED] | **CC-BY-SA-4.0**, flagged | human | QEC code taxonomy | 15k |
| 11 | [samuellimabraz/quantum-assistant](https://huggingface.co/datasets/samuellimabraz/quantum-assistant) (text rows) | hf (Q/A + code + tests) | 2025-12-07 / 2026-08-13 | 8,366 rows | Apache-2.0 per card; **derived from SA docs (laundering risk)** | unnamed LLM/VLM pipeline [UNVERIFIED] | chat-format Qiskit | 20k |
| 12 | *Quantum Computing for the Quantum Curious* (Springer 2021, OAPEN) | book | 2021 | ~150 pp | CC-BY-4.0 per catalogue ([DOAB](https://openaccess.library.uitm.edu.my/Record/doab67917)) [UNVERIFIED on the PDF] | human | intro theory | 10k |
| 13 | local: private owner agent sessions on quantum tasks (planned; not released) | agentic | — | — | private | claude / spine | agentic format | 20k |
| | **total** | | | | | | | **400k** |

Optional extras, all permissive: [qutip](https://github.com/qutip/qutip) BSD-3, [Quantinuum/tket](https://github.com/Quantinuum/tket) Apache-2.0, [braket SDK](https://github.com/amazon-braket/amazon-braket-sdk-python) Apache-2.0, [classiq-library](https://github.com/Classiq/classiq-library) MIT, [qBraid](https://github.com/qBraid/qBraid) Apache-2.0, [dwave-ocean-sdk](https://github.com/dwavesystems/dwave-ocean-sdk) Apache-2.0, [tensorflow/quantum](https://github.com/tensorflow/quantum) Apache-2.0, [qiskit-nature](https://github.com/qiskit-community/qiskit-nature) Apache-2.0. [Qiskit/textbook](https://github.com/Qiskit/textbook) (Apache-2.0, archived 2024-01) is **superseded by row 1**.

---

## B. Fine-tuning sources

| rank | id / URL | type | created / updated | size | licence | generator | sub-areas | verifier | quality notes | contamination risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [EstelYang/QCircuitBench](https://github.com/EstelYang/QCircuitBench) ([2410.07961](https://arxiv.org/pdf/2410.07961), NeurIPS 2025) | RL env + data | 2024-10-10 / 2026-05-06 | 3 suites, 25 algorithms, 120,290 data points | MIT | programmatic (Qiskit → OpenQASM 3) | oracles, algorithm design, circuit synthesis | **built-in verification functions** | best executable quantum RL env | hold out the 28-task Harbor subset |
| 2 | Stim + PyMatching QEC task generator (build), seeded by **StabilizerBench**'s three task types ([arXiv 2604.21287](https://arxiv.org/abs/2604.21287), 2026-04-23: state-prep synthesis, constrained optimisation, FT circuit synthesis; Gottesman–Knill verification) | RL env (local) | 2026 | unlimited | Apache-2.0 tooling. StabilizerBench code/data **not located** [UNVERIFIED] | programmatic | QEC | logical error rate / stabilizer equivalence | fills the biggest public gap | none |
| 3 | MIT [QuantumKatas](https://github.com/microsoft/QuantumKatas) / [qdk](https://github.com/microsoft/qdk) katas, **re-translated by us** to Qiskit/Cirq/PennyLane | SFT + tests | MIT originals | 350 | MIT | human (Q#) | algorithms, measurement, oracles | statevector equivalence | replaces NC-SA [Qiskit/Qiskit-QuantumKatas](https://huggingface.co/datasets/Qiskit/Qiskit-QuantumKatas) (2026-04-30) | do not eval on Qiskit-QuantumKatas then |
| 4 | [samuellimabraz/quantum-assistant](https://huggingface.co/datasets/samuellimabraz/quantum-assistant) | SFT (code-gen with `test_code`, QA; multimodal) | 2025-12-07 / 2026-08-13 | 8,366 (2,653 code-gen) | Apache-2.0 (SA-derived) | unnamed LLM pipeline | Qiskit 2.0 | unit tests | executable-filtered | overlaps QHE source docs |
| 5 | [arasyi/quantum-api-drift](https://huggingface.co/datasets/arasyi/quantum-api-drift) **(new 2026-07)** | `benchmark` + `version_anchored` configs | 2026-07-15 / 07-15 | small [UNVERIFIED] | Apache-2.0 | [UNVERIFIED] | Qiskit/SDK API-version drift | [UNVERIFIED] | targets the deprecated-API failure mode; content not inspected | low |
| 6 | [munich-quantum-toolkit/bench](https://github.com/munich-quantum-toolkit/bench) + [pnnl/QASMBench](https://github.com/pnnl/QASMBench) | circuit corpora | 2021-11-05 / 2026-10-31; 2020-05-29 / 2025-01-20 | thousands | MIT / PNNL BSD-style | programmatic | compilation, mapping | equivalence checking (mqt.qcec) + gate/depth | transpiler RL env | low |
| 7 | [PennyLaneAI/demos](https://github.com/PennyLaneAI/demos) → tasks | SFT | 2026-10-30 | ~250 | Apache-2.0 | human | QML, VQE | re-execute cells | human exemplars | QuanBench+ PennyLane overlap |
| 8 | [itsalissonsilva/quantum-circuit-repair-synthetic](https://huggingface.co/datasets/itsalissonsilva/quantum-circuit-repair-synthetic) + [-specs-synthetic](https://huggingface.co/datasets/itsalissonsilva/quantum-circuit-specs-synthetic) **(new 2026-10-31)** | synthetic | 2026-10-31 | n<10K | CC-BY-4.0 | [UNVERIFIED] | circuit repair, specs | [UNVERIFIED] | brand new, 50 downloads; sample before use | unknown |
| 9 | [Neura-parse/quantum-computing](https://huggingface.co/datasets/Neura-parse/quantum-computing) (+ 17 siblings, all created 2026-07-01) | mixed | 2026-07-01 / 07-07 | 117,041 rows | CC-BY-4.0 | undisclosed pipeline ("expert-generated" claim [UNVERIFIED]) | broad | per-row `source_url` | templated mill signature; audit before use | unknown |
| 10 | [BoltzmannEntropy/QuantumLLMInstruct](https://huggingface.co/datasets/BoltzmannEntropy/QuantumLLMInstruct) | SFT | 2024-10-09 / 2024-12-27 | claims 500k; size tag n<10K [conflict] | Apache-2.0 | Qwen-2.5-Coder et al. | Hamiltonians, QPE, VQE | none | **low trust, pre-Qiskit-2.0**; superseded by rows 4 and 5 | low |
| 11 | [mlfoundations-dev/stackexchange_quantumcomputing](https://huggingface.co/datasets/mlfoundations-dev/stackexchange_quantumcomputing) | Q&A | 2024-12-23 | 10K–100K | StackExchange content **CC-BY-SA**, flagged | human | practitioner Q&A | — | real chat-format questions | low |
| 12 | Surface-code data: [willow-surface-code-detection-events](https://huggingface.co/datasets/ShayManor/willow-surface-code-detection-events) (Apache-2.0, 2026-06-19 / 07-10), [ai4sci-surface-code-decoding](https://huggingface.co/datasets/sunweiwei/ai4sci-surface-code-decoding) (CC-BY-4.0, 2026-09-24) | numeric | 2026 | 10M–100M rows | as listed | hardware / sim | decoder tasks | decoding accuracy | tool-use only | low |

No IBM Qiskit Code Assistant training data has been published. The Qiskit HF org (checked 2026-11-01) has 3 datasets (QHE, QHE-hard, QuantumKatas) and models such as [mistral-small-3.2-24b-qiskit](https://huggingface.co/Qiskit/mistral-small-3.2-24b-qiskit) (Apache-2.0) and the 2026 AI-transpiler models ([routing](https://huggingface.co/Qiskit/ai-transpiler_routing), 2026-03-06).

---

## C. Evaluation

| benchmark | created / updated | tasks | metric | best published (source) | harness / licence | Thor? |
|---|---|---|---|---|---|---|
| **Qiskit HumanEval Hard** ([HF](https://huggingface.co/datasets/Qiskit/qiskit_humaneval_hard)) | 2025-08-25 / 2026-10-13 | 151 | pass@1 (unit tests) | 63.8% GPT-6 Astra; 58.1% Gemini 3.1 Pro Preview ([arXiv 2610.28678](https://arxiv.org/html/2610.28678v1), 2026-10) | [qiskit-human-eval](https://github.com/qiskit-community/qiskit-human-eval) (2024-11-08 / 2026-10-16), Apache-2.0 | Yes. **Primary Qiskit eval; supersedes standard QHE** |
| Qiskit HumanEval ([HF](https://huggingface.co/datasets/Qiskit/qiskit_humaneval)) | 2024-11-13 / 2026-10-13 | 151 | pass@1 | 98.7% GPT-6 Astra ([2610.28678](https://arxiv.org/html/2610.28678v1)); best Qiskit fine-tune QSpark 56.29% ([2507.12642](https://arxiv.org/pdf/2507.12642)) | same | Yes. **Saturated at the frontier**; regression check only |
| **QuanBench+** ([arXiv 2604.08570](https://arxiv.org/abs/2604.08570), ICLR 2026 workshop) | repo 2026-04-13 | 42 tasks × Qiskit/PennyLane/Cirq | pass@1 | 59.5% Qiskit / 54.8% Cirq / 42.9% PennyLane | [JawadKotaichh/quanbench-plus](https://github.com/JawadKotaichh/quanbench-plus), MIT [repo↔paper UNVERIFIED] | Yes. Cross-framework |
| QuanBench 44 / 117 ([2510.16779](https://arxiv.org/html/2510.16779)) | repo 2025-09-27 / 2026-10-20 | 44 / 117 | pass@1 | 60.0% / 84.6% ([2610.28678](https://arxiv.org/html/2610.28678v1)) | [GuoXiaoYu1125/Quanbench](https://github.com/GuoXiaoYu1125/Quanbench), MIT [official? UNVERIFIED] | Yes. **The audit found evaluator bugs: test classes never run, and an empty candidate inherits the canonical solution. Patch first** |
| **QCircuitBench** | 2024-10-10 / 2026-05-06 | 25 algorithms; Harbor subset 28 ([Harbor](https://hub.harborframework.com/datasets/harbor-index/harbor-index-1.0/latest)) | verification-function pass | per paper ([2410.07961](https://arxiv.org/pdf/2410.07961)) | MIT | Yes |
| StabilizerBench ([arXiv 2604.21287](https://arxiv.org/abs/2604.21287)) **(new 2026-04)** | 2026-04-23 | 3 QEC task types | Gottesman–Knill verified | "discriminates across" 3 frontier agents; no numbers in the abstract | code/data not located [UNVERIFIED] | Yes once obtained or rebuilt (B2) |
| QuantumBenchEval (QBE) ([2610.28678](https://arxiv.org/html/2610.28678v1)) **(new 2026-10)** | 2026-10 | 96 (T1–T6) | pass@1 / rubric | 94.1% T1 (GPT-6 Astra / Claude Fable 5.1); no model >32% on T4 | release **[UNVERIFIED]** | — |
| QCoder | — | 67 | LLM judge | 43.0% Claude Fable 5.1 ([2610.28678](https://arxiv.org/html/2610.28678v1)) | [UNVERIFIED] | not executable |
| Qiskit Code Migration ([arXiv 2606.20173](https://arxiv.org/pdf/2606.20173)) (2026-06) | 2026-06 | migration tasks [UNVERIFIED] | — | — | repo not located | candidate API-drift eval |
| QuantumBench ([2511.00092](https://arxiv.org/abs/2511.00092)) | 2025-10-15 / 11-05 | 769 MCQ | accuracy | [alphaXiv board](https://alphaxiv.org/benchmarks/nvidia/quantumbench) | [shunyaist/QuantumBench](https://github.com/shunyaist/QuantumBench): **no LICENSE; NC-SA sources** | internal knowledge probe only; not executable |
| Qiskit-QuantumKatas ([HF](https://huggingface.co/datasets/Qiskit/Qiskit-QuantumKatas), [2605.27210](https://arxiv.org/pdf/2605.27210)) | 2026-04-30 / 10-20 | 350 | pass@1 | — | **CC-BY-NC-SA-4.0, EXCLUDED**; rebuild from the MIT originals (B3) | — |
| In-house QEC eval (Stim/PyMatching) | to build | ~50 | logical error rate vs reference | — | Apache-2.0 tooling | Yes |

---

## 2026 releases and what they supersede

| new (2026) | supersedes | why |
|---|---|---|
| Benchmark audit [arXiv 2610.28678](https://arxiv.org/html/2610.28678v1) (10) | trusting QuanBench/QHE scores as published | found evaluator bugs; 530-task cross-suite scores for current models |
| QHE-Hard re-scored (via 2610.28678) | standard QHE (saturated at 98.7%) | still discriminative (63.8%) |
| QuanBench+ (04) | single-framework QHE for breadth | Qiskit + PennyLane + Cirq aligned |
| StabilizerBench (04) | none (first QEC synthesis benchmark) | QEC coverage |
| Qiskit-QuantumKatas (04) | Q#-only katas for Qiskit eval | but NC-SA, so rebuild from MIT originals |
| quantum-api-drift (07), Qiskit Code Migration (06) | QuantumLLMInstruct-era pre-2.0 data | targets Qiskit 1.x→2.x drift |
| Qiskit documentation (updated daily, 2026-11-01) | Qiskit/textbook (archived 2024) | current API + Watrous courses |

## Gaps
1. **No licensed agentic or chat quantum trajectories** in 2026 date-sorted HF search. The plan relies on private agent sessions (A13; not released).
2. **No released QEC code-gen dataset.** StabilizerBench code is not located, so build a Stim/PyMatching environment.
3. **No CUDA-Q or Q# executable eval** apart from the MIT katas.
4. **API drift:** pin Qiskit ≥2 in every verifier. Pre-2025 sets teach removed APIs (`execute`, `BasicAer`).
5. **Open CC-BY textbooks are scarce.** TU Delft OCW and OpenLearn are NC-SA ([TU Delft](https://ocw.tudelft.nl/?p=31227)). The de Wolf notes ([1907.09415](https://arxiv.org/abs/1907.09415), v6 2026-08-27) carry only the arXiv non-exclusive licence (checked).

## Licence red flags
- **Excluded:** Qiskit-QuantumKatas (CC-BY-NC-SA); TU Delft/OpenLearn (NC-SA); QuantumBench (no licence, NC-SA sources); arXiv papers without CC licences.
- **Share-alike:** Qiskit docs prose, eczoo_data, StackExchange, CC-BY-SA arXiv rows.
- **Copyleft code:** [mitiq](https://github.com/unitaryfoundation/mitiq) (GPL-3.0); QuantumChem/QuantumChem-200k-new (GPL-3.0, 2026-07).
- **Laundering or opaque:** quantum-assistant (Apache over SA); QuantumLLMInstruct; Neura-parse; secemp9/arxiv-complete (`other`, mixed).

## 5 highest-value additions
1. **Stim + PyMatching QEC RL environment** built on StabilizerBench's three task types (Apache tooling, aarch64 CPU). It covers the largest gap.
2. **QCircuitBench** (MIT, 120k points, built-in verifiers) as the main RL pool with a held-out split.
3. **Qiskit/documentation courses + guides** (live 2026-11-01; Qiskit 2.x; SA flagged) as the core calibration text.
4. **QHE-Hard + QuanBench+** as the executable eval pair, using 2610.28678's evaluator patches. Plain QHE is saturated.
5. **MIT QuantumKatas re-translated with statevector tests**: a licence-clean replacement for the 2026 NC-SA Qiskit-QuantumKatas.

---

## Search log (2026-11-01; auditable)
- HF datasets, `sort=createdAt&direction=-1&limit=100` (pacing needed after HTTP 429): `quantum`, `qiskit`, `qasm`, `pennylane`, `cirq`, `quantum error`, `surface code`, `stabilizer`. Also `sort=downloads`: `quantum circuit`, `qcircuitbench`, `quantumbench`, `quant-ph`, `openqasm`, `stim`, `QuantumLLMInstruct`, `quantum computing`, `quantum code`, `quantum algorithm`, `QuantumKatas`, `qec`, `quantum-assistant`, `qhack`, `q#`. Author listing `author=Qiskit` (datasets and models).
- HF models, `sort=createdAt` (likes ≥3): `qiskit`, `quantum`. Only the Qiskit AI-transpiler models are relevant.
- arXiv export API: quant-ph LLM-benchmark queries returned HTTP 429 on 2026-11-01. Covered instead by web search ("arXiv 2026 benchmark LLM quantum error correction code generation …") and direct arXiv abs fetches (2604.21287, 1907.09415, 2610.28678).
- GitHub (`gh search repos`): `StabilizerBench`, `QuanBench`, `QHackBench`, `qiskit humaneval`, `qiskit llm`, `quantum code generation`, `qiskit code migration`; `gh api repos/X` (created/pushed/licence) for every repo cited.
- Web: QuantumBench, Qiskit HumanEval 2026 scores, quantum code-gen benchmarks 2026, CC-BY quantum textbooks, the *Quantum Curious* licence.
