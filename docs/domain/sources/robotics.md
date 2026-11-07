# robotics — simulation, ROS 2, planning, control, manipulation/locomotion, VLA: source catalogue

Domain (bucket 14, PLAN.md "Domain set v3"): physics simulators and scene description
(MuJoCo/MJX/Warp/Playground/mjlab, Newton, Genesis, Isaac Lab, ManiSkill, Gazebo, PyBullet, Drake;
MJCF/URDF/SDF/USD), ROS 2 (MoveIt 2, Nav2, ros2_control), motion planning (OMPL, MoveIt, cuRobo,
TAMP), control/MPC/trajectory optimisation (acados, CasADi, crocoddyl, Pinocchio, Drake),
manipulation, locomotion/whole-body control, reward design and RL training code, VLA/embodied
foundation models and data, robot task planning and code generation, sim-to-real and
perception-for-control. Software only: the owner has no robot. Every verifier must run in a
simulator or as a pure computation. The served model is text-only, so image, video and action data
count only for their text (task language, code, metadata).

**Search date: 2026-11-01.** Every row comes from live queries run that day (see the search log at
the end), not from recall. Rows reused from an earlier catalogue of the original domain set (not released) (A10, A11, A12, B14 MuJoCo
part, B17, B18, B19, C12, C13, C15, gap 2, licence flags) were verified live earlier the same day
and are marked "(earlier catalogue)". Licences were read live from HF `cardData.license` or the GitHub licence
API / LICENSE file unless marked **UNVERIFIED**. "card: none" means the HF card declares no
licence, so all rights are reserved by default. Dates are written `c` = created, `u` = last updated
(HF `createdAt`/`lastModified`; for GitHub, `pushed_at`). Row counts come from
`datasets-server.huggingface.co/size`. Nothing was downloaded beyond metadata, READMEs and
licence files.

Licence policy (PLAN.md, 2026-11-01 refinement): **NC/ND sources may be used only for private
evaluation, never for calibration or training**. **SA/copyleft sources are allowed for private use
but flagged `[SA]`/`[GPL]`**. Unlicensed sources are private use at most. `generator: claude` rows
can be switched off. **NEW-2026** marks 2026 releases; "supersedes" names the older source each one
replaces.

### Thor portability summary (checked live 2026-11-01)

The host is Ubuntu 24.04.3 (noble), aarch64, sm_110 (`/etc/os-release`). PyPI facts come from
`pypi.org/pypi/<pkg>/json`. ROS facts come from the `packages.ros.org` apt indices.

| tool | Thor verdict | evidence |
|---|---|---|
| MuJoCo 3.17.0 | **yes**: native aarch64 wheel | PyPI (u 2026-10-22) |
| MJX / MuJoCo Warp 3.17.0 | **yes, probably**: pure-Python packages on `jaxlib` 0.12.0.1 + `jax-cuda13-plugin` (aarch64 wheels) / `warp-lang` 1.18.0 (aarch64 wheel). sm_110 kernels not exercised (**UNVERIFIED**) | PyPI |
| MuJoCo Playground | git install only (not on PyPI); runs on MJX/Warp | [repo](https://github.com/google-deepmind/mujoco_playground) |
| mjlab 1.6.0 (Isaac Lab API on MuJoCo Warp) | **yes, probably**: pure Python on mujoco-warp | PyPI; [repo](https://github.com/mujocolab/mjlab) Apache-2.0 (u 2026-11-01) |
| Newton 1.7.0 | **yes, probably**: pure Python on warp. Its extras select `usd-exchange` on aarch64 and skip open3d there, so aarch64 is a handled target | PyPI `requires_dist` |
| Genesis 1.4.5 | **CPU/GPU core probably yes**: depends on `quadrants` 1.3.3 (aarch64 wheel). The batch renderer `gs-madrona` is x86-only, so no batch rendering | PyPI `requires_dist` |
| Isaac Lab 3.0.0 | **UNVERIFIED on Thor**. 3.0 runs kit-less on Newton by default and documents Linux aarch64 *DGX Spark* (CUDA 13) | [install docs](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html) |
| Isaac Sim 6.2.0 | aarch64 wheel exists, but the only official aarch64 target is DGX Spark. Render problems are reported on IGX Thor | PyPI; earlier catalogue C notes |
| ManiSkill 3.0.1 | **NO (x86 needed)**: needs `sapien>=3.0.0`, whose 3.0.3 wheels are x86_64/mac/win only | PyPI |
| PyBullet 3.2.7 | sdist only on aarch64 (last release 2025-01-30), so it has to be built from source (**UNVERIFIED** on Thor) | PyPI |
| Drake 1.58.0 | **yes**: aarch64 wheel | PyPI (u 2026-10-15) |
| Pinocchio (`pin` 4.1.0), crocoddyl 3.2.1, CasADi 3.8.2, proxsuite 0.8.2, OSQP 1.1.3, coal 3.0.3, mink 1.4.0, placo 0.10.2, roboticstoolbox 1.5.0, tesseract-robotics 0.35.0 | **yes**: aarch64 wheels | PyPI |
| OMPL 2.0.1 | **yes**: official aarch64 wheels (new in 2026) | PyPI (u 2026-06-23) |
| acados | source build only (plain C + `acados_template`); aarch64 **UNVERIFIED** | [repo](https://github.com/acados/acados) BSD-2 |
| cuRoboV2 | git/source build; sm_110 **UNVERIFIED**. The licence is now **Apache-2.0** (v1 was not) | [repo](https://github.com/NVlabs/curobo) (u 2026-09-10) |
| ROS 2 Jazzy / Kilted | **yes**: arm64 debs on noble for `moveit`, `navigation2`, `ros2-control`, `gz-sim-vendor` | `packages.ros.org/ros2/ubuntu/dists/noble/main/binary-arm64` |
| ROS 2 Lyrical (2026 LTS) | arm64 is **Tier 1 on Ubuntu 26.04** (resolute) and has the same packages; noble is Tier 3 | [supported platforms](https://docs.ros.org/en/jazzy/Releases/lyrical/supported-platforms.html); resolute arm64 index |
| Gazebo (OSRF apt) | **yes**: noble arm64 | `packages.osrfoundation.org` Release (2026-10-29) |
| robosuite 1.5.2 / dm_control 1.0.50 / Gymnasium 1.4.0 / LeRobot 0.6.1 | yes: pure Python (MuJoCo / torch 2.15 aarch64 underneath) | PyPI |
| TextWorld 1.7.0 (ALFWorld text) | sdist only on aarch64 (**UNVERIFIED** build) | PyPI |
| Fast Downward (`up-fast-downward`) | x86-only wheel; build from source | PyPI |
| Isaac Gym (Eureka, DrEureka) | **NO**: x86-only, deprecated | earlier catalogue B19 |
| habitat-sim | **NO**: no Linux wheel | PyPI |

## Sub-areas and token floors

Bucket budget ~400k calibration tokens. Every sub-area has a floor of at least 8% (32k tokens). The
target column is the realised share in table A; the next section shows which rows supply each one.
Self-traces (PLAN.md gate 2, ~15% extra rows of the A192 spine's own traces) come on top and are
split the same way.

| id | sub-area | floor | target share (~tokens) |
|---|---|---|---|
| S1 | Simulators and scene description (MJCF/URDF/SDF/USD, MuJoCo/Warp/Newton/Genesis/Gazebo APIs) | 8% | 12% (~48k) |
| S2 | ROS 2 and middleware (rclpy/rclcpp, launch, tf2, QoS, Nav2, ros2_control, Gazebo bridge) | 8% | 13% (~52k) |
| S3 | Motion planning and kinematics (OMPL, MoveIt 2, cuRobo, IK, collision) | 8% | 10% (~40k) |
| S4 | Control, MPC and trajectory optimisation (acados, CasADi, crocoddyl, Pinocchio, Drake, LQR) | 8% | 12% (~48k) |
| S5 | Manipulation (grasping, code-as-policy, contact-rich tasks) | 8% | 8% (~32k) |
| S6 | Locomotion and whole-body control (legged and humanoid RL, retargeting) | 8% | 8% (~32k) |
| S7 | Reward design and RL training code (Isaac Lab/mjlab/Playground task configs, Eureka-style) | 8% | 10% (~40k) |
| S8 | VLA and embodied foundation models and data (the language side: instructions, model/training code) | 8% | 8% (~32k) |
| S9 | Robot task planning and agentic code generation (PDDL/TAMP, behavior trees, LLM-as-planner) | 8% | 10% (~40k) |
| S10 | Sim-to-real and perception/estimation for control (domain randomisation, system ID, state estimation, SLAM) | 8% | 9% (~36k) |
| | **total** | | **100%** |

Supply per sub-area (rows of table A): S1 = A1 10 + A10 2; S2 = A2 9 + A3 2 + A12 2;
S3 = A2 1 + A4 7 + A12 1 + A18 1; S4 = A5 8 + A6 2 + A12 2; S5 = A6 2 + A9 4 + A11 2;
S6 = A8 4 + A12 1 + A13 2 + A16 1; S7 = A7 8 + A16 2; S8 = A13 2 + A14 3 + A15 3;
S9 = A9 2 + A11 2 + A16 2 + A17 4; S10 = A12 2 + A13 4 + A16 1 + A18 2. Dose-response check
(PLAN.md gate 1) applies per bucket.

## A. Calibration sources (routing statistics; ~400k tokens)

Mix: ~60% code and docs (`git:` rows), ~20% Q&A and literature, ~20% agentic. No eval content
from section C.

| # | id / URL | dates | type | approx size | licence (verified 2026-11-01) | generator | sub-areas | share |
|---|---|---|---|---|---|---|---|---|
| A1 | `git:` simulators (earlier catalogue A10, extended): [mujoco](https://github.com/google-deepmind/mujoco) Apache-2.0 (u 2026-11-01; XML reference + Python docs), [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) Apache-2.0 (u 2026-11-01), [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) MJCF **per-model licences, permissive models only** (u 2026-10-31), [newton](https://github.com/newton-physics/newton) Apache-2.0 (u 2026-11-01), [genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) Apache-2.0 (u 2026-10-29), [gz-sim](https://github.com/gazebosim/gz-sim) SDF examples Apache-2.0 (u 2026-10-30), [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) Apache-2.0 (u 2026-10-28), **NEW-2026** [wuji-description](https://github.com/wuji-technology/wuji-description) MIT (c 2026-02-09; URDF+MJCF+MJX+USD of one robot), **NEW-2026** [allex_model](https://github.com/wirobotics-rih/allex_model) BSD-3 (c 2026-03-12) | see cell | code, docs, MJCF/URDF/SDF | tens of MB | as listed | human | S1 | 10% |
| A2 | `git:` ROS 2 docs and tutorials: [ros2_documentation](https://github.com/ros2/ros2_documentation) CC-BY-4.0 (earlier catalogue A11), [moveit2_tutorials](https://github.com/moveit/moveit2_tutorials) BSD-3 (u 2026-09-16), [docs.nav2.org](https://github.com/ros-navigation/docs.nav2.org) Apache-2.0 (u 2026-10-28), [control.ros.org](https://github.com/ros-controls/control.ros.org) Apache-2.0 (u 2026-11-01), [ros2_control_demos](https://github.com/ros-controls/ros2_control_demos) Apache-2.0 (u 2026-11-01), [navigation2](https://github.com/ros-navigation/navigation2) source (Apache-2.0 **and LGPL-2.1+ portions** `[GPL]`, u 2026-10-30), [gazebosim/docs](https://github.com/gazebosim/docs) CC-BY-4.0 | see cell | docs, launch/URDF/YAML, C++/Python nodes | ~10s MB | as listed | human | S2 (9), S3 (1) | 10% |
| A3 | **NEW-2026** ROS 2 agent tooling (MCP servers and tool schemas): [mergeos-bounties/ros2-mcp](https://github.com/mergeos-bounties/ros2-mcp) MIT (c 2026-07-12), [ros-claw/rosclaw](https://github.com/ros-claw/rosclaw) MIT (c 2026-03-16 / u 10-30), [TmxjTmxj/ros2-agent-workflow](https://github.com/TmxjTmxj/ros2-agent-workflow) MIT (c 2026-08-11; FastMCP + Gazebo TurtleBot3), [emgena ROS 2 MCP teasers](https://huggingface.co/datasets/emgena/emgena_ros2_tf2_coordinate_transform_guard_mcp_teaser) apache-2.0 (c 2026-09-28; tf2/QoS/executor/Nav2-costmap variants, tiny) | 2026-03 – 10 | agentic tool schemas + code | small | MIT / Apache | human / **UNVERIFIED** | S2 | 2% |
| A4 | `git:` motion planning: [ompl](https://github.com/ompl/ompl) BSD-3 LICENSE (u 2026-11-01), [moveit2](https://github.com/moveit/moveit2) BSD-3 (u 2026-10-28), [curobo](https://github.com/NVlabs/curobo) (cuRoboV2) Apache-2.0 (u 2026-09-10), [mink](https://github.com/kevinzakka/mink) Apache-2.0 (u 2026-11-01), [pink](https://github.com/pink-kinematics/pink) Apache-2.0 (u 2026-09-27), [pybullet-planning](https://github.com/caelan/pybullet-planning) MIT (u 2024-11-22) | see cell | code + docs + examples | tens of MB | permissive | human | S3 | 7% |
| A5 | `git:` control/optimisation: [acados](https://github.com/acados/acados) BSD-2 LICENSE (u 2026-10-29), [casadi](https://github.com/casadi/casadi) LGPL-3.0 `[GPL]` (u 2026-10-31), [crocoddyl](https://github.com/loco-3d/crocoddyl) BSD-3 (u 2026-10-26), [pinocchio](https://github.com/stack-of-tasks/pinocchio) BSD-2 (u 2026-11-01), [drake](https://github.com/RobotLocomotion/drake) examples/tutorials BSD-3 LICENSE (u 2026-11-01), [python-control](https://github.com/python-control/python-control) BSD-3 (earlier catalogue A12) | see cell | code + notebooks | tens of MB | as listed | human | S4 | 8% |
| A6 | [RussTedrake/manipulation](https://github.com/RussTedrake/manipulation) BSD-3 (u 2026-10-27), [RussTedrake/underactuated](https://github.com/RussTedrake/underactuated) BSD-3 for "all included code" (u 2026-10-24): **code and notebooks only**; licence of the book prose **UNVERIFIED** | see cell | course code (Drake) | small | BSD-3 (code) | human | S5 (2), S4 (2) | 4% |
| A7 | `git:` RL tasks and rewards: [IsaacLab](https://github.com/isaac-sim/IsaacLab) task/reward configs BSD-3 LICENSE (u 2026-11-01), [mjlab](https://github.com/mujocolab/mjlab) Apache-2.0 (u 2026-11-01), [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) Apache-2.0 (u 2026-10-31), [Eureka](https://github.com/eureka-research/Eureka) reward prompts + generated rewards MIT (u 2024-05-03), [text2reward](https://github.com/xlang-ai/text2reward) (**no licence: private only**, u 2024-12-17) | see cell | code (reward functions, env configs) | small–medium | as listed | human + LLM (Eureka outputs) | S7 | 8% |
| A8 | `git:` locomotion/whole-body: **NEW-2026** [mjlab_playground](https://github.com/mujocolab/mjlab_playground) Apache-2.0 (c 2026-04-03), **NEW-2026** [leggedrobotics/foothold-prediction-model](https://github.com/leggedrobotics/foothold-prediction-model) MIT (c 2026-10-14), plus the locomotion/humanoid task folders of A1/A7 (Playground `locomotion/`, Isaac Lab `locomotion/`) | see cell | code | small | permissive | human | S6 | 4% |
| A9 | **NEW-2026** code-as-policy agent harnesses (code and READMEs only): [capgym/cap-x](https://github.com/capgym/cap-x) MIT (c 2026-03-25, [arXiv 2603.22435](https://huggingface.co/papers/2603.22435)), [waltstephen/CaPtain](https://github.com/waltstephen/CaPtain) + [CaPtain-Harness](https://github.com/waltstephen/CaPtain-Harness) MIT (c 2026-09-16 / 10-28), [HexaFuture/HexaShell](https://github.com/HexaFuture/HexaShell) Apache-2.0 (c 2026-10-09, [2610.23479](https://huggingface.co/papers/2610.23479)), [BinceQu/RoboHarness](https://github.com/BinceQu/RoboHarness) MIT (c 2026-10-07, [2610.14018](https://huggingface.co/papers/2610.14018)), [robot-agent-exploration/rax](https://github.com/robot-agent-exploration/rax) Apache-2.0 (c 2026-10-13), [tomsilver/robocode](https://github.com/tomsilver/robocode) MIT (c 2026-02-14, [2609.30233](https://huggingface.co/papers/2609.30233)), [kindergarden](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) MIT (c 2026-03-10, RSS 2026) | 2026-02 – 10 | code + skill/API docs | small | permissive | human | S5 (4), S9 (2) | 6% |
| A10 | **NEW-2026** agent skill files (SKILL.md format): [coolbeevip/mujoco-skills](https://github.com/coolbeevip/mujoco-skills) MIT (c 2026-06-07), [earthtojake/urdf-skill](https://github.com/earthtojake/urdf-skill) MIT (c 2026-04-22; URDF generation + validation). [rax skills-lib](https://github.com/robot-agent-exploration/skills-lib) has **no licence: excluded** | 2026-04 – 09 | agentic instructions | small | MIT | human/LLM-assisted (**UNVERIFIED**) | S1 | 2% |
| A11 | **NEW-2026** [robot-agent-exploration/robot-session-logs](https://huggingface.co/datasets/robot-agent-exploration/robot-session-logs) (RAX, [arXiv 2610.22496](https://arxiv.org/abs/2610.22496)): full Claude Code transcripts, MCP robot tool calls/results/errors, agent-written code and skills from 15 sessions (Franka, RB-Y1, simulated G1). **Text parts only** (strip base64 images and videos) | c 2026-10-16 / u 10-31 | agentic transcripts | 15 sessions | cc-by-4.0 | **claude** (third party's Claude Code) | S5 (2), S9 (2) | 4% |
| A12 | **NEW-2026 dump** [flymin/stackexchange_20260331](https://huggingface.co/datasets/flymin/stackexchange_20260331) (earlier catalogue A13): `robotics` site (it absorbed ROS Answers in the 2023 migration), plus control/estimation questions from `engineering`, `electronics`, `dsp` | c/u 2026-07-03 (dump 2026-03-31) | Q&A (accepted answers) | site `.7z` files | upstream CC BY-SA 4.0 `[SA]`; **card: none** | human | S2 (2), S3 (1), S4 (2), S6 (1), S10 (2) | 8% |
| A13 | arXiv **cs.RO abstracts 2024-01 → 2026-10** (arXiv metadata is CC0) via OAI-PMH, or the HF Papers index while the export API returns 429. Full text only for CC-BY/CC0 papers ([common-pile/arxiv_papers_filtered](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered), per-paper licence in `metadata.license`; it has **no category field**, so join on a cs.RO id list such as [cometadata cs.RO list](https://huggingface.co/datasets/cometadata/202508-arxiv-cs-ai-cv-ro-lg-ma-cl-dois) cc0-1.0) | abstracts to 2026-10; common-pile c 2024-09 / u 2025-06 | literature | ~10k abstracts | CC0 (abstracts); per-paper (full text, some `[SA]`) | human | S10 (4), S6 (2), S8 (2) | 8% |
| A14 | VLA task language, deduplicated and rendered as text tables (instruction, embodiment, subtasks): [OXE mirror jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) cc-by-4.0 (c 2023-10), [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) apache-2.0 (u 2026-06-25), **NEW-2026** [nvidia/PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) cc-by-4.0 (c 2026-02-06 / u 09-16), **NEW-2026** [tencent/Hy-Embodied-0.5-VLA-Data](https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data) cc-by-4.0 (c 2026-06-11), **NEW-2026** [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) mit (c 2026-06-30), [RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) mit (u 2026-09-22) | see cell | short instructions + metadata | metadata only (no frames) | as listed | human / teleop | S8 | 3% |
| A15 | VLA code and docs: [openpi](https://github.com/Physical-Intelligence/openpi) Apache-2.0 (u 2026-08-24), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) Apache-2.0 (u 2026-10-29), [lerobot](https://github.com/huggingface/lerobot) Apache-2.0 (u 2026-11-01), **NEW-2026** [AgenticRoboRL](https://github.com/xinxinliu-ai/AgenticRoboRL) Apache-2.0 (c 2026-10-16) | see cell | code + docs | tens of MB | Apache-2.0 | human | S8 | 3% |
| A16 | Private owner agent sessions on Thor (planned; not released) with real tools (MuJoCo/mjlab, Drake, Pinocchio, OMPL, acados, ROS 2 Jazzy container) on held-out tasks | to be generated | agentic transcripts | ~60–80 sessions | owner's own; ToS caveat per PLAN.md | **claude** | S7 (2), S9 (2), S6 (1), S10 (1) | 6% |
| A17 | Planning and behavior trees: **NEW-2026** [BTGenBot-2](https://github.com/AIRLab-POLIMI/BTGenBot-2) MIT (c 2026-05-26, [2602.01870](https://huggingface.co/papers/2602.01870)), [AIRLab-POLIMI/btgenbot](https://huggingface.co/datasets/AIRLab-POLIMI/btgenbot) mit (c 2024-03-29), [BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP) docs/examples MIT (u 2026-10-26), [unified-planning](https://pypi.org/project/unified-planning/) Apache (PyPI 1.3.0, 2025-12-17) PDDL examples, [pddlstream](https://github.com/caelan/pddlstream) GPL-3.0 `[GPL]` | see cell | code, PDDL, BT XML | small | as listed | human / LLM (btgenbot) | S9 | 4% |
| A18 | Estimation and kinematics tutorials: [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) MIT LICENSE (u 2026-10-26; localisation, SLAM, path planning), [robotics-toolbox-python](https://github.com/petercorke/robotics-toolbox-python) MIT (u 2026-11-01) | see cell | code + docs | small | MIT | human | S10 (2), S3 (1) | 3% |
| | **total** | | | | | | | **100%** (10+10+2+7+8+4+8+4+6+2+4+8+8+3+3+6+4+3) |

If the `claude` rows are switched off: A16 (6%) goes to A12 +2 (S10/S6), A13 +2 (S10/S6) and
A17 +2 (S9), and S7 keeps its floor through A7 (8%). A11 (4%) goes to A9 +2 (S5) and A17 +2 (S9),
so S5 stays at 8%.
**Supersedes:** A9 (2026 code-as-policy harnesses) replaces Code-as-Policies-era prompts as the
code-gen corpus. A13 takes over the role of the 2024 cs.RO slices; A3/A10 are the first public ROS
2 / URDF agent tool schemas (2026).

## B. Fine-tuning sources (SFT, trajectories, RL environments)

| # | id / URL | dates | type | size | licence (2026-11-01) | generator | verifier | quality | contamination risk |
|---|---|---|---|---|---|---|---|---|---|
| B1 | **Self-built RL environments (recommended core; see "Top-10 additions" 1–5)**: MJCF/URDF compile + settle, ROS 2 colcon/launch, OMPL/MoveIt planning, mjlab/Playground reward rollouts, acados/CasADi MPC on a MuJoCo plant, Pinocchio/Drake dynamics and IK checks. Extends earlier catalogue B14 (MuJoCo part) | tools u 2026-10/11 | RL env | unbounded | permissive tools (CasADi LGPL `[GPL]`) | programmatic | the simulator or solver itself | high | none with disjoint seeds/assets |
| B2 | **NEW-2026** [YijiaFan/CaPtain-SFT-Data](https://huggingface.co/datasets/YijiaFan/CaPtain-SFT-Data) | c 2026-10-28 / u 10-30 | agent SFT: a VLM writes Python against a robot API (LIBERO family) | 24,440 rows (15,921 agent rows, 2,036 episodes; 797M tokens incl. images) | mit | **GPT-6 Astra / GPT-6.x Sol** (teacher-output terms) | CaPtain-Harness sim success (MuJoCo/robosuite) | high, but **multimodal**: stripping images removes the grounding. Use the code/tool-call turns, or replay with privileged text state | **high vs C4/C5** (LIBERO-PRO/Plus) |
| B3 | **NEW-2026** [tomsilver/robocode](https://github.com/tomsilver/robocode) + [KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) (MIT) as an RL env: write one program that solves held-out generalized-TAMP instances | c 2026-02 / u 10-07 | program-synthesis env (28 envs: 25 KinDER + 3 PDDLStream) | 100 held-out instances per env | MIT (pddlstream GPL-3 `[GPL]`) | programmatic | frozen program × held-out instances | high | **total vs C3**: split the envs between train and eval |
| B4 | **NEW-2026** [robot-session-logs](https://huggingface.co/datasets/robot-agent-exploration/robot-session-logs) (A11) | c 2026-10-16 | long-horizon agent trajectories with operator turns | 15 sessions | cc-by-4.0 | claude | none we can rerun (real robots); HumanoidArena sessions are simulated | medium (rich, tiny) | low |
| B5 | [Eureka](https://github.com/eureka-research/Eureka) / [DrEureka](https://github.com/eureka-research/DrEureka) reward-design loop (earlier catalogue B19) ported to mjlab / Playground on MuJoCo Warp | u 2024 | reward-design RL pattern | — | MIT | — | short PPO run → task fitness | high once ported | none |
| B6 | Behavior-tree SFT: [btgenbot](https://huggingface.co/datasets/AIRLab-POLIMI/btgenbot) (2024) + [BTGenBot-2](https://github.com/AIRLab-POLIMI/BTGenBot-2) (2026; whether its training data is released is **UNVERIFIED**) | 2024-03 / 2026-05 | NL → BehaviorTree.CPP XML | small | MIT | LLM-generated | XML parse + BehaviorTree.CPP / py_trees tick in a mock world (self-built) | medium | low |
| B7 | MJCF/URDF ground truth for conversion and repair tasks: [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) Apache-2.0, permissive [Menagerie](https://github.com/google-deepmind/mujoco_menagerie) models, [Kitchen-MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF) cc-by-4.0 (earlier catalogue B17), **NEW-2026** [senthurayyappan/robot-descriptions](https://huggingface.co/datasets/senthurayyappan/robot-descriptions) (card apache-2.0, c 2026-08-30; MJCF+URDF generated from 128,883 CAD assemblies, sources Fusion 360 Gallery / Automate-Onshape: **upstream licences UNVERIFIED, private only**) | see cell | assets for generated tasks (e.g. "write the MJCF for…", "fix this URDF") | thousands of models | as listed | human / programmatic | `mujoco.MjModel.from_xml_*` compile + simulated settle + joint-limit checks | high | low with held-out robots |
| B8 | **NEW-2026** [Gene829/gene-robotics-instruct](https://huggingface.co/datasets/Gene829/gene-robotics-instruct) | c 2026-06-13 / u 06-23 | arXiv-grounded robotics Q&A | 190 rows | cc-by-4.0 | Qwen3-4B (Q4) + judge | judge only | low | low |
| B9 | [StackExchange robotics](https://huggingface.co/datasets/flymin/stackexchange_20260331) accepted answers with score ≥ 3 (A12) | 2026-07-03 | SFT | thousands | CC BY-SA `[SA]` (card: none) | human | votes only | medium | low |
| B10 | [zichao22/robo-instruct](https://huggingface.co/datasets/zichao22/robo-instruct) (earlier catalogue B18) | 2024-06-05 | robot programs | 5,000 | **card none; CodeBotler forbids ML training: EXCLUDED** | LLM + sim | RoboEval | medium | high |
| B11 | EmbodiedSWE-Gen trajectories ([EmbodiedSWE](https://github.com/EmbodiedSWE/EmbodiedSWE) Apache-2.0, c 2026-06-13; [2609.27308](https://huggingface.co/papers/2609.27308)) | 2026-09 | coding-agent solutions → VLA demos | — | Apache-2.0 (assets [robobench-assets](https://huggingface.co/datasets/EmbodiedSWE/robobench-assets) apache-2.0) | frontier coding agents | Isaac Sim/Lab: **not runnable on Thor (unconfirmed)** | — | — |

**Supersedes (B):** B2/B3 (2026) are the first open, licensed, simulator-verified code-as-policy and
generalized-TAMP training sources; they replace Robo-Instruct (B10, excluded). No 2026 ROS 2,
MoveIt or URDF/MJCF SFT corpus with a verifier exists (gap 1), so B1 and B7 are self-built.
**What VLA/robot-learning data is worth to a text LLM:** OXE, DROID, AgiBot World, Open-H,
Hy-Embodied, BEHAVIOR-1K, RoboTwin, the 2026 LeRobot dumps and the KinDER demos
([kinder-bench/kinder-datasets](https://huggingface.co/datasets/kinder-bench/kinder-datasets) mit)
are frames and actions. Their text is short imperative instructions plus embodiment and subtask
metadata, which is useful only as a few-percent prompt source (A14) and for planner-decomposition
SFT targets. They are not useful as supervision for a text-only spine.

## C. Evaluation (executable verifiers; Linux aarch64 / Jetson Thor)

| # | benchmark | dates | tasks | metric | best published (open / frontier) | harness + licence | Thor aarch64 |
|---|---|---|---|---|---|---|---|
| C1 | **Embodied Agent Interface (EAI)**: [repo](https://github.com/embodied-agent-interface/embodied-agent-interface) MIT (u 2025-03-06), data [Inevitablevalor/EmbodiedAgentInterface](https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface) apache-2.0 | 2024 (NeurIPS D&B) | 438 (VirtualHome 338 + BEHAVIOR 100) × 4 modules | goal F1; subgoal/action trajectory SR; transition-model planner SR | e.g. BEHAVIOR action sequencing o1-preview 81.0 goal SR; VirtualHome subgoals o1-preview 89.4 ([site](https://embodied-agent-interface.github.io/)); no 2026 or DeepSeek rows | `eai-eval` 1.0.5 (pure Python, PyPI) | **yes**: text-only symbolic simulators. Partly saturated; use as a regression floor |
| C2 | **NEW-2026 Coding Agents for Generalized TAMP (AgenticGenPlan)**: [tomsilver/robocode](https://github.com/tomsilver/robocode) MIT (u 2026-10-07), [2609.30233](https://huggingface.co/papers/2609.30233) | 2026-09 | 28 envs × 100 held-out instances (980 programs, 98k episodes in the paper) | held-out success of a frozen program | Claude Code (Opus 5) and Codex (GPT-6 Astra); LLMGenPlan with source access 28% (Opus 5, $20 budget). Per-env table **UNVERIFIED** (site charts did not load) | MIT; strict black box: Python+NumPy+SciPy only; Docker/Apptainer sandbox | **probably yes**: MuJoCo aarch64; PyBullet + Fast Downward source builds (**UNVERIFIED**). Best fit for a text spine (render tools are optional) |
| C3 | **NEW-2026 CaP-Bench (CaP-X)**: [capgym/cap-x](https://github.com/capgym/cap-x) MIT (c 2026-03-25), [2603.22435](https://huggingface.co/papers/2603.22435) | 2026-03 | 39 tasks (Robosuite, LIBERO-PRO, BEHAVIOR); tiers S1–S4 single-turn, M1–M4 multi-turn | success rate | frontier "over 30% average"; Qwen2.5-Coder-7B 20% → 72% after CaP-RL ([project](https://capgym.github.io/)); per-model table **UNVERIFIED** | MIT; needs a CUDA GPU | Robosuite/LIBERO part **probably yes** (MuJoCo); BEHAVIOR (OmniGibson/Isaac) **no**. Text-only tiers (privileged state) **UNVERIFIED** |
| C4 | **NEW-2026 CaPtain-Harness LIBERO family**: [CaPtain](https://github.com/waltstephen/CaPtain) MIT | 2026-10 | LIBERO held-out 240, LIBERO-PRO 600/800/160, LIBERO-Plus 700, LIBERO-MAX 160 episodes | host / strict success % | **CaPtain-4B (open, Qwen3.5-4B) 73.8/67.5** LIBERO held-out, 78.5/71.5 LIBERO-PRO 6-cell; base Qwen3.5-4B 0.0 (README) | MIT; README states Linux x86_64 + EGL; transfer benches need SAPIEN/Isaac | MuJoCo backends **UNVERIFIED on aarch64**; VLM inputs, so a text spine needs a privileged-state adapter. **Contamination: total vs B2** |
| C5 | **NEW-2026 RLE-Bench**: [RLE-Bench/RLE-Bench](https://github.com/RLE-Bench/RLE-Bench) MIT (c 2026-09-14 / u 11-01), [2609.34210](https://huggingface.co/papers/2609.34210) | 2026-09 | 9 task families (control, policy learning, perception, mechanical design) | RLE Index | [leaderboard](https://rle-bench.github.io/): Gemini 4 Argon ranks #2; GPT-6.1 Sol, Claude Opus 5.5, Fable 5.1, GPT-6 Luna added 2026-10-02. Scores **UNVERIFIED** (charts did not render); no open-weights rows found | MIT; Harbor + Docker | tasks 01–05 need a GPU via Docker; 06–09 CPU. Image arch **UNVERIFIED** (likely amd64). Multimodal feedback |
| C6 | **NEW-2026 Embodied First Exam**: [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam) MIT (c 2026-09-18 / u 10-31), assets [MLL-Lab/embodied-first-exam-public](https://huggingface.co/datasets/MLL-Lab/embodied-first-exam-public) (licence "per-folder") | 2026-10-19 | 449 core tasks (1,161 task-modes; 35 benchmarks; 9 original suites) | per-task success | no published scores | code MIT; **some content NC** (LabUtopia CC BY-NC 4.0; AutoBio unlicensed): **private eval only** | **no**: x86_64 Docker (+ RTX for some tasks). Remote x86 only |
| C7 | **NEW-2026 Agentic Robotics Benchmark ("Agentic Robotics Last Exam")**: [repo](https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark) Apache-2.0 (c 2026-09-08 / u 10-31), assets [agentic-robotics-last-exam](https://huggingface.co/datasets/grounded-intelligence-initiative/agentic-robotics-last-exam) cc-by-4.0 (u 2026-11-01) | 2026-10 | **1 published task** today (space / fault-adaptive docking); 12 platforms planned | reward = clamp(measured/anchor) on hidden seeds | — | Apache-2.0; ALE engine + Docker | image arch **UNVERIFIED**. **Watch list**: the design (artifact grading on hidden seeds, controllers/URDF/plans) is the closest match to this bucket |
| C8 | RoboEval / [CodeBotler](https://github.com/ut-amrl/codebotler) (earlier catalogue C12) | 2023–24 | service-robot programs | pass@1 (symbolic sim) | [Robo-Instruct paper](https://arxiv.org/abs/2405.20179) | **BSD-3 + ML-training ban**: eval only | yes |
| C9 | **NEW-2026 RoboDojo, "LLM as policy"**: [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) MIT (u 2026-10-30), [2607.04434](https://huggingface.co/papers/2607.04434); LLM evaluation [2609.24170](https://huggingface.co/papers/2609.24170) | 2026-07 / 09 | 42 sim tasks | average success / Score | **GPT-6 Astra 22.48% SR** (above all 40 public policies); GPT-5.5 0.88%; **"DeepSeek-Flash" 1.92%** (10 episodes per task; exact version **UNVERIFIED**) | MIT; Isaac Sim 5.1 / Isaac Lab 2.3 | **no** (Isaac Sim). It is still the only published DeepSeek-Flash robotics baseline: rerun remotely if wanted |
| C10 | **NEW-2026** [RobotAgentBench](https://github.com/AetherLabsAI/RobotAgentBench) (earlier catalogue C13; Apache-2.0, u 2026-10-29) / [EmbodiedSWE-Bench](https://github.com/EmbodiedSWE/EmbodiedSWE) (Apache-2.0) | 2026-09/10 | RAB: 90 atomic + 18 short + 8 long + 4 open-ended; ESWE: contact-rich, deformable, long-horizon | success | — | Apache-2.0; RAB training data card: none | **no** (Isaac Sim/Lab USD) |
| C11 | **NEW-2026** [SimReadyBench](https://huggingface.co/datasets/hcltech-robotics/simreadybench) ([repo](https://github.com/hcltech-robotics/simreadybench) MIT, c 2026-10-20) | 2026-10 | sim-ready asset reconstruction (nv25 / nv100 references) | formal / visual / kinematic fidelity, `simready_reward` | per-run results in the dataset (10,399 rows); aggregate **UNVERIFIED** | MIT + **NVIDIA asset licence** (`license_link`); Harbor | image-conditioned: only partly applicable to a text spine; arch **UNVERIFIED** |
| C12 | ALFWorld (text) [alfworld/alfworld](https://github.com/alfworld/alfworld) MIT (u 2026-02-08) | 2020 | 134 unseen eval games | success | many 2026 RL papers report it (e.g. [wckwan eval dumps](https://huggingface.co/datasets/wckwan/Alfworld-Qwen3-8B-GiGPO-s2000-eval)); SOTA **UNVERIFIED**, likely near-saturated | MIT | TextWorld sdist build **UNVERIFIED**. Household planning, not robotics proper: low weight |
| C13 | Not applicable / excluded: [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) (repo **no licence**, MLLM vision; [EB-ALFRED](https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED) apache-2.0), [VLABench](https://github.com/OpenMOSS/VLABench) (MIT, VLA), LIBERO/LIBERO-PRO/[LIBERO-Pro2.0](https://huggingface.co/papers/2610.24587)/[LIBERO-Industry](https://huggingface.co/papers/2610.16782) (VLA; Pro2.0 train CC BY-NC), [HumanoidToolBench](https://huggingface.co/papers/2610.02089) (teleop CC BY-NC), [XPlanner-Benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark) (video; card none), [Words to World](https://huggingface.co/papers/2610.15228) (real hardware), Eureka (Isaac Gym x86) | — | — | — | LIBERO > 90% largely memorised ([LIBERO-PRO](https://arxiv.org/abs/2510.03827)) | — | **no** for a text spine on Thor |
| C14 | **Self-built held-out suites** (from B1): MJCF/URDF generation, ROS 2 package build + launch test, OMPL/MoveIt planning, reward design, MPC closed loop, IK/dynamics | to build | ~50 tasks per sub-area, disjoint seeds/assets from B1 | pass rate / normalised reward | baseline = A192 spine before vs after the mask | permissive | **yes** (all aarch64-native; ROS 2 Jazzy debs) |

Recommended local suite (aarch64, no x86 Docker): **C2 + C14 + C1 + C8** (eval-only), plus the
MuJoCo parts of **C3/C4** once a privileged-state text adapter exists. Remote x86 only, if the owner
wants frontier comparability: C5, C6 (NC content, eval-only), C9 (DeepSeek-Flash baseline). Watch
C7 for multi-arch images and more tasks.

## Gaps

1. **No ROS 2 / MoveIt / Nav2 code-generation benchmark or SFT corpus with an executable
   verifier was found** (HF datasets `ros2`/`moveit`/`nav2`: only teleop bags, MCP teasers and
   VLA dumps; 2026 papers: none). Build B1/C14 with a colcon + `launch_testing` verifier in a Jazzy
   container.
2. **No URDF/MJCF-generation benchmark with an executable verifier** (still true, earlier catalogue gap 2).
   [Articraft](https://huggingface.co/papers/2605.15187) (2026-05) published no data that we found;
   URDF-Anything+ is NC. Self-build from B7.
3. **2026 robotics agent benchmarks are Isaac-centric** (RoboDojo, RobotAgentBench,
   EmbodiedSWE, RLE GPU tasks, BEHAVIOR) or **x86 Docker** (Embodied First Exam, Harbor images).
   Isaac Sim on Thor is unconfirmed. MuJoCo-based ones (KinDER/robocode, Robosuite/LIBERO in
   CaP-X/CaPtain) are the portable ones.
4. **Most 2026 embodied benchmarks are multimodal** (camera feedback). A text-only spine needs
   privileged-state adapters; without them the scores measure the missing modality.
5. **The largest 2026 robot datasets are NC** (AgiBot World 2026 and its challenge set CC BY-NC-SA;
   Galaxea CC BY-NC-SA; GR00T-Teleop-Sim/GRAIL CC BY-NC). Their text value is low anyway.
6. **Teacher-output terms**: the best 2026 agentic SFT (CaPtain) is GPT-6 Astra/Sol output; RAX
   is Claude Code output (`generator: claude`).
7. **Textbooks**: Tedrake's *Manipulation*/*Underactuated* prose licence and *Modern Robotics*
   (Lynch & Park) redistribution terms are **UNVERIFIED**; only BSD code is used.
8. **arXiv export API returned 429 again**; 2026 literature was found through the HF Papers mirror.
   Rerun the cs.RO harvest for A13 once it clears.

## Licence red flags

- **EXCLUDED from calibration/training (NC/ND or ML ban; eval-only at most)**:
  [AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) (CC BY-NC-SA, c 2026-03-11, 736,502 rows),
  [AgiBotWorldChallenge-2026](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026) (CC BY-NC-SA) and mirrors;
  [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) (CC BY-NC-SA);
  NVIDIA [GR00T-Teleop-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim) and
  [Locomanipulation-GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) (CC BY-NC);
  [loco-mujoco-datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (CC BY-NC-ND);
  [URDF-Anything-plus](https://huggingface.co/datasets/URDF-Anything-plus/Dataset) (CC BY-NC);
  [LIBERO-Pro2.0-Train](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Train),
  [humanoidtoolbench-teleop](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop),
  [EmbodiedMemoryBench](https://huggingface.co/datasets/lzLiang/EmbodiedMemoryBench),
  [qvac/GenesisIII](https://huggingface.co/datasets/qvac/GenesisIII) (CC BY-NC);
  [leggedrobotics/hoi-retarget](https://huggingface.co/datasets/leggedrobotics/hoi-retarget) and
  [qwen_robotics_open_dataset](https://huggingface.co/datasets/Jinyan0924/qwen_robotics_open_dataset) (CC BY-NC-SA);
  [codebotler](https://github.com/ut-amrl/codebotler) (ML-training ban; also Robo-Instruct);
  Embodied First Exam NC sub-suites (LabUtopia CC BY-NC).
- **Unlicensed (private at most)**: Robo-Instruct; [text2reward](https://github.com/xlang-ai/text2reward);
  [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) repo;
  [RobotAgentBench-TrainingData](https://huggingface.co/datasets/AetherLabs-AI/RobotAgentBench-TrainingData) (card none);
  [XPlanner-Benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark);
  [rax skills-lib](https://github.com/robot-agent-exploration/skills-lib);
  [AgiBot-World code](https://github.com/OpenDriveLab/AgiBot-World) and
  [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) (card none, gated);
  [InternData-A1/M1](https://huggingface.co/datasets/InternRobotics/InternData-A1) (card none, gated);
  [tencent/PlanningBench](https://huggingface.co/datasets/tencent/PlanningBench) (licence file missing; general planning, not robotics);
  flymin SE dump card (content CC BY-SA upstream).
- **Copyleft / SA, flagged**: StackExchange (A12/B9); CasADi (LGPL-3); navigation2 LGPL-2.1+
  portions; pddlstream and `up-fast-downward` (GPL-3); common-pile per-paper CC BY-SA.
- **Custom / proprietary**: Isaac Sim (NVIDIA proprietary; Isaac Lab itself BSD-3); MuJoCo
  Menagerie per-model; SimReadyBench NVIDIA asset licence; [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) `openmdw-1.1`;
  senthurayyappan/robot-descriptions CAD sources (Fusion 360 Gallery terms **UNVERIFIED**).
- **Teacher-output terms**: CaPtain-SFT-Data (GPT-6 Astra/Sol, card MIT); RAX logs and A16
  (`generator: claude`); Gene robotics-instruct (Qwen3-4B); Eureka generated rewards (GPT-4).

## Top-10 additions (highest value first)

1. **MJCF/URDF verifier env** (B1/B7/C14): `mujoco` compile → 2 s settle (no NaN, no
   interpenetration above a threshold, base stays upright) → joint-limit, mass and inertia sanity →
   URDF round-trip via `yourdfpy`. Ground truth comes from held-out `robot_descriptions` and
   Menagerie robots. Covers S1 and closes gap 2.
2. **ROS 2 headless verifier** (B1/C14): Jazzy arm64 container; task = write a package (node,
   launch file, params, URDF/xacro). Verified by `colcon build` + `launch_testing` + topic/tf
   assertions, `ros2_control` mock hardware for controllers, and Nav2 + Gazebo headless for
   navigation tasks. Covers S2 and closes gap 1.
3. **Motion-planning verifier** (B1/C14): OMPL 2.0 Python (aarch64 wheels) and MoveIt 2 `moveit_py`.
   Scores planning success, collision-free check, path length and time against an OMPL reference.
   cuRoboV2 is optional once sm_110 is confirmed. Covers S3.
4. **Reward-design rollout verifier** (B5/B1): an Eureka-style loop on mjlab / MuJoCo Playground
   (Warp). The LLM writes the reward, a short PPO run follows, and fitness = task success on
   held-out seeds. Covers S7 and S6 (locomotion tasks).
5. **Control and dynamics verifiers** (B1/C14): acados/CasADi MPC in closed loop on a MuJoCo plant
   (tracking error, constraint violations); Pinocchio/Drake RNEA/ABA, Jacobian and IK answers
   checked numerically; python-control for linear design. Covers S4.
6. **robocode/KinDER generalized TAMP** (B3/C2; MIT, 2026): split the 28 envs between training and
   evaluation. It is the most text-compatible executable robotics eval found. Covers S9 and S5.
7. **CaP-X / CaPtain on Robosuite-LIBERO with a privileged-state text adapter** (C3/C4, B2):
   code-as-policy SFT plus RL with simulator rewards. Covers S5. Watch contamination between B2
   and C4.
8. **EAI** (C1) as a cheap, pure-Python planning regression floor, and **RoboEval** (C8) eval-only.
9. **A13 arXiv cs.RO 2024–2026 CC0 abstracts + A12 StackExchange robotics**: the only broad
   sources for S10 (sim-to-real, estimation) and S6 literature.
10. **A16 private agent sessions + PLAN.md self-traces** produced in this harness on held-out tasks
    from 1–5, so the routing statistics see in-format tool use for every sub-area.

Watch list: C7 Agentic Robotics Benchmark (more tasks, multi-arch images), C5 RLE-Bench CPU tasks
06–09 on aarch64, Isaac Lab 3.0 kit-less on Thor (would make Isaac Lab tasks and C9/C10
reachable).

## Search log (auditable; all run 2026-11-01)

- **HF datasets, newest first**: helper `s/hf.py datasets` (`/api/datasets?search=<t>&sort=createdAt&direction=-1&limit=100&full=true`,
  authenticated; keeps created ≥ 2026-01-01 with downloads ≥ 30 or likes ≥ 2). Terms: `ros2`, `ros`,
  `moveit`, `urdf`, `mjcf`, `mujoco`, `isaac`, `maniskill`, `genesis`, `robot code`,
  `robot planning`, `embodied`, `vla`, `open-x`, `droid`, `agibot`, `reward`, `locomotion`,
  `manipulation`, `sim2real`, `gazebo`, `pybullet`, `drake`, `robotics`, `robot instruct`,
  `robotics qa`, `embodied agent`, `robot task`, `alfred`, `robotics sft`, `robot sft`, `ros2 code`,
  `ros 2`, `ros-`, `robotics bench`, `robot bench`, `embodied bench`, `embodiedbench`, `vlabench`,
  `robocodex`, `code as policies`, `codeaspolicies`, `robot program`, `robotics reasoning`,
  `robot reasoning`, `robotics instruct`, `physical ai`, `motion planning`,
  `trajectory optimization`, `mpc`, `inverse kinematics`, `kinematics`, `pinocchio`, `urdf gen`,
  `robot description`, `task planning`, `pddl`, `llm planning`, `planning bench`, `eureka`,
  `text2reward`, `reward design`, `reward function`, `humanoid`, `quadruped`, `legged`,
  `whole-body`, `isaaclab`, `isaac lab`, `newton`, `warp`, `mujoco playground`, `libero`,
  `robocasa`, `behavior-1k`, `calvin`, `simpler`, `habitat`, `alfworld`, `virtualhome`,
  `robotics-qa`, `robot qa`, `robotics knowledge`, `robotics textbook`, `robotics papers`,
  `ros2 dataset`, `rclpy`, `launch file`, `nav2`, `colcon`, `urdf dataset`, `mjcf dataset`, `sdf`,
  `usd`, `omniverse`, `isaac sim`, `robot agent`, `embodied reasoning`, `embodied planning`,
  `robot code generation`, `robot llm`, `llm robot`, `robotic`, `robot-`, `control code`,
  `controller`, `pid`, `trajectory`, `grasp`, `dexterous`, `drone`, `uav`, `px4`, `ardupilot`,
  `autonomous`. Unfiltered (all 2026 rows, limit 200): `ros2`, `moveit`, `urdf`, `mjcf`, `nav2`,
  `robotics instruct`, `robocodex`, `robot code`, `embodiedbench`, `isaaclab`, `gazebo`,
  `pybullet`, `drake`, `pinocchio`, `ompl`, `curobo`. Download-sorted lookups: `BTGenBot`,
  `gensim`, `robogen`, `alfworld`, `embodied_agent_interface`, `ERQA`, `RoboVQA`, `robospatial`,
  `Cosmos-Reason`, `ros_answers`, `robotics stack`, `arxiv cs.RO`, `robotics_papers`,
  `Nemotron-Post-Training`, `PlanBench`, `planning bench`, `blocksworld`, `text2reward`, `eureka`,
  `robotwin`, `open-x-embodiment`, `droid`, `embodiedbench`, `VLABench`, `ManiSkill`,
  `physical reasoning`, `Cosmos-Reason2`, `common-pile/arxiv`, `arxiv_papers`,
  `stackexchange_2026`, `AgenticRoboRL`, `CaPtain`, `HexaShell`, `RoboHarness`, `EmbodiedSWE`,
  `cap-x`, `capgym`, `RLE-Bench`, `kinder`, `robot-agent`. Cards and licences via
  `/api/datasets/<id>`, `raw/main/README.md`, datasets-server `/size` and `/first-rows`. HF name
  search is dominated by LeRobot episode dumps and the `rose33300/*-submission` noise; those were
  filtered out.
- **arXiv**: export API (`export.arxiv.org/api/query?search_query=cat:cs.RO+AND+abs:"large language model"+AND+abs:benchmark&sortBy=submittedDate&sortOrder=descending`)
  returned **HTTP 429** on both attempts (20 s backoff). Substitute: HF Papers
  (`/api/papers/search?q=<q>`, publishedAt ≥ 2026-01-01). Queries: `robot code generation benchmark LLM`,
  `embodied agent benchmark LLM`, `robotics LLM benchmark`, `ROS 2 LLM`, `ROS code generation`,
  `URDF generation LLM`, `MJCF generation`, `reward design LLM robot`,
  `Eureka reward function generation`, `LLM task planning robot benchmark`,
  `vision-language-action model`, `VLA dataset 2026`, `robot learning dataset large-scale`,
  `sim-to-real reinforcement learning humanoid`, `MuJoCo Warp`, `Newton physics engine`,
  `motion planning LLM`, `model predictive control LLM`, `robotics coding agent`,
  `simulation environment generation LLM`, `robot operating system LLM agent`,
  `ROS2 benchmark language model`, `PDDL planning LLM benchmark`,
  `text-based embodied planning LLM`, `URDF`, `robot design large language model`,
  `reward function code generation reinforcement learning LLM`,
  `LLM generate simulation tasks robot`, `robotics question answering benchmark LLM`,
  `robotics knowledge benchmark`, `inverse kinematics LLM`, `trajectory optimization code LLM`,
  `humanoid control LLM agent`, `spatial reasoning text LLM robot`. Hits used: 2609.34210,
  2610.28589, 2610.23479, 2610.14018, 2610.15228, 2609.30233, 2609.27308, 2609.24170, 2607.04434,
  2605.15187, 2610.34649, 2603.22435, 2602.01870, 2610.22496, 2610.24587, 2610.16782, 2610.02089,
  2605.20873. **Rerun the arXiv API once the 429 clears.**
- **GitHub, recent** (`gh search repos <q> --created ">=2026-01-01" --sort stars`): `robot benchmark llm`,
  `robotics llm`, `embodied benchmark`, `robot code generation`, `ros2 llm`, `ros2 agent`,
  `ros2 mcp`, `urdf generation`, `mjcf`, `mujoco llm`, `reward design llm`, `code as policies`,
  `embodied agent`, `sim2real`, `urdf llm`, `vla benchmark`. The search API hit its 30/min limit
  twice; these queries **did not run**: `eureka reward`, `robotics agent`, `isaac lab llm`,
  `moveit llm`, `llm planning robot`, `mujoco warp`, `robot skills claude`, `robotics skill`,
  `coding agent robot`, `robot agent benchmark`, `ros2 skill`, `ros2 claude`, `mujoco agent`,
  `isaac lab agent`, `reward llm robot`. Licences and dates via `gh api repos/<r>` and
  `repos/<r>/license` (NOASSERTION repos read from the LICENSE text: acados, ompl, navigation2,
  drake, IsaacLab, underactuated, PythonRobotics).
- **Web**: RLE-Bench leaderboard and arXiv abstract (scores did not render), AgenticGenPlan site,
  CaP-X site, Embodied Agent Interface site, Embodied First Exam README, Isaac Lab 3.0 install docs
  (kit-less Newton, DGX Spark aarch64), ROS 2 Lyrical supported platforms, cuRobo on Jetson (no
  result).
- **Portability**: PyPI JSON (`/pypi/<pkg>/json`, wheel tags + `requires_dist`) for 60 packages;
  apt `Release` and `binary-arm64/Packages.gz` indices of `packages.ros.org` (noble, resolute) and
  `packages.osrfoundation.org` (noble); `/etc/os-release` on the host.
