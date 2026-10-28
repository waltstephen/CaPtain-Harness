# Installing CaPtain-Harness

This page covers the harness itself, the agent's sandboxed Python, the perception services and the policy server, then
one section per backend. Every backend uses the same harness, agent Python and services; only the simulator side differs.
Run the commands from the repository root (`<harness>` = this checkout). Model weights are covered in
[Models](#4-models-and-perception-services); `scripts/download_models.sh` downloads them.

| Backend | Suite(s) | Harness interpreter | Simulator environment | GPU / rendering |
|---|---|---|---|---|
| [LIBERO](#libero) | `libero_{spatial,object,goal,10,90}` | `.venv-libero` (Python 3.10) | the same venv (robosuite 1.4, MuJoCo 2.3.7) | NVIDIA EGL |
| [LIBERO-PRO / Plus / MAX](#libero-pro--libero-plus--libero-max) | same suites + `--libero-variant`, `--max-case` | `.venv-libero` | `.venv-libero-pro`, `.venv-libero-plus` (Python 3.10) | NVIDIA EGL |
| [RoboTwin 2.0](#robotwin-20) | `robotwin2_sim` | `.venv-transfer` (Python 3.11) | RoboTwin @ ea8b211, SAPIEN 3 (Python 3.10) | Vulkan |
| [RoboDojo](#robodojo) | `robodojo_sim` | `.venv-transfer` | RoboDojo @ cb2a37d, Isaac Sim 5.1 + Isaac Lab (Python 3.11) | Vulkan raster |
| [SimplerEnv](#simplerenv) | `simplerenv_sim` | `.venv-transfer` | SimplerEnv @ eddb569, ManiSkill2_real2sim @ ef7a4d4, SAPIEN 2.2.2 (Python 3.11) | Vulkan + EGL |
| [CaP-Bench](#cap-bench) | `capgym_{lift,stack,restack,nut,wipe}` | `.venv-libero` | none (rebuilt on robosuite 1.4) | NVIDIA EGL |
| [Hardware](#hardware-real-robots) | `hardware_real` | `.venv-transfer` | your robot driver | none for the harness |

## 1. System requirements

- Linux x86_64. The paper used NVIDIA GPUs (80 GB for the policy server; one more GPU for perception and rendering).
- `git`, Python 3.10 and Python 3.11 (3.11.16 for the agent's interpreter).
- `bubblewrap` ≥ 0.5 (`bwrap`). The agent's Python runs in a bwrap sandbox, which needs `--clearenv`.
- An NVIDIA driver with EGL for MuJoCo. If your system's glvnd lists only Mesa, point
  `__EGL_VENDOR_LIBRARY_FILENAMES` at an NVIDIA vendor file. `~/.config/glvnd/10_nvidia.json` is picked up automatically.
- Vulkan (the NVIDIA ICD) only for the SAPIEN and Isaac Sim backends.
- About one CPU core per running episode, for MuJoCo and the sandboxed Python.

## 2. The harness (environment server and ReAct driver)

```bash
git clone https://github.com/waltstephen/CaPtain-Harness.git && cd CaPtain-Harness
scripts/setup_third_party.sh                       # LIBERO @ 8f1084e into third_party/libero (--capx: also CaP-X + sam3, for the SAM 3 service)

python3.10 -m venv .venv-libero
.venv-libero/bin/pip install -r requirements-libero.txt          # pinned: mujoco 2.3.7, robosuite 1.4.0, bddl 1.0.1, torch 2.6.0, numpy 2.2.6, ...
.venv-libero/bin/pip install -e third_party/libero --no-deps
.venv-libero/bin/pip install -e . --no-deps                      # package captain-harness, import name captain_harness
```

The install must be editable (`pip install -e .`), because the harness reads `configs/`, `data/` and `third_party/` from
the checkout. The extras `.[simulator]`, `.[transfer]` and `.[test]` list the same dependencies loosely; the
`requirements-*.txt` files hold the exact versions of the paper runs. LIBERO writes its `config.yaml` to
`<harness>/.libero-config` on the first episode, or to `$LIBERO_CONFIG_PATH` if that is set.

## 3. The agent's Python (sandboxed interpreter)

The model's `python` tool runs this interpreter inside bwrap. CaPtain-4B was trained and evaluated with these exact
versions, and the driver checks them before every episode.

```bash
python3.11 -m venv ~/envs/student-py311 && ~/envs/student-py311/bin/pip install -r requirements-student.txt  # numpy 1.26.4, scipy 1.13.1, Pillow 10.4.0
export STUDENT_PYTHON=~/envs/student-py311/bin/python
```

## 4. Models and perception services

| Service | Port | Model | Download | Needed for |
|---|---|---|---|---|
| SAM 3 segmentation (CaP-X FastAPI server) | 8114 | `facebook/sam3` (`sam3.pt`), **gated** | `scripts/models/download_sam3.sh` | every backend |
| Molmo2-8B pointing (vLLM) | 8122 | `allenai/Molmo2-8B` @ `e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b` | `scripts/models/download_molmo2.sh` | every backend |
| Policy (vLLM, OpenAI-compatible) | 8600 | CaPtain-4B / CaPtain-4B-SFT, or any model | `scripts/models/download_policy.sh` | every backend |
| MoGe-2 monocular depth | 8130 | `Ruicheng/moge-2-vitl-normal` @ `cb0e8bbd6b1e243589717c78e750b1ba4c093acf` | `scripts/models/download_moge.sh` | only `hardware_real` cameras without a depth sensor |

```bash
export HF_TOKEN=...                                 # an account that accepted the SAM License on huggingface.co/facebook/sam3
CAPTAIN_MODELS_DIR=/data/models scripts/download_models.sh            # sam3 + molmo2 (default); add `moge` / `policy`; --dry-run prints only
```

The scripts can be re-run safely: a finished download is skipped (`FORCE=1` downloads again). They read `HF_TOKEN` from the
environment and never store it. `sam3.pt` and MoGe's `model.pt` are checked against the sha256 of the files used for the paper.

**SAM 3 service.** It needs an environment with CaP-X and its `sam3` submodule (`scripts/setup_third_party.sh --capx`;
install as described in `third_party/cap-x/README.md`). **Molmo2-8B** needs a vLLM environment that supports it.

```bash
TOOLS_PYTHON=/path/to/capx-env/bin/python VLLM_BIN=/path/to/vllm-env/bin/vllm \
  SAM3_CHECKPOINT=$CAPTAIN_MODELS_DIR/sam3/sam3.pt MOLMO_MODEL=$CAPTAIN_MODELS_DIR/Molmo2-8B \
  SAM3_GPU=1 MOLMO_GPU=1 services/start_perception.sh                 # SAM 3 :8114 + Molmo2-8B :8122, ~30 GB together
```

**Policy server.** vLLM 0.29.0 (cu129, torch 2.13) with transformers 5.17 was used for Qwen3.5-based models:

```bash
VLLM_BIN=/path/to/vllm-env/bin/vllm POLICY_GPU=0 services/serve_policy.sh $CAPTAIN_MODELS_DIR/CaPtain-4B 8600
```

Instead of serving a local model, you can use any OpenAI-compatible chat endpoint: set `POLICY_URL=http://<host>:<port>/v1`
and `POLICY_MODEL=<served name>`. If the endpoint needs a key, export `CAPTAIN_API_KEY` (or `OPENAI_API_KEY`) and append
`--auth bearer` to the episode command.

**MoGe-2 depth service** (optional). It needs torch, the MoGe code, fastapi and uvicorn:

```bash
pip install "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" fastapi uvicorn
python services/depth_server.py --checkpoint $CAPTAIN_MODELS_DIR/moge-2-vitl-normal/model.pt --port 8130 --device cuda:0
```

## 5. Per-backend requirements

### LIBERO

- **Environment.** `.venv-libero` from section 2, with LIBERO at the commit pinned in `third_party/manifest.json`. LIBERO's
  assets ship with the repository. The official init states come from LIBERO.
- **Seeds.** Seeds 11–16 select the official init states (`init_states[seed % 50]`), which are the paper's held-out
  states. Seeds ≥ 1000 select the fresh initial states in `data/libero_fresh_init/`, which are used for training-style rollouts.
- **Rendering.** MuJoCo renders headless through EGL (`EGL_DEVICE` picks the GPU).
- **Inputs the agent uses.** LIBERO's agentview and wrist RGB-D cameras with intrinsics and extrinsics, measured joint
  state and gripper opening, and a calibrated wrist force signal (`configs/calibration/libero_franka_v*.json`).

### LIBERO-PRO / LIBERO-Plus / LIBERO-MAX

Each of these forks is a drop-in `libero` package in its own Python 3.10 venv. The harness looks for checkouts and venvs under
`$CAPTAIN_BENCHMARKS_HOME` (default `~/benchmarks`), or in `.venv-libero-pro` / `.venv-libero-plus` at the harness root.

```bash
export CAPTAIN_BENCHMARKS_HOME=~/benchmarks && mkdir -p $CAPTAIN_BENCHMARKS_HOME && cd $CAPTAIN_BENCHMARKS_HOME
# LIBERO-PRO (evaluated at eafdb80); bddl_files + init_files from the HF dataset zhouxueyang/LIBERO-Pro, as in its README
git clone https://github.com/Zxy-MLlab/LIBERO-PRO && git -C LIBERO-PRO checkout eafdb809426b13153aa1e4c42d6601844217dfec
python3.10 -m venv .venv-libero-pro && .venv-libero-pro/bin/pip install -r <harness>/requirements-libero.txt \
  && .venv-libero-pro/bin/pip install -e LIBERO-PRO --no-deps
# LIBERO-plus @ 4976dc3; assets.zip from the HF dataset Sylvest/LIBERO-plus unzipped into LIBERO-plus/libero/libero
git clone https://github.com/sylvestf/LIBERO-plus && git -C LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
python3.10 -m venv .venv-libero-plus && .venv-libero-plus/bin/pip install -r <harness>/requirements-libero.txt wand==0.7.2 \
  && .venv-libero-plus/bin/pip install -e LIBERO-plus --no-deps
# LIBERO-MAX (evaluated at 01625e2): imported from the checkout, runs in the LIBERO-plus venv
git clone https://github.com/liberomax/LIBERO-MAX && git -C LIBERO-MAX checkout 01625e290b1aa07511bdc820781e635e851353cc
```

- **LIBERO-plus.** Its sensor-noise perturbation needs ImageMagick 6 for `wand`. Use the system package, or unpack the
  libraries under `~/local/magick` (see `server/libero_variants.variant_env`). With numpy 2, two numpy-1 idioms in
  `libero/libero/envs/env_wrapper.py` must be replaced: `np.fromstring` → `np.frombuffer` and `np.float_` → `np.float64`.
- **Frozen assets.** The PRO Pos+Task frozen init states (800 `.npy` files, sha256 in the task list) and the 80 LIBERO-MAX
  cases come from the CaPtain artifacts (`eval/`, released with the CaPtain models). Pass them with
  `eval/run_jobs.py --artifacts <dir>`. `$LIBERO_MAX_ROOT` overrides the location of the LIBERO-MAX checkout.
- **Inputs the agent uses.** The same as LIBERO. The perturbation is applied by the fork; the agent is not told which one.

### RoboTwin 2.0

- **Environment.** RoboTwin @ `ea8b211` in a Python 3.10 venv with torch 2.4.1 / torchvision 0.19.1 and RoboTwin's
  `scripts/requirements.txt`.
  - Apply `eval/transfer/robotwin2/robotwin.patch` and run `patch_unstable.py`.
  - Apply RoboTwin's own sapien `urdf_loader` / mplib source fixes.
  - Unzip `embodiments.zip` + `objects.zip` from the HF dataset `TianxingChen/RoboTwin2.0` into `assets/`.
- **cuRobo.** cuRobo v0.7.8 is needed only to rebuild the official seed list (`make_official_tasks.sh`, RoboTwin's expert check).
- **Harness side.** `.venv-transfer` (below), and `ROBOTWIN_ROOT` / `ROBOTWIN_PYTHON` set.
- **GPU.** SAPIEN 3 renders with Vulkan (`SIM_GPU`).
- **Inputs the agent uses**, beyond RoboTwin's official observation:
  - simulated depth from the head and wrist cameras;
  - joint velocities;
  - the measured finger opening;
  - a wrist contact-force signal (summed contact impulses on the gripper links);
  - SAM 3 + Molmo2-8B.
- **Embodiment.** Two Franka Pandas (RoboTwin's `franka-panda` embodiment) are placed on the official instances.

```bash
python3.11 -m venv .venv-transfer && .venv-transfer/bin/pip install -r requirements-transfer.txt   # mujoco 3.3.7 for the kinematic twins
```

### RoboDojo

- **Environment.**
  - Isaac Sim 5.1 in a Python 3.11 venv: `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`.
  - RoboDojo @ `cb2a37d` with the Isaac Lab and cuRobo revisions it pins, installed into that venv as described in RoboDojo's README.
  - RoboDojo's assets from Hugging Face, including `Assets/Eval_Layout/RoboDojo/arx_x5/0`.
  - A small script that activates the venv and puts the RoboDojo checkout on `PYTHONPATH`; pass it as `ROBODOJO_ENV_SH`. Also set `ROBODOJO_ROOT`.
- **Harness side.** `.venv-transfer`.
- **GPU.** One per episode process. Isaac Sim's camera annotators need a GPU with Vulkan raster support; GPUs without it need
  a vendor-specific workaround that is not included. Isaac Sim writes logs to `$TMPDIR` (`RD_TMPDIR`, default `~/robodojo/tmp`).
- **Inputs the agent uses**, beyond RoboDojo's official observation:
  - simulated depth images;
  - camera intrinsics and extrinsics;
  - a simulated wrist force/torque reading (the PhysX joint-6 wrench; the X5 has no wrist sensor);
  - the measured gripper aperture;
  - SAM 3 + Molmo2-8B.
- **Budgets.** 120 model requests and 3,600 s per episode.

### SimplerEnv

- **Environment.**
  - SimplerEnv @ `eddb569` (with `--recurse-submodules`) and ManiSkill2_real2sim @ `ef7a4d4`, in a Python 3.11 venv.
  - Install `"setuptools<70" -r eval/transfer/simplerenv/requirements-simplerenv.txt` (sapien 2.2.2, numpy 1.24.4, gymnasium 0.29.1), then `pip install -e` both checkouts.
  - Set `SIMPLERENV_ROOT` / `SIMPLERENV_PYTHON`, and optionally `SIMPLERENV_EGL_VENDOR_FILE`.
- **Harness side.** `.venv-transfer`.
- **GPU.** SAPIEN 2.2 needs the NVIDIA Vulkan ICD and the glvnd EGL vendor file.
- **Inputs the agent uses**, beyond SimplerEnv's single RGB camera:
  - the rendered depth of that camera, with its intrinsics and extrinsics;
  - an **added simulated wrist camera** (512×512, fovy 75°, RGB-D; not part of the official protocol);
  - a wrist contact-force signal;
  - SAM 3 + Molmo2-8B.
- **Control.** The official delta end-effector controller is bypassed. The harness commands joint targets through the robot's own PD drives.

### CaP-Bench

- **Environment.** Nothing extra: the five single-arm CaP-Gym tasks are rebuilt on robosuite 1.4 (`adapters/capgym`) and
  run in `.venv-libero`. CaP-X itself pins robosuite 1.5, which conflicts with LIBERO.
- **GPU.** EGL, as for LIBERO.
- **Inputs the agent uses.** The LIBERO harness inputs: head RGB-D at CaP-Gym's camera pose, robosuite's wrist camera, and
  calibrated wrist force. No privileged poses.

### Hardware (real robots)

- **Environment.** `.venv-transfer` (MuJoCo 3 for the kinematic twin) plus your robot's own SDK.
- **Driver.** You write a driver for the protocol in `adapters/hardware/facade.py`: `reset`, `read` (measured joints,
  velocities, gripper opening, optional wrist force), `command` (one control tick), `hold` and `frames` (RGB, metric
  depth or None, intrinsics, optional extrinsics).
- **Config.** Write a robot config (twin MJCF, cameras with their links and depth source, control rate, workspace box,
  calibration profile) and a task file. Load the driver with `--hardware-driver module:factory`.
- **Depth.** Cameras without a depth sensor can use `"depth": "mono"`: the MoGe-2 service (`services/depth_server.py`,
  port 8130, `--hardware-depth-url` or `$CAPTAIN_DEPTH_URL`). One scale per frame is fitted to the support plane.
- **Inputs the agent uses.** Whatever your driver returns: RGB, depth (sensor or MoGe-2), intrinsics/extrinsics, measured
  joints and gripper, and optional wrist force. There is no success predicate; label outcomes after the episode.

## 6. Check the install

```bash
.venv-libero/bin/python -m pytest -q tests/test_react_*.py tests/test_server_env_server.py \
  tests/test_import_boundaries.py tests/test_transfer_wiring.py         # CPU, seconds
scripts/download_models.sh --dry-run all                                # what the model scripts would do
curl -s http://127.0.0.1:8600/v1/models                                 # the policy server is up
```
