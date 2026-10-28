# AGENTS.md — guide for coding agents working in this repository

This file is for AI coding agents (Codex, Claude Code, Cursor, …). Humans: see [README.md](README.md).

## What this repository is

CaPtain-Harness is the environment a code-as-policy agent acts in. A vision-language model, served behind an
OpenAI-compatible chat endpoint, controls a robot by writing Python in a persistent sandboxed session. The session gives it
22 low-level capabilities plus `finish()`, and every robot call returns a text receipt. The same API runs on LIBERO
(+ PRO / Plus / MAX), RoboTwin 2.0, RoboDojo, SimplerEnv, CaP-Bench and real robots. CaPtain-4B was trained and evaluated
in it; the training pipeline lives in [CaPtain](https://github.com/waltstephen/CaPtain), which includes this repository as a
pinned submodule at `harness/`.

## Map

| You want to… | Go to |
|---|---|
| run one episode | `eval/run_episode.sh <suite> <task_id> <seed> <out_dir>`; transfer benchmarks: `eval/transfer/run_episode.sh <benchmark> <task.json> <out_root>` |
| run / score a task list | `eval/run_jobs.py <tasks.tsv> --out <dir>`, `eval/summarize.py <tasks.tsv> <dir>`; `eval/transfer/{run_jobs,summarize}.py <benchmark> ...` |
| change the agent loop, context compaction, model client | `src/captain_harness/react/` (driver entry point `scripts/react_episode.py`) |
| change what one episode does (budgets, completion gate, receipts plumbing, HTTP API) | `src/captain_harness/server/` (`env_server.py`, `episode_launch.py`, `task_markdown.py`) |
| change robot tools, receipts, the completion gate rules | `src/captain_harness/adapters/capx/`, contracts in `src/captain_harness/ports/` |
| change a simulator backend | `src/captain_harness/adapters/{libero,robotwin2,robodojo,simplerenv,capgym}/`; LIBERO variants in `server/libero_variants.py`, `server/libero_max_events.py` |
| connect real hardware | driver protocol in `src/captain_harness/adapters/hardware/facade.py`, episode builder `adapters/hardware/episode.py` |
| configs | `configs/benchmarks/` (templates with `${VAR}` paths), `configs/calibration/` (per-robot profiles), `configs/protocols/` |
| perception / policy / depth services | `services/start_perception.sh`, `services/serve_policy.sh`, `services/depth_server.py` |
| model weights | `scripts/download_models.sh`, one script per asset in `scripts/models/` |
| install a benchmark | `INSTALL.md` (one section per backend), `eval/transfer/README.md` (official protocols) |
| task lists | `eval/tasks/*.tsv` (LIBERO family), `eval/transfer/<benchmark>/` |
| tests | `tests/` |

## Rules (do not break these)

1. **Layering.** `adapters`/`ports` know the environment; `server` wraps one episode; `react` knows only workspaces, tools,
   text, images and a model API. `react` must not import `server` or `adapters` and must not contain robot, LIBERO or task
   vocabulary. `tests/test_import_boundaries.py` enforces this. The `server` package may touch `check_success` only at
   adapter construction, so backend-specific episode builders live in `adapters/<backend>/episode.py`.
2. **Hidden state stays hidden.** Nothing served to the model (HTTP responses, receipts, TASK.md, images) may carry object
   poses, contacts or the success predicate. `check_success` is wired once, at adapter construction. The episode output
   directory holds hidden audit files (`hidden_*`), so it is hidden from the sandbox (`--sandbox-hide`). The completion gate
   checks `finish()` only against the agent's own recorded measurements.
3. **Budgets live in the server.** The request cap, wall time and action budget are enforced server-side,
   and every request counts. The driver stops on the deadline, the cap, or when the model stops calling tools.
4. **Model-visible text is frozen for the released checkpoints.** When evaluating CaPtain-4B / CaPtain-4B-SFT, do not edit
   TASK.md templates (`server/task_markdown.py`), tool descriptions, receipts or `react/prompts/captain.request.json`.
   The template is a captured Codex-CLI request, including its system prompt, and the models were trained on exactly this
   text. If you change it for a new model, say so; the released checkpoints are then off-distribution.
5. **No hard-coded machines.** Paths, hosts and keys come from environment variables (`POLICY_URL`, `POLICY_MODEL`,
   `STUDENT_PYTHON`, `EGL_DEVICE`, `SIM_GPU`, `CAPTAIN_MODELS_DIR`, `CAPTAIN_BENCHMARKS_HOME`, `ROBOTWIN_ROOT`,
   `SIMPLERENV_ROOT`, `ROBODOJO_ROOT`, `CAPTAIN_DEPTH_URL`, `CAPTAIN_API_KEY`, `HF_TOKEN`, …). Benchmark configs use
   `${VAR}` placeholders rendered by `eval/transfer/render_config.py`. Never commit keys, tokens or weights.
6. **RL hooks are default-off.** `--mark-requests`, live marks (`.live_request_marks`), `CAPTAIN_FAST_LOOP` and
   `CAPTAIN_LAZY_RENDER` must not change behaviour when off.
7. **Backend options are opt-in.** Shared LIBERO / CaP-X modules carry backend-specific behaviour (for example the RoboDojo
   force model and IK-branch check) as attributes that are inactive unless that backend's builder sets them. Keep the
   LIBERO code path unchanged when adding options for another backend.

## Adding a backend (simulator or benchmark)

1. Put the scene side in its own environment if its dependencies clash with the harness. RoboTwin, RoboDojo and SimplerEnv
   run a scene server under the simulator's own Python and talk to the harness over a pipe. Implement the **driver
   protocol** of `adapters/hardware/facade.py`: `reset`, `read`, `command` (one control tick), `hold`, `frames` (RGB,
   metric depth or None, intrinsics, optional extrinsics). Reuse `XR1FacadeEnv` with a MuJoCo kinematic twin of the robot.
2. Write `adapters/<backend>/episode.py`, which builds the adapter and wires `check_success` once. Then register the suite in
   the env server's `episode_builder(suite)` dispatch and add `--<backend>-*` arguments.
3. Add a calibration profile in `configs/calibration/` and a config template in `configs/benchmarks/`, with paths as `${VAR}`.
4. Add the task list and a scorer to `eval/transfer/` (`run_episode.sh`, `run_jobs.py`, `summarize.py`), following the
   benchmark's own official protocol, and document it in `eval/transfer/README.md` and `INSTALL.md`. That documentation
   must state plainly which inputs go beyond the benchmark's official policy observation.
5. Extend `tests/test_transfer_wiring.py` and run the checks below.

## Adding a real robot

Write a driver object with the protocol above for your robot SDK and a factory `module:make_driver`. Then write a robot
config (twin MJCF, cameras with their links and depth source `sensor` or `mono`, control rate, workspace box, calibration
profile) and a task file, and run `eval/run_episode.sh hardware_real 0 0 <out> --server-arg=--hardware-config ...`
(see `docs/harness.md`). Safety limits belong in the driver and on the robot side. There is no success predicate, so
outcomes are labelled after the episode.

## Checks before you hand back

```bash
.venv-libero/bin/python -m pytest -q tests/test_react_*.py tests/test_server_env_server.py \
  tests/test_import_boundaries.py tests/test_transfer_wiring.py        # seconds, CPU
MUJOCO_GL=osmesa .venv-libero/bin/python -m pytest -q                  # full suite, ~20 min CPU (known failures: RELEASE_NOTES.md)
for f in scripts/*.sh scripts/models/*.sh services/*.sh eval/*.sh eval/transfer/*.sh; do bash -n "$f" || echo "FAIL $f"; done
scripts/download_models.sh --dry-run all
```

## Conventions

- Python 3.10 for the LIBERO harness interpreter (`.venv-libero`), Python 3.11 for the transfer harness interpreter
  (`.venv-transfer`) and for the agent's sandboxed interpreter (exact versions in `requirements-student.txt`).
- Success metrics: *host* = goal predicate on the final state; *strict* = host and the agent called `finish()`.
- Receipts are one line (≤ 300 characters) and factual; the policy learns from them.
- Episode outputs, logs and weights stay out of git (`runs/`, `logs/`, `models/`).
