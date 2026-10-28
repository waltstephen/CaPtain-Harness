# Release notes (v1.0.0)

CaPtain-Harness v1.0.0 is the harness of the CaPtain v1.0.0 release (previously its `harness/` directory), published as a
standalone repository. The code is unchanged. What is new in this repository: `pyproject.toml` metadata and package data,
the model download scripts (`scripts/download_models.sh`, `scripts/models/`), the MoGe-2 depth service
(`services/depth_server.py`), `INSTALL.md`, `AGENTS.md` and the documentation in `docs/`.

## What was verified

- **Evaluation tooling.** `eval/summarize.py`, run on the original CaPtain-4B evaluation episodes with the task lists in
  `eval/tasks/`, reproduces every number of the paper's main table: LIBERO 73.8 / 67.5; LIBERO-PRO Pos+Task 78.5 / 71.5
  (6-cell) and 68.6 / 58.9 (8-cell); LIBERO-Plus 66.1 / 59.0; LIBERO-PRO 4-category 68.8 / 60.6; LIBERO-MAX Dynamic / Base
  61.2 / 70.0. `eval/run_episode.sh` uses exactly the command-line protocol of those evaluations.
- **Harness.** The release harness is the evaluated snapshot, with these changes:
  - packages and identifiers renamed;
  - the provider-specific Responses backend removed (`--auth none|bearer`);
  - the LIBERO-PRO / Plus / MAX variant options added;
  - the default-off RL hooks added (request marks, live marks, fast loop, visual variation).
  None of these changes touch model-visible text. The driver and server tests, the import-boundary test and the
  transfer-wiring test pass.
- **Transfer backends** (`adapters/{robotwin2,robodojo,simplerenv,capgym,hardware}`, `eval/transfer/`). Each backend is the
  code of the evaluated snapshot for that benchmark, renamed; source hashes were checked against the run records.
  `eval/transfer/summarize.py` reproduces the RoboTwin 2.0 numbers (22.6 / 30.0 ever) from the saved per-episode rows and the
  CaP-Bench numbers (mean 75.0) from the original episode directories.

## Known limitations

- **Full test suite.** Some adapter tests have long-standing failures that already exist in the evaluated snapshot. Most
  are legacy adapter tests for code paths the python-runtime contract no longer uses. On a CPU-only machine (no GL renderer) the full suite gives
  1,122 passed, 99 failed, 26 errors, the same failing set as the evaluated snapshot. The fast CPU subset (see `AGENTS.md`)
  and `tests/test_transfer_wiring.py` (with MuJoCo 3) pass.
- **No GPU end-to-end run of this tree.** vLLM serving, EGL episodes and the transfer simulators (SAPIEN 3, Isaac Sim 5.1,
  SAPIEN 2.2) were not re-run during release preparation. The backends were checked by imports, CPU wiring tests and
  source hashes. RoboDojo needs a GPU with Vulkan raster support for Isaac Sim's camera annotators.
- **Model download scripts and the depth service** were checked with `bash -n`, `--dry-run` and against the model ids and
  revisions used by the code. The weights were not downloaded during release preparation.
- **RoboTwin 2.0 task files.** The official seed list is shipped (`official_hard_seeds.tsv`). The task files with RoboTwin's
  instructions are rebuilt with RoboTwin's own expert check (`make_official_tasks.sh`, which needs cuRobo and a GPU).
- **RoboDojo.** The shipped list is the full official 2,100-episode protocol. Unstable layouts are replaced with
  `replace_unstable.py`, not automatically.
- **Real robot.** The hardware backend interface and an episode builder (`hardware_real`, driver loaded from
  `--hardware-driver`) are included. The robot driver, the robot's twin model and the embodiment-adapted checkpoint used
  for the paper's real-robot episodes are not.
- **Request template.** `react/prompts/captain.request.json` is a captured Codex-CLI code-mode request, including its
  system prompt ("You are Codex, an agent based on GPT-6 ..."). CaPtain-4B was trained on this exact text, so it is kept verbatim.
- **Benchmark licenses.** LIBERO-plus and LIBERO-MAX state no license (see `THIRD_PARTY_LICENSES.md`).
- **LIBERO-10 task 5.** Its goal predicate has a ~3.5 mm offset in z, so a correct flat placement is scored as a failure.
  The task is kept in all evaluations, as in the paper.
