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
  code of the evaluated snapshot for that benchmark, renamed.
  `eval/transfer/summarize.py` reproduces the RoboTwin 2.0 numbers (22.6 / 30.0 ever) from the saved per-episode rows and the
  CaP-Bench numbers (mean 75.0) from the original episode directories.

## Notes

- **RoboTwin 2.0 task files.** The official seed list is shipped (`official_hard_seeds.tsv`). The task files with RoboTwin's
  instructions are rebuilt with RoboTwin's own expert check (`make_official_tasks.sh`, which needs cuRobo and a GPU).
- **RoboDojo.** The shipped list is the full official 2,100-episode protocol. Unstable layouts are replaced with
  `replace_unstable.py`. RoboDojo needs a GPU with Vulkan raster support for Isaac Sim's camera annotators.
- **Real robot.** The hardware backend interface and an episode builder (`hardware_real`) are included; load your robot
  driver with `--hardware-driver`.
- **Request template.** `react/prompts/captain.request.json` is a captured Codex-CLI code-mode request, including its
  system prompt ("You are Codex, an agent based on GPT-6 ..."). CaPtain-4B was trained on this exact text, so it is kept verbatim.
