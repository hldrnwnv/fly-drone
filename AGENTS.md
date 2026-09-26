# Repository Guidelines

## Project Structure & Module Organization

Python package code lives in `src/fly_drone/`: `brain.py` adapts MaleCNS, `wing_motors.py` reads wing motor neurons, `wing_adapter.py` maps wing intent to quad forces and moments, `perception.py` runs optional depth and flow networks, and `sensory_adapter.py` encodes their outputs. `sim.py` contains the planar prototype, `fpv.py` contains MuJoCo physics, `chase.py` handles moving-target pursuit, and `vision.py`/`benchmark.py` handle camera comparisons. Research probes and genetic search live in `scripts/`; model limitations are in `docs/`. The quad model is in `assets/fpv_quad.xml`; tests are in `tests/`. Downloaded connectome files belong in `data/flybrain/`; generated trajectories, metrics, and videos belong in `runs/`. Keep published neural data separate from the task adapter and quad model.

## Build, Test, and Development Commands

Create an environment with `python3 -m venv .venv`, then install `.venv/bin/python -m pip install -e '.[fpv]'`; add `perception` for pretrained vision. Set `FLY_DATA="$PWD/data/flybrain"` when running `fly-drone` or any `fly-fpv` experiment, including `wing-direct`. The first run downloads about 260 MB of neural data. Use `--output runs/trial` to keep trials separate. Run checks with `.venv/bin/python -m unittest discover -s tests -v`.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public functions, `snake_case` for Python names, and `PascalCase` for classes. Keep simulation math deterministic and isolate calls to `flybrain` in `brain.py` and `wing_motors.py`. Add dependencies to `pyproject.toml`; avoid committing virtual environments or downloaded data. No formatter or linter is configured yet, so keep changes consistent with nearby code.

## Testing Guidelines

Name test files `test_*.py` and test methods `test_*`. Tests cover geometry, command limits, camera bearings, gate crossing, and moving-target pursuit without downloading the connectome. For neural changes, inspect `results.json`: compare held-out accuracy, gate success or pursuit time, open-loop decisions on identical observations, and decision latency. Keep image processing, neural steering, and speed control distinct; one successful flight is not general proof.

## Research Questions & GitHub Issues

Track open questions in [GitHub Issues](https://github.com/hldrnwnv/fly-drone/issues) and link them from `docs/research-results.md`. Before experiments, reuse or create an issue stating the baseline, hypothesis, controls, held-out conditions, and artifacts. Save methods, traces, and negative results in `runs/`; update the issue and summary. After each research step, comment on its issue with the exact files or code changed, experiment commands, conditions, observed metrics, interpretation, and remaining work. Update the issue status automatically: close it when every stated check and its result are documented, including a failed hypothesis; leave it open with an explicit remaining check when incomplete, and reopen it if later evidence invalidates closure.

Every research report must cite the sources found during the investigation, preferably the original dataset, paper, or maintainer documentation for each substantive external claim. Record dataset version and file hashes when they affect reproducibility; distinguish source facts, implementation choices, observations, and interpretations. Include an infographic in every research report and link or embed it beside the relevant results. The infographic must show the experiment's inputs, controls, and key outcomes, label units and sample counts, remain legible without color alone, and be generated from saved data with its source code retained. Report negative and inconclusive results in the same way as positive ones.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, such as `Add DNa02 steering readout`. Pull requests should explain the changed behavior, list commands run, and include a replay or key metrics for simulation changes. Link an issue when applicable.

Before creating any pull request or merge request, ask the independent, read-only `next_issue_scout` subagent defined in `.codex/agents/next-issue-scout.toml` to inspect the finished diff, current results and artifacts, and existing issues. Give it the evidence and issue link without suggesting a desired conclusion. Wait for its response, then include a **Proposed next issues** section in the PR/MR description with its evidence-backed candidates, or state that it found none. Each candidate needs a distinct question, a link to the result that motivates it, a baseline, controls, a falsifiable check, and expected artifacts. Check for duplicates before proposing creation. The subagent proposes; it does not create issues, edit the PR/MR, or change issue status. Keep its independent conclusion separate from the author's interpretation.
