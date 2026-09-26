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

Track open questions in [GitHub Issues](https://github.com/hldrnwnv/fly-drone/issues) and link them from `docs/research-results.md`. Before experiments, reuse or create an issue stating the baseline, hypothesis, controls, held-out conditions, and artifacts. Save methods, traces, and negative results in `runs/`; update the issue and summary. Close only after the stated check is documented, including a failed hypothesis.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, such as `Add DNa02 steering readout`. Pull requests should explain the changed behavior, list commands run, and include a replay or key metrics for simulation changes. Link an issue when applicable.
