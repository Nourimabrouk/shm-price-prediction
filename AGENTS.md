<!-- 
⚠️ NOTICE: This file contains AI agent instructions and configuration
This document exists primarily to guide AI coding agents working in this codebase.
Human developers may find it useful for understanding system architecture,
but it is specifically designed for automated AI agent collaboration.
-->

# Repository Guidelines

## Project Structure & Modules
- `internal_prototype/` (prototype): early pipeline modules — `preprocessing.py`, `feature_engineering.py`, `modeling.py`, `evaluation.py`, `pipeline.py`. Kept for compatibility; new work should prefer `src/` modules.
- `data/raw/`: committed input CSVs for out‑of‑the‑box runs (default `data/raw/Bit_SHM_data.csv`).
- `plots/` and `analysis_output/`: generated charts and artifacts (created on run).
- `src/viz_suite.py`, `src/viz_theme.py`: plotting helpers and theme (canonical viz location).
- `tests/`: test suite and smoke tests (e.g., `tests/test_viz.py`).

## Build, Test, and Dev Commands
- Create venv: `py -3 -m venv .venv` and activate PowerShell: `.\.venv\Scripts\Activate.ps1`.
- Install deps: `python -m pip install -r requirements.txt`.
- Run pipeline (time‑aware CatBoost + reports, legacy): `python -m internal_prototype.pipeline` or `python -m internal_prototype.pipeline --file data/raw/Bit_SHM_data.csv`.
- Legacy analysis script: `python internal/data_analysis.py`.
- Generate example plots: `python generate_plots.py` (or `generate_plots_efficient.py`).
- Smoke test visuals: `python tests/test_viz.py` (writes to `test_output/`).

## Coding Style & Naming
- Python 3.10+; follow PEP 8, 4‑space indentation, type hints (used across `src/` and `internal_prototype/`).
- Names: modules and functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Prefer pure functions and clear I/O boundaries; avoid hard‑coding paths (use `--file`).

## Testing Guidelines
- Use `python tests/test_viz.py` for quick environment validation.
- For data pipeline, a successful run prints metrics (RMSLE, MAPE, MAE) and saves `plots/model_performance.png` and `analysis_output/predictions_with_intervals.csv`.
- Suggested (optional): add pytest tests under `tests/` with files named `test_*.py`.

## Commit & Pull Requests
- Commits: use imperative mood; scope prefix optional (e.g., `feat: add PCA plot`, `fix: handle missing dates`, `chore: setup venv`).
- PRs must include: purpose/summary, key changes, how to run (`commands`), before/after screenshots of plots (if visuals), and any data path assumptions.
- Link related issues; keep diffs focused and documented in code where non‑obvious.

## Security & Configuration Tips
- Committed demo dataset: `data/raw/` is versioned to ensure notebooks run from a fresh clone.
- Reproducibility: pin deps via `requirements.txt`; avoid notebook‑only logic in pipeline.
- Matplotlib runs headless (`Agg`); artifacts are written to `plots/` and `analysis_output/` for review.
