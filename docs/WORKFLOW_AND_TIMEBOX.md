# Workflow, Timebox, and Scope

## Purpose

This note provides a brief, professional overview of how this solution was planned and executed within the 3–5 hour timebox for the SHM price prediction tech case. It complements the instructions in `planning/bit-tech-case-instructions.txt` and is intended for reviewers to understand scope, workflow, and assumptions.

## Scope and timebox

- **Time constraint**: 3–5 hours total (EDA, modeling, reporting) as specified.
- **Deliverable level**: A curated, demo‑grade prototype that runs quickly end‑to‑end, suitable for review and discussion—not a production system.
- **Execution mode**: “Quick” paths favor small samples and defaults to keep wall‑clock runs to minutes.

## Working model and roles (agentic, parallel)

To stay within the timebox, we followed a parallel, role‑based workflow. Responsibilities were split along clear seams so tasks could proceed concurrently:

- **Data Analysis**: Rapid EDA to identify up to 5 key findings, with basic preprocessing guidance (missing hours treatment, `YearMade` sanity checks, temporal context).
- **Modeling**: Baseline Random Forest and an advanced CatBoost variant trained on sampled data with a time‑aware split and business‑oriented metrics (±15%/±25% tolerance).
- **Visualization**: Professional plots from a shared theme plus optional interactive dashboards; static figures prioritized for speed.
- **Migration/Packaging**: Repository structuring, CLI entrypoints, requirements, and concise documentation.
- **Orchestration**: Simple modes (`quick`, `analysis`, `modeling`, `full`) to demonstrate functionality within minutes.
- **Technical Analysis**: Brief rationale for choices, risks, and a pragmatic roadmap.

This division enabled simultaneous progress and short feedback loops (critique–test–validate), reducing elapsed time.

## Reuse and inspiration

- **Related work**: The approach is informed by well‑known patterns from the Blue Book for Bulldozers dataset (similar domain, problem structure, and temporal considerations).
- **Templates and utilities**: Shared visualization themes, evaluation scaffolds, and CLI wiring were reused to avoid reinventing standard components.

## What is prototype‑grade vs. deferred

- **Included in timebox**:
  - Data loading with robust column aliasing and basic feature engineering
  - Time‑aware splitting with simple audit prints
  - Baseline RF and CatBoost on samples with default or light‑touch parameters
  - Core plots (distribution, age vs price, temporal, product groups) using a consistent theme
  - CLI modes, quick runs, and minimal smoke‑style tests
- **Deferred beyond timebox**:
  - Full hyperparameter optimization across the entire dataset
  - Production APIs, CI/CD, and hardening
  - Comprehensive unit testing and performance benchmarking

## Disclosures and constraints

- **Performance numbers**: Any figures shown should be treated as illustrative from sample‑based runs in quick mode; full‑dataset, fully‑tuned results were out of scope for the strict timebox.
- **Interpretation**: Plots and metrics emphasize clarity for business stakeholders within the given time; further depth (e.g., SHAP, drift monitoring) is future work.

## How to reproduce the demo quickly

1) Create a virtual environment and install requirements per `README.md`.
2) Run a quick end‑to‑end demo:

```bash
python main.py --mode quick
```

This generates core figures and validation prints in minutes on a typical laptop.

## Alignment to the assignment brief

- Up to 5 key findings are derived via rapid EDA and reported in the notebooks/CLI output.
- Preprocessing steps, model choice (baseline + advanced), and evaluation methodology are documented and demonstrated.
- Time allocation and scoping decisions reflect the guidance in `planning/bit-tech-case-instructions.txt`.

## Closing note

This repository demonstrates a focused, business‑oriented prototype aligned to a 3–5 hour assessment. The structure intentionally favors clarity, speed, and demonstrability. With additional time, the same architecture can be extended for full optimization, broader validation, and productionization.
