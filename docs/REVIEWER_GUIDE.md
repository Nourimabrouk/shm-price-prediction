# Reviewer Guide

## Goal

Help reviewers navigate the repository in minutes and reproduce the demo.

## Fast path (5 minutes)

1. Install requirements (see README Quick Start).
2. Run the quick demo (front-and-center quick path):

```bash
python main.py --mode quick
```

3. Open `notebooks/03_final_executive_analysis.ipynb` for the executive view.

Optional one-click executive view (HTML):

```bash
python tools/run_notebooks.py --pattern "03_final_executive_analysis.ipynb" --html
```

Then open: `outputs/notebooks_html/03_final_executive_analysis.html`

Outputs:

- Figures in `outputs/figures/`
- Console metrics: RMSE, ±15%/±25% accuracy

## What to look at

- `docs/WORKFLOW_AND_TIMEBOX.md`: Scope, timebox, and what is prototype‑grade.
- `src/data_loader.py`: Robust aliasing, basic features, temporal parsing.
- `src/models.py`: RF and CatBoost training with time‑aware split and business metrics.
- `src/evaluation.py`: Business‑focused metrics and plots.
- `src/viz_suite.py`: Professional static visualization suite.
- `main.py` / `src/cli.py`: Orchestration and quick/analysis/modeling modes.

## Notes on results

- Performance numbers in README are illustrative from quick‑mode, sampled runs to fit the timebox.
- Full‑dataset, fully‑tuned results are out of scope for this assessment.

## Suggested review flow

1) Run quick demo → confirm figures/metrics generate.
2) Skim `src/models.py` temporal split and metrics.
3) Review 3–5 static figures (distribution, age vs price, temporal).
4) Skim `docs/WORKFLOW_AND_TIMEBOX.md` to see constraints and scope decisions.

## Out of scope (timebox)

- Full HPO on 400K+ records
- Production APIs/CI/CD
- Extensive unit test coverage and benchmarking
