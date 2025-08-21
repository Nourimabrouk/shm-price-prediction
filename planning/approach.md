# Technical Approach (Draft)

- Goal: Professional, minimal pipeline for SHM heavy‑equipment price prediction within ~5 hours.
- Models: RandomForest (baseline) and CatBoost (advanced) with business metrics MAE, MAPE, RMSLE.
- Validation: Time‑aware holdout if a sale date exists; otherwise non‑shuffled order split.
- Preprocessing: Median imputation for numeric; explicit "Unknown" for categorical; preserve `machinehourscurrentmeter == 0` as valid signal.
- Features: Lightweight additions (sale_year, sale_month, equipment_age) when columns exist.
- Artifacts: `plots/model_performance.png`, `analysis_output/predictions_with_intervals.csv` (with simple residual‑based intervals).

Run: `python -m internal_prototype.pipeline --file data/raw/Bit_SHM_data.csv`

Known limitations: No heavy hyperparameter tuning; intervals are heuristic; feature set intentionally small for clarity.
