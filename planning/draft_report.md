SHM Heavy‑Equipment Price Prediction — Solution‑First Report (Draft)

Answer first — We will replace the retiring expert’s “price sense” with a time‑aware CatBoost model trained on log‑price with age, hours, usage‑rate, product hierarchy, and geography as first‑class signals. We validate chronologically to avoid leakage and report MAPE/RMSLE plus a “within ±15%” business accuracy and 80% prediction intervals for decision‑grade quoting. The pipeline is already implemented as a modular CLI + Python package with EDA, temporal splitting, model training, evaluation, and professional visualizations.

0) Executive summary

What we built: A compact pricing pipeline: Data audit → Temporal EDA → Time‑aware modeling (RF baseline, CatBoost primary) → Evaluation (MAPE/RMSLE, ±15% accuracy) → Uncertainty (80% PI) → client‑ready visuals and CLI.

Why it fits this case: The dataset structure mirrors Kaggle’s Blue Book for Bulldozers—noisy auctions with strong time effects and high‑cardinality categoricals—where chronological splits and boosted trees repeatedly win.

What SHM gets on day one: A model that is accurate, auditable, and safe against leakage, with prediction intervals to manage risk. Success target: ≥80% of quotes within ±15% of actuals on a held‑out future window.

Business value: Faster quotes, consistent pricing, and quantified uncertainty for negotiation, while preserving explainability and upgrade paths (per‑group models, conformal intervals, external indices).

Placeholders to be filled after final runs:
– Topline result: RMSLE = [TBD], MAPE = [TBD]%, Within ±15% = [TBD]%, R² = [TBD] (Val: [YEAR] window).
– Key plot: plots/model_comparison.png; plots/catboost_performance.png.
– Intervals: 80% coverage = [TBD]%, avg width = $[TBD].

1) Highlights & design choices (internally meta‑ranked)

Meta‑ranking scale (importance → impact → confidence). Higher is better.

Temporal validation as a hard rule — prevents leakage; aligns with Bulldozers best practice; foundation for trust. [Rank: A+]. Implemented in both models.EquipmentPricePredictor.temporal_split_with_audit and hybrid_pipeline.temporal_split_with_comprehensive_audit.

CatBoost on log‑price — natively handles missing + high‑cardinality categoricals; competitive on tabular auctions; interpretable importances. [A].

RF baseline + comparison plotting — establishes a credible floor and sanity check. [A].

Prediction intervals (residual‑quantile heuristic; upgrade path to conformal) — elevates to decision‑grade pricing. [A‑].

Business‑aware EDA + visuals — practical findings (missing hours, crisis years, geography) and client‑grade figures. [A‑].

Hybrid pipeline (competition‑grade + production guardrails) — combines optimization and audits. [B+].

WeAreBit fit (exec‑first, prototyping mindset, B‑Corp lens) — report/storytelling aligned to consultancy context. [B+].

2) What stands out in the data (max 5 findings)

Extracted via eda.py (missingness, high cardinality, temporal volatility, age/usage anomalies, geography). These are the five to call out in the client deck:

Usage data is sparse: a large share of records miss MachineHoursCurrentMeter (zeros ≈ “unknown”); affects depreciation modeling and needs careful imputation/encoding. Business impact: high. Action: treat zeros as missing; build usage‑rate, add missingness flags.

High‑cardinality categoricals (e.g., ModelID, ModelDescription): >100 levels; specialized handling required (CatBoost or regularized target encodings). Impact: medium‑high.

Market regime volatility (crisis years): pronounced swings around 2008–2010; mandates chronological validation and possibly regime‑aware models. Impact: high.

Age & hours anomalies (implausible YearMade, extreme hours): requires sanity rules and robustness. Impact: medium. Action: cap ages; set YearMade<1930→NaN.

Geographic premia: state‑level price variation (filter low‑sample states). Impact: medium. Action: include state_of_usage features.

Figure placeholders
[Fig‑E1] Missingness radar + yearly heatmap (plots/06_missingness.png)
[Fig‑E2] Temporal trends (volume + median price) with shaded splits (plots/04_temporal_trends.png)
[Fig‑E3] State premia error‑bar chart (plots/09_state_premia.png)

3) Preprocessing steps (what & why)

Column normalization (snake_case; robust candidates for date/target/hours/IDs). Why: consistent feature plumbing. Implemented in data_loader.SHMDataloader.

Date parsing → time features: year, month, quarter, dow; age_years = sale_year − year_made (clip negatives & >50). Why: depreciation & seasonality.

Usage features: log1p_hours, hours_per_year = hours / max(age, 0.5). Why: normalize heavy‑tailed hours; capture intensity.

Missing value strategy:
– Numerics → median;
– Categoricals → "Unknown" sentinel;
– Hours zeros → NaN (auction convention). Why: aligns with Bulldozers signal; prevents bias.

Unit parsing (feet/inches) where beneficial coverage exists (screen size, widths). Why: salvage numeric signal from texty columns.

Keep lean v1 feature set (age, hours, usage‑rate, product hierarchy, geography, auctioneer, sale‑time). Why: clarity + speed under a 4‑hour constraint.

Note on internal consistency: early draft suggested “preserve hours==0 as valid signal” while the finalized loader converts zeros→NaN. We will standardize on zeros→NaN with a missingness flag (empirically strongest on Bulldozers).

4) Model choice & rationale

Baseline: Random Forest (fast, robust); establishes a credible benchmark and sanity check.

Primary: CatBoost Regressor on log1p(price), early stopping, depth 6–9; good with missing and high‑card categoricals; validated by Bulldozers literature.

Optimization (optional): time‑budgeted coarse‑to‑fine tuning around depth, learning‑rate, l2, iterations; lifted into train_optimized_catboost.

Alternatives: XGBoost/LightGBM with target/frequency encoding for the highest‑cardinality features; ensemble later if ROI is justified. (Strategic note retained for stakeholder Q&A.)

WeAreBit alignment: practical, interpretable, and shippable within a tight window—prototyping mindset over academic maximalism.

5) Evaluation methodology (business‑first)

Split: Strict chronological hold‑out (train≤T₁, val=T₂, test=T₃) with an audit log confirming no temporal overlap; option to test regime years separately. Implemented in both model and hybrid pipelines.

Metrics:
– RMSLE (primary model metric on log‑price);
– MAPE/MAE/RMSE/R²;
– Within ±10/15/25% accuracy (sales‑friendly KPI). Implemented in evaluation.ModelEvaluator.

Uncertainty: 80% prediction intervals via residual quantiles (heuristic), with a roadmap to conformal for calibrated PIs.

Visualization: model comparison grid; actual vs predicted with ±15% bands; error histogram; interval coverage. Auto‑export to plots/.

Placeholders
[Tbl‑R1] Validation metrics (RF vs CatBoost) — to be inserted
[Fig‑R1] plots/model_comparison.png (auto‑generated)
[Fig‑R2] plots/catboost_performance.png (actual vs predicted + tolerances)

6) Implementation overview (what exists, how to run)

CLI: python -m src.cli --file data/Bit_SHM_data.csv [--quick | --optimize --budget 30 | --eda-only] — discovery, EDA, training, evaluation, reporting.

Core modules:
– data_loader.py: robust loading, normalization, feature engineering, audits.
– eda.py: missingness, cardinality, temporal patterns, anomalies, geography; top‑5 findings utility.
– models.py: RF/CatBoost, temporal split with audit, optimization routines.
– evaluation.py: business metrics, ± tolerance, intervals, and plots.
– hybrid_pipeline.py: end‑to‑end orchestration + comprehensive audits.
– viz_suite.py + viz_theme.py: professional visual pack.

Artifacts (expected)
– plots/model_comparison.png, plots/catboost_feature_importance.png, plots/catboost_performance.png
– analysis_output/predictions_with_intervals.csv (row‑aligned, with 80% PI)
– docs/findings.md (top‑5), docs/approach.md (method), notebook with outputs.

7) Results (to be populated)

Validation window: [YYYY]

CatBoost: RMSLE = [TBD], MAPE = [TBD]%, Within ±15% = [TBD]%, R² = [TBD]

Random Forest: RMSLE = [TBD], MAPE = [TBD]%, Within ±15% = [TBD]%, R² = [TBD]

Intervals (80%): coverage=[TBD]% (target≈80%), avg width=$[TBD]

Top drivers (importance): Age, ProductGroup/ModelID, Hours, State, SaleMonth. [Insert feature‑importance figure]

8) Risks, constraints & mitigations

Temporal leakage → enforce chronological splits + encoded transforms fit on train only (audited).

Regime shifts (e.g., 2008–2010) → regime‑aware analysis; recent‑year weighting; per‑segment models if needed.

Missing hours → zeros→NaN + missingness flags; rely on age/usage‑rate & hierarchy; evaluate sensitivity.

High‑cardinality → CatBoost native handling now; target/frequency encoding alternatives available.

Over‑cleaning → retain “useful noise” characteristic of auctions (Bulldozers lesson).

9) Why this is a good decision for SHM (and Bit)

Reliable now: protects against classic traps (time leakage), delivers business‑readable KPIs, and runs end‑to‑end from a single CLI.

Explainable enough: feature importances, tolerance bands, and prediction intervals support sales conversations.

Upgradeable later: per‑ProductGroup models, conformal intervals, external market indices, text features from descriptions, and ensembles—without rethinking the architecture.

Consultancy‑fit: packaged as a professional, minimal prototype that communicates clearly to mixed stakeholders—exactly how WeAreBit profiles its client work.

10) Time allocation (realistic 4–6 h assessment)
Block	Minutes	Output
EDA & audit	60–90	Key findings, visuals (missingness, trends, geography)
Preprocessing v1	40–60	Normalized table, engineered features
Modeling	60–90	RF baseline; CatBoost primary; (opt) coarse tuning
Evaluation	30–45	Metrics, ±15% accuracy, intervals, plots
Report packaging	45–60	This report + notebook with outputs

This aligns with the case brief guidance.

11) Roadmap & next steps

Week 1: lock data scope; run full pipeline; finalize metrics + interval calibration (optionally conformal).

Month 1: per‑ProductGroup ablation; add lightweight external indices (if provided) for market normalization; dashboard export.

Quarter: ensemble & regime‑aware models; human‑in‑the‑loop review flows; A/B in quoting tool; margin impact tracking.

12) Appendix (operational notes)

How to run:
python -m src.cli --file data/Bit_SHM_data.csv --optimize --budget 30 (full pipeline)
python -m src.cli --quick --model catboost (fast sample)

Deliverables list (expected on final turn‑in):
– Notebook: EDA → Modeling → Evaluation → Intervals (all outputs included).
– Report: This document (PDF/MD).
– Plots: /plots/*.png (performance, comparison, importances, EDA).
– CSV: predictions with 80% PI.

Open questions for SHM: currency/tax normalization, channel mix (“datasource”), preferred KPI (MAPE vs tolerance band), and pricing reference year for historical normalization.

References (context & priors)

Case requirements & structure (Bit brief).

Technical approach draft & plan (our earlier blueprint).

Kaggle Bulldozers insights (temporal splits, high‑cardinality, “useful noise”).

WeAreBit culture & expectations (storytelling, prototyping).

13) Critical evaluation of our current implementation (concise)

Strengths
– Proper temporal splitting with audits; leakage risk addressed.
– Clean metric suite (MAPE, RMSLE, tolerance) + intervals with clear coverage stats.
– Production‑like ergonomics (CLI, modular modules, plots).

Gaps to tighten before final

Hours==0 policy: unify (zeros→NaN + flag) across loader and models; re‑run sensitivity.

Feature list contract: lock the “lean set” and ensure consistent handling in preprocess_data.

Interval calibration: the residual‑quantile heuristic is fine for v1; add conformal upgrade note in the report and (if time) a quick calibration check.

Per‑group variant: schedule a short ablation (global vs per‑ProductGroup) and keep only if ROI > modest threshold.

Visualization wiring: ensure viz_suite uses the same cleaned table as modeling to avoid column name drift.

End matter

This draft is structured to drop in the final numbers and figures with minimal edits. Once your latest training run completes, paste the metrics into §7, export the listed plots, and the deck is client‑ready for Bit/SHM review.