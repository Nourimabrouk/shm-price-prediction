from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import feature_engineering as fe
from . import preprocessing as pp
from . import modeling as mdl
from . import evaluation as ev


def time_aware_split(
    X: pd.DataFrame, y: Optional[pd.Series], date_col: Optional[str], test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    if date_col and date_col in X:
        order = X[date_col].argsort(kind="mergesort")
        cutoff = int(len(X) * (1 - test_size))
        train_idx = order[:cutoff]
        valid_idx = order[cutoff:]
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = y.iloc[train_idx] if y is not None else None
        y_valid = y.iloc[valid_idx] if y is not None else None
        # Split audit
        start_train, end_train = X_train[date_col].min(), X_train[date_col].max()
        start_val, end_val = X_valid[date_col].min(), X_valid[date_col].max()
        print(f"[split] Train: {start_train} → {end_train} | Valid: {start_val} → {end_val}")
        return X_train, X_valid, y_train, y_valid
    else:
        # Fallback: simple holdout (no shuffle to avoid accidental leakage in ordered files)
        n = len(X)
        cutoff = int(n * (1 - test_size))
        X_train, X_valid = X.iloc[:cutoff], X.iloc[cutoff:]
        y_train = y.iloc[:cutoff] if y is not None else None
        y_valid = y.iloc[cutoff:] if y is not None else None
        print("[split] No date column detected — using simple holdout by order.")
        return X_train, X_valid, y_train, y_valid


def ensure_dirs():
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("analysis_output").mkdir(parents=True, exist_ok=True)


def main(file: str):
    ensure_dirs()

    print(f"[load] Reading data from: {file}")
    df = pp.load_csv(file)
    meta = pp.detect_columns(df)

    if not meta.target_col:
        print("[warn] No target column detected. Proceeding without training.")

    # Parse dates and add light features
    df = pp.parse_dates(df, meta.date_col)
    df = fe.add_basic_features(df, date_col=meta.date_col, yearmade_col=meta.yearmade_col)

    # Impute (preserve zeros in machinehours)
    df = pp.apply_basic_imputation(df, meta)

    # Prepare X/y
    X, y = pp.prepare_features(df, meta)

    # Identify columns lists fresh from X post-FE
    categorical_cols = [c for c in meta.categorical_cols if c in X.columns]
    numeric_cols = [c for c in meta.numeric_cols if c in X.columns]

    # Split
    X_train, X_valid, y_train, y_valid = time_aware_split(X, y, meta.date_col)

    results: Dict[str, Dict[str, float]] = {}
    preds_for_plot: Dict[str, np.ndarray] = {}
    rows_out = []

    # Baseline RF
    if y_train is not None:
        print("[fit] Training baseline RandomForest…")
        rf = mdl.train_baseline_rf(
            X_train, y_train, categorical_cols=categorical_cols, numeric_cols=numeric_cols
        )
        p_val = mdl.predict_baseline(rf, X_valid)
        preds_for_plot["RF"] = p_val
        results["RandomForest"] = ev.evaluate_all(y_valid.values, p_val)
        y_lower, y_upper = ev.residual_intervals(y_valid.values, p_val)
        # Choose an id column if present
        id_series = (
            df.loc[X_valid.index, meta.id_col] if meta.id_col and meta.id_col in df else X_valid.index
        )
        for idx, yt, yp, yl, yu in zip(id_series, y_valid.values, p_val, y_lower, y_upper):
            rows_out.append({
                "id": idx,
                "y_true": float(yt),
                "y_pred": float(yp),
                "y_lower": float(yl),
                "y_upper": float(yu),
                "model": "RandomForest",
            })

    # Advanced CatBoost
    if y_train is not None:
        cb_model = mdl.train_catboost(
            X_train, y_train, categorical_cols=categorical_cols
        )
        if cb_model is not None:
            print("[fit] Training advanced CatBoost…")
            p_val_cb = mdl.predict_advanced(cb_model, X_valid, categorical_cols)
            preds_for_plot["CatBoost"] = p_val_cb
            results["CatBoost"] = ev.evaluate_all(y_valid.values, p_val_cb)
            y_lower, y_upper = ev.residual_intervals(y_valid.values, p_val_cb)
            id_series = (
                df.loc[X_valid.index, meta.id_col] if meta.id_col and meta.id_col in df else X_valid.index
            )
            for idx, yt, yp, yl, yu in zip(id_series, y_valid.values, p_val_cb, y_lower, y_upper):
                rows_out.append({
                    "id": idx,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "y_lower": float(yl),
                    "y_upper": float(yu),
                    "model": "CatBoost",
                })
        else:
            print("[warn] CatBoost not available. Skipping advanced model.")

    # Save artifacts
    if y_train is not None and len(preds_for_plot) > 0:
        ev.plot_actual_vs_pred(y_valid.values, preds_for_plot, "plots/model_performance.png")

        out_df = pd.DataFrame(rows_out)
        out_df.to_csv("analysis_output/predictions_with_intervals.csv", index=False)

        # Print metrics table
        print("\n[metrics]")
        for name, m in results.items():
            print(
                f"- {name}: MAE={m['MAE']:.2f} | MAPE={m['MAPE']:.2f}% | RMSLE={m['RMSLE']:.4f}"
            )


def _default_data_path() -> str:
    # Try the default file under data/
    default = Path("data/raw/Bit_SHM_data.csv")
    if default.exists():
        return str(default)
    # Fallback: first CSV under data*
    for root in [Path("data"), Path(".")]:
        if root.exists():
            for p in root.rglob("*.csv"):
                return str(p)
    raise SystemExit("No CSV file found. Provide --file <path>.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHM price prediction pipeline")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV data file")
    args = parser.parse_args()

    data_path = args.file or _default_data_path()
    main(data_path)

