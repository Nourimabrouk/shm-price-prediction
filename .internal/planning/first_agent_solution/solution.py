"""
solution.py
===========

This script contains a short exploratory analysis and a baseline model for
predicting the sales price of second‑hand machinery in the SHM data set.

It performs the following steps:

1. **Load the data** – read the CSV provided in the challenge.
2. **Basic cleaning and feature engineering** – parse dates, handle
   missing or implausible values, and derive a few temporal features
   from the sales date.  The year ``1000`` is used as a sentinel for
   unknown build years, so it is set to missing.
3. **Missing value imputation** – median imputation for numeric columns
   and ``'Unknown'`` for categoricals.
4. **Train/test split** – build a simple hold‑out split to estimate
   generalisation performance.
5. **Model training** – fit a ``CatBoostRegressor``, which natively
   handles mixed numeric and categorical data, on a random sample of
   50 k rows to demonstrate feasibility without consuming excessive
   compute.  Categorical features are passed in by index.
6. **Evaluation** – report both root mean squared error (RMSE) and
   root mean squared logarithmic error (RMSLE) on the validation set.
7. **Feature importance** – display the top 10 important features
   according to the fitted model.

Running this script will print descriptive statistics, missing value
percentages, and model performance metrics.  It is intended as a
starting point for a more comprehensive analysis.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error

try:
    from catboost import CatBoostRegressor
except ImportError as e:
    raise SystemExit(
        "CatBoost is required for this script. Please install it via\n"
        "    pip install catboost\n"
        "before running."
    )


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV file and return a pandas DataFrame."""
    # Use low_memory=False to suppress dtype warnings due to mixed types
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic feature engineering and cleaning."""
    df = df.copy()

    # Convert sales date to datetime and extract useful components
    df['Sales date'] = pd.to_datetime(df['Sales date'], errors='coerce')
    df['sale_year'] = df['Sales date'].dt.year
    df['sale_month'] = df['Sales date'].dt.month
    df['sale_dayofyear'] = df['Sales date'].dt.dayofyear
    df.drop(columns=['Sales date'], inplace=True)

    # Replace sentinel year 1000 with NaN – indicates unknown manufacture year
    df.loc[df['Year Made'] < 1900, 'Year Made'] = np.nan

    return df


def impute_missing(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Impute missing values.  Numeric columns are filled with the median,
    categorical columns with ``'Unknown'``.  Returns the imputed DataFrame
    and the list of categorical column names.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Remove target from numeric columns if present
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Median imputation for numeric columns
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Fill categorical missing with 'Unknown'
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    return df, cat_cols


def run_model(df: pd.DataFrame, target_col: str, cat_cols: list[str], sample_frac: float = 0.12) -> None:
    """
    Train a CatBoostRegressor on a sampled subset of the data and
    evaluate performance.  Prints performance metrics and top
    feature importances.

    ``sample_frac`` controls what fraction of rows to use for
    training/validation.  A value of 0.12 (~50k rows) strikes a
    balance between speed and representativeness.  Use 1.0 for the
    full dataset if compute allows.
    """
    # Drop ID-like columns that have no predictive value
    drop_cols = ['Sales ID', 'Machine ID', 'Model ID', 'Unnamed: 0']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Sample a subset for quick training; shuffle beforehand to ensure randomness
    df_sampled = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    X = df_sampled.drop(columns=[target_col])
    y = df_sampled[target_col]

    # Determine categorical feature indices for CatBoost
    cat_features = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    # Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and fit the model
    # Use a modest number of trees to keep the runtime reasonable for the case study
    model = CatBoostRegressor(
        iterations=150,
        learning_rate=0.1,
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    # Predictions and metrics
    preds = model.predict(X_valid)
    # To compute RMSLE we need to ensure predictions are positive
    preds_clipped = np.clip(preds, a_min=1, a_max=None)
    rmsle = np.sqrt(mean_squared_log_error(y_valid, preds_clipped))
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    print(f"Validation RMSE: {rmse:,.2f}")
    print(f"Validation RMSLE: {rmsle:,.4f}")

    # Show feature importances
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    top10 = importance_df.sort_values('importance', ascending=False).head(10)
    print("\nTop 10 important features:")
    for _, row in top10.iterrows():
        print(f"{row['feature']}: {row['importance']:.2f}")


def main():
    # Determine the location of the CSV.  When running from a notebook
    # ``__file__`` may be undefined, so fall back to the current working
    # directory.  The data directory is assumed to be ``shm_data`` alongside
    # this script.
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path('.')
    data_path = base_dir / 'shm_data' / 'Bit_SHM_data.csv'
    df = load_data(str(data_path))
    print(f"Loaded data shape: {df.shape}")

    # Descriptive statistics and missing value overview
    desc = df['Sales Price'].describe()
    print("\nSales Price descriptive statistics:")
    print(desc)

    # Show top missing columns
    missing = df.isna().sum().sort_values(ascending=False)
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'missing': missing, 'missing_percent': missing_percent})
    print("\nTop 10 columns by missing values:")
    print(missing_df.head(10))

    df = engineer_features(df)
    df, cat_cols = impute_missing(df, target_col='Sales Price')

    # Run the baseline model on a sample of the data
    run_model(df, target_col='Sales Price', cat_cols=cat_cols, sample_frac=0.12)


if __name__ == '__main__':
    main()