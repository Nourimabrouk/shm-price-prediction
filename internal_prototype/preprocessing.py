from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import find_column


DATE_CANDIDATES = ["saledate", "sale_date", "SaleDate", "date", "Date"]
TARGET_CANDIDATES = ["SalePrice", "saleprice", "Price", "price", "target"]
YEARMADE_CANDIDATES = ["YearMade", "yearmade", "year_made", "year"]
MACHINEHOURS_CANDIDATES = [
    "MachineHoursCurrentMeter",
    "machinehourscurrentmeter",
    "MachineHours",
    "machine_hours",
]
ID_CANDIDATES = ["SalesID", "salesid", "id", "ID"]


@dataclass
class DatasetMeta:
    target_col: Optional[str]
    date_col: Optional[str]
    yearmade_col: Optional[str]
    machinehours_col: Optional[str]
    id_col: Optional[str]
    categorical_cols: List[str]
    numeric_cols: List[str]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with basic dtype inference and safe NA handling."""
    df = pd.read_csv(path, low_memory=False)
    return df


def detect_columns(df: pd.DataFrame) -> DatasetMeta:
    target = find_column(TARGET_CANDIDATES, df.columns)
    date = find_column(DATE_CANDIDATES, df.columns)
    yearmade = find_column(YEARMADE_CANDIDATES, df.columns)
    machinehours = find_column(MACHINEHOURS_CANDIDATES, df.columns)
    id_col = find_column(ID_CANDIDATES, df.columns)

    # Infer dtypes for cat/num split
    categorical_cols = [
        c for c in df.columns if df[c].dtype == "object" and c not in {target}
    ]
    numeric_cols = [
        c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in {target}
    ]

    return DatasetMeta(
        target_col=target,
        date_col=date,
        yearmade_col=yearmade,
        machinehours_col=machinehours,
        id_col=id_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


def parse_dates(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def apply_basic_imputation(
    df: pd.DataFrame, meta: DatasetMeta, preserve_zero_for_hours: bool = True
) -> pd.DataFrame:
    """Apply light imputation: median for numeric, 'Unknown' for categoricals.

    Preserves zeros for `machinehourscurrentmeter` (never treated as missing).
    """
    # Preserve zero counts for machine hours (sanity check after imputation)
    zero_hours_before = None
    if preserve_zero_for_hours and meta.machinehours_col and meta.machinehours_col in df:
        zero_hours_before = (df[meta.machinehours_col] == 0).sum()

    # Numeric: fill NaN with median (but never convert zeros to NaN)
    for col in meta.numeric_cols:
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)

    # Categoricals: fill NaN with explicit 'Unknown'
    for col in meta.categorical_cols:
        df[col] = df[col].astype("object").fillna("Unknown")

    # Sanity check: ensure zero counts preserved for machinehours
    if zero_hours_before is not None and meta.machinehours_col in df:
        zero_hours_after = (df[meta.machinehours_col] == 0).sum()
        assert (
            zero_hours_after == zero_hours_before
        ), "machinehourscurrentmeter zeros must be preserved during preprocessing"

    return df


def prepare_features(
    df: pd.DataFrame, meta: DatasetMeta
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return features X and target y (if available). Does not drop rows.

    Excludes the target column from X.
    """
    y = df[meta.target_col] if meta.target_col and meta.target_col in df else None
    feature_cols = [c for c in df.columns if c != meta.target_col]
    X = df[feature_cols].copy()
    return X, y

