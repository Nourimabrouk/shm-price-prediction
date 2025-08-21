from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore


@dataclass
class BaselineModel:
    pipeline: Pipeline


def train_baseline_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    categorical_cols: List[str],
    numeric_cols: List[str],
    random_state: int = 42,
) -> BaselineModel:
    """RandomForest baseline with simple preprocessing.

    - Numeric: median impute
    - Categorical: fill NA with Unknown + OneHot (rare bucketing via min_frequency)
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                sparse_output=True,
            ),
        ),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in numeric_cols if c in X_train.columns]),
            ("cat", cat_pipe, [c for c in categorical_cols if c in X_train.columns]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    pipe = Pipeline([("pre", pre), ("rf", rf)])
    pipe.fit(X_train, y_train)
    return BaselineModel(pipeline=pipe)


@dataclass
class AdvancedModel:
    model: object
    cat_features: List[int]
    feature_names: List[str]


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    categorical_cols: List[str],
    random_state: int = 42,
) -> Optional[AdvancedModel]:
    """Train a CatBoostRegressor using native categorical handling.

    Returns None if CatBoost is unavailable.
    """
    if CatBoostRegressor is None:
        return None

    # Fill missing categoricals with 'Unknown'. Leave numeric as-is (CatBoost tolerates NaN)
    X_train_cb = X_train.copy()
    for col in categorical_cols:
        if col in X_train_cb.columns:
            X_train_cb[col] = X_train_cb[col].astype("object").fillna("Unknown")

    feature_names = list(X_train_cb.columns)
    cat_features_idx = [i for i, c in enumerate(feature_names) if c in categorical_cols]

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.08,
        loss_function="RMSE",
        random_seed=random_state,
        iterations=600,
        verbose=False,
    )
    model.fit(X_train_cb, y_train, cat_features=cat_features_idx)
    return AdvancedModel(model=model, cat_features=cat_features_idx, feature_names=feature_names)


def predict_baseline(model: BaselineModel, X: pd.DataFrame) -> np.ndarray:
    return model.pipeline.predict(X)


def predict_advanced(model: AdvancedModel, X: pd.DataFrame, categorical_cols: List[str]) -> np.ndarray:
    X_cb = X.copy()
    for col in categorical_cols:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype("object").fillna("Unknown")
    return model.model.predict(X_cb)

