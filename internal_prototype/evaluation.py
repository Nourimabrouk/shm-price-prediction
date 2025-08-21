from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.maximum(0.0, np.asarray(y_true))
    y_pred = np.maximum(0.0, np.asarray(y_pred))
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
    }


def residual_intervals(y_true: np.ndarray, y_pred: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    resid = y_true - y_pred
    lo = np.quantile(resid, lower_q)
    hi = np.quantile(resid, upper_q)
    y_lower = np.maximum(0.0, y_pred + lo)
    y_upper = np.maximum(y_lower, y_pred + hi)
    return y_lower, y_upper


def plot_actual_vs_pred(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    out_path: str,
    sample: int = 2000,
):
    n = len(y_true)
    idx = np.arange(n)
    if n > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=sample, replace=False)

    plt.figure(figsize=(7, 6))
    max_val = float(np.percentile(y_true, 99))
    for name, p in preds.items():
        plt.scatter(y_true[idx], p[idx], s=10, alpha=0.6, label=name)
    plt.plot([0, max_val], [0, max_val], "k--", lw=1, label="ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Model Performance: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

