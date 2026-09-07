from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype="float64").clip(min=0)


def nwrmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, weights: np.ndarray | pd.Series | None = None) -> float:
    y_true_arr = _clip(y_true)
    y_pred_arr = _clip(y_pred)
    if weights is None:
        weights_arr = np.ones_like(y_true_arr, dtype="float64")
    else:
        weights_arr = np.asarray(weights, dtype="float64")
    squared = (np.log1p(y_pred_arr) - np.log1p(y_true_arr)) ** 2
    return float(np.sqrt(np.sum(weights_arr * squared) / np.sum(weights_arr)))


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return nwrmsle(y_true, y_pred)


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(_clip(y_true) - _clip(y_pred))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(np.mean((_clip(y_true) - _clip(y_pred)) ** 2)))


def wmape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, eps: float = 1e-9) -> float:
    y_true_arr = _clip(y_true)
    y_pred_arr = _clip(y_pred)
    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / max(np.sum(np.abs(y_true_arr)), eps))


def evaluate_predictions(
    frame: pd.DataFrame,
    target_col: str = "unit_sales",
    pred_col: str = "prediction",
    weight_col: str = "weight",
) -> dict[str, float]:
    weights = frame[weight_col] if weight_col in frame.columns else None
    return {
        "nwrmsle": nwrmsle(frame[target_col], frame[pred_col], weights),
        "rmsle": rmsle(frame[target_col], frame[pred_col]),
        "mae": mae(frame[target_col], frame[pred_col]),
        "rmse": rmse(frame[target_col], frame[pred_col]),
        "wmape": wmape(frame[target_col], frame[pred_col]),
    }
