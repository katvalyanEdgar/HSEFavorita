from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import DATE_COL, ID_COLS, TARGET_COL


def predict_naive(frame: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    if frame.empty or history.empty:
        return pd.Series(np.zeros(len(frame), dtype="float32"), index=frame.index)

    last_values = (
        history.sort_values([*ID_COLS, DATE_COL])
        .groupby(ID_COLS, as_index=False, observed=True)
        .tail(1)[[*ID_COLS, TARGET_COL]]
        .rename(columns={TARGET_COL: "prediction"})
    )
    pred = frame[[*ID_COLS]].merge(last_values, on=ID_COLS, how="left")["prediction"]
    return pred.fillna(0).clip(lower=0).astype("float32")


def predict_seasonal_naive(frame: pd.DataFrame, history: pd.DataFrame, season_length: int = 7) -> pd.Series:
    if frame.empty or history.empty:
        return pd.Series(np.zeros(len(frame), dtype="float32"), index=frame.index)

    cutoff = history[DATE_COL].max()
    lookup = history[[DATE_COL, *ID_COLS, TARGET_COL]].copy()
    lookup = lookup.rename(columns={TARGET_COL: "prediction", DATE_COL: "source_date"})

    work = frame[[DATE_COL, *ID_COLS]].copy()
    horizon_step = (work[DATE_COL] - cutoff).dt.days.clip(lower=1)
    seasonal_step = ((horizon_step - 1) % int(season_length)) + 1
    work["source_date"] = pd.to_datetime(cutoff) - pd.to_timedelta(int(season_length) - seasonal_step, unit="D")

    pred = work.merge(lookup, on=[*ID_COLS, "source_date"], how="left")["prediction"]
    return pred.fillna(0).clip(lower=0).astype("float32")


def predict_statsforecast(
    history: pd.DataFrame,
    future_frame: pd.DataFrame,
    model_names: Iterable[str],
    season_length: int = 7,
    n_jobs: int = 1,
) -> dict[str, pd.Series]:
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:
        raise RuntimeError("установите statsmodels, чтобы запустить auto_theta и auto_ets.") from exc

    requested = list(model_names)
    if not requested:
        return {}

    history_end = history[DATE_COL].max()
    horizon = int((future_frame[DATE_COL].max() - history_end).days)
    future_dates = pd.date_range(history_end + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_lookup = future_frame[[DATE_COL, *ID_COLS]].copy()
    result = {name: pd.Series(np.zeros(len(future_frame), dtype="float32"), index=future_frame.index) for name in requested}

    def fallback_forecast(y: pd.Series) -> np.ndarray:
        if y.empty:
            return np.zeros(horizon, dtype="float32")
        tail = y.tail(season_length).to_numpy(dtype="float32")
        if len(tail) == 0:
            return np.zeros(horizon, dtype="float32")
        return np.resize(tail, horizon).clip(min=0).astype("float32")

    def forecast_theta(y: pd.Series) -> np.ndarray:
        if len(y) < season_length * 2 or float(y.sum()) == 0.0:
            return fallback_forecast(y)
        try:
            fitted = ThetaModel(y, period=season_length, deseasonalize=True, method="auto").fit()
            return np.asarray(fitted.forecast(horizon), dtype="float32").clip(min=0)
        except Exception:
            return fallback_forecast(y)

    def forecast_ets(y: pd.Series) -> np.ndarray:
        if len(y) < season_length * 2 or float(y.sum()) == 0.0:
            return fallback_forecast(y)
        candidates = [
            {"trend": None, "seasonal": None},
            {"trend": "add", "seasonal": None},
            {"trend": None, "seasonal": "add"},
            {"trend": "add", "seasonal": "add"},
        ]
        best_score = np.inf
        best_forecast: np.ndarray | None = None
        for params in candidates:
            try:
                seasonal_periods = season_length if params["seasonal"] is not None else None
                model = ExponentialSmoothing(
                    y,
                    trend=params["trend"],
                    seasonal=params["seasonal"],
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True)
                score = float(getattr(fitted, "aic", np.inf))
                if not np.isfinite(score):
                    score = float(getattr(fitted, "sse", np.inf))
                if score < best_score:
                    best_score = score
                    best_forecast = np.asarray(fitted.forecast(horizon), dtype="float32")
            except Exception:
                continue
        if best_forecast is None:
            return fallback_forecast(y)
        return best_forecast.clip(min=0).astype("float32")

    grouped = list(history.groupby(ID_COLS, observed=True))
    for key, group in tqdm(grouped, desc="классические ряды", unit="ряд", leave=False):
        key = key if isinstance(key, tuple) else (key,)
        y = (
            group.groupby(DATE_COL, observed=True)[TARGET_COL]
            .sum()
            .sort_index()
            .asfreq("D", fill_value=0)
            .clip(lower=0)
            .astype("float64")
        )
        forecasts = {}
        if "auto_theta" in requested:
            forecasts["auto_theta"] = forecast_theta(y)
        if "auto_ets" in requested:
            forecasts["auto_ets"] = forecast_ets(y)

        forecast_frame = pd.DataFrame({DATE_COL: future_dates})
        for id_col, value in zip(ID_COLS, key):
            forecast_frame[id_col] = value
        matched = future_lookup.reset_index().merge(forecast_frame, on=[DATE_COL, *ID_COLS], how="inner")
        if matched.empty:
            continue
        offsets = (matched[DATE_COL] - future_dates[0]).dt.days.to_numpy()
        for name, values in forecasts.items():
            result[name].loc[matched["index"].to_numpy()] = values[offsets]

    for name in result:
        result[name] = result[name].fillna(0).clip(lower=0).astype("float32")
    return result
