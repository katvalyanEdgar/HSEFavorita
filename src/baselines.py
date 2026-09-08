from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

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
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, AutoTheta, Naive
    except ImportError as exc:
        raise RuntimeError("установите statsforecast, чтобы запустить бейзлайны auto_theta и auto_ets.") from exc

    requested = list(model_names)
    models = []
    if "auto_theta" in requested:
        models.append(AutoTheta(season_length=season_length))
    if "auto_ets" in requested:
        models.append(AutoETS(season_length=season_length))
    if not models:
        return {}

    train_long = history[[DATE_COL, *ID_COLS, TARGET_COL]].copy()
    train_long["unique_id"] = train_long["store_nbr"].astype(str) + "_" + train_long["item_nbr"].astype(str)
    train_long = train_long.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})[["unique_id", "ds", "y"]]
    train_long["y"] = train_long["y"].fillna(0).clip(lower=0).astype("float32")

    horizon = int((future_frame[DATE_COL].max() - history[DATE_COL].max()).days)
    sf = StatsForecast(models=models, freq="D", n_jobs=n_jobs, fallback_model=Naive())
    forecast = sf.forecast(df=train_long, h=horizon).reset_index()

    future = future_frame[[DATE_COL, *ID_COLS]].copy()
    future["unique_id"] = future["store_nbr"].astype(str) + "_" + future["item_nbr"].astype(str)
    merged = future.merge(forecast, left_on=["unique_id", DATE_COL], right_on=["unique_id", "ds"], how="left")

    result: dict[str, pd.Series] = {}
    for requested_name in requested:
        candidates = {
            "auto_theta": ["AutoTheta", "AutoTheta_season_length-7"],
            "auto_ets": ["AutoETS", "AutoETS_season_length-7"],
        }[requested_name]
        column = next((col for col in candidates if col in merged.columns), None)
        if column is None:
            matching = [col for col in merged.columns if col.lower().startswith(requested_name.replace("auto_", "auto"))]
            column = matching[0] if matching else None
        if column is None:
            result[requested_name] = pd.Series(np.zeros(len(future_frame), dtype="float32"), index=future_frame.index)
        else:
            result[requested_name] = merged[column].fillna(0).clip(lower=0).astype("float32")
    return result
