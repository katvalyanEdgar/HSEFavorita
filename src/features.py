from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from config import DATE_COL, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS, FORECAST_HORIZON, ID_COLS, TARGET_COL
from src.data import FavoritaData


def select_series(train: pd.DataFrame, max_series: int | None = None) -> pd.DataFrame:
    series = train[ID_COLS].drop_duplicates()
    if max_series is None or len(series) <= max_series:
        return series.sort_values(ID_COLS).reset_index(drop=True)

    recent_start = train[DATE_COL].max() - pd.Timedelta(days=90)
    scores = (
        train.loc[train[DATE_COL] >= recent_start]
        .groupby(ID_COLS, observed=True)[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(max_series)
        .reset_index()[ID_COLS]
    )
    return scores.sort_values(ID_COLS).reset_index(drop=True)


def build_dense_grid(series_index: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    dates_df = pd.DataFrame({DATE_COL: pd.to_datetime(list(dates))})
    series = series_index[ID_COLS].drop_duplicates().copy()
    series["_key"] = 1
    dates_df["_key"] = 1
    grid = series.merge(dates_df, on="_key", how="inner").drop(columns="_key")
    return grid[[DATE_COL, *ID_COLS]].sort_values([*ID_COLS, DATE_COL]).reset_index(drop=True)


def build_target_grid(train: pd.DataFrame, series_index: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    grid = build_dense_grid(series_index, dates)
    if grid.empty:
        return grid.assign(**{TARGET_COL: np.float32(0.0), "onpromotion": np.int8(0)})

    min_date = grid[DATE_COL].min()
    max_date = grid[DATE_COL].max()
    observed = train.loc[
        train[DATE_COL].between(min_date, max_date),
        [DATE_COL, *ID_COLS, TARGET_COL, "onpromotion"],
    ].copy()
    observed = (
        observed.groupby([DATE_COL, *ID_COLS], as_index=False, observed=True)
        .agg({TARGET_COL: "sum", "onpromotion": "max"})
    )
    grid = grid.merge(observed, on=[DATE_COL, *ID_COLS], how="left")
    grid[TARGET_COL] = grid[TARGET_COL].fillna(0).astype("float32")
    grid["onpromotion"] = grid["onpromotion"].fillna(0).astype("int8")
    return grid


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dt = result[DATE_COL].dt
    iso = dt.isocalendar()
    result["dow"] = dt.dayofweek.astype("int8")
    result["day"] = dt.day.astype("int8")
    result["week"] = iso.week.astype("int16")
    result["month"] = dt.month.astype("int8")
    result["year"] = dt.year.astype("int16")
    result["is_weekend"] = (result["dow"] >= 5).astype("int8")
    result["is_month_start"] = dt.is_month_start.astype("int8")
    result["is_month_end"] = dt.is_month_end.astype("int8")
    result["is_payday"] = ((result["day"] == 15) | result["is_month_end"].astype(bool)).astype("int8")
    return result


def _holiday_aggregates(holidays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = holidays.loc[~holidays["transferred"]].copy()
    active["holiday_count"] = 1
    active["event_count"] = (active["type"] == "Event").astype("int8")

    national = (
        active.loc[active["locale"] == "National"]
        .groupby(DATE_COL, as_index=False)
        .agg(national_holiday_count=("holiday_count", "sum"), national_event_count=("event_count", "sum"))
    )
    regional = (
        active.loc[active["locale"] == "Regional"]
        .rename(columns={"locale_name": "state"})
        .groupby([DATE_COL, "state"], as_index=False)
        .agg(regional_holiday_count=("holiday_count", "sum"))
    )
    local = (
        active.loc[active["locale"] == "Local"]
        .rename(columns={"locale_name": "city"})
        .groupby([DATE_COL, "city"], as_index=False)
        .agg(local_holiday_count=("holiday_count", "sum"))
    )
    return national, regional, local


def add_exogenous_features(frame: pd.DataFrame, data: FavoritaData) -> pd.DataFrame:
    result = frame.copy()
    result = result.merge(data.items, on="item_nbr", how="left")
    result = result.merge(data.stores, on="store_nbr", how="left")
    result = result.merge(data.oil[[DATE_COL, "dcoilwtico"]], on=DATE_COL, how="left")

    national, regional, local = _holiday_aggregates(data.holidays)
    result = result.merge(national, on=DATE_COL, how="left")
    result = result.merge(regional, on=[DATE_COL, "state"], how="left")
    result = result.merge(local, on=[DATE_COL, "city"], how="left")

    holiday_cols = [
        "national_holiday_count",
        "national_event_count",
        "regional_holiday_count",
        "local_holiday_count",
    ]
    for col in holiday_cols:
        result[col] = result[col].fillna(0).astype("int16")
    result["any_holiday"] = (result[holiday_cols].sum(axis=1) > 0).astype("int8")

    result["dcoilwtico"] = result["dcoilwtico"].ffill().bfill()
    if result["dcoilwtico"].isna().any():
        result["dcoilwtico"] = result["dcoilwtico"].fillna(result["dcoilwtico"].median())

    result["onpromotion"] = result["onpromotion"].fillna(0).astype("int8")
    result["perishable"] = result["perishable"].fillna(0).astype("int8")
    result["class"] = result["class"].fillna(-1).astype("int32")
    result["cluster"] = result["cluster"].fillna(-1).astype("int16")
    return result


def add_lag_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    result = frame.copy()
    source = history[[DATE_COL, *ID_COLS, TARGET_COL]].copy()
    source[TARGET_COL] = source[TARGET_COL].fillna(0).clip(lower=0).astype("float32")

    for lag in lags:
        lagged = source.copy()
        lagged[DATE_COL] = lagged[DATE_COL] + pd.to_timedelta(int(lag), unit="D")
        lagged = lagged.rename(columns={TARGET_COL: f"lag_{lag}"})
        result = result.merge(lagged, on=[DATE_COL, *ID_COLS], how="left")

    source = source.sort_values([*ID_COLS, DATE_COL])
    for window in rolling_windows:
        col = f"rolling_mean_{window}_lag_{horizon}"
        rolled = source[[DATE_COL, *ID_COLS]].copy()
        rolled[col] = (
            source.groupby(ID_COLS, observed=True)[TARGET_COL]
            .transform(lambda values: values.rolling(int(window), min_periods=1).mean())
            .astype("float32")
        )
        rolled[DATE_COL] = rolled[DATE_COL] + pd.to_timedelta(int(horizon), unit="D")
        result = result.merge(rolled, on=[DATE_COL, *ID_COLS], how="left")

    lag_cols = [col for col in result.columns if col.startswith("lag_") or col.startswith("rolling_mean_")]
    for col in lag_cols:
        result[col] = result[col].fillna(0).astype("float32")
    return result


def make_feature_frame(
    base_frame: pd.DataFrame,
    history: pd.DataFrame,
    data: FavoritaData,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    result = base_frame.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL])
    result = add_calendar_features(result)
    result = add_exogenous_features(result, data)
    result = add_lag_features(result, history=history, lags=lags, rolling_windows=rolling_windows, horizon=horizon)

    object_cols = result.select_dtypes(include=["object"]).columns
    for col in object_cols:
        result[col] = result[col].fillna("нет_данных").astype("category")
    return result


def competition_weights(frame: pd.DataFrame) -> pd.Series:
    if "perishable" not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype="float32")
    return np.where(frame["perishable"].fillna(0).astype("int8") == 1, 1.25, 1.0).astype("float32")
