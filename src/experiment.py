from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    DATE_COL,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_TRAIN_WINDOW_DAYS,
    DEFAULT_VALID_DAYS,
    FORECAST_HORIZON,
    ID_COLS,
    RESULTS_DIR,
    TARGET_COL,
)
from src.baselines import predict_naive, predict_seasonal_naive, predict_statsforecast
from src.data import load_favorita
from src.features import build_target_grid, competition_weights, make_feature_frame, select_series
from src.metrics import evaluate_predictions
from src.models_dl import predict_torch_mlp
from src.models_ml import predict_catboost
from src.utils import ensure_dir, timestamp, write_json


BASELINE_MODELS = {"naive", "seasonal_naive", "auto_theta", "auto_ets"}
SUPPORTED_MODELS = {"naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"}


def _date_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def _parse_models(models: list[str]) -> list[str]:
    unknown = sorted(set(models) - SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"неизвестные модели: {', '.join(unknown)}. поддерживаются: {', '.join(sorted(SUPPORTED_MODELS))}.")
    return models


def _history_start(target_start: pd.Timestamp, lags: tuple[int, ...], rolling_windows: tuple[int, ...]) -> pd.Timestamp:
    lookback = max(max(lags), FORECAST_HORIZON + max(rolling_windows)) + 1
    return target_start - pd.Timedelta(days=int(lookback))


def _write_predictions(path: Path, frame: pd.DataFrame, prediction: pd.Series) -> None:
    ensure_dir(path.parent)
    output = frame[[DATE_COL, *ID_COLS, TARGET_COL]].copy() if TARGET_COL in frame.columns else frame[[DATE_COL, *ID_COLS]].copy()
    if "id" in frame.columns:
        output.insert(0, "id", frame["id"].to_numpy())
    output["prediction"] = prediction.to_numpy(dtype="float32")
    output.to_csv(path, index=False)


def run_validation(
    raw_dir: Path,
    results_dir: Path = RESULTS_DIR,
    models: list[str] | None = None,
    max_series: int | None = None,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    valid_days: int = DEFAULT_VALID_DAYS,
    nrows: int | None = None,
) -> pd.DataFrame:
    model_names = _parse_models(models or ["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"])
    run_id = timestamp()
    run_dir = ensure_dir(results_dir / f"validation_{run_id}")

    data = load_favorita(raw_dir, include_test=False, nrows=nrows)
    valid_end = data.train[DATE_COL].max()
    valid_start = valid_end - pd.Timedelta(days=int(valid_days) - 1)
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=int(train_window_days) - 1)

    train_history = data.train.loc[data.train[DATE_COL] <= train_end].copy()
    series_index = select_series(train_history, max_series=max_series)

    feature_history_start = _history_start(train_start, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS)
    history_dates = _date_range(feature_history_start, train_end)
    train_dates = _date_range(train_start, train_end)
    valid_dates = _date_range(valid_start, valid_end)

    history_grid = build_target_grid(train_history, series_index, history_dates)
    train_grid = build_target_grid(train_history, series_index, train_dates)
    valid_grid = build_target_grid(data.train, series_index, valid_dates)

    train_features = make_feature_frame(train_grid, history_grid, data)
    valid_features = make_feature_frame(valid_grid, history_grid, data)
    valid_features["weight"] = competition_weights(valid_features)

    train_features.to_parquet(run_dir / "train_features.parquet", index=False)
    valid_features.to_parquet(run_dir / "valid_features.parquet", index=False)

    metrics_rows = []
    statsforecast_models = [name for name in model_names if name in {"auto_theta", "auto_ets"}]
    statsforecast_predictions = {}
    if statsforecast_models:
        statsforecast_predictions = predict_statsforecast(history_grid, valid_features, statsforecast_models)

    for model_name in model_names:
        if model_name == "naive":
            prediction = predict_naive(valid_features, history_grid)
        elif model_name == "seasonal_naive":
            prediction = predict_seasonal_naive(valid_features, history_grid, season_length=7)
        elif model_name in statsforecast_predictions:
            prediction = statsforecast_predictions[model_name]
        elif model_name == "catboost":
            prediction = predict_catboost(train_features, valid_features)
        elif model_name == "torch_mlp":
            prediction = predict_torch_mlp(train_features, valid_features)
        else:
            continue

        eval_frame = valid_features[[TARGET_COL, "weight"]].copy()
        eval_frame["prediction"] = prediction.to_numpy(dtype="float32")
        row = {"model": model_name, **evaluate_predictions(eval_frame)}
        metrics_rows.append(row)
        _write_predictions(run_dir / f"predictions_{model_name}.csv", valid_features, prediction)

    metrics = pd.DataFrame(metrics_rows).sort_values("nwrmsle").reset_index(drop=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    write_json(
        run_dir / "run_config.json",
        {
            "mode": "validation",
            "models": model_names,
            "max_series": max_series,
            "train_window_days": train_window_days,
            "valid_days": valid_days,
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "run_dir": str(run_dir),
        },
    )
    return metrics


def run_submission(
    raw_dir: Path,
    results_dir: Path = RESULTS_DIR,
    model_name: str = "catboost",
    max_series: int | None = None,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    nrows: int | None = None,
) -> Path:
    _parse_models([model_name])
    run_id = timestamp()
    run_dir = ensure_dir(results_dir / f"submission_{run_id}")

    data = load_favorita(raw_dir, include_test=True, nrows=nrows)
    if data.test is None:
        raise RuntimeError("для режима submission нужен файл test.csv.")

    train_end = data.train[DATE_COL].max()
    train_start = train_end - pd.Timedelta(days=int(train_window_days) - 1)
    test_frame = data.test.copy()

    series_index = test_frame[ID_COLS].drop_duplicates().reset_index(drop=True)
    if max_series is not None:
        series_index = series_index.head(max_series)
        test_frame = test_frame.merge(series_index, on=ID_COLS, how="inner")

    feature_history_start = _history_start(train_start, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS)
    history_dates = _date_range(feature_history_start, train_end)
    train_dates = _date_range(train_start, train_end)

    history_grid = build_target_grid(data.train.loc[data.train[DATE_COL] <= train_end], series_index, history_dates)
    train_grid = build_target_grid(data.train, series_index, train_dates)
    train_features = make_feature_frame(train_grid, history_grid, data)
    test_features = make_feature_frame(test_frame, history_grid, data)

    if model_name == "naive":
        prediction = predict_naive(test_features, history_grid)
    elif model_name == "seasonal_naive":
        prediction = predict_seasonal_naive(test_features, history_grid, season_length=7)
    elif model_name in {"auto_theta", "auto_ets"}:
        prediction = predict_statsforecast(history_grid, test_features, [model_name])[model_name]
    elif model_name == "catboost":
        prediction = predict_catboost(train_features, test_features)
    elif model_name == "torch_mlp":
        prediction = predict_torch_mlp(train_features, test_features)
    else:
        raise ValueError(f"модель не поддерживается для submission: {model_name}")

    submission = test_features[["id"]].copy()
    submission[TARGET_COL] = prediction.to_numpy(dtype="float32").clip(min=0)
    submission_path = run_dir / f"kaggle_submission_{model_name}.csv"
    submission.to_csv(submission_path, index=False)

    write_json(
        run_dir / "run_config.json",
        {
            "mode": "submission",
            "model": model_name,
            "max_series": max_series,
            "train_window_days": train_window_days,
            "submission_path": str(submission_path),
        },
    )
    return submission_path
