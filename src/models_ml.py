from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    CATBOOST_PARAMS,
    DATE_COL,
    DEFAULT_CATBOOST_MAX_TRAIN_ROWS,
    DEFAULT_CATBOOST_PREDICTION_CHUNK_SIZE,
    ID_COLS,
    RANDOM_SEED,
    TARGET_COL,
)


EXCLUDED_FEATURES = {"id", DATE_COL, TARGET_COL, "prediction", "weight"}


def split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [col for col in frame.columns if col not in EXCLUDED_FEATURES]
    cat_cols = [
        col
        for col in feature_cols
        if str(frame[col].dtype) in {"category", "object", "bool"} or col in {"store_nbr", "item_nbr", "class", "cluster"}
    ]
    return feature_cols, cat_cols


def _sample_training_rows(train_features: pd.DataFrame, max_train_rows: int | None) -> pd.DataFrame:
    if max_train_rows is None or int(max_train_rows) <= 0 or len(train_features) <= int(max_train_rows):
        return train_features
    return train_features.sample(n=int(max_train_rows), random_state=RANDOM_SEED).sort_index()


def _prepare_feature_frame(frame: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    features = frame.loc[:, feature_cols].copy()
    for col in cat_cols:
        features[col] = features[col].astype(str).fillna("no_data")
    return features


def predict_catboost(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    params: dict | None = None,
    max_train_rows: int | None = DEFAULT_CATBOOST_MAX_TRAIN_ROWS,
    prediction_chunk_size: int = DEFAULT_CATBOOST_PREDICTION_CHUNK_SIZE,
) -> pd.Series:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("Install catboost to run the catboost model.") from exc

    model_params = dict(CATBOOST_PARAMS)
    if params:
        model_params.update(params)

    feature_cols, cat_cols = split_feature_columns(train_features)
    train_data = _sample_training_rows(train_features, max_train_rows=max_train_rows)
    if len(train_data) < len(train_features):
        print(f"CatBoost train rows: {len(train_data):,} sampled from {len(train_features):,}")

    x_train = _prepare_feature_frame(train_data, feature_cols, cat_cols)
    y_train = np.log1p(train_data[TARGET_COL].clip(lower=0).astype("float32"))
    train_pool = Pool(x_train, y_train, cat_features=cat_cols)

    model = CatBoostRegressor(**model_params)
    model.fit(train_pool)

    del x_train, y_train, train_pool, train_data
    gc.collect()

    chunk_size = max(1, int(prediction_chunk_size))
    pred = np.empty(len(predict_features), dtype="float32")
    ranges = range(0, len(predict_features), chunk_size)
    for start in tqdm(ranges, desc="catboost predict", unit="chunk", leave=False):
        end = min(start + chunk_size, len(predict_features))
        x_pred = _prepare_feature_frame(predict_features.iloc[start:end], feature_cols, cat_cols)
        pred_pool = Pool(x_pred, cat_features=cat_cols)
        pred[start:end] = np.expm1(model.predict(pred_pool)).clip(min=0).astype("float32")
        del x_pred, pred_pool

    return pd.Series(pred, index=predict_features.index)
