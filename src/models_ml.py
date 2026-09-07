from __future__ import annotations

import numpy as np
import pandas as pd

from config import CATBOOST_PARAMS, DATE_COL, ID_COLS, TARGET_COL


EXCLUDED_FEATURES = {"id", DATE_COL, TARGET_COL, "prediction", "weight"}


def split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [col for col in frame.columns if col not in EXCLUDED_FEATURES]
    cat_cols = [
        col
        for col in feature_cols
        if str(frame[col].dtype) in {"category", "object", "bool"} or col in {"store_nbr", "item_nbr", "class", "cluster"}
    ]
    return feature_cols, cat_cols


def predict_catboost(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    params: dict | None = None,
) -> pd.Series:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("установите catboost, чтобы запустить модель catboost.") from exc

    model_params = dict(CATBOOST_PARAMS)
    if params:
        model_params.update(params)

    feature_cols, cat_cols = split_feature_columns(train_features)
    x_train = train_features[feature_cols].copy()
    x_pred = predict_features[feature_cols].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].astype(str).fillna("нет_данных")
        x_pred[col] = x_pred[col].astype(str).fillna("нет_данных")

    y_train = np.log1p(train_features[TARGET_COL].clip(lower=0).astype("float32"))
    train_pool = Pool(x_train, y_train, cat_features=cat_cols)
    pred_pool = Pool(x_pred, cat_features=cat_cols)

    model = CatBoostRegressor(**model_params)
    model.fit(train_pool)
    pred = np.expm1(model.predict(pred_pool)).clip(min=0)
    return pd.Series(pred.astype("float32"), index=predict_features.index)
