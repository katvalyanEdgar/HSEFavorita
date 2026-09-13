from __future__ import annotations

import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import RANDOM_SEED, TARGET_COL, TORCH_MLP_PARAMS
from src.models_ml import EXCLUDED_FEATURES, split_feature_columns


def _embedding_dim(cardinality: int) -> int:
    return int(min(64, max(4, round(1.6 * math.sqrt(max(cardinality, 2))))))


class _TabularEncoder:
    def __init__(self, cat_cols: list[str], num_cols: list[str]) -> None:
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.category_maps: dict[str, dict[str, int]] = {}
        self.num_mean: pd.Series | None = None
        self.num_std: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "_TabularEncoder":
        for col in self.cat_cols:
            values = frame[col].astype(str).fillna("нет_данных")
            uniques = pd.Index(values.unique())
            self.category_maps[col] = {value: idx + 1 for idx, value in enumerate(uniques)}
        numeric = frame[self.num_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        self.num_mean = numeric.mean()
        self.num_std = numeric.std().replace(0, 1).fillna(1)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_arrays = []
        for col in self.cat_cols:
            mapping = self.category_maps[col]
            encoded = frame[col].astype(str).fillna("нет_данных").map(mapping).fillna(0).astype("int64")
            cat_arrays.append(encoded.to_numpy())
        cat_matrix = np.vstack(cat_arrays).T if cat_arrays else np.zeros((len(frame), 0), dtype="int64")

        numeric = frame[self.num_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if self.num_mean is not None and self.num_std is not None:
            numeric = (numeric - self.num_mean) / self.num_std
        return cat_matrix.astype("int64"), numeric.to_numpy(dtype="float32")

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[col]) + 1 for col in self.cat_cols]


def predict_torch_mlp(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    params: dict | None = None,
) -> pd.Series:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("установите torch, чтобы запустить нейросетевую модель.") from exc

    run_params = dict(TORCH_MLP_PARAMS)
    if params:
        run_params.update(params)

    feature_cols, cat_cols = split_feature_columns(train_features)
    num_cols = [col for col in feature_cols if col not in cat_cols and col not in EXCLUDED_FEATURES]

    encoder = _TabularEncoder(cat_cols=cat_cols, num_cols=num_cols).fit(train_features)
    x_cat, x_num = encoder.transform(train_features)
    p_cat, p_num = encoder.transform(predict_features)
    y = np.log1p(train_features[TARGET_COL].clip(lower=0).to_numpy(dtype="float32"))
    target_log_cap = max(float(np.quantile(y, 0.999)), 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_ds = TensorDataset(
        torch.from_numpy(x_cat),
        torch.from_numpy(x_num),
        torch.from_numpy(y).view(-1, 1),
    )
    loader = DataLoader(train_ds, batch_size=int(run_params["batch_size"]), shuffle=True, num_workers=0)

    class TabularMLP(nn.Module):
        def __init__(self, cardinalities: list[int], n_numeric: int) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(card, _embedding_dim(card)) for card in cardinalities])
            input_dim = sum(embedding.embedding_dim for embedding in self.embeddings) + n_numeric
            layers: list[nn.Module] = []
            for hidden in run_params["hidden_units"]:
                layers.extend([nn.Linear(input_dim, int(hidden)), nn.ReLU(), nn.Dropout(float(run_params["dropout"]))])
                input_dim = int(hidden)
            layers.append(nn.Linear(input_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, cat_values: torch.Tensor, num_values: torch.Tensor) -> torch.Tensor:
            if self.embeddings:
                embedded = [embedding(cat_values[:, idx]) for idx, embedding in enumerate(self.embeddings)]
                x = torch.cat([*embedded, num_values], dim=1)
            else:
                x = num_values
            return self.net(x)

    model = TabularMLP(encoder.cardinalities, len(num_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(run_params["learning_rate"]))
    loss_fn = nn.MSELoss()

    model.train()
    for _ in tqdm(range(int(run_params["epochs"])), desc="обучение torch mlp", unit="эпоха"):
        for batch_cat, batch_num, batch_y in tqdm(loader, desc="батчи", unit="батч", leave=False):
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_cat, batch_num), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    predictions = []
    pred_ds = TensorDataset(torch.from_numpy(p_cat), torch.from_numpy(p_num))
    pred_loader = DataLoader(pred_ds, batch_size=int(run_params["batch_size"]), shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch_cat, batch_num in tqdm(pred_loader, desc="прогноз torch mlp", unit="батч", leave=False):
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            predictions.append(model(batch_cat, batch_num).cpu().numpy())

    pred_log = np.vstack(predictions).reshape(-1)
    pred_log = np.clip(pred_log, 0, target_log_cap)
    pred = np.expm1(pred_log).clip(min=0)
    return pd.Series(pred.astype("float32"), index=predict_features.index)
