from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from config import DATE_COL, ID_COLS, RAW_FILES, TARGET_COL


@dataclass
class FavoritaData:
    train: pd.DataFrame
    items: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    transactions: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    sample_submission: pd.DataFrame | None = None


def check_raw_files(raw_dir: Path, include_test: bool) -> None:
    required = ["train", "items", "stores", "oil", "holidays"]
    if include_test:
        required.extend(["test", "sample_submission"])

    missing = [RAW_FILES[name] for name in required if not (raw_dir / RAW_FILES[name]).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"в {raw_dir} не найдены файлы favorita: {joined}. "
            "скачайте данные соревнования kaggle и распакуйте csv в data/raw/."
        )


def train_date_bounds(raw_dir: Path, nrows: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    reader = pd.read_csv(raw_dir / RAW_FILES["train"], usecols=[DATE_COL], chunksize=2_000_000, nrows=nrows)
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    for chunk in tqdm(reader, desc="чтение дат train.csv", unit="чанк", leave=False):
        dates = pd.to_datetime(chunk[DATE_COL])
        chunk_min = dates.min()
        chunk_max = dates.max()
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)
    if min_date is None or max_date is None:
        raise RuntimeError("train.csv пустой или не содержит колонку date.")
    return min_date, max_date


def select_series_from_train_file(
    raw_dir: Path,
    max_series: int,
    score_start_date: pd.Timestamp,
    score_end_date: pd.Timestamp,
    nrows: int | None = None,
) -> pd.DataFrame:
    dtype = {
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
        "onpromotion": "object",
    }
    parts = []
    usecols = [DATE_COL, *ID_COLS, TARGET_COL]
    reader = pd.read_csv(raw_dir / RAW_FILES["train"], dtype=dtype, usecols=usecols, chunksize=2_000_000, nrows=nrows)
    for chunk in tqdm(reader, desc="выбор рядов train.csv", unit="чанк", leave=False):
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL])
        mask = chunk[DATE_COL].between(score_start_date, score_end_date)
        chunk = chunk.loc[mask]
        if chunk.empty:
            continue
        parts.append(chunk.groupby(ID_COLS, observed=True)[TARGET_COL].sum().reset_index())

    if not parts:
        return pd.DataFrame(columns=ID_COLS)

    scores = (
        pd.concat(parts, ignore_index=True)
        .groupby(ID_COLS, observed=True)[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(max_series)
        .reset_index()[ID_COLS]
    )
    return scores.sort_values(ID_COLS).reset_index(drop=True)


def _read_train(
    path: Path,
    nrows: int | None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    series_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    dtype = {
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
    }
    usecols = [DATE_COL, "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    if start_date is None and end_date is None:
        train = pd.read_csv(path, parse_dates=[DATE_COL], dtype=dtype, usecols=usecols, nrows=nrows)
    else:
        chunks = []
        reader = pd.read_csv(path, dtype=dtype, usecols=usecols, chunksize=2_000_000, nrows=nrows)
        for chunk in tqdm(reader, desc="чтение train.csv", unit="чанк", leave=False):
            chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL])
            mask = pd.Series(True, index=chunk.index)
            if start_date is not None:
                mask &= chunk[DATE_COL] >= start_date
            if end_date is not None:
                mask &= chunk[DATE_COL] <= end_date
            chunk = chunk.loc[mask]
            if series_index is not None and not chunk.empty:
                chunk = chunk.merge(series_index[ID_COLS], on=ID_COLS, how="inner")
            if not chunk.empty:
                chunks.append(chunk)
        if chunks:
            train = pd.concat(chunks, ignore_index=True)
        else:
            train = pd.DataFrame(columns=usecols)
    if "onpromotion" in train.columns:
        train["onpromotion"] = train["onpromotion"].fillna(False).astype(bool).astype("int8")
    else:
        train["onpromotion"] = 0
    train[TARGET_COL] = train[TARGET_COL].clip(lower=0).astype("float32")
    return train


def _read_test(path: Path) -> pd.DataFrame:
    dtype = {"id": "int64", "store_nbr": "int16", "item_nbr": "int32"}
    test = pd.read_csv(path, parse_dates=[DATE_COL], dtype=dtype)
    test["onpromotion"] = test["onpromotion"].fillna(False).astype("int8")
    return test


def _read_items(path: Path) -> pd.DataFrame:
    items = pd.read_csv(path)
    items["item_nbr"] = items["item_nbr"].astype("int32")
    items["class"] = items["class"].astype("int32")
    items["perishable"] = items["perishable"].astype("int8")
    return items


def _read_stores(path: Path) -> pd.DataFrame:
    stores = pd.read_csv(path)
    stores["store_nbr"] = stores["store_nbr"].astype("int16")
    stores = stores.rename(columns={"type": "store_type"})
    return stores


def _read_oil(path: Path) -> pd.DataFrame:
    oil = pd.read_csv(path, parse_dates=[DATE_COL])
    oil = oil.sort_values(DATE_COL)
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill().astype("float32")
    return oil


def _read_holidays(path: Path) -> pd.DataFrame:
    holidays = pd.read_csv(path, parse_dates=[DATE_COL])
    holidays["transferred"] = holidays["transferred"].fillna(False).astype(bool)
    return holidays


def _read_transactions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    transactions = pd.read_csv(path, parse_dates=[DATE_COL])
    transactions["store_nbr"] = transactions["store_nbr"].astype("int16")
    transactions["transactions"] = transactions["transactions"].astype("float32")
    return transactions


def load_favorita(
    raw_dir: Path,
    include_test: bool = False,
    nrows: int | None = None,
    train_start_date: pd.Timestamp | None = None,
    train_end_date: pd.Timestamp | None = None,
    series_index: pd.DataFrame | None = None,
) -> FavoritaData:
    check_raw_files(raw_dir, include_test=include_test)

    test = _read_test(raw_dir / RAW_FILES["test"]) if include_test else None
    sample = (
        pd.read_csv(raw_dir / RAW_FILES["sample_submission"])
        if include_test and (raw_dir / RAW_FILES["sample_submission"]).exists()
        else None
    )

    return FavoritaData(
        train=_read_train(
            raw_dir / RAW_FILES["train"],
            nrows=nrows,
            start_date=train_start_date,
            end_date=train_end_date,
            series_index=series_index,
        ),
        items=_read_items(raw_dir / RAW_FILES["items"]),
        stores=_read_stores(raw_dir / RAW_FILES["stores"]),
        oil=_read_oil(raw_dir / RAW_FILES["oil"]),
        holidays=_read_holidays(raw_dir / RAW_FILES["holidays"]),
        transactions=_read_transactions(raw_dir / RAW_FILES["transactions"]),
        test=test,
        sample_submission=sample,
    )
