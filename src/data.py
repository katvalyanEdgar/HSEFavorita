from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATE_COL, RAW_FILES, TARGET_COL


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


def _read_train(path: Path, nrows: int | None) -> pd.DataFrame:
    dtype = {
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
    }
    train = pd.read_csv(path, parse_dates=[DATE_COL], dtype=dtype, nrows=nrows)
    if "onpromotion" in train.columns:
        train["onpromotion"] = train["onpromotion"].fillna(False).astype("int8")
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


def load_favorita(raw_dir: Path, include_test: bool = False, nrows: int | None = None) -> FavoritaData:
    check_raw_files(raw_dir, include_test=include_test)

    test = _read_test(raw_dir / RAW_FILES["test"]) if include_test else None
    sample = (
        pd.read_csv(raw_dir / RAW_FILES["sample_submission"])
        if include_test and (raw_dir / RAW_FILES["sample_submission"]).exists()
        else None
    )

    return FavoritaData(
        train=_read_train(raw_dir / RAW_FILES["train"], nrows=nrows),
        items=_read_items(raw_dir / RAW_FILES["items"]),
        stores=_read_stores(raw_dir / RAW_FILES["stores"]),
        oil=_read_oil(raw_dir / RAW_FILES["oil"]),
        holidays=_read_holidays(raw_dir / RAW_FILES["holidays"]),
        transactions=_read_transactions(raw_dir / RAW_FILES["transactions"]),
        test=test,
        sample_submission=sample,
    )
