from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"

DATE_COL = "date"
TARGET_COL = "unit_sales"
ID_COLS = ["store_nbr", "item_nbr"]

RANDOM_SEED = 42
FORECAST_HORIZON = 16
DEFAULT_VALID_DAYS = 16
DEFAULT_TRAIN_WINDOW_DAYS = 180
DEFAULT_CATBOOST_MAX_TRAIN_ROWS = 2_000_000
DEFAULT_CATBOOST_PREDICTION_CHUNK_SIZE = 250_000

DEFAULT_LAGS = (16, 17, 18, 21, 28, 35, 42, 56, 91, 182, 364)
DEFAULT_ROLLING_WINDOWS = (7, 14, 28, 56)

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "items": "items.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
    "sample_submission": "sample_submission.csv",
}

CATBOOST_PARAMS = {
    "iterations": 1200,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_SEED,
    "allow_writing_files": False,
    "verbose": 100,
}

TORCH_MLP_PARAMS = {
    "epochs": 30,
    "batch_size": 4096,
    "learning_rate": 5e-4,
    "hidden_units": (256, 128),
    "dropout": 0.10,
}
