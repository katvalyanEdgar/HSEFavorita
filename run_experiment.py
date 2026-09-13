from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_CATBOOST_MAX_TRAIN_ROWS,
    DEFAULT_CATBOOST_PREDICTION_CHUNK_SIZE,
    DEFAULT_TRAIN_WINDOW_DAYS,
    DEFAULT_VALID_DAYS,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Favorita sales forecasting experiments.")
    parser.add_argument("mode", nargs="?", choices=["validate", "submit"], default="validate")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Directory with Favorita csv files.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory for metrics and submissions.")
    parser.add_argument("--max-series", type=int, default=None, help="Limit store-item series for local validation/debug only.")
    parser.add_argument("--train-window-days", type=int, default=DEFAULT_TRAIN_WINDOW_DAYS)
    parser.add_argument("--catboost-max-train-rows", type=int, default=DEFAULT_CATBOOST_MAX_TRAIN_ROWS)
    parser.add_argument("--catboost-prediction-chunk-size", type=int, default=DEFAULT_CATBOOST_PREDICTION_CHUNK_SIZE)
    parser.add_argument("--nrows", type=int, default=None, help="Optional train.csv row limit for smoke tests.")
    parser.add_argument("--valid-days", type=int, default=DEFAULT_VALID_DAYS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"],
        help="Models: naive seasonal_naive auto_theta auto_ets catboost torch_mlp",
    )
    parser.add_argument(
        "--model",
        default="catboost",
        choices=["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"],
        help="Model for submit mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = args.mode

    if mode == "validate":
        from src.experiment import run_validation

        metrics = run_validation(
            raw_dir=args.raw_dir,
            results_dir=args.results_dir,
            models=args.models,
            max_series=args.max_series,
            train_window_days=args.train_window_days,
            valid_days=args.valid_days,
            nrows=args.nrows,
            catboost_max_train_rows=args.catboost_max_train_rows,
            catboost_prediction_chunk_size=args.catboost_prediction_chunk_size,
        )
        print(metrics.to_string(index=False))
    elif mode == "submit":
        from src.experiment import run_submission

        path = run_submission(
            raw_dir=args.raw_dir,
            results_dir=args.results_dir,
            model_name=args.model,
            max_series=args.max_series,
            train_window_days=args.train_window_days,
            nrows=args.nrows,
            catboost_max_train_rows=args.catboost_max_train_rows,
            catboost_prediction_chunk_size=args.catboost_prediction_chunk_size,
        )
        print(f"Submission file saved: {path}")
    else:
        parser.error(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
