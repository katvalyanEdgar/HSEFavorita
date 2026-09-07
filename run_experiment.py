from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_TRAIN_WINDOW_DAYS, DEFAULT_VALID_DAYS, RAW_DATA_DIR, RESULTS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="эксперименты по прогнозированию продаж favorita.")
    parser.add_argument("mode", nargs="?", choices=["validate", "submit"], default="validate")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="папка с csv-файлами favorita.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="папка для метрик и submission-файлов.")
    parser.add_argument("--max-series", type=int, default=None, help="ограничение числа store-item рядов для быстрых запусков.")
    parser.add_argument("--train-window-days", type=int, default=DEFAULT_TRAIN_WINDOW_DAYS)
    parser.add_argument("--nrows", type=int, default=None, help="необязательное ограничение строк csv для smoke-тестов.")
    parser.add_argument("--valid-days", type=int, default=DEFAULT_VALID_DAYS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"],
        help="модели: naive seasonal_naive auto_theta auto_ets catboost torch_mlp",
    )
    parser.add_argument(
        "--model",
        default="catboost",
        choices=["naive", "seasonal_naive", "auto_theta", "auto_ets", "catboost", "torch_mlp"],
        help="модель для режима submit.",
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
        )
        print(f"submission-файл сохранен: {path}")
    else:
        parser.error(f"режим не поддерживается: {mode}")


if __name__ == "__main__":
    main()
