# Прогнозирование продаж Favorita с экзогенными признаками


Задача: предсказать unit_sales для пар store_nbr/item_nbr на 16 дней вперед. Основная метрика - NWRMSLE. Для скоропортящихся товаров применяется вес 1.25, для остальных - 1.0.

## Структура

- config.py - общие пути и настройки
- downloadData.py - загрузка данных
- run_experiment.py - запуск валидации и формирование submission
- src - код подготовки данных, признаков, моделей и метрик
- data/raw - исходные данные
- results - результаты запусков
- reports/report.md - итоговый отчет

## Данные

pip install -r requirements.txt
python downloadData.py

## Запуск

Быстрая проверка:

python run_experiment.py validate --max-series 200 --train-window-days 90 --models naive seasonal_naive catboost

Полный локальный запуск:

python run_experiment.py validate --models naive seasonal_naive auto_theta auto_ets catboost torch_mlp

## Результаты

Результаты сохраняются в results/validation_<timestamp>: metrics.csv, predictions_<model>.csv, train_features.parquet, valid_features.parquet и run_config.json.

Submission сохраняется в results/submission_<timestamp>.

## Отчет

Итоговый отчет находится в reports/report.md.
