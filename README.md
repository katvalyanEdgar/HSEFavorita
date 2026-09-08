# прогнозирование продаж favorita

репозиторий для домашнего задания по прогнозированию продаж favorita с экзогенными признаками.

задача: спрогнозировать `unit_sales` для пар `store_nbr` / `item_nbr` на горизонт 16 дней. основная метрика соответствует соревнованию kaggle: `NWRMSLE`, где товары с `perishable = 1` получают вес `1.25`, остальные товары вес `1.0`. модели обучаются на лог-шкале там, где это уместно, но метрики считаются после восстановления прогноза в исходную шкалу.

## структура

```text
.
├── config.py
├── downloadData.py
├── run_experiment.py
├── src/
│   ├── baselines.py
│   ├── data.py
│   ├── experiment.py
│   ├── features.py
│   ├── metrics.py
│   ├── models_dl.py
│   └── models_ml.py
├── data/
│   └── raw/
├── results/
│   └── analysis_results.ipynb
├── reports/
│   └── report.md
└── requirements.txt
```

## данные

данные скачиваются из соревнования kaggle `corporacion favorita grocery sales forecasting`.

короткий вариант:

```powershell
pip install -r requirements.txt
python downloadData.py
```

если `kagglehub` попросит авторизацию:

```powershell
py -3.11 -c "import kagglehub; kagglehub.login()"
```

перед скачиванием нужно принять правила соревнования на kaggle.

ожидаемые файлы в `data/raw/`:

```text
train.csv
test.csv
items.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

`transactions.csv` читается как дополнительный файл, но не используется в основных признаках, потому что будущие транзакции неизвестны для test-периода. внешние признаки в экспериментах: календарь, `onpromotion`, нефть, праздники, магазины и товары.

## установка

рекомендуемая версия python: `3.11`. текущий pycharm venv может быть создан на более новой версии python, но `catboost` и `torch` обычно надежнее ставятся на python 3.10-3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## быстрый запуск

для проверки пайплайна на ограниченном числе рядов:

```powershell
python run_experiment.py validate --max-series 200 --train-window-days 90 --models naive seasonal_naive catboost
```

если не хватает оперативной памяти, начинайте с меньшего запуска:

```powershell
python run_experiment.py validate --max-series 20 --train-window-days 20 --models naive seasonal_naive
```

для полного набора требуемых моделей:

```powershell
python run_experiment.py validate --models naive seasonal_naive auto_theta auto_ets catboost torch_mlp
```

`auto_theta` и `auto_ets` запускаются через `statsforecast` как независимые per-series модели. на полном числе `store-item` рядов они могут работать долго, поэтому для локальной отладки используется `--max-series`, а финальный запуск делается без ограничения или с явно описанным ограничением в отчете.

## посылка в kaggle

пример посылки `catboost`:

```powershell
python run_experiment.py submit --model catboost
```

файл будет сохранен в `results/submission_<timestamp>/kaggle_submission_catboost.csv`.

## валидация

протокол:

- последний доступный блок длиной 16 дней используется как holdout;
- обучение идет на предыдущем окне длиной `--train-window-days`;
- лаговые признаки имеют лаг не меньше 16 дней, поэтому не используют значения из validation/test горизонта;
- `NWRMSLE`, `RMSLE`, `MAE`, `RMSE` и `WMAPE` считаются по прогнозу в исходной шкале.

результаты каждого запуска сохраняются в `results/validation_<timestamp>/`:

- `metrics.csv`;
- `predictions_<model>.csv`;
- `train_features.parquet`;
- `valid_features.parquet`;
- `run_config.json`.

## отчет

черновик отчета лежит в `reports/report.md`. после финальных запусков туда нужно перенести таблицу из `metrics.csv`, описание kaggle submission и место в лидерборде. pdf можно собрать через pandoc:

```powershell
.\scripts\make_report.ps1
```
