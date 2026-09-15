# Прогнозирование продаж Favorita

Проект для прогноза продаж unit_sales на 16 дней вперед по данным Kaggle Favorita.

## Установка

pip install -r requirements.txt
python downloadData.py

Для полного набора моделей:

pip install -r requirements-full.txt

## Запуск

Быстрая проверка:

python run_experiment.py validate --max-series 200 --train-window-days 90

Короткий запуск с классическими моделями:

python run_experiment.py validate --max-series 20 --train-window-days 60 --classic-jobs 8 --models naive seasonal_naive auto_theta auto_ets

Полный запуск:

python run_experiment.py validate --models naive seasonal_naive auto_theta auto_ets catboost torch_mlp

## Файлы

Код находится в src, результаты сохраняются в results, отчет лежит в reports/report.md, там же находится score kaggle.
