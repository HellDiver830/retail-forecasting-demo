# Данные

В репозитории лежит небольшой синтетический пример `sample_sales.csv`.

Для более реалистичного теста используйте Kaggle **Store Item Demand Forecasting Challenge** (train.csv):
https://www.kaggle.com/competitions/demand-forecasting-kernels-only

Скопируйте `train.csv` в эту папку и запускайте обучение так:
```bash
python -m src.train --data data/train.csv --horizon 14
```