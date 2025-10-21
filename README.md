# 1221Systems — Retail Forecasting Demo (Time Series)

Небольшой демонстрационный проект под вакансию Junior Data Scientist: прогнозирование продаж в ретейле.
Акцент на **Python + pandas + scikit-learn + FastAPI + SQL** и валидацию на временных рядах.

## Что внутри
- **EDA/фичи**: лаги (1, 7, 14), скользящие средние (7, 28), календарные признаки (день недели, месяц, праздники-заглушки).
- **Модели**: базовая Seasonal Naive, `HistGradientBoostingRegressor`. Если доступны — LightGBM / CatBoost / Prophet (опционально, через `try/except`).
- **Валидация**: `TimeSeriesSplit` и холдаут по последним неделям. Метрики: **RMSE/MAE/MAPE**.
- **Инференс**: FastAPI (`/forecast`) + простая демо-логика на 14 дней.
- **SQL**: минимальная схема витрин и примеры аналитических запросов (`schema.sql`, `queries.sql`).
- **Docker**: контейнер для запуска API.

## Данные
По умолчанию используется синтетический датасет `data/sample_sales.csv` (3 магазина × 5 товаров × ~180 дней).
Для реалистичного теста положите в папку `data/` датасет Kaggle **Store Item Demand Forecasting Challenge**
(train.csv) и укажите `--data data/train.csv` при обучении.

## Быстрый старт
```bash
# 1) Установка
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Обучение (на sample или вашем train.csv)
python -m src.train --data data/sample_sales.csv --horizon 14

# 3) Запуск API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4) Запрос прогноза
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" -d '{"store": 1, "item": 3, "horizon": 14}'
```

**Docker**
```bash
docker build -t retail-forecasting-demo .
docker run -p 8000:8000 retail-forecasting-demo
```

## Структура
```text
1221systems-retail-forecasting-demo/
├─ app/
│  └─ main.py
├─ src/
│  ├─ features.py
│  └─ train.py
├─ data/
│  ├─ sample_sales.csv
│  └─ README.md
├─ models/
│  └─ README.md
├─ tests/
│  └─ test_features.py
├─ schema.sql
├─ queries.sql
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
└─ README.md
```

## Заметки
- Код «бережно» относится к зависимостям: если LightGBM/CatBoost/Prophet не установлены, обучение всё равно пройдёт на `sklearn`.
- Эндпоинт `/forecast` в демо-режиме использует сохранённые метаданные/средние, чтобы не усложнять продовую логику рекурсивного прогноза.
- Для боевого кейса: добавить калибровку, holiday-календарь, стабильные витрины из DWH, CI/CD (GitLab), мониторинг метрик дрейфа.