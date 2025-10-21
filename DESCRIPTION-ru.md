# Описание для команды (коротко)

**Цель:** показать навыки, релевантные задачам прогнозирования спроса в ретейле и стеку из вакансии (Python, pandas, sklearn, FastAPI, SQL, Docker).

**Что демонстрируется:**
- формирование признаков для временных рядов (лаги, скользящие окна, календарь);
- корректная временная валидация (`TimeSeriesSplit`), метрики RMSE/MAE/MAPE;
- базовые модели + градиентный бустинг (sklearn; при наличии — LightGBM/CatBoost);
- API-инференс на FastAPI (`/forecast`), Dockerfile;
- SQL-схема и пример аналитических запросов.

**Как проверить:**
1. `pip install -r requirements.txt`
2. `python -m src.train --data data/sample_sales.csv --horizon 14`
3. `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. POST `/forecast` с `{"store":1,"item":3,"horizon":14}`

**Kaggle-данные:** можно положить `train.csv` из *Store Item Demand Forecasting Challenge* в папку `data/` и переобучить модель на реальных рядах.