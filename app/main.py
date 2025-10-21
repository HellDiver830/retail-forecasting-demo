import os, json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from datetime import timedelta

app = FastAPI(title="Retail Forecasting Demo")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
META_PATH = MODELS_DIR / "meta.json"
SERVED_PATH = MODELS_DIR / "served_predictions.csv"

class ForecastRequest(BaseModel):
    store: int
    item: int
    horizon: int = 14

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    # Demo: use served_predictions.csv as a placeholder prediction source
    if SERVED_PATH.exists():
        df = pd.read_csv(SERVED_PATH, parse_dates=["date"])
        df = df[(df['store'] == req.store) & (df['item'] == req.item)].copy()
        if df.empty:
            return {"detail": "No recent data for this store/item in demo artifacts."}
        df = df.tail(req.horizon)
        return {
            "store": req.store,
            "item": req.item,
            "horizon": req.horizon,
            "predictions": [
                {"date": d.strftime("%Y-%m-%d"), "yhat": float(y)}
                for d, y in zip(df["date"], df["yhat_demo"])
            ]
        }
    return {"detail": "Model artifacts not found. Run training first."}