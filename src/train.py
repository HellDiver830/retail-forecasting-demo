import argparse, json, os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib

from .features import build_train_frame

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.maximum(1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }

def try_model(name, cls, X_train, y_train, X_val, y_val, params=None):
    params = params or {}
    model = cls(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = evaluate(y_val, preds)
    return name, model, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sample_sales.csv")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--models_dir", type=str, default="models")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df['date'] = pd.to_datetime(df['date'])

    # Build supervised frame
    frame = build_train_frame(df)
    feature_cols = [c for c in frame.columns if c not in ("date","sales")]
    X = frame[feature_cols]
    y = frame["sales"]

    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=5)
    best = {"name": None, "model": None, "metrics": {"RMSE": 1e18}}

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Baseline: seasonal naive ~ lag-7 as a regressor approximation
        _, baseline_model, baseline_metrics = try_model(
            "Dummy(last7-mean)",
            DummyRegressor,
            X_train[["lag_7"]].fillna(method="bfill"),
            y_train,
            X_val[["lag_7"]].fillna(method="ffill"),
            y_val,
            {"strategy": "mean"},
        )

        # Sklearn HGB
        _, hgb, hgb_metrics = try_model(
            "HistGradientBoosting",
            HistGradientBoostingRegressor,
            X_train.fillna(0),
            y_train,
            X_val.fillna(0),
            y_val,
            {"max_depth": 6, "l2_regularization": 0.0},
        )

        # Keep best on this fold
        for name, model, metrics in [("Dummy", baseline_model, baseline_metrics),
                                     ("HistGradientBoosting", hgb, hgb_metrics)]:
            if metrics["RMSE"] < best["metrics"]["RMSE"]:
                best = {"name": name, "model": model, "metrics": metrics}

    # Optional: LightGBM
    try:
        import lightgbm as lgb
        name, lgbm, lgbm_metrics = try_model(
            "LightGBM",
            lgb.LGBMRegressor,
            X.fillna(0),
            y,
            X.fillna(0).iloc[-args.horizon*2:],  # simple proxy val
            y.iloc[-args.horizon*2:],
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": -1},
        )
        if lgbm_metrics["RMSE"] < best["metrics"]["RMSE"]:
            best = {"name": "LightGBM", "model": lgbm, "metrics": lgbm_metrics}
    except Exception as ex:
        print(f"[warn] LightGBM not used: {ex}")

    # Optional: CatBoost
    try:
        from catboost import CatBoostRegressor
        name, cbr, cbr_metrics = try_model(
            "CatBoost",
            CatBoostRegressor,
            X.fillna(0),
            y,
            X.fillna(0).iloc[-args.horizon*2:],
            y.iloc[-args.horizon*2:],
            {"depth": 6, "learning_rate": 0.05, "iterations": 300, "verbose": False},
        )
        if cbr_metrics["RMSE"] < best["metrics"]["RMSE"]:
            best = {"name": "CatBoost", "model": cbr, "metrics": cbr_metrics}
    except Exception as ex:
        print(f"[warn] CatBoost not used: {ex}")

    # Save best
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(best["model"], os.path.join(args.models_dir, "model.pkl"))
    meta = {
        "best_model": best["name"],
        "metrics": best["metrics"],
        "feature_columns": feature_cols,
        "horizon": args.horizon,
        "data_path": args.data
    }
    json.dump(meta, open(os.path.join(args.models_dir, "meta.json"), "w"), indent=2)

    # Simple served predictions (demo): last horizon days via baseline
    last_df = frame.sort_values("date").tail(args.horizon)
    served = last_df[["date","store","item"]].copy()
    served["yhat_demo"] = last_df["lag_7"].fillna(method="ffill").values
    served.to_csv(os.path.join(args.models_dir, "served_predictions.csv"), index=False)

    print("Saved:", os.path.join(args.models_dir, "model.pkl"))
    print("Meta:", json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()