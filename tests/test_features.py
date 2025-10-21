import pandas as pd
from src.features import build_train_frame

def test_build_train_frame():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "store": [1]*30,
        "item": [1]*30,
        "sales": range(30)
    })
    out = build_train_frame(df)
    assert {"lag_1","lag_7","lag_14","roll_mean_7","roll_mean_28"}.issubset(set(out.columns))