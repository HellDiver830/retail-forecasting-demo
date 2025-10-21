import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    return df

def make_lags(df: pd.DataFrame, lags=(1,7,14)) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(['store','item','date'])
    for lag in lags:
        out[f'lag_{lag}'] = out.groupby(['store','item'])['sales'].shift(lag)
    return out

def make_rollings(df: pd.DataFrame, windows=(7,28)) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(['store','item','date'])
    for w in windows:
        out[f'roll_mean_{w}'] = out.groupby(['store','item'])['sales'].shift(1).rolling(w).mean()
        out[f'roll_std_{w}']  = out.groupby(['store','item'])['sales'].shift(1).rolling(w).std()
    return out

def build_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = make_lags(df)
    df = make_rollings(df)
    df = df.dropna().reset_index(drop=True)
    return df