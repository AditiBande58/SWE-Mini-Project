import pandas as pd
import numpy as np
import joblib
from pathlib import Path

WINDOW_SIZE = 200
STEP = 50
channel_prefix = "channel"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    channel_cols = [c for c in df.columns if c.startswith(channel_prefix)]
    data = df[channel_cols].values
    n = len(data)
    features = []

    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        window = data[start:start + WINDOW_SIZE, :]

        feat = {}
        for i, ch in enumerate(channel_cols):
            x = window[:, i]
            dx = np.diff(x)
            feat[f'{ch}_RMS'] = np.sqrt(np.mean(x**2))
            feat[f'{ch}_MAV'] = np.mean(np.abs(x))
            feat[f'{ch}_WL']  = np.sum(np.abs(np.diff(x)))
            feat[f'{ch}_VAR'] = np.var(x)
            feat[f'{ch}_ZCR'] = np.sum((x[:-1] * x[1:]) < 0)
            feat[f'{ch}_SSC'] = np.sum((dx[:-1] * dx[1:]) < 0)

        features.append(feat)

    return pd.DataFrame(features)

SCALER_PATH = DATA_DIR / "emg_3class_scaler.pkl"
CLF_PATH = DATA_DIR / "emg_3class_logreg.pkl"

scaler = joblib.load(SCALER_PATH)
clf = joblib.load(CLF_PATH)


# Take in raw EMG samples (df_window) and return predicted label for the window
def predict_window(df_window: pd.DataFrame):
    X_new = extract_features_from_df(df_window)

    if X_new.empty:
        return None, None

    X_new_s = scaler.transform(X_new)
    y_pred = clf.predict(X_new_s)
    y_proba = clf.predict_proba(X_new_s)

    labels_series = pd.Series(y_pred)
    majority_label = labels_series.mode().iloc[0]

    return majority_label, y_proba