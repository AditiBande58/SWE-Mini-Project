import pandas as pd
import numpy as np
import joblib

raw_df = pd.read_csv(".csv") # Read in raw data

WINDOW_SIZE = 200
STEP = 50
channel_prefix = "channel"

def extract_features_from_df(df):
    channel_cols = [c for c in df.columns if c.startswith(channel_prefix)]
    data = df[channel_cols].values
    n = len(data)
    features = []

    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        window = data[start:start + WINDOW_SIZE, :]

        feat = {}
        for i, ch in enumerate(channel_cols):
            x  = window[:, i]
            dx = np.diff(x)
            feat[f'{ch}_RMS'] = np.sqrt(np.mean(x**2))
            feat[f'{ch}_MAV'] = np.mean(np.abs(x))
            feat[f'{ch}_WL']  = np.sum(np.abs(np.diff(x)))
            feat[f'{ch}_VAR'] = np.var(x)
            feat[f'{ch}_ZCR'] = np.sum((x[:-1] * x[1:]) < 0)
            feat[f'{ch}_SSC'] = np.sum((dx[:-1] * dx[1:]) < 0)

        features.append(feat)

    return pd.DataFrame(features)

scaler = joblib.load("emg_3class_scaler.pkl")
clf = joblib.load("emg_3class_logreg.pkl")

X_new = extract_features_from_df(raw_df)
X_new_s = scaler.transform(X_new)

y_pred = clf.predict(X_new_s)
y_pred_proba = clf.predict_proba(X_new_s)

X_new["pred_label"] = y_pred
X_new.to_csv("filtered_emg_pred_windows.csv", index=False)