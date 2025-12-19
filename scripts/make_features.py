import pandas as pd
import numpy as np

df = pd.read_csv("filtered_emg.csv")
channel_cols = [c for c in df.columns if c.startswith("channel")]

WINDOW_SIZE = 200
STEP = 100

features = []

for label, group in df.groupby('control_label'):
    data = group[channel_cols].values
    n = len(data)
    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        window = data[start:start + WINDOW_SIZE, :]

        feat = {}
        for i, ch in enumerate(channel_cols):
            x = window[:, i]
            feat[f'{ch}_RMS'] = np.sqrt(np.mean(x**2))
            feat[f'{ch}_MAV'] = np.mean(np.abs(x))
            feat[f'{ch}_WL']  = np.sum(np.abs(np.diff(x)))

        feat['control_label'] = label
        features.append(feat)

feat_df = pd.DataFrame(features)
feat_df.to_csv("emg_car_windows_3class.csv", index=False)
print(feat_df['control_label'].value_counts())
