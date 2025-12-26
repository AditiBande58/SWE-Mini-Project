import pandas as pd
import numpy as np

df = pd.read_csv("EMG-data-one-channel.csv") 

# Mapping based on dataset description:
# 2 - fist, 3 - wrist flexion, 4 - wrist extension
def map_to_control_label(c):
    if c == 2:
        return "stop"      # clenched/fist
    elif c == 3:
        return "forward"   # wrist flexion
    elif c == 4:
        return "backward"  # wrist extension
    else:
        return np.nan      # all other classes removed

df['control_label'] = df['class'].apply(map_to_control_label)
df = df.dropna(subset=['control_label'])

channel_cols = [c for c in df.columns if c.startswith("channel")]
all_zero_mask = (df[channel_cols].abs().sum(axis=1) == 0)
df = df[~all_zero_mask]

df.to_csv("filtered_emg.csv", index=False)
print(df['control_label'].value_counts())
