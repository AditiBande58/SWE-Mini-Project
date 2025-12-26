import time
import numpy as np
import pandas as pd
from pathlib import Path
from predict_emg import predict_window, WINDOW_SIZE, channel_prefix

def acquire_live_emg(num_samples: int) -> pd.DataFrame:
    # EXAMPLE: fake data with 1 channel; replace with real live data
    # TO DELETE: From here
    t = np.arange(num_samples)
    fake_signal = 0.01 * np.random.randn(num_samples)
    # to here
    df = pd.DataFrame({
        "time": t,
        f"{channel_prefix}1": fake_signal,
    })
    return df

# Continuously 1) Acquires EMG window, 2) predicts motion 3) Prints the command
def live_predict_loop(sleep_s: float = 0.1):
    print("Starting live EMG prediction. Press Ctrl+C to stop.")

    try:
        while True:
            df_window = acquire_live_emg(WINDOW_SIZE)
            label, proba = predict_window(df_window)

            if label is None:
                print("Not enough data for a full window yet.")
            else:
                print(f"Predicted command: {label}")

            time.sleep(sleep_s) # Wait a bit before grabbing the next window

    except KeyboardInterrupt:
        print("\nLive prediction loop stopped.")


if __name__ == "__main__":
    live_predict_loop()