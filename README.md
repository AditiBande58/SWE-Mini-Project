# SWE Match: Brain Controlled RC Car

This project uses a Logistic Regression model to identify commands of forward, backward, and stop based on EMG (muscle activity) from your arm to control a robotic car.

## Requirements

- Python 3.9+
- Required packages:
  ```bash
  pip install pandas scikit-learn joblib
  ```

## Usage

Before running any code, ensure that:

- Inside the `datasets` folder, you place your **labeled EMG raw data** file.

There are **two main phases** to this project:

### 1) Clean raw data, create features, and train the model

From the `Swe-Mini-Project/datasets` directory, run the scripts in the following order:

- `inspect_emg.py`
- `make_features.py`
- `train_model.py`

This will:

- Clean and filter the raw EMG data.
- Generate feature representations.
- Train and save the classification model.

### 2) Take in live data and predict

After the model is trained, you can run live prediction using:

```
python main.py
```

This script will:

- Continuously accept live EMG input.
- Call the prediction pipeline.
- Output the predicted motion/command labels in real time.
