import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("emg_car_windows_3class.csv")

X = df.drop(columns=['control_label'])
y = df['control_label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train_s, y_train)

print("Validation report:")
print(classification_report(y_val, clf.predict(X_val_s)))
print("Test confusion matrix:")
print(confusion_matrix(y_test, clf.predict(X_test_s)))

joblib.dump(clf, "emg_3class_logreg.pkl")
joblib.dump(scaler, "emg_3class_scaler.pkl")
