"""Train the crop recommendation model and save it to disk.
Run once:  python train_model.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("Crop_recommendation.csv")

x = data.iloc[:, :-1]  # features: N, P, K, temperature, humidity, ph, rainfall
y = data.iloc[:, -1]   # label: crop name

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test))

joblib.dump(model, "crop_model.joblib")
print("Model saved to crop_model.joblib")
