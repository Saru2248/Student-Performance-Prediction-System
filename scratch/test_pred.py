import os
import joblib
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

regressor = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))
classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

data = {
    "age": 18,
    "gender": "Male",
    "study_hours_per_day": 8.0,
    "attendance_pct": 98.0,
    "assignments_done": 20,
    "previous_score": 95.0,
    "parental_education": "Graduate",
    "internet_access": "Yes",
    "extracurricular": "Yes",
    "sleep_hours": 8.0,
    "stress_level": "Medium",
    "library_visits": 15,
    "tuition_classes": "No"
}

df = pd.DataFrame([data])
binary_map = {"Yes": 1, "No": 0, "Male": 0, "Female": 1, "Other": 2}
stress_map = {"Low": 0, "Medium": 1, "High": 2}
edu_map = {"None": 0, "High School": 1, "Graduate": 2, "Postgraduate": 3}

df["gender"] = df["gender"].map(binary_map)
df["parental_education"] = df["parental_education"].map(edu_map)
df["internet_access"] = df["internet_access"].map(binary_map)
df["extracurricular"] = df["extracurricular"].map(binary_map)
df["stress_level"] = df["stress_level"].map(stress_map)
df["tuition_classes"] = df["tuition_classes"].map(binary_map)

with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "r") as f:
    metrics = json.load(f)
    feature_names = metrics["feature_names"]
    
df = df[feature_names]
X = scaler.transform(df)

score_pred = regressor.predict(X)[0]
grade_pred_idx = classifier.predict(X)[0]
grade_pred = label_encoder.inverse_transform([grade_pred_idx])[0]

print(f"Predicted Score: {score_pred}")
print(f"Predicted Grade: {grade_pred}")
