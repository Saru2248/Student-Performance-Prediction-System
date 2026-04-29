"""
preprocessor.py
---------------
Data cleaning, encoding, and feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[SUCCESS] Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, handle missing values."""
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols     = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[cat_cols]     = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True):
    """
    Encode categoricals + scale numerics.
    Returns (X, y_score, y_grade, y_pass, encoders, scaler)
    """
    df = df.copy()

    # Drop ID and targets
    drop_cols  = ["student_id", "final_score", "performance_grade", "pass_fail"]
    target_reg = df["final_score"]
    target_clf = df["performance_grade"]
    target_bin = df["pass_fail"]

    X = df.drop(columns=drop_cols, errors="ignore")

    # Binary / ordinal categoricals
    binary_map  = {"Yes": 1, "No": 0, "Male": 0, "Female": 1, "Other": 2,
                   "Pass": 1, "Fail": 0}
    stress_map  = {"Low": 0, "Medium": 1, "High": 2}
    edu_map     = {"None": 0, "High School": 1, "Graduate": 2, "Postgraduate": 3}

    for col in X.select_dtypes(include="object").columns:
        if col == "stress_level":
            X[col] = X[col].map(stress_map)
        elif col == "parental_education":
            X[col] = X[col].map(edu_map)
        elif col in ("gender",):
            X[col] = X[col].map(binary_map)
        else:
            X[col] = X[col].map(binary_map)

    X = X.astype(float)

    # Scale
    scaler = StandardScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    else:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        X_scaled = scaler.transform(X)

    # Encode grade label
    le = LabelEncoder()
    if fit:
        y_grade_enc = le.fit_transform(target_clf)
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    else:
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        y_grade_enc = le.transform(target_clf)

    y_pass_enc = target_bin.map({"Pass": 1, "Fail": 0}).values

    return (
        pd.DataFrame(X_scaled, columns=X.columns),
        target_reg.values,
        y_grade_enc,
        y_pass_enc,
        le.classes_.tolist(),
        X.columns.tolist()
    )


if __name__ == "__main__":
    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
    df   = load_data(DATA)
    df   = clean_data(df)
    X, y_reg, y_clf, y_bin, classes, feat_names = encode_and_scale(df)
    print("Feature columns:", feat_names)
    print("Grade classes  :", classes)
    print("X shape        :", X.shape)
