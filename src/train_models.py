"""
train_models.py
---------------
Trains and evaluates multiple ML models:
  - Regression  → predict final_score
  - Multi-class → predict performance_grade (A/B/C/D/F)
  - Binary      → predict pass_fail

Saves best models + metrics JSON for the web API.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import (RandomForestRegressor, RandomForestClassifier,
                                     GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics         import (mean_absolute_error, r2_score, mean_squared_error,
                                     accuracy_score, classification_report,
                                     confusion_matrix)
import joblib, warnings
warnings.filterwarnings("ignore")

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "..", "data",    "students.csv")
MODELS  = os.path.join(BASE, "..", "models")
OUTPUTS = os.path.join(BASE, "..", "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ── Local imports ─────────────────────────────────────────────────────────────
from preprocessor import load_data, clean_data, encode_and_scale


def train_regression(X_tr, X_te, y_tr, y_te, feat_names):
    print("\n=== REGRESSION (predict final_score) ===")
    models = {
        "Linear Regression"        : LinearRegression(),
        "Decision Tree Regressor"  : DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor"  : RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting Reg"    : GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    results = {}
    best_r2, best_name, best_model = -np.inf, None, None

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)
        results[name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}
        print(f"  {name:35s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    print(f"\n  [BEST] Best Regressor -> {best_name}  (R²={best_r2:.4f})")
    joblib.dump(best_model, os.path.join(MODELS, "regressor.pkl"))

    # Feature importance
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        feat_imp = dict(sorted(zip(feat_names, fi.tolist()), key=lambda x: -x[1]))

    return results, feat_imp


def train_classification(X_tr, X_te, y_tr, y_te, classes):
    print("\n=== CLASSIFICATION (predict grade A/B/C/D/F) ===")
    models = {
        "Logistic Regression"        : LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree Classifier"   : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest Classifier"   : RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting Clf"      : GradientBoostingClassifier(n_estimators=150, random_state=42),
    }
    results = {}
    best_acc, best_name, best_model = -np.inf, None, None

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        results[name] = {"Accuracy": round(acc, 4)}
        print(f"  {name:35s}  Accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    print(f"\n  [BEST] Best Classifier -> {best_name}  (Acc={best_acc:.4f})")
    joblib.dump(best_model, os.path.join(MODELS, "classifier.pkl"))

    # Confusion matrix
    y_pred_best = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred_best).tolist()

    return results, cm


def train_binary(X_tr, X_te, y_tr, y_te):
    print("\n=== BINARY CLASSIFICATION (Pass / Fail) ===")
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    print(f"  Pass/Fail Accuracy: {acc:.4f}")
    joblib.dump(model, os.path.join(MODELS, "binary_clf.pkl"))
    return {"Accuracy": round(acc, 4)}


def run_training():
    # 1. Load + clean
    df = clean_data(load_data(DATA))

    # 2. Encode
    X, y_reg, y_clf, y_bin, classes, feat_names = encode_and_scale(df, fit=True)

    X_arr = X.values

    # 3. Split
    X_tr, X_te, yr_tr, yr_te = train_test_split(X_arr, y_reg, test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X_arr, y_clf, test_size=0.2, random_state=42)
    _,    _,    yb_tr, yb_te = train_test_split(X_arr, y_bin, test_size=0.2, random_state=42)

    # 4. Train
    reg_results,   feat_imp = train_regression(X_tr, X_te, yr_tr, yr_te, feat_names)
    clf_results,   cm       = train_classification(X_tr, X_te, yc_tr, yc_te, classes)
    bin_results             = train_binary(X_tr, X_te, yb_tr, yb_te)

    # 5. Save metadata
    meta = {
        "feature_names"          : feat_names,
        "grade_classes"          : classes,
        "regression_results"     : reg_results,
        "classification_results" : clf_results,
        "binary_results"         : bin_results,
        "feature_importance"     : feat_imp,
        "confusion_matrix"       : cm,
        "n_students"             : len(df),
    }
    with open(os.path.join(OUTPUTS, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[SUCCESS] metrics.json saved -> {OUTPUTS}")

    # 6. Dataset stats for dashboard
    stats = {
        "pass_rate"    : round((df["pass_fail"] == "Pass").mean() * 100, 1),
        "avg_score"    : round(df["final_score"].mean(), 1),
        "grade_dist"   : df["performance_grade"].value_counts().to_dict(),
        "gender_dist"  : df["gender"].value_counts().to_dict(),
        "stress_dist"  : df["stress_level"].value_counts().to_dict(),
        "avg_study_hrs": round(df["study_hours_per_day"].mean(), 2),
        "avg_attend"   : round(df["attendance_pct"].mean(), 2),
        "score_bins"   : {
            "labels": ["0-20","20-40","40-60","60-80","80-100"],
            "counts": pd.cut(df["final_score"],
                             bins=[0,20,40,60,80,100],
                             labels=["0-20","20-40","40-60","60-80","80-100"]
                             ).value_counts().sort_index().tolist()
        }
    }
    with open(os.path.join(OUTPUTS, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[SUCCESS] stats.json saved  -> {OUTPUTS}")
    print("\n[SUCCESS] ALL MODELS TRAINED SUCCESSFULLY!")


if __name__ == "__main__":
    run_training()
