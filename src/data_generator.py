"""
data_generator.py
-----------------
Generates a realistic synthetic student performance dataset.
Uses statistical correlations to ensure data quality.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

N_STUDENTS = 1000


def generate_student_data(n=N_STUDENTS, save=True) -> pd.DataFrame:
    """
    Generate a synthetic student performance dataset with realistic correlations.

    Features:
        - student_id         : Unique student identifier
        - age                : Student age (15–22)
        - gender             : Male / Female / Other
        - study_hours_per_day: Average daily study hours (0–10)
        - attendance_pct     : Attendance percentage (40–100)
        - assignments_done   : Number of assignments completed (0–20)
        - previous_score     : Average score in previous term (30–100)
        - parental_education : Parental education level (None/High School/Graduate/Postgraduate)
        - internet_access    : Whether student has internet access (Yes/No)
        - extracurricular    : Participates in extracurricular activities (Yes/No)
        - sleep_hours        : Average sleep hours per night (4–10)
        - stress_level       : Self-reported stress level (Low/Medium/High)
        - library_visits     : Monthly library visits (0–20)
        - tuition_classes    : Attends private tuition (Yes/No)
        - final_score        : Final exam score — TARGET (0–100)
        - performance_grade  : Grade derived from final_score (A/B/C/D/F)
        - pass_fail          : Binary pass/fail (Pass if final_score >= 40)
    """

    # --- Base features ---
    study_hours = np.clip(np.random.normal(4.5, 2.0, n), 0, 10).round(1)
    attendance  = np.clip(np.random.normal(75, 15, n), 40, 100).round(1)
    prev_score  = np.clip(np.random.normal(62, 15, n), 30, 100).round(1)
    assignments = np.clip(np.random.normal(13, 4, n), 0, 20).astype(int)
    sleep_hours = np.clip(np.random.normal(7, 1.2, n), 4, 10).round(1)
    lib_visits  = np.clip(np.random.poisson(5, n), 0, 20)

    # --- Categorical features ---
    gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    parental_edu = np.random.choice(
        ["None", "High School", "Graduate", "Postgraduate"],
        n, p=[0.10, 0.30, 0.40, 0.20]
    )
    internet    = np.random.choice(["Yes", "No"], n, p=[0.78, 0.22])
    extra       = np.random.choice(["Yes", "No"], n, p=[0.55, 0.45])
    stress      = np.random.choice(["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25])
    tuition     = np.random.choice(["Yes", "No"], n, p=[0.40, 0.60])
    age         = np.random.randint(15, 23, n)

    # --- Encoded influences for target calculation ---
    internet_boost = np.where(internet == "Yes", 3, 0)
    extra_boost    = np.where(extra == "Yes", 2, -1)
    stress_penalty = np.where(stress == "High", -5, np.where(stress == "Medium", -2, 2))
    tuition_boost  = np.where(tuition == "Yes", 4, 0)
    edu_map        = {"None": -3, "High School": 0, "Graduate": 2, "Postgraduate": 4}
    edu_boost      = np.array([edu_map[e] for e in parental_edu])

    # --- Final score (correlated with features + noise) ---
    noise       = np.random.normal(0, 5, n)
    final_score = (
        0.45 * prev_score
        + 0.20 * (study_hours * 10)
        + 0.15 * attendance
        + 0.05 * (assignments * 5)
        + internet_boost
        + extra_boost
        + stress_penalty
        + tuition_boost
        + edu_boost
        + noise
    )
    final_score = np.clip(final_score.round(1), 0, 100)

    # --- Grade and pass/fail ---
    def score_to_grade(s):
        if s >= 80: return "A"
        elif s >= 65: return "B"
        elif s >= 50: return "C"
        elif s >= 40: return "D"
        else: return "F"

    grades    = [score_to_grade(s) for s in final_score]
    pass_fail = ["Pass" if s >= 40 else "Fail" for s in final_score]

    df = pd.DataFrame({
        "student_id"           : [f"STU{str(i+1).zfill(4)}" for i in range(n)],
        "age"                  : age,
        "gender"               : gender,
        "study_hours_per_day"  : study_hours,
        "attendance_pct"       : attendance,
        "assignments_done"     : assignments,
        "previous_score"       : prev_score,
        "parental_education"   : parental_edu,
        "internet_access"      : internet,
        "extracurricular"      : extra,
        "sleep_hours"          : sleep_hours,
        "stress_level"         : stress,
        "library_visits"       : lib_visits,
        "tuition_classes"      : tuition,
        "final_score"          : final_score,
        "performance_grade"    : grades,
        "pass_fail"            : pass_fail,
    })

    if save:
        out_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[SUCCESS] Dataset saved -> {os.path.abspath(out_path)}  ({n} rows)")

    return df


if __name__ == "__main__":
    df = generate_student_data()
    print(df.head())
    print(df.describe())
