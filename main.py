# Server reload trigger
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Student Performance Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load models and preprocessing objects globally
try:
    regressor = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))
    classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
    binary_clf = joblib.load(os.path.join(MODELS_DIR, "binary_clf.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("[SUCCESS] All models loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Could not load models. Did you run train_models.py? Error: {e}")

class StudentData(BaseModel):
    age: int
    gender: str
    study_hours_per_day: float
    attendance_pct: float
    assignments_done: int
    previous_score: float
    parental_education: str
    internet_access: str
    extracurricular: str
    sleep_hours: float
    stress_level: str
    library_visits: int
    tuition_classes: str

def preprocess_input(data: StudentData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Apply the same mapping as in preprocessor.py
    binary_map = {"Yes": 1, "No": 0, "Male": 0, "Female": 1, "Other": 2}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    edu_map = {"None": 0, "High School": 1, "Graduate": 2, "Postgraduate": 3}
    
    df["gender"] = df["gender"].map(binary_map)
    df["parental_education"] = df["parental_education"].map(edu_map)
    df["internet_access"] = df["internet_access"].map(binary_map)
    df["extracurricular"] = df["extracurricular"].map(binary_map)
    df["stress_level"] = df["stress_level"].map(stress_map)
    df["tuition_classes"] = df["tuition_classes"].map(binary_map)
    
    # Ensure column order matches the scaler
    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
        feature_names = metrics["feature_names"]
        
    df = df[feature_names]
    
    # Scale features
    scaled_features = scaler.transform(df)
    return scaled_features

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Load stats if available
    stats = {}
    try:
        with open(os.path.join(OUTPUTS_DIR, "stats.json"), "r") as f:
            stats = json.load(f)
    except:
        pass
        
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})

@app.get("/api/dashboard-data")
async def dashboard_data():
    try:
        with open(os.path.join(OUTPUTS_DIR, "stats.json"), "r") as f:
            stats = json.load(f)
        with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "r") as f:
            metrics = json.load(f)
        return {"stats": stats, "metrics": metrics}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/predict")
async def predict(data: StudentData):
    try:
        # Preprocess
        X = preprocess_input(data)
        
        # Predict
        score_pred = regressor.predict(X)[0]
        score_pred = max(0, min(100, score_pred)) # clip between 0-100
        
        grade_pred_idx = classifier.predict(X)[0]
        grade_pred = label_encoder.inverse_transform([grade_pred_idx])[0]
        
        pass_fail_pred = binary_clf.predict(X)[0]
        pass_fail = "Pass" if pass_fail_pred == 1 else "Fail"
        
        # Insights generation based on input
        insights = []
        if data.attendance_pct < 75:
            insights.append("Low attendance is likely impacting performance.")
        if data.study_hours_per_day < 3:
            insights.append("Consider increasing daily study hours.")
        if data.stress_level == "High":
            insights.append("High stress level detected. Stress management might help.")
        if not insights:
            insights.append("Student is on a good track! Keep up the current habits.")
            
        return {
            "predicted_score": round(score_pred, 1),
            "predicted_grade": grade_pred,
            "pass_fail": pass_fail,
            "insights": insights
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
