# 🎓 Student Performance Prediction System

An end-to-end Machine Learning system designed to predict student academic performance, identify at-risk students, and generate actionable insights for improving learning outcomes.

---

## 📌 Problem Statement

Educational institutions often struggle to identify students who are likely to underperform until it is too late.

This project solves that problem by building a predictive system that:

* Forecasts student performance
* Detects potential failures early
* Provides data-driven insights for intervention

---

## 🎯 Key Features

* 📊 Predicts **final student score** using ML models
* ⚠️ Identifies **at-risk students**
* 📈 Provides **feature importance insights**
* 🧠 Uses **synthetic yet realistic dataset simulation**
* 🔍 Includes **exploratory data analysis (EDA)**
* ⚙️ End-to-end ML pipeline (data → model → prediction)

---

## 🧠 Machine Learning Approach

### Models Used

* Linear Regression (Baseline)
* Random Forest Regressor (Primary Model)

### Why Random Forest?

* Handles non-linear relationships
* Robust to noise
* Provides feature importance

---

## 📊 Input Features

* Study Hours
* Attendance (%)
* Previous Marks
* Assignments Completed
* Sleep Hours

---

## 📈 Output

* Predicted Final Score (Regression)

---

## 🏗️ Project Architecture

```
Student Data
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Insights & Visualization
```

---

## 📂 Folder Structure

```
Student-Performance-Prediction/
│
├── data/               # Dataset (synthetic)
├── notebooks/          # EDA & experiments
├── src/                # Core scripts
├── models/             # Saved models
├── outputs/            # Results & metrics
├── images/             # Graphs & visuals
├── README.md
├── requirements.txt
└── main.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

---

## 🚀 Running the Project

```bash
python main.py
```

---

## 📊 Model Performance

| Metric   | Value  |
| -------- | ------ |
| MAE      | ~2-5   |
| R² Score | ~0.85+ |

*(Values may vary due to synthetic data)*

---

## 🔬 Dataset Strategy (Important)

Since real student data is private, this project uses **synthetic data generation**.

The dataset simulates real-world behavior:

* More study hours → higher scores
* Higher attendance → better performance
* Random noise → realistic variation

This ensures:

* Realistic training patterns
* Interview-ready explanation

---

## 📈 Visualizations

* Feature Importance Graph
* Score Distribution
* Correlation Heatmap

---

## 💡 Key Insights

* Study hours and previous marks are the strongest predictors
* Attendance has a significant impact on performance
* Assignments improve consistency in results

---

## 📌 Use Cases

* Schools & Colleges → Early intervention
* EdTech Platforms → Personalized learning
* Government → Educational analytics

---

## 🔮 Future Improvements

* Add classification (Pass/Fail / Grades)
* Deploy using Streamlit or Flask
* Integrate real datasets
* Hyperparameter tuning
* Add recommendation system

---

## 🧑‍💻 Author

**Sarthak Dhumal**

---

