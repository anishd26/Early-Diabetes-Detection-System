# Diabetes Prediction — Web Application

**ML Lab Project | Anish Deshmukh & Aayush Chandel**

## How to Run Locally

### Step 1 — Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### Step 2 — Start the server
```bash
cd webapp
python app.py
```

### Step 3 — Open in browser
```
http://localhost:5050
```

## What the website does
- Takes 8 clinical inputs (Glucose, BMI, Age, etc.)
- Runs **Random Forest** + **Logistic Regression** predictions simultaneously
- Shows risk level, contributing factors, feature importance bars
- Includes a Viva Q&A section with all answers

## Files
- `app.py`            — Flask backend + prediction API
- `templates/index.html` — Full frontend (no external dependencies needed)
- `model_rf.pkl`      — Trained Random Forest model
- `model_lr.pkl`      — Trained Logistic Regression model
- `scaler.pkl`        — StandardScaler for LR preprocessing

## Dataset
Pima Indians Diabetes Database — Kaggle / NIDDK
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
