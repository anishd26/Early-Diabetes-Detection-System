from flask import Flask, request, jsonify, render_template
import pickle, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
rf    = pickle.load(open(os.path.join(BASE, 'model_rf.pkl'), 'rb'))
lr    = pickle.load(open(os.path.join(BASE, 'model_lr.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE, 'scaler.pkl'), 'rb'))

FEATURE_NAMES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']

FEATURE_IMPORTANCE = {
    'Glucose': 0.3115, 'DiabetesPedigreeFunction': 0.1174,
    'Pregnancies': 0.1166, 'BMI': 0.1115, 'BloodPressure': 0.1034,
    'SkinThickness': 0.0818, 'Insulin': 0.0818, 'Age': 0.0762
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [float(data[f]) for f in FEATURE_NAMES]
        arr = np.array(features).reshape(1, -1)

        # Random Forest prediction
        rf_pred = int(rf.predict(arr)[0])
        rf_prob = float(rf.predict_proba(arr)[0][1])

        # Logistic Regression prediction
        arr_sc = scaler.transform(arr)
        lr_pred = int(lr.predict(arr_sc)[0])
        lr_prob = float(lr.predict_proba(arr_sc)[0][1])

        # Risk level
        prob = rf_prob
        if prob < 0.3:   risk = "Low Risk"
        elif prob < 0.6: risk = "Moderate Risk"
        else:            risk = "High Risk"

        # Contributing factors
        glucose = features[1]
        bmi     = features[5]
        age     = features[7]
        dpf     = features[6]

        factors = []
        if glucose > 140: factors.append({"label": "High Glucose", "level": "high", "value": f"{glucose} mg/dL"})
        elif glucose > 100: factors.append({"label": "Elevated Glucose", "level": "medium", "value": f"{glucose} mg/dL"})
        else: factors.append({"label": "Normal Glucose", "level": "low", "value": f"{glucose} mg/dL"})

        if bmi > 30: factors.append({"label": "Obese BMI", "level": "high", "value": f"{bmi}"})
        elif bmi > 25: factors.append({"label": "Overweight BMI", "level": "medium", "value": f"{bmi}"})
        else: factors.append({"label": "Healthy BMI", "level": "low", "value": f"{bmi}"})

        if age > 45: factors.append({"label": "Age Risk", "level": "high", "value": f"{int(age)} yrs"})
        elif age > 35: factors.append({"label": "Age Factor", "level": "medium", "value": f"{int(age)} yrs"})
        else: factors.append({"label": "Young Age", "level": "low", "value": f"{int(age)} yrs"})

        if dpf > 0.8: factors.append({"label": "High Genetic Risk", "level": "high", "value": f"{dpf:.3f}"})
        elif dpf > 0.4: factors.append({"label": "Moderate Genetic Risk", "level": "medium", "value": f"{dpf:.3f}"})
        else: factors.append({"label": "Low Genetic Risk", "level": "low", "value": f"{dpf:.3f}"})

        return jsonify({
            "rf_prediction": rf_pred,
            "rf_probability": round(rf_prob * 100, 1),
            "lr_prediction": lr_pred,
            "lr_probability": round(lr_prob * 100, 1),
            "risk_level": risk,
            "factors": factors,
            "feature_importance": FEATURE_IMPORTANCE
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5050)
