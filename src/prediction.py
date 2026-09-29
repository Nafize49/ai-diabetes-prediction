"""
prediction.py
-------------
Inference pipeline connecting patient input parameters to the trained model.

Features expected by the trained pipeline:
- Age
- BMI
- BloodPressure
- Glucose

Clinical tracking parameters:
- Physical Activity Level
- Total Cholesterol
"""

import os
import joblib
import pandas as pd
import numpy as np


def load_model():
    """Loads the trained Random Forest pipeline and metadata from disk."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "model", "diabetes_model.pkl"),
        os.path.join(base_dir, "diabetes_model.pkl"),
    ]
    for model_path in candidates:
        if os.path.exists(model_path):
            return joblib.load(model_path)

    raise FileNotFoundError(
        f"Trained model file not found in: {candidates}. Run 'python train_model.py' to generate it."
    )


def predict_patient_risk(
    age: float,
    bmi: float,
    physical_activity: str,
    blood_pressure: float,
    cholesterol: float,
    glucose: float,
    model_data=None
) -> dict:
    """
    Executes the prediction pipeline for a single patient record using the actual trained model.

    Parameters:
    -----------
    age : float
        Patient age in years
    bmi : float
        Body Mass Index in kg/m²
    physical_activity : str
        'Low', 'Moderate', or 'High'
    blood_pressure : float
        Diastolic / resting blood pressure in mm Hg
    cholesterol : float or str
        Total blood cholesterol level in mg/dL
    glucose : float
        Fasting blood glucose concentration in mg/dL
    model_data : dict or Pipeline, optional
        Pre-loaded model data from Streamlit cache
    """
    # 1. Acquire model data
    if model_data is None:
        model_data = load_model()

    # 2. Extract the actual trained Scikit-learn Pipeline
    if isinstance(model_data, dict):
        pipeline = model_data.get("pipeline") or model_data.get("model")
        feature_names = model_data.get("features") or model_data.get("feature_names") or ["Age", "BMI", "BloodPressure", "Glucose"]
    else:
        pipeline = model_data
        feature_names = ["Age", "BMI", "BloodPressure", "Glucose"]

    if pipeline is None:
        raise ValueError("Could not extract a valid Scikit-Learn pipeline from model file.")

    # 3. Form input DataFrame for the trained Pipeline
    # Maps user inputs directly to the 4 core clinical features
    df_core = pd.DataFrame([{
        "Age": float(age),
        "BMI": float(bmi),
        "BloodPressure": float(blood_pressure),
        "Glucose": float(glucose)
    }])[feature_names]

    # 4. Predict using the real trained pipeline
    probs = pipeline.predict_proba(df_core)[0]
    predicted_class = int(pipeline.predict(df_core)[0])

    prob_diabetic = float(probs[1])
    prob_non_diabetic = float(probs[0])
    risk_percentage = round(prob_diabetic * 100.0, 1)

    class_label = "Diabetic" if predicted_class == 1 else "Non-Diabetic"

    recommendation = (
        "Model prediction indicates an elevated likelihood of diabetes based on the provided clinical metrics."
        if predicted_class == 1 else
        "Model prediction indicates a lower likelihood of diabetes based on the provided clinical metrics."
    )

    return {
        "predicted_class": predicted_class,
        "class_label": class_label,
        "risk_percentage": risk_percentage,
        "base_model_probability": risk_percentage,
        "probability_negative": round(prob_non_diabetic * 100.0, 1),
        "probability_positive": risk_percentage,
        "recommendation": recommendation,
        "inputs_received": {
            "Age (years)": age,
            "BMI (kg/m²)": bmi,
            "Physical Activity Level": physical_activity,
            "Blood Pressure (mm Hg)": blood_pressure,
            "Total Cholesterol (mg/dL)": cholesterol,
            "Glucose Level (mg/dL)": glucose
        },
        "disclaimer": "This system is intended for academic and educational purposes and is not a medical diagnosis."
    }
