"""
test_predictions.py
-------------------
Validates prediction across 3 distinct patient profiles.
"""

from src.prediction import predict_patient_risk

tests = [
    {
        "name": "Test 1 (Standard / Baseline Patient)",
        "age": 35,
        "bmi": 26.4,
        "physical_activity": "Moderate",
        "blood_pressure": 75,
        "cholesterol": 195,
        "glucose": 110
    },
    {
        "name": "Test 2 (Older Adult / High Risk)",
        "age": 58,
        "bmi": 34.2,
        "physical_activity": "Low",
        "blood_pressure": 92,
        "cholesterol": 245,
        "glucose": 185
    },
    {
        "name": "Test 3 (Young Athlete / Very Low Risk)",
        "age": 24,
        "bmi": 21.5,
        "physical_activity": "High",
        "blood_pressure": 68,
        "cholesterol": 160,
        "glucose": 85
    }
]

for t in tests:
    res = predict_patient_risk(
        age=t["age"],
        bmi=t["bmi"],
        physical_activity=t["physical_activity"],
        blood_pressure=t["blood_pressure"],
        cholesterol=t["cholesterol"],
        glucose=t["glucose"]
    )
    print(f"=== {t['name']} ===")
    print(f"  Class: {res['predicted_class']} ({res['class_label']})")
    print(f"  Estimated Risk Probability: {res['risk_percentage']}%")
    print(f"  Base Model Probability:     {res['base_model_probability']}%")
    print(f"  Inputs: {res['inputs_received']}")
    print(f"  Recommendation: {res['recommendation']}")
    print()
