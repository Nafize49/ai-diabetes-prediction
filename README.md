# AI-Based Diabetes Prediction System

**Student:** Nafize Ali  
**Branch:** Artificial Intelligence & Data Science  
**Technology:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Random Forest, Joblib  
**Project:** B.Tech Minor Project  

---

## 1. Project Overview
Diabetes Mellitus represents one of the fastest-growing metabolic health challenges globally. Early identification of individuals at elevated risk allows for timely lifestyle modifications and clinical guidance.

This project delivers an end-to-end, scientifically grounded AI/ML system developed as a B.Tech Minor Project in Artificial Intelligence & Data Science. The application provides an interactive, easy-to-use clinical risk estimation interface backed by a validated machine learning pipeline trained with rigorous data hygiene principles.

---

## 2. Features
- **Interactive Multi-Page Streamlit UI:** Five intuitive modules covering Home, Prediction, Data Insights, Model Performance, and About.
- **Human-Centric Clinical Form:** Two-column input layout with clear, high-contrast dark charcoal (`#1F2937`) labels and readable input controls.
- **Dynamic Probability & Risk Estimation:** Generates model probability percentages, class assignment (Diabetic / Non-Diabetic), and contextual clinical observations.
- **Transparent Data Insights:** Real dataset distributions, Pearson correlation heatmaps, and class imbalance metrics.
- **Empirical Model Performance:** True benchmark comparison across multiple algorithms with ROC curves, confusion matrices, and feature importance rankings.
- **Responsible AI Disclaimer:** Prominent academic notifications emphasizing that the tool is intended for educational demonstrations.

---

## 3. Technologies Used
- **Programming Language:** Python 3.9+
- **Web Dashboard:** Streamlit
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-learn (`Pipeline`, `RandomForestClassifier`, `SimpleImputer`, `StandardScaler`)
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

---

## 4. Machine Learning Approach
1. **Zero Data Leakage:** Imputation and feature scaling are encapsulated within a Scikit-learn `Pipeline` fitted strictly on the training partition ($80\%$) and applied to the test partition ($20\%$).
2. **Biological Zero Treatment:** Physiological features such as Glucose, Blood Pressure, and BMI where a reading of 0 is physiologically impossible are converted to `NaN` and imputed with partition medians.
3. **Class Imbalance Handling:** Configured `class_weight='balanced'` in the Random Forest to ensure optimal sensitivity and recall on minority diabetic cases.
4. **Ensemble Modeling:** Employs a 100-tree Random Forest classifier to minimize individual decision tree variance and ensure strong generalization.

---

## 5. Dataset
The project is built on the benchmark Pima Indians Diabetes Dataset:
- **Total Records:** 284 clinical observations
- **Target Distribution:** 177 Non-Diabetic ($62.3\%$) / 107 Diabetic ($37.7\%$)
- **File:** `diabetes_binary_health.csv`

---

## 6. Input Features
The prediction interface evaluates six patient parameters:

| Parameter | Type | Valid Range | Clinical Description |
| :--- | :--- | :--- | :--- |
| **Age** | Numerical | 1 – 120 years | Patient age in years |
| **Blood Pressure** | Numerical | 40 – 200 mm Hg | Resting diastolic blood pressure |
| **BMI** | Numerical | 10.0 – 70.0 kg/m² | Body Mass Index ($\text{weight in kg}/(\text{height in m})^2$) |
| **Glucose Level** | Numerical | 40 – 300 mg/dL | Fasting plasma glucose concentration |
| **Physical Activity Level** | Categorical | Low, Moderate, High | Routine physical activity habit |
| **Total Cholesterol** | Numerical | 100 – 400 mg/dL | Total serum cholesterol level |

---

## 7. Prediction Output
The prediction system outputs:
- **Predicted Condition:** Diabetic (Class 1) or Non-Diabetic (Class 0)
- **Model Risk Probability:** Estimated probability percentage from the Random Forest ensemble
- **Contextual Recommendation:** Clear, plain-language guidance based on the risk level
- **Academic Disclaimer:** Explicit notification stating that output is an educational machine-learning estimate and not a medical diagnosis.

---

## 8. Model Evaluation
Models were evaluated on an identical stratified $20\%$ holdout test partition ($N = 57$):

| Model | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 70.18% | 57.14% | 76.19% | 65.31% | 0.8135 |
| **Decision Tree** | 64.91% | 51.52% | 80.95% | 62.96% | 0.6812 |
| **Random Forest (Selected)** | **71.93%** | **60.00%** | **71.43%** | **65.22%** | **0.7817** |

*Random Forest was chosen as the primary deployment algorithm due to its optimal balance between Recall ($71.43\%$), Precision ($60.00\%$), and variance reduction.*

---

## 9. Project Structure
```text
ai-diabetes-prediction/
│
├── app.py                             # Interactive Streamlit application entry point
├── requirements.txt                   # Production Python package dependencies
├── README.md                          # Project documentation & reference guide
├── diabetes_model.pkl                 # Serialized trained ML pipeline & metadata
├── diabetes_binary_health.csv         # Verified clinical dataset
│
├── src/
│   ├── __init__.py                    # Python package initializer
│   ├── data_preprocessing.py          # Preprocessing import compatibility
│   ├── prediction.py                  # Inference pipeline & risk evaluation
│   ├── preprocessing.py               # Data loading, zero handling & pipeline builder
│   └── train_model.py                 # Model training, evaluation & artifact exporter
│
├── .streamlit/
│   └── config.toml                    # Theme and server configuration
│
├── dataset/
│   └── diabetes_binary_health.csv     # Dataset copy for structured layout
│
├── model/
│   └── diabetes_model.pkl             # Model copy for structured layout
│
├── screenshots/                       # Application UI screenshots
│   ├── home.png
│   ├── prediction.png
│   ├── data_insights.png
│   └── model_performance.png
│
├── visualizations/                    # Evaluation and EDA charts
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── key_features_distribution.png
│   └── roc_curves.png
│
└── VIVA_QUESTIONS.md                  # Viva Voce preparation guide
```

---

## 10. How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Nafize49/ai-diabetes-prediction.git
cd ai-diabetes-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the application
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## 11. How to Deploy (Streamlit Community Cloud)

1. Fork or push this repository to your GitHub account: `https://github.com/Nafize49/ai-diabetes-prediction`.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**.
4. Configure the deployment settings:
   - **Repository:** `Nafize49/ai-diabetes-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**.

---

## 12. Disclaimer
**Educational & Academic Use Only:** This application is developed solely as an educational demonstration for a B.Tech Minor Project in Artificial Intelligence & Data Science. It is not certified for clinical diagnostic use and does not constitute medical advice. For clinical health evaluations, consult a licensed medical professional.
