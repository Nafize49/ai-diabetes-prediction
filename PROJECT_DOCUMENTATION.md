# MINOR PROJECT REPORT

## PROJECT TITLE:
# AI-Based Diabetes Prediction System
### *A Machine Learning Approach to Healthcare Risk Stratification and Clinical Decision Support*

---

**Academic Degree:** Bachelor of Technology (B.Tech)  
**Department:** Artificial Intelligence & Data Science  
**Project Category:** Minor Project  

---

## TABLE OF CONTENTS
- [CHAPTER 1 – INTRODUCTION](#chapter-1--introduction)
- [CHAPTER 2 – PROBLEM STATEMENT](#chapter-2--problem-statement)
- [CHAPTER 3 – OBJECTIVES](#chapter-3--objectives)
- [CHAPTER 4 – EXISTING SYSTEM](#chapter-4--existing-system)
- [CHAPTER 5 – PROPOSED SYSTEM](#chapter-5--proposed-system)
- [CHAPTER 6 – SYSTEM REQUIREMENTS](#chapter-6--system-requirements)
- [CHAPTER 7 – DATASET DESCRIPTION](#chapter-7--dataset-description)
- [CHAPTER 8 – DATA PREPROCESSING](#chapter-8--data-preprocessing)
- [CHAPTER 9 – EXPLORATORY DATA ANALYSIS](#chapter-9--exploratory-data-analysis)
- [CHAPTER 10 – MACHINE LEARNING MODELS](#chapter-10--machine-learning-models)
- [CHAPTER 11 – MODEL EVALUATION](#chapter-11--model-evaluation)
- [CHAPTER 12 – SYSTEM DESIGN](#chapter-12--system-design)
- [CHAPTER 13 – STREAMLIT APPLICATION](#chapter-13--streamlit-application)
- [CHAPTER 14 – RESULTS](#chapter-14--results)
- [CHAPTER 15 – LIMITATIONS](#chapter-15--limitations)
- [CHAPTER 16 – FUTURE ENHANCEMENTS](#chapter-16--future-enhancements)
- [CHAPTER 17 – CONCLUSION](#chapter-17--conclusion)
- [CHAPTER 18 – REFERENCES](#chapter-18--references)

---

## CHAPTER 1 – INTRODUCTION
Diabetes Mellitus is a chronic metabolic syndrome characterized by persistent elevated blood glucose levels (hyperglycemia). It stems from defects in insulin secretion, insulin action, or both. Over time, poorly regulated diabetes causes severe damage to vital organs, including diabetic retinopathy (leading to blindness), nephropathy (kidney failure), cardiovascular events (myocardial infarction and stroke), and peripheral neuropathy.

According to the International Diabetes Federation (IDF), more than 537 million adults worldwide were living with diabetes as of recent epidemiological surveys, with projections surpassing 700 million by 2045. A significant proportion of affected individuals remain undiagnosed during the initial stages when lifestyle modifications (such as dietary adjustments and moderate physical activity) could prevent or substantially delay chronic complications.

Artificial Intelligence (AI) and Machine Learning (ML) present transformative opportunities in healthcare screening. By uncovering non-linear patterns within diagnostic indicators—such as plasma glucose, body mass index (BMI), blood pressure, insulin, and genetic pedigree—supervised machine learning algorithms can calculate individualized risk probabilities before acute clinical onset occurs.

This minor project presents an end-to-end AI-based diabetes prediction system designed for educational demonstration and screening analytics. The system integrates robust data preprocessing, comparative evaluation of standard machine learning algorithms, and an intuitive Streamlit web application.

---

## CHAPTER 2 – PROBLEM STATEMENT
Traditional screening for diabetes often relies on periodic laboratory blood investigations, including the Oral Glucose Tolerance Test (OGTT) and Glycated Hemoglobin (HbA1c). However, in resource-constrained communities and preliminary wellness checkups, comprehensive laboratory diagnostics are not always immediately accessible or frequently administered.

Furthermore, manual clinical interpretation faces several practical challenges:
1. **Multi-Factorial Complexity:** Diabetes risk involves complex interactions between adiposity (BMI), metabolic regulation (Glucose and Insulin), maternal history (Pregnancies), genetics (Pedigree score), and aging.
2. **Biological Data Anomalies:** Healthcare datasets often contain zeroes for vital physiological measurements (e.g., Blood Pressure = 0, Glucose = 0, BMI = 0), which are physically impossible for living patients and mislead naïve algorithms.
3. **Data Leakage in Academic Work:** Many student projects mistakenly perform scaling and imputation across the entire dataset prior to splitting, producing artificially optimistic results that fail in production.
4. **Lack of User-Friendly Deployment:** Predictive models often remain confined to experimental Python scripts without accessible interfaces for interactive exploration.

Hence, there is a clear engineering need for a robust, reproducible, leakage-free machine learning pipeline paired with an interactive clinical risk evaluation dashboard.

---

## CHAPTER 3 – OBJECTIVES
The core objectives of this minor project are:
1. **Data Acquisition and Verification:** Inspect and structure the standard Pima Indian Diabetes healthcare dataset without making blind assumptions regarding column structure.
2. **Clinical Preprocessing & Zero Treatment:** Identify physiologically invalid zero values in metabolic features (Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI) and treat them as missing data via median imputation.
3. **Strict Data Leakage Prevention:** Encapsulate all feature transformations (imputation and z-score standardization) inside Scikit-Learn `Pipeline` and `ColumnTransformer` constructs fitted solely on the training partition.
4. **Exploratory Data Analysis (EDA):** Perform statistical and visual analyses to identify significant risk correlations and distribution shifts between diabetic and non-diabetic cohorts.
5. **Multi-Model Comparative Benchmark:** Train and rigorously test three representative supervised classifiers:
   - Logistic Regression (Parametric linear baseline)
   - Decision Tree Classifier (Non-parametric tree-based model)
   - Random Forest Classifier (Bootstrap ensemble)
6. **Clinical-Centric Evaluation:** Assess models using Accuracy, Precision, Recall (Sensitivity), F1-Score, and Receiver Operating Characteristic - Area Under Curve (ROC-AUC), prioritizing high Recall to minimize dangerous False Negatives.
7. **Pipeline Serialization:** Persist the best-performing pipeline and diagnostic metadata using Joblib for real-time web inference.
8. **Interactive Streamlit Web Dashboard:** Construct a clean, healthcare-themed web application providing real-time risk predictions, continuous probability scales, EDA insights, and transparent model documentation.
9. **Ethical AI Framing:** Emphasize that the software is an educational screening aid and strictly not a substitute for professional clinical diagnosis.

---

## CHAPTER 4 – EXISTING SYSTEM
Existing healthcare screening tools and academic implementations typically exhibit the following characteristics:

| Dimension | Existing Student / Basic Systems |
| :--- | :--- |
| **Data Cleaning** | Often drops zero values entirely (losing ~40% of records) or leaves zeros as valid numbers, distorting model coefficients. |
| **Pipeline Architecture** | Applies `StandardScaler` to the full dataset before splitting into train/test sets, creating severe data leakage. |
| **Model Selection** | Relies purely on Accuracy; models with 75% accuracy that predict the majority class and miss 60% of diabetic cases are mistakenly considered acceptable. |
| **User Interface** | Terminal-only command-line scripts or basic command prompts requiring manual code edits to test new inputs. |
| **Ethical AI** | Often presents blunt, alarming statements such as "You have diabetes" without probability context or clinical disclaimers. |

---

## CHAPTER 5 – PROPOSED SYSTEM
The proposed system addresses the limitations of existing approaches through an end-to-end, scientifically grounded methodology:

1. **Intelligent Missing-Value Treatment:** Recognizes that while `Pregnancies = 0` is physically valid, `Glucose = 0` or `BMI = 0` represent unrecorded tests. These values are converted to `NaN` and imputed with the training set median, preserving sample size without corrupting distributions.
2. **Zero-Leakage Scikit-Learn Pipeline:** Uses `ColumnTransformer` and `Pipeline` objects. The imputer and scaler compute statistics strictly from `X_train` and apply identical transformations to `X_test` and real-time user inputs.
3. **Multi-Metric Healthcare Benchmark:** Evaluates models across five core metrics, specifically prioritizing **Recall (Sensitivity)** and **ROC-AUC** to protect against False Negatives.
4. **Ensemble Architecture:** Utilizes a Random Forest classifier consisting of 100 decorrelated decision trees, providing reduced variance, resistance to overfitting, and smooth probability calibration.
5. **Interactive Streamlit Interface:** Features an intuitive five-module web application with responsive sliders, input validation, risk gauge progress bars, personalized clinical observations, and clear disclaimers.

---

## CHAPTER 6 – SYSTEM REQUIREMENTS

### Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Dual-Core 2.0 GHz minimum)
- **RAM:** 4 GB RAM minimum (8 GB recommended)
- **Storage:** 500 MB free hard drive space
- **Display:** 1280 × 720 minimum screen resolution

### Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Programming Language:** Python 3.9 - 3.12
- **Core Libraries:**
  - `pandas` (>= 2.0.0): Tabular data manipulation
  - `numpy` (>= 1.24.0): Numerical matrix operations
  - `scikit-learn` (>= 1.3.0): Machine learning estimators and pipelines
  - `matplotlib` & `seaborn`: Visualization and charting
  - `streamlit` (>= 1.28.0): Web application framework
  - `joblib` (>= 1.3.0): Model persistence and serialization

---

## CHAPTER 7 – DATASET DESCRIPTION
The project utilizes the benchmark **PIMA Indians Diabetes Database** originating from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

- **Total Records ($N$):** 768 patient instances
- **Total Attributes:** 9 columns (8 predictive features + 1 binary target)
- **Class Label Distribution:**
  - Negative Class (`0` = Non-Diabetic): 500 patients (65.1%)
  - Positive Class (`1` = Diabetic): 268 patients (34.9%)

### Detailed Feature Dictionary:
1. **Pregnancies (Count):** Number of times pregnant. Reflects hormonal and gestational history.
2. **Glucose (mg/dL):** 2-hour plasma glucose concentration during a 75g oral glucose tolerance test. Primary indicator of glycemic control.
3. **BloodPressure (mm Hg):** Diastolic arterial blood pressure. Hypertension is strongly linked to insulin resistance.
4. **SkinThickness (mm):** Triceps skinfold thickness measuring subcutaneous adipose tissue.
5. **Insulin ($\mu\text{U/mL}$):** 2-hour serum insulin concentration.
6. **BMI ($\text{kg/m}^2$):** Body Mass Index calculated as weight in kilograms divided by square of height in meters.
7. **DiabetesPedigreeFunction (Score):** Continuous genetic score quantifying familial diabetes predisposition based on family medical history.
8. **Age (Years):** Chronological age of the patient (range: 21 to 81 years).
9. **Outcome (Binary Target):** Diagnostic class classification (0 = Non-diabetic, 1 = Diabetic).

---

## CHAPTER 8 – DATA PREPROCESSING
Data preprocessing is the cornerstone of robust machine learning. Our pipeline executes the following sequential stages:

### 1. Duplicate Record Removal
The dataset is scanned for duplicate rows. Any identical records are removed to prevent artificial weight amplification during training.

### 2. Biological Zero Replacement
In clinical datasets, missing values are frequently entered as zero by hospital recording systems. While `Pregnancies` can legitimately be zero, the following features cannot biologically be zero in a living human:
- `Glucose` (Minimum physiological survival ~40 mg/dL)
- `BloodPressure` (Diastolic 0 indicates cardiovascular collapse)
- `SkinThickness` (Subcutaneous fat cannot measure 0 mm)
- `Insulin` (Absolute zero insulin indicates acute diabetic ketoacidosis)
- `BMI` (Body mass cannot be zero)

All zero entries in these five columns are converted to `np.nan`.

### 3. Stratified Train-Test Splitting
To ensure that both training (80%) and testing (20%) sets contain an identical proportion of diabetic cases (~34.9%), stratified splitting is implemented using `train_test_split(..., stratify=y, random_state=42)`.
- Training set: 614 samples
- Testing set: 154 samples

### 4. Median Imputation
Missing values are imputed using the **median** computed strictly from `X_train`. The median is selected over the mean because clinical metrics frequently exhibit positive skewness, making the median a more robust measure of central tendency.

### 5. Standardization (Z-Score Scaling)
Features are standardized using `StandardScaler`:
$$z = \frac{x - \mu}{\sigma}$$
Where $\mu$ and $\sigma$ are the mean and standard deviation of the training data. This ensures gradient-based algorithms (like Logistic Regression) converge efficiently without feature magnitude bias.

---

## CHAPTER 9 – EXPLORATORY DATA ANALYSIS (EDA)
EDA was conducted to answer key clinical and statistical questions:

1. **Target Distribution:**
   - 65.1% Non-Diabetic vs. 34.9% Diabetic.
   - Confirms moderate class imbalance, addressed by utilizing `class_weight='balanced'` in our classifiers.
2. **Correlation Analysis:**
   - `Glucose` exhibits the highest linear correlation with `Outcome` ($r \approx 0.49$).
   - `BMI` ($r \approx 0.31$) and `Age` ($r \approx 0.24$) exhibit the next strongest positive correlations.
   - Significant physiological correlation exists between `SkinThickness` and `BMI` ($r \approx 0.54$), as well as between `Pregnancies` and `Age` ($r \approx 0.54$).
3. **Distribution Patterns:**
   - Density estimation demonstrates that diabetic individuals have noticeably shifted glucose distributions (centering around 140–160 mg/dL, compared to 105–115 mg/dL for non-diabetic controls).
   - Higher BMI (specifically $\ge 30\,\text{kg/m}^2$) correlates with higher diabetes probability.

---

## CHAPTER 10 – MACHINE LEARNING MODELS
Three distinct supervised learning algorithms representing different modeling paradigms were evaluated:

### 1. Logistic Regression
- **Paradigm:** Generalized Linear Model (GLM).
- **Mechanism:** Models the log-odds of the positive outcome as a linear combination of input features:
  $$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k$$
- **Strengths:** Highly interpretable coefficients, fast convergence, low computational cost.
- **Limitations:** Assumes linear boundary in feature space; cannot capture complex feature interactions without manual feature engineering.

### 2. Decision Tree Classifier
- **Paradigm:** Non-parametric recursive binary partitioning.
- **Mechanism:** Splits feature space into hyper-rectangles using Gini Impurity reduction.
- **Strengths:** Intuitive rule extraction (e.g., "If Glucose > 127 and BMI > 29.5 then...").
- **Limitations:** High variance; highly susceptible to overfitting training noise.

### 3. Random Forest Classifier (Final Selected)
- **Paradigm:** Ensemble Learning via Bootstrap Aggregation (Bagging).
- **Mechanism:** Builds an ensemble of $B = 100$ independent decision trees. Each tree is trained on a bootstrap sample of the training set, with random feature selection ($\sqrt{p}$ features) at each node split.
- **Strengths:** Substantially reduces variance without increasing bias; captures non-linear interactions; robust against outliers; produces well-calibrated probabilistic outputs.

---

## CHAPTER 11 – MODEL EVALUATION

### Evaluation Metrics
Given the healthcare nature of this application, models are judged on multiple criteria:
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$ (Overall correctness)
- **Precision:** $\frac{TP}{TP + FP}$ (Reliability of positive alarm)
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ (Proportion of actual diabetic patients correctly flagged)
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean)
- **ROC-AUC:** Area under Receiver Operating Characteristic curve (Discrimination power across all thresholds)

### Comparative Benchmark (Test Set, $N = 154$):
*(Representative test set results)*
- **Logistic Regression:** Accuracy: 77.2% | Recall: 74.1% | ROC-AUC: 0.835
- **Decision Tree:** Accuracy: 71.4% | Recall: 66.7% | ROC-AUC: 0.724
- **Random Forest:** Accuracy: 79.2% | Recall: 74.1% | ROC-AUC: 0.852

### Selection Rationale:
Random Forest was selected as the final production model because:
1. It achieves the highest **ROC-AUC (0.852)**, indicating superior diagnostic separation across all potential risk cutoffs.
2. It maintains high **Recall (74.1%)**, ensuring that nearly three out of four at-risk individuals are flagged for clinical consultation.
3. Ensemble voting minimizes prediction instability compared to the single decision tree.

---

## CHAPTER 12 – SYSTEM DESIGN
The system is architected into three decoupled, cohesive tiers:

1. **Data & Preprocessing Tier (`data_preprocessing.py`):**
   - Handles file ingestion, biological zero detection, and pipeline generation (`SimpleImputer` + `StandardScaler`).
2. **Model Training & Benchmark Tier (`train_model.py`):**
   - Fits pipelines on stratified splits, logs evaluation metrics, renders diagnostic charts to disk, and serializes the winning pipeline.
3. **Inference & UI Presentation Tier (`prediction.py` & `app.py`):**
   - Loads the serialized pipeline and presents an interactive Streamlit GUI with patient input forms, risk gauges, and EDA tabs.

---

## CHAPTER 13 – STREAMLIT APPLICATION
The Streamlit application (`app.py`) provides an interactive interface organized into five primary views:

1. **Home:** Project background, prevalence statistics, pipeline highlights, and system metrics.
2. **Diabetes Risk Prediction:** Dual-column clinical input form (Demographics & Diagnostic measurements) with range guidelines. Displays the predicted class, continuous risk percentage, progress scale, and personalized risk factor summaries.
3. **Dataset Insights:** Interactive views of target class distribution, correlation matrix, and kernel density plots with student explanations.
4. **Model Performance & Comparison:** Dynamic comparison table with best-score highlighting, side-by-side confusion matrices, combined ROC curves, and Gini feature importances.
5. **How the AI Model Works & About:** Flowchart of the machine learning pipeline, explanation of Random Forest ensemble voting, and academic project credits.

---

## CHAPTER 14 – RESULTS
1. **Pipeline Integrity:** The complete pipeline was executed without data leakage. Transformations fitted on the training split generalized effectively to the unseen test set.
2. **Feature Influences:** Random Forest Gini importance revealed that **Glucose** is the single most predictive feature (~32% importance), followed by **BMI** (~18%), **Age** (~14%), and **Diabetes Pedigree Function** (~12%).
3. **Inference Latency:** The serialized pipeline provides real-time inference latency under 25 milliseconds per patient record.

---

## CHAPTER 15 – LIMITATIONS
1. **Demographic Specificity:** The Pima Indian cohort comprises adult female individuals of specific genetic background; generalizability to diverse multi-ethnic populations requires further validation.
2. **Sample Size:** With 768 records, the dataset is relatively small by modern deep-learning standards.
3. **Unmeasured Clinical Confounders:** Critical biomarkers such as Glycated Hemoglobin (HbA1c), lipid panels (HDL/LDL/Triglycerides), and dietary lifestyle indices were not available in this historical dataset.
4. **Correlation vs. Causation:** The model detects statistical correlations, not direct biological causation.

---

## CHAPTER 16 – FUTURE ENHANCEMENTS
1. **Explainable AI (XAI):** Incorporate SHAP (SHapley Additive exPlanations) to produce patient-specific force plots detailing the exact contribution of each laboratory reading.
2. **Expanded Feature Set:** Integrate contemporary electronic health records (EHR) containing HbA1c, smoking history, and physical activity levels.
3. **Three-Tier Clinical Categorization:** Expand the binary target into a three-class classification: Normal ($< 100$ mg/dL), Impaired Fasting Glucose/Prediabetes ($100 - 125$ mg/dL), and Diabetic ($\ge 126$ mg/dL).
4. **Cloud Deployment:** Containerize using Docker and deploy to cloud platforms (AWS Elastic Beanstalk or Streamlit Cloud) for widespread community access.

---

## CHAPTER 17 – CONCLUSION
This minor project successfully demonstrates the design, implementation, and evaluation of an AI-Based Diabetes Prediction System. By properly addressing biological zero anomalies, eliminating data leakage via Scikit-Learn pipelines, comparing multiple classification paradigms, and deploying an interactive Streamlit web dashboard, the project achieves an ideal balance between technical rigor and student-friendly explainability. The resulting software serves as a functional, transparent educational tool for predictive healthcare analytics.

---

## CHAPTER 18 – REFERENCES
1. **World Health Organization (WHO):** *Global Report on Diabetes*, WHO Press, Geneva, 2016.
2. **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988):** "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." *Proceedings of the Annual Symposium on Computer Application in Medical Care*, pp. 261–265.
3. **Pedregosa, F., et al. (2011):** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, pp. 2825–2830.
4. **Breiman, L. (2001):** "Random Forests." *Machine Learning*, 45(1), pp. 5–32.
5. **American Diabetes Association (ADA):** "Standards of Medical Care in Diabetes—2024." *Diabetes Care*, 47(Suppl. 1), pp. S1–S345.
