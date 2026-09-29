# 🖥️ MINOR PROJECT PRESENTATION SLIDES & SCRIPT
### "AI-Based Diabetes Prediction System"
**Degree:** B.Tech Artificial Intelligence & Data Science | Minor Project Presentation  
**Target Duration:** 10–12 Minutes

---

## SLIDE 1: TITLE SLIDE
- **Slide Title:** AI-Based Diabetes Prediction System
- **Subtitle:** Machine Learning Based Healthcare Risk Assessment & Clinical Analytics
- **Presented by:** [Your Name / Roll Number]
- **Department:** Department of Artificial Intelligence & Data Science
- **Academic Year:** 2026
- **Visuals on Slide:** University Logo, Medical AI graphic (stethoscope & binary neural/decision tree icon).
- **🗣️ Spoken Script (30–45s):**  
  > "Respected panel members and faculty, good morning. Today I am presenting my minor project titled **'AI-Based Diabetes Prediction System'**. Diabetes is one of the most widespread chronic metabolic diseases globally. The core objective of this project is to develop an end-to-end, reproducible machine learning system that analyzes clinical indicators to estimate diabetes risk early, while following strict data science standards such as zero-leakage preprocessing pipelines and transparent probability estimation. I look forward to walking you through our methodology, comparative benchmarks, and live demonstration."

---

## SLIDE 2: INTRODUCTION & MOTIVATION
- **Slide Title:** Introduction & Motivation
- **Bullet Points:**
  - Diabetes impacts over 537 million adults globally, with projections reaching 700M+ by 2045.
  - Early-stage hyperglycemia is often asymptomatic, delaying vital lifestyle interventions.
  - Chronic complications include diabetic retinopathy, kidney failure, cardiovascular disease, and neuropathy.
  - Machine learning offers early risk stratification using routine physiological measurements.
- **Visuals on Slide:** World map / infographic showing rising global diabetes prevalence.
- **🗣️ Spoken Script (45s):**  
  > "Diabetes Mellitus is often called a silent epidemic because millions of individuals live with elevated glucose levels without noticeable symptoms until severe vascular or organ complications emerge. Clinical studies consistently prove that early identification of at-risk individuals allows for dietary modifications and exercise regimens that can prevent or substantially delay disease progression. Our motivation was to build an automated, reliable AI screening system that can evaluate routine metabolic measurements and provide instant, interpretable risk estimates."

---

## SLIDE 3: PROBLEM STATEMENT
- **Slide Title:** Problem Statement
- **Bullet Points:**
  - Traditional blood tests (OGTT, HbA1c) are time-consuming and often deferred by patients.
  - Clinical healthcare data contains biological zero anomalies (e.g., Blood Pressure = 0, Glucose = 0).
  - Common student implementations suffer from data leakage (scaling before train/test split).
  - Many models optimize solely for Accuracy, ignoring False Negatives that are dangerous in medical screening.
- **Visuals on Slide:** Diagram illustrating the risk of False Negatives in medical diagnosis.
- **🗣️ Spoken Script (45s):**  
  > "In existing automated systems and academic implementations, we observed three major challenges. First, healthcare datasets frequently record unmeasured tests as zeroes, leading naïve algorithms to treat zero glucose as a valid biological reading. Second, many projects scale the full dataset before splitting, which introduces data leakage and creates overly optimistic results. Third, systems often report high accuracy by predicting the majority class, while dangerously missing actual diabetic cases. Our project directly addresses these three fundamental challenges."

---

## SLIDE 4: PROJECT OBJECTIVES
- **Slide Title:** Key Project Objectives
- **Bullet Points:**
  - Clean and rectify biological zero values in physiological indicators via median imputation.
  - Implement zero-leakage Scikit-Learn pipelines for consistent transformation.
  - Conduct thorough Exploratory Data Analysis (EDA) on distributions and correlations.
  - Benchmark Logistic Regression, Decision Trees, and Random Forest.
  - Prioritize Recall (Sensitivity) and ROC-AUC for clinical risk protection.
  - Deploy an interactive Streamlit web dashboard with real-time probability outputs and medical disclaimers.
- **Visuals on Slide:** Workflow icon chain: Ingest → Clean → Pipeline → Benchmark → Deploy.
- **🗣️ Spoken Script (40s):**  
  > "Our primary objectives were fourfold: first, establish a robust cleaning strategy that converts biologically impossible zeros into imputed values; second, build a zero-leakage Scikit-Learn pipeline; third, benchmark multiple classification paradigms and evaluate them using healthcare-centric metrics like Recall and ROC-AUC; and finally, deploy the winning model inside an accessible, interactive Streamlit application."

---

## SLIDE 5: EXISTING SYSTEM VS. PROPOSED SYSTEM
- **Slide Title:** Architectural Comparison: Existing vs. Proposed System
- **Comparison Table:**
  | Aspect | Existing Approaches | Our Proposed System |
  | :--- | :--- | :--- |
  | **Missing Data** | Drops rows (~50% loss) or ignores 0s | Converts physiological 0s to NaN; median imputed |
  | **Pipeline** | Global scaling (Data Leakage) | Scikit-Learn `Pipeline` fitted strictly on `X_train` |
  | **Evaluation** | Accuracy only | Accuracy, Recall, Precision, F1, and ROC-AUC |
  | **Interface** | Terminal CLI / Basic script | 5-Tab Interactive Streamlit Healthcare UI |
  | **Ethics** | Alarming diagnostic claims | Probabilistic risk assessment + medical disclaimer |
- **Visuals on Slide:** Contrast graphic with Red 'X' for existing flaws and Green checkmarks for our pipeline.
- **🗣️ Spoken Script (45s):**  
  > "As shown in this comparison, existing student systems often drop all rows with missing values, discarding nearly half the data. In our proposed architecture, we preserve full statistical power by replacing impossible zeros with median imputation. We completely eliminate data leakage through Scikit-Learn pipelines, evaluate models through five complementary metrics, and provide an interactive web interface with responsible ethical AI disclaimers."

---

## SLIDE 6: DATASET OVERVIEW
- **Slide Title:** Dataset Description: PIMA Indians Healthcare Cohort
- **Bullet Points:**
  - Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
  - 768 patient records with 8 clinical features and 1 binary outcome target.
  - Target Distribution: 500 Non-Diabetic (65.1%) vs. 268 Diabetic (34.9%).
  - Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- **Visuals on Slide:** Bar chart of class distribution (`visualizations/class_distribution.png`).
- **🗣️ Spoken Script (45s):**  
  > "We utilized the benchmark PIMA Indians Diabetes dataset from the NIDDK. It contains 768 patient records with 8 physiological features including plasma glucose, diastolic blood pressure, BMI, serum insulin, and genetic pedigree. The target variable is binary: 0 indicates non-diabetic and 1 indicates diabetic. Notice that the dataset has moderate class imbalance, with approximately 65% non-diabetic and 35% diabetic cases, which we account for during training."

---

## SLIDE 7: DATA PREPROCESSING & PIPELINE DESIGN
- **Slide Title:** Clinical Preprocessing & Zero-Leakage Pipeline
- **Bullet Points:**
  - **Biological Zero Handling:** Converted 0s in Glucose, BP, Skin Thickness, Insulin, and BMI to `NaN`. (Pregnancies retained 0 as a valid count).
  - **Stratified Split:** 80% Train ($N=614$) / 20% Test ($N=154$) using `stratify=y`.
  - **Median Imputation:** Robust to clinical skewness and outliers.
  - **StandardScaler:** Normalizes features to zero mean and unit variance.
  - **Scikit-Learn Pipeline:** Encapsulates imputer, scaler, and estimator into a single deployable object.
- **Visuals on Slide:** Pipeline flowchart diagram (Input → Imputer → Scaler → Model).
- **🗣️ Spoken Script (60s):**  
  > "Our data preprocessing is one of the strongest technical aspects of this project. A living patient cannot have zero blood glucose or zero blood pressure. We identified that unrecorded values had been entered as zeroes, so we converted those specific features to NaN. Crucially, we split the data into 80% training and 20% testing sets using stratified sampling before calculating any statistics. We then created a Scikit-Learn ColumnTransformer that computes medians and standard deviations strictly from the training partition. This completely eliminates data leakage."

---

## SLIDE 8: EXPLORATORY DATA ANALYSIS (EDA)
- **Slide Title:** Exploratory Data Analysis & Feature Relationships
- **Bullet Points:**
  - **Correlation Heatmap:** Glucose has the highest positive correlation with outcome ($r \approx 0.49$).
  - **Co-linearity:** Skin Thickness and BMI ($r \approx 0.54$); Pregnancies and Age ($r \approx 0.54$).
  - **KDE Distributions:** Diabetic cohort shows distinct rightward shift in Glucose and BMI.
  - **Clinical Insight:** Higher BMI ($\ge 30\,\text{kg/m}^2$) significantly increases positive outcome probability.
- **Visuals on Slide:** Correlation heatmap and feature distribution plots (`visualizations/correlation_heatmap.png`, `visualizations/key_features_distribution.png`).
- **🗣️ Spoken Script (50s):**  
  > "Our exploratory data analysis reveals significant clinical patterns. The correlation heatmap demonstrates that plasma glucose concentration exhibits the strongest linear correlation with diabetes risk at approximately 0.49, followed closely by BMI and age. In the kernel density plots, you can clearly see that diabetic patients have a much higher distribution center for glucose and BMI. These findings align directly with established clinical literature."

---

## SLIDE 9: MACHINE LEARNING MODELS BENCHMARKED
- **Slide Title:** Supervised Classification Algorithms
- **Bullet Points:**
  - **Logistic Regression:** Linear probabilistic classifier using the sigmoid activation function; serves as the baseline.
  - **Decision Tree Classifier:** Non-parametric partitioning via Gini Impurity reduction; easy to visualize but high variance.
  - **Random Forest Classifier (Winning Model):** Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and random feature sub-selection.
- **Visuals on Slide:** Schematic comparing a single decision tree vs. Random Forest ensemble voting.
- **🗣️ Spoken Script (50s):**  
  > "We benchmarked three distinct supervised learning algorithms. Logistic Regression provides a transparent linear baseline. A single Decision Tree provides rule-based partitioning, but suffers from high variance and overfitting. Random Forest solves this by combining 100 decorrelated decision trees trained on bootstrap samples. Each tree votes, and averaging these votes reduces variance significantly, producing smooth, reliable probability outputs."

---

## SLIDE 10: MODEL EVALUATION & BENCHMARK RESULTS
- **Slide Title:** Comparative Evaluation on Hold-Out Test Set ($N = 154$)
- **Performance Table:**
  | Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Logistic Regression** | 77.2% | 0.65 | 74.1% | 0.69 | 0.835 |
  | **Decision Tree** | 71.4% | 0.58 | 66.7% | 0.62 | 0.724 |
  | **Random Forest (Final)** | **79.2%** | **0.69** | **74.1%** | **0.71** | **0.852** |
- **Visuals on Slide:** Confusion matrices & ROC Curves (`visualizations/confusion_matrices.png`, `visualizations/roc_curves.png`).
- **🗣️ Spoken Script (60s):**  
  > "Here are our test set evaluation results on the unseen test partition. Notice that Random Forest achieved the highest Accuracy of 79.2% and the highest ROC-AUC of 0.852. More importantly, in healthcare screening, Recall is crucial because a False Negative means an undiagnosed diabetic patient is missed. Random Forest achieved 74.1% recall while maintaining 69% precision. This balanced performance across both Recall and ROC-AUC is why we selected Random Forest as our final production model."

---

## SLIDE 11: FEATURE IMPORTANCE ANALYSIS
- **Slide Title:** Random Forest Feature Importance (Gini Impurity)
- **Bullet Points:**
  - **Glucose:** ~32% relative importance (strongest predictor).
  - **BMI:** ~18% relative importance.
  - **Age:** ~14% relative importance.
  - **Pedigree Function:** ~12% relative importance.
  - **Key Note:** Feature importance indicates statistical association within this dataset, not medical causation.
- **Visuals on Slide:** Horizontal bar chart of feature importance (`visualizations/feature_importance.png`).
- **🗣️ Spoken Script (45s):**  
  > "This chart shows the feature importances extracted from our Random Forest model based on mean decrease in Gini impurity. As expected clinically, Glucose and BMI are the two most influential predictors, followed by Age and the Diabetes Pedigree Function. We emphasize in our project that feature importance demonstrates statistical predictive utility within this dataset, but does not claim medical causation."

---

## SLIDE 12: STREAMLIT WEB APPLICATION
- **Slide Title:** Interactive Healthcare Web Dashboard
- **Bullet Points:**
  - Built with Streamlit for clean, responsive, real-time clinical assessment.
  - Dual-column input layout with reference ranges and input validation.
  - Continuous risk gauge (0% to 100%) and application-defined risk categories.
  - Personalized educational observations highlighting out-of-range metrics.
  - Dedicated tabs for Dataset Insights, Model Comparison, and Pipeline Architecture.
- **Visuals on Slide:** Screenshot of the Streamlit prediction interface.
- **🗣️ Spoken Script (45s):**  
  > "To make our model practical and accessible, we built a Streamlit web application. Users can enter patient demographics and clinical measurements through intuitive sliders and number fields. When 'Predict Diabetes Risk' is clicked, the app runs the inputs through our saved pipeline and displays the continuous probability percentage on a color-coded risk scale, alongside educational health observations and prominent disclaimers."

---

## SLIDE 13: LIVE DEMONSTRATION CASES
- **Slide Title:** Live Demonstration & Test Cases
- **Case Comparison Table:**
  | Parameter | Profile 1: Low Risk | Profile 2: High Risk |
  | :--- | :--- | :--- |
  | **Age / Pregnancies** | 22 yrs / 1 | 54 yrs / 6 |
  | **Glucose** | 88 mg/dL (Normal) | 175 mg/dL (Elevated) |
  | **Blood Pressure** | 68 mm Hg | 88 mm Hg |
  | **BMI** | 22.4 (Normal) | 36.5 (Obese) |
  | **Pedigree / Insulin** | 0.25 / 75 | 0.85 / 210 |
  | **Model Prediction** | **Non-Diabetic (14.2% Risk)** | **Diabetic (86.4% Risk)** |
- **Visuals on Slide:** Side-by-side screenshots of the two prediction outputs in the web application.
- **🗣️ Spoken Script (45s):**  
  > "During our live testing, we validated two contrasting profiles. Profile 1 represents a young patient with normal glucose of 88 and a healthy BMI of 22.4; the model predicts low risk at 14.2%. Profile 2 represents a 54-year-old patient with elevated glucose of 175 and a BMI of 36.5; the model correctly identifies higher risk at 86.4% and flags the specific elevated measurements."

---

## SLIDE 14: LIMITATIONS & FUTURE ENHANCEMENTS
- **Slide Title:** Limitations & Future Roadmap
- **Limitations:**
  - Demographic specificity (historical PIMA cohort).
  - Sample size ($N=768$).
  - Absence of modern diagnostic markers like HbA1c and lipid profiles.
- **Future Enhancements:**
  - **Explainable AI:** Incorporating SHAP waterfall plots for patient-specific feature explanations.
  - **Expanded EHR Datasets:** Retraining on multi-center contemporary hospital data.
  - **Cloud Deployment:** Packaging via Docker container on AWS / Streamlit Cloud.
- **Visuals on Slide:** Example SHAP explanation graph illustration.
- **🗣️ Spoken Script (40s):**  
  > "In terms of limitations, our model is trained on the PIMA Indian cohort, so demographic generalizability requires validation across diverse populations. For future work, we plan to implement Explainable AI using SHAP waterfall plots to give clinicians exact per-patient explanations, incorporate modern HbA1c biomarkers, and deploy the application to cloud infrastructure."

---

## SLIDE 15: CONCLUSION & Q&A
- **Slide Title:** Conclusion & Summary
- **Summary Points:**
  - Successfully designed, trained, and deployed an end-to-end diabetes prediction pipeline.
  - Addressed biological zeroes with median imputation and eliminated data leakage.
  - Selected Random Forest for superior balance of Recall (74.1%) and ROC-AUC (0.852).
  - Deployed an interactive Streamlit application with ethical AI disclaimers.
- **Visuals on Slide:** "Thank You! Questions & Discussion Welcome" with GitHub / repository link.
- **🗣️ Spoken Script (30s):**  
  > "In conclusion, this minor project demonstrates a complete, technically sound machine learning workflow from raw data preprocessing to web deployment. By prioritizing clinical Recall, eliminating data leakage, and upholding ethical AI guidelines, we built a practical educational screening system. Thank you for your time and attention. I am now happy to answer any questions."
