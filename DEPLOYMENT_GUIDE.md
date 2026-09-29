# 🌐 Streamlit Community Cloud Deployment Guide
**Project:** AI-Based Diabetes Prediction System  
**Student:** Nafize Ali (B.Tech AI & Data Science)

---

## 📌 Deployment Readiness Status
- **Status:** **Deployment-Ready**
- **Public URL Note:** Deployment-ready; public URL must be generated after connecting your GitHub repository to Streamlit Community Cloud. (Localhost `http://localhost:8501` is strictly for local machine execution).

---

## 🚀 Step-by-Step Guide to Deploy Online for Free

### Step 1: Create a GitHub Repository
1. Log into your GitHub account at [github.com](https://github.com).
2. Click the **`+`** icon at the top right and select **New repository**.
3. Name your repository: `AI-Based-Diabetes-Prediction-System`.
4. Choose **Public** visibility (required for free Streamlit Community Cloud hosting).
5. Leave "Initialize with README" unchecked (since we already have a complete project).
6. Click **Create repository**.

---

### Step 2: Push Your Project Files to GitHub
Open your terminal / PowerShell in the project directory (`AI-Based-Diabetes-Prediction-System`) and run:

```bash
# 1. Initialize git
git init

# 2. Add all project files
git add .

# 3. Commit the project
git commit -m "Initial commit: Complete AI-Based Diabetes Prediction System"

# 4. Set the main branch
git branch -M main

# 5. Add your remote GitHub repository URL (replace with your GitHub username)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/AI-Based-Diabetes-Prediction-System.git

# 6. Push code to GitHub
git push -u origin main
```

---

### Step 3: Deploy on Streamlit Community Cloud
1. Open [share.streamlit.io](https://share.streamlit.io/) in your browser.
2. Sign in with your **GitHub account**.
3. Click the **"New app"** button in the top right corner.
4. Fill in the deployment form:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/AI-Based-Diabetes-Prediction-System`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Choose a custom subdomain like `diabetes-ai-nafize.streamlit.app`
5. Click **"Deploy!"**.

---

### Step 4: Obtain and Share Your Public URL
- Streamlit Cloud will automatically install all dependencies listed in `requirements.txt` and launch the app in 1–2 minutes.
- Once deployed, your app will be live at a public URL like:
  ```
  https://diabetes-ai-nafize.streamlit.app
  ```
- You can send this public URL to your project mentors, evaluators, and teammates!

---

## 🛠️ Verification Checklist for Cloud Deployment
- [x] `requirements.txt` includes all runtime dependencies (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `reportlab`).
- [x] `.streamlit/config.toml` enforces light theme and neutral UI aesthetics.
- [x] `model/diabetes_model.pkl` is committed and loaded automatically via `@st.cache_resource`.
- [x] Relative file paths used in `app.py` ensure compatibility across Linux / cloud containers.
