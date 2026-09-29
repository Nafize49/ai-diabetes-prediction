"""
train_model.py
--------------
Model Training, Comparative Evaluation, and Pipeline Serialization.

Author: Nafize Ali (B.Tech AI & Data Science)
Project: AI-Based Diabetes Prediction System

Models Compared:
1. Logistic Regression (Linear parametric baseline)
2. Decision Tree Classifier (Single-tree interpretable baseline)
3. Random Forest Classifier (Ensemble bagging, winning final model)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.pipeline import Pipeline

try:
    from src.preprocessing import (
        load_dataset,
        inspect_dataset,
        clean_dataset,
        prepare_features_and_target,
        build_preprocessor,
        get_train_test_split,
        CORE_FEATURES
    )
except ImportError:
    try:
        from preprocessing import (
            load_dataset,
            inspect_dataset,
            clean_dataset,
            prepare_features_and_target,
            build_preprocessor,
            get_train_test_split,
            CORE_FEATURES
        )
    except ImportError:
        from data_preprocessing import (
            load_dataset,
            inspect_dataset,
            clean_dataset,
            prepare_features_and_target,
            build_preprocessor,
            get_train_test_split,
            CORE_FEATURES
        )

# Academic styling: clean white backgrounds, muted slate and blue tones, no neon
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#1e293b",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "font.family": "sans-serif",
    "font.size": 10
})


def generate_eda_charts(df_clean: pd.DataFrame, viz_dir: str):
    """Generates clean, academic exploratory visualizations."""
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    counts = df_clean["Outcome"].value_counts()
    bars = ax.bar(["Non-Diabetic (0)", "Diabetic (1)"], [counts[0], counts[1]],
                  color=["#3b82f6", "#ef4444"], width=0.5, edgecolor="#94a3b8")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 3,
                f"{int(h)} ({h/len(df_clean)*100:.1f}%)", ha="center", va="bottom",
                fontsize=9, color="#1e293b", fontweight="bold")
    ax.set_title("Diabetes Class Distribution in Dataset", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Patient Count", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # 2. Correlation Heatmap (Core Features + Outcome)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr_cols = CORE_FEATURES + ["Outcome"]
    corr = df_clean[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax,
                linewidths=0.5, linecolor="#e2e8f0", annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()

    # 3. Density Distributions for Glucose, BMI, Age
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    plot_cols = ["Glucose", "BMI", "Age"]
    for ax, col in zip(axes, plot_cols):
        for label, grp in df_clean.groupby("Outcome"):
            name = "Diabetic" if label == 1 else "Non-Diabetic"
            c = "#ef4444" if label == 1 else "#3b82f6"
            sns.kdeplot(grp[col].dropna(), ax=ax, label=name, color=c, fill=True, alpha=0.25)
        ax.set_title(f"{col} Distribution", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "key_features_distribution.png"), dpi=200)
    plt.close()


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, viz_dir: str):
    """Trains and compares Logistic Regression, Decision Tree, and Random Forest."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    }

    results = []
    fitted_pipes = {}
    roc_info = {}
    cm_info = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_info[name] = (fpr, tpr, auc)
        cm_info[name] = confusion_matrix(y_test, y_pred)

    df_results = pd.DataFrame(results)

    # Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (name, cm) in zip(axes, cm_info.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrices.png"), dpi=200)
    plt.close()

    # ROC Curves Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = ["#2563eb", "#10b981", "#d97706"]
    for (name, (fpr, tpr, auc)), col in zip(roc_info.items(), palette):
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=col, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_title("ROC Curves Comparison (Hold-Out Test Set)", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "roc_curves.png"), dpi=200)
    plt.close()

    return df_results, fitted_pipes


def save_feature_importance(rf_pipe, feature_names: list, viz_dir: str):
    """Saves Random Forest Gini feature importances horizontal bar chart."""
    rf = rf_pipe.named_steps["classifier"]
    importances = rf.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="#0284c7", edgecolor="#cbd5e1", height=0.55)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.set_title("Random Forest Feature Importance", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=9)
    ax.set_xlim(0, max(df_imp["Importance"]) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_importance.png"), dpi=200)
    plt.close()

    return df_imp


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("="*60)
    print("AI-BASED DIABETES PREDICTION SYSTEM - MODEL TRAINING")
    print("Author: Nafize Ali | B.Tech AI & Data Science")
    print("="*60)

    # 1. Load and inspect
    df_raw = load_dataset()
    stats = inspect_dataset(df_raw)
    print(f"\n[Dataset] Loaded: {stats['num_rows']} rows, {stats['num_cols']} columns")
    print(f"[Dataset] Features in CSV: {stats['columns']}")
    print(f"[Dataset] Target distribution: Non-Diabetic={stats['negative_cases']}, Diabetic={stats['positive_cases']}")
    print(f"[Dataset] Core mapped clinical features: {CORE_FEATURES}")

    # 2. Clean data & generate EDA
    df_clean = clean_dataset(df_raw)
    generate_eda_charts(df_clean, viz_dir)

    # 3. Train-test split (80/20, stratified)
    X, y = prepare_features_and_target(df_clean, feature_subset=CORE_FEATURES)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n[Split] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Build pipeline & train
    preprocessor = build_preprocessor(CORE_FEATURES)
    df_results, fitted_pipes = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, viz_dir)

    print("\n--- MODEL PERFORMANCE COMPARISON (UNSEEN TEST SET) ---")
    print(df_results.to_string(index=False))

    # 5. Feature importance
    df_imp = save_feature_importance(fitted_pipes["Random Forest"], CORE_FEATURES, viz_dir)

    # 6. Save winning model
    winning_model_name = "Random Forest"
    best_pipe = fitted_pipes[winning_model_name]
    best_metrics = df_results[df_results["Model"] == winning_model_name].iloc[0].to_dict()

    save_payload = {
        "model": best_pipe,
        "pipeline": best_pipe,
        "model_name": winning_model_name,
        "features": CORE_FEATURES,
        "feature_names": CORE_FEATURES,
        "all_metrics": df_results,
        "best_metrics": best_metrics,
        "feature_importances": df_imp.to_dict(orient="records"),
        "total_records": stats["num_rows"],
        "class_distribution": stats["class_distribution"],
        "negative_cases": stats["negative_cases"],
        "positive_cases": stats["positive_cases"],
        "biological_zeros": stats["biological_zeros"],
        "author": "Nafize Ali",
        "department": "B.Tech AI & Data Science"
    }

    model_path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump(save_payload, model_path)
    print(f"\n[Saved] Final trained pipeline saved to: {model_path}")
    print("[Done] Pipeline ready for Streamlit deployment.\n")


if __name__ == "__main__":
    main()
