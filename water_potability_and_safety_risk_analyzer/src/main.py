"""
main.py

End-to-end pipeline for the Water Potability and Safety Risk Analyzer.

Steps:
1. Load raw data
2. Clean / impute missing values
3. Run EDA and save plots
4. Train baseline + tuned classification models
5. Evaluate models (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
6. Extract feature importance
7. Save the best model + scaler for reuse in the Streamlit app
8. Run a couple of sample scenario checks through the risk analyzer

Run with:
    python main.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from preprocessing import load_raw_data, missing_value_report, clean_data, train_test_split_data
from reference_ranges import FEATURE_COLUMNS, TARGET_COLUMN
from utils import flag_parameters, risk_score_from_flags, build_recommendation

DATA_PATH = "../data/water_potability.csv"
OUTPUT_DIR = "../outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

sns.set_theme(style="whitegrid")


def run_eda(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Target class balance
    plt.figure(figsize=(5, 4))
    df[TARGET_COLUMN].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Potability Class Balance (0 = Not Potable, 1 = Potable)")
    plt.xlabel("Potability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_balance.png"), dpi=150)
    plt.close()

    # 2. Distribution of each feature split by Potability
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.kdeplot(data=df, x=col, hue=TARGET_COLUMN, fill=True, ax=ax, common_norm=False)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # 4. Boxplots to spot outliers
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.boxplot(data=df, y=col, ax=ax, color="#55A868")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outlier_boxplots.png"), dpi=150)
    plt.close()

    print(f"EDA plots saved to {output_dir}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted_models[name] = model
        print(f"\n{name}")
        print(classification_report(y_test, preds, target_names=["Not Potable", "Potable"]))

    return results, fitted_models, scaler


def plot_feature_importance(model, feature_names, output_dir):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {output_dir}")


def run_sample_scenarios(best_model, scaler, model_name, df):
    """Demonstrate the risk analyzer on a couple of real rows from the dataset."""
    print("\n--- Sample Scenario Checks ---")
    sample_rows = df.sample(3, random_state=7)

    for i, row in sample_rows.iterrows():
        sample = row[FEATURE_COLUMNS].to_dict()
        X_sample = pd.DataFrame([sample])

        if model_name == "Logistic Regression":
            X_input = scaler.transform(X_sample)
        else:
            X_input = X_sample

        pred = int(best_model.predict(X_input)[0])
        flags = flag_parameters(sample)
        risk_info = risk_score_from_flags(flags)
        recommendation = build_recommendation(pred, risk_info, flags)

        print(f"\nSample row index {i} (actual Potability={row[TARGET_COLUMN]}):")
        print(f"  Model prediction: {'Potable' if pred == 1 else 'Not Potable'}")
        print(f"  Risk score: {risk_info['risk_score']} ({risk_info['risk_band']})")
        print(f"  Recommendation: {recommendation}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: Loading raw data...")
    raw_df = load_raw_data(DATA_PATH)
    print(f"  Raw shape: {raw_df.shape}")

    print("\nStep 2: Missing value report (before cleaning)...")
    print(missing_value_report(raw_df))

    print("\nStep 3: Cleaning data (class-wise median imputation)...")
    clean_df = clean_data(raw_df, strategy="median_by_class")
    print(f"  Clean shape: {clean_df.shape}, remaining NaNs: {clean_df.isnull().sum().sum()}")
    clean_df.to_csv(os.path.join(OUTPUT_DIR, "clean_water_potability.csv"), index=False)

    print("\nStep 4: Running EDA...")
    run_eda(clean_df, OUTPUT_DIR)

    print("\nStep 5: Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split_data(clean_df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nStep 6: Training models...")
    results, fitted_models, scaler = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Pick best model by F1 score (potability classes are imbalanced)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} -> {results[best_name]}")

    plot_feature_importance(best_model, FEATURE_COLUMNS, OUTPUT_DIR)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    print(f"\nBest model saved to {MODEL_PATH}")

    run_sample_scenarios(best_model, scaler, best_name, clean_df)


if __name__ == "__main__":
    main()
