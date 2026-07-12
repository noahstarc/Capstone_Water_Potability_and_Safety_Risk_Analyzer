"""
app.py

Streamlit dashboard for the Water Potability and Safety Risk Analyzer.

Run with:
    streamlit run app.py
"""

import os
import sys
import json

import joblib
import pandas as pd
import streamlit as st

# allow importing from ../src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from reference_ranges import FEATURE_COLUMNS, SAFE_RANGES
from utils import flag_parameters, risk_score_from_flags, build_recommendation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
MODEL_NAME_PATH = os.path.join(OUTPUT_DIR, "best_model_name.txt")
METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

st.set_page_config(page_title="Water Potability & Safety Risk Analyzer", page_icon="💧", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(MODEL_NAME_PATH) as f:
        model_name = f.read().strip()
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, scaler, model_name, metrics


def predict_sample(model, scaler, model_name, sample: dict):
    X_sample = pd.DataFrame([sample])[FEATURE_COLUMNS]
    if model_name == "Logistic Regression":
        X_input = scaler.transform(X_sample)
    else:
        X_input = X_sample
    pred = int(model.predict(X_input)[0])
    prob = float(model.predict_proba(X_input)[0][1])
    return pred, prob


def main():
    st.title("💧 Water Potability and Safety Risk Analyzer")
    st.caption(
        "Enter water quality test results to get a potability prediction, "
        "parameter-level safety flags, and a plain-English recommendation."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found. Please run `python src/main.py` first "
            "to train and save the model."
        )
        st.stop()

    model, scaler, model_name, metrics = load_artifacts()

    with st.sidebar:
        st.header("Model Info")
        st.write(f"**Active model:** {model_name}")
        if model_name in metrics:
            m = metrics[model_name]
            st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            st.metric("F1 Score", f"{m['f1_score']:.2f}")
            st.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        st.markdown("---")
        st.markdown(
            "**Limitations & Responsible Use**\n\n"
            "- Trained on a public Kaggle dataset; may not generalize to "
            "all real-world water sources.\n"
            "- Reference safe-ranges are simplified for educational use.\n"
            "- This tool does NOT replace certified lab testing. "
            "Always confirm with a local water-testing authority before "
            "making health decisions."
        )

    tab1, tab2 = st.tabs(["🔬 Analyze a Water Sample", "📊 About the Parameters"])

    with tab1:
        st.subheader("Enter Water Quality Parameters")

        col1, col2, col3 = st.columns(3)
        defaults = {
            "ph": 7.0, "Hardness": 196.0, "Solids": 20000.0, "Chloramines": 7.0,
            "Sulfate": 333.0, "Conductivity": 425.0, "Organic_carbon": 14.0,
            "Trihalomethanes": 66.0, "Turbidity": 4.0,
        }

        inputs = {}
        columns = [col1, col2, col3]
        for i, col_name in enumerate(FEATURE_COLUMNS):
            target_col = columns[i % 3]
            low, high, unit, desc = SAFE_RANGES[col_name]
            with target_col:
                inputs[col_name] = st.number_input(
                    f"{col_name} ({unit})" if unit else col_name,
                    value=float(defaults[col_name]),
                    help=f"{desc}. Reference safe range: {low}–{high} {unit}".strip(),
                )

        if st.button("Analyze Sample", type="primary"):
            pred, prob = predict_sample(model, scaler, model_name, inputs)
            flags = flag_parameters(inputs)
            risk_info = risk_score_from_flags(flags)
            recommendation = build_recommendation(pred, risk_info, flags)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.subheader("Model Prediction")
                if pred == 1:
                    st.success(f"✅ Potable (confidence: {prob*100:.1f}%)")
                else:
                    st.error(f"🚫 Not Potable (confidence: {(1-prob)*100:.1f}%)")

                st.subheader("Overall Risk Score")
                band_colors = {
                    "Low Risk": "green", "Moderate Risk": "orange",
                    "High Risk": "red", "Very High Risk": "red",
                }
                st.markdown(
                    f"**{risk_info['risk_score']}/100** — "
                    f":{band_colors[risk_info['risk_band']]}[{risk_info['risk_band']}]  \n"
                    f"({risk_info['unsafe_parameter_count']} of "
                    f"{risk_info['total_parameters_checked']} parameters outside safe range)"
                )

            with res_col2:
                st.subheader("Recommendation")
                st.info(recommendation)

            st.subheader("Parameter-Level Safety Flags")
            flags_df = pd.DataFrame(flags)
            flags_df["safe_range"] = flags_df["safe_range"].apply(lambda r: f"{r[0]}–{r[1]}")
            flags_df = flags_df[["parameter", "value", "unit", "safe_range", "status", "note"]]

            def highlight_status(row):
                color = "background-color: #ffe1e1" if row["status"] == "Unsafe" else "background-color: #e1ffe1"
                return [color] * len(row)

            st.dataframe(flags_df.style.apply(highlight_status, axis=1), use_container_width=True)

    with tab2:
        st.subheader("Reference Safe Ranges")
        ref_rows = [
            {"Parameter": k, "Safe Min": v[0], "Safe Max": v[1], "Unit": v[2], "Notes": v[3]}
            for k, v in SAFE_RANGES.items()
        ]
        st.table(pd.DataFrame(ref_rows))
        st.caption(
            "These reference ranges are simplified for this educational project "
            "and loosely follow WHO / EPA style drinking-water guidance."
        )


if __name__ == "__main__":
    main()
