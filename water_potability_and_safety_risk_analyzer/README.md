# 💧 Water Potability and Safety Risk Analyzer

## 1. Project Title
Water Potability and Safety Risk Analyzer

## 2. Problem Statement
Unsafe drinking water causes serious health issues. This project builds a system that
analyzes water quality parameters (pH, hardness, solids, chloramines, sulfate,
conductivity, organic carbon, trihalomethanes, turbidity) and:
- Predicts whether a water sample is **potable** (safe to drink) using a trained
  classification model
- Flags **individual parameters** that fall outside recommended safe ranges
- Produces a plain-English **risk score and recommendation** so a non-technical user
  (e.g. a field health worker or local authority) knows what action to take next

## 3. Dataset / Reference Source
- **Name:** Water Quality and Potability (Kaggle)
- **Link:** https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
- **File used:** `data/water_potability.csv` (3,276 rows, 9 features + target `Potability`)
- Missing values were present in `ph` (~15%), `Sulfate` (~24%), and `Trihalomethanes` (~5%)
  and were handled during preprocessing (see below).

## 4. Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Jupyter, joblib

## 5. Project Workflow
```
Raw CSV → Clean & impute missing values → EDA (distributions, correlation, outliers)
        → Train/test split → Train Logistic Regression / Decision Tree / Random Forest
        → Evaluate (accuracy, precision, recall, F1, ROC-AUC) → Select best model
        → Feature importance → Save model → Streamlit app for interactive predictions
        → Rule-based parameter flagging + risk score → Plain-English recommendation
```

## 6. AI / ML Component
- **Classification model:** three models are trained and compared — Logistic
  Regression, Decision Tree, and Random Forest. The best model (by F1-score) is
  automatically selected and saved.
- **Feature importance:** the Random Forest's `feature_importances_` are plotted to
  show which water parameters most influence the potability prediction.
- **Unsafe-parameter flagging:** a rule-based layer (`src/utils.py`) compares each
  input parameter against reference safe ranges (`src/reference_ranges.py`) and flags
  which ones are outside the safe zone — this is **not** the ML model, it is a
  transparent, explainable companion to it.
- **Why this combination is useful:** the ML model captures non-linear interactions
  between parameters that a simple rule-based system alone would miss (e.g. a sample
  can be predicted "Not Potable" even if no single parameter is extreme, due to a
  *combination* of moderately poor values). The rule-based flags then make the
  model's decision explainable to a non-technical user by pointing at *which*
  parameters are concerning.

## 7. How to Run the Project

### Setup
```bash
cd water_potability_and_safety_risk_analyzer
pip install -r requirements.txt
```

### Step 1 — Run the full pipeline (clean data, EDA, train & save model)
```bash
cd src
python main.py
```
This produces, inside `outputs/`:
- `clean_water_potability.csv` — cleaned dataset
- `class_balance.png`, `feature_distributions.png`, `correlation_heatmap.png`,
  `outlier_boxplots.png`, `feature_importance.png` — EDA plots
- `model_metrics.json` — accuracy/precision/recall/F1/ROC-AUC for all 3 models
- `best_model.joblib`, `scaler.joblib`, `best_model_name.txt` — saved model artifacts

### Step 2 — Explore interactively (optional)
```bash
jupyter notebook notebooks/exploration_or_modeling.ipynb
```

### Step 3 — Launch the Streamlit app
```bash
cd ../app
streamlit run app.py
```
Enter a water sample's parameter values and get:
- Potability prediction with confidence
- Overall safety risk score (0–100) and risk band
- Parameter-by-parameter safe/unsafe flags
- A plain-English recommendation

## 8. Demo Screenshots
See `docs/` folder (add screenshots after running the Streamlit app locally).

## 9. Results and Insights
On the held-out test set (20% split, stratified):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.61 | 0.00* | 0.00* | 0.00* | ~0.51 |
| Decision Tree | ~0.76 | 0.75 | 0.57 | 0.65 | ~0.75 |
| **Random Forest (best)** | **~0.80** | **0.82** | **0.63** | **0.71** | **~0.88** |

\* Logistic Regression struggles because potability is **not linearly separable**
from these features — this itself is an insight: water safety depends on
non-linear combinations of parameters, which is why a tree-based model performs
much better.

Key insight from feature importance: `Sulfate`, `ph`, and `Solids` tend to be
among the more influential features in the Random Forest model (exact ranking can
be viewed in `outputs/feature_importance.png` after running the pipeline).

## 10. Limitations
- Trained on a single public Kaggle dataset; may not generalize to all real-world
  water sources or regions.
- The reference "safe ranges" used for parameter flagging are simplified
  approximations for educational purposes and should not be treated as an official
  regulatory standard.
- Missing values were imputed statistically (class-wise median); this is an
  approximation, not a substitute for actual measurement.
- Model performance (F1 ≈ 0.71 for the Potable class) means there will be false
  negatives/positives — **this tool must not be the sole basis for a real
  drinking-water safety decision.**

## 11. Future Improvements
- Try gradient boosting models (XGBoost/LightGBM) and hyperparameter tuning
  (GridSearchCV) for higher recall on the Potable class.
- Add SHAP-based per-sample explainability instead of only global feature importance.
- Allow CSV batch upload in the Streamlit app to score many samples at once.
- Incorporate region-specific regulatory limits (e.g. BIS 10500 for India) as a
  selectable reference-range profile.

## 12. Team Members
Noah (Individual submission)

## Responsible Use Note
This is a student/educational project. Predictions and risk scores are for
learning and demonstration purposes only. For any real decision about drinking
water safety, always use a certified water-testing laboratory and consult local
public health authorities.

## Repository Structure
```
water_potability_and_safety_risk_analyzer/
│
├── data/
│   └── water_potability.csv
│
├── notebooks/
│   └── exploration_or_modeling.ipynb
│
├── src/
│   ├── main.py              # end-to-end pipeline
│   ├── preprocessing.py     # load, clean, split
│   ├── reference_ranges.py  # safe-range reference table
│   └── utils.py             # flagging, risk scoring, recommendations
│
├── app/
│   └── app.py                # Streamlit dashboard
│
├── outputs/                  # generated: plots, metrics, saved model
│
├── docs/
│   └── project_report.md
│
├── requirements.txt
└── README.md
```
