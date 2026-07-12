# Project Report: Water Potability and Safety Risk Analyzer

## 1. Introduction
Access to safe drinking water is a fundamental public health need. Manually testing
and interpreting every water quality parameter is time-consuming for field workers
and local authorities. This project builds a decision-support tool that combines a
machine learning classifier with a transparent, rule-based safety-flagging layer to
help a non-expert quickly understand whether a water sample is safe and *why*.

## 2. Users and Stakeholders
- **Field health workers / NGOs** testing water sources in communities
- **Municipal / local water authorities** doing routine quality checks
- **Households** with private borewell/well water wanting a quick sanity check
  (with the clear caveat that lab confirmation is still required)

## 3. Data
- Source: Kaggle "Water Quality and Potability" dataset (public)
- 3,276 samples, 9 numeric features (ph, Hardness, Solids, Chloramines, Sulfate,
  Conductivity, Organic_carbon, Trihalomethanes, Turbidity) and a binary target
  `Potability` (0 = not potable, 1 = potable)
- Missing values: `Sulfate` (23.8%), `ph` (15.0%), `Trihalomethanes` (4.95%)
- Imputation strategy: median value computed **within each Potability class**, so the
  imputed value reflects typical values for water of that safety class rather than a
  single global average.

## 4. Exploratory Data Analysis — Key Observations
- The target classes are imbalanced: roughly 61% "Not Potable" vs 39% "Potable".
- Individual feature distributions for potable vs non-potable water overlap heavily —
  no single parameter cleanly separates the two classes, which explains why a purely
  linear model (Logistic Regression) performs close to random guessing on this
  dataset.
- Correlations between the 9 features are generally weak, suggesting potability is
  driven by a *combination* of factors rather than one dominant parameter.
- Boxplots show a modest number of statistical outliers in `Solids`, `Sulfate`, and
  `Trihalomethanes`, consistent with natural variation across different water sources.

(See `outputs/feature_distributions.png`, `outputs/correlation_heatmap.png`,
`outputs/outlier_boxplots.png` after running `src/main.py`.)

## 5. Modeling Approach
Three classifiers were trained and compared on an 80/20 stratified train/test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.610 | 0.000 | 0.000 | 0.000 | 0.554 |
| Decision Tree (max_depth=8) | 0.758 | 0.746 | 0.574 | 0.649 | 0.819 |
| **Random Forest (300 trees, max_depth=12, class_weight="balanced")** | **0.799** | **0.816** | **0.625** | **0.708** | **0.884** |

The Random Forest was selected as the final model based on the highest F1-score for
the minority ("Potable") class, since correctly identifying potable water (and not
missing unsafe water) matters more than raw accuracy on an imbalanced dataset.

**Why Logistic Regression underperforms:** it can only draw a linear decision
boundary in feature space. Because potability depends on non-linear interactions
between parameters, it defaults to always predicting the majority class here. This
is itself a useful finding: it confirms that tree-based / non-linear models are
necessary for this problem.

## 6. Feature Importance
The Random Forest's built-in feature importance highlights which parameters
contribute most to the potability decision (see `outputs/feature_importance.png`).
This gives water authorities a sense of which parameters to prioritize when they
have limited testing budget/time.

## 7. Safety Risk Analyzer (Explainability Layer)
Beyond the ML prediction, every sample is also checked against reference "safe
ranges" for each of the 9 parameters (`src/reference_ranges.py`, loosely based on
WHO/EPA-style drinking water guidance). This produces:
- A list of which specific parameters are outside the safe range
- A **risk score** (0–100, percentage of parameters that are unsafe) and a risk band
  (Low / Moderate / High / Very High Risk)
- A **plain-English recommendation** combining both the ML prediction and the
  rule-based flags (e.g. flagging cases where the model says "Potable" but individual
  parameters are still borderline, prompting the user to double check)

## 8. Validation
- Standard train/test split with stratification on the target to preserve class
  balance
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix for
  all three models (`outputs/model_metrics.json`)
- Scenario checks: several real rows from the dataset were run through the full
  pipeline (model prediction → parameter flags → risk score → recommendation) to
  confirm the explanation logic behaves sensibly (see the notebook, Section 9)

## 9. Limitations and Responsible Use
- The dataset, while widely used for this task, is limited in size and source
  diversity; the trained model's accuracy (~80%) means it will sometimes be wrong.
- Reference safe ranges are simplified for this project and are not a substitute for
  official regulatory limits (e.g. BIS 10500 in India, WHO guidelines).
- This tool is intended for **educational and decision-support purposes only**. It
  must not be used as the sole basis for real drinking water safety decisions —
  always confirm with a certified water testing laboratory.

## 10. Future Improvements
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) and gradient boosting
  models for improved recall on the Potable class
- Per-sample explainability (e.g. SHAP values) instead of only global feature
  importance
- Batch CSV scoring in the Streamlit app
- Region-specific safe-range profiles (e.g. Indian BIS standards)
