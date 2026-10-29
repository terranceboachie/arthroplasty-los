# Explainable Length-of-Stay Prediction after Hip and Knee Arthroplasty

Leakage-controlled, interpretable machine learning pipeline that predicts **prolonged postoperative stay (> 7 days)** after elective hip and knee arthroplasty, using only information available before or during surgery. Built on the public [INSPIRE](https://physionet.org/) perioperative dataset as part of my MSc Computer Science dissertation (Birmingham City University).

> Research prototype. Not validated for clinical use.

## Headline results

Stratified 70/30 split, held-out test set (n = 767, prevalence of prolonged stay = 77%).

| Model | AUROC | AUPRC (baseline = 0.77) |
|---|---|---|
| Regularised logistic regression | 0.88 | 0.95 |
| XGBoost | 0.87 | 0.95 |

XGBoost confusion matrix at the default 0.5 threshold:

| | Predicted: no long stay | Predicted: long stay |
|---|---|---|
| **Actual: no long stay** | 92 | 87 |
| **Actual: long stay** | 30 | 558 |

Recall 94.9% · precision 86.5% · specificity 51.4% · accuracy 84.7%.

**Main finding:** a transparent logistic regression baseline matched XGBoost, so extra model complexity added nothing here.

## What this project is about

Many published length-of-stay models look strong because they include postoperative or discharge-dependent variables. This pipeline is built to avoid that:

- **Allowlisted predictors only:** age, sex, BMI, ASA physical status, anaesthetic type, department, ICD-10-PCS procedure code and operation duration.
- **Excluded on purpose:** `discharge_time`, `opstart_time` and `opend_time` (LOS is derived from discharge time), the LOS column itself, and all identifiers. An assertion in the notebook fails if any of them re-enters the feature set.
- **Cohort restricted to elective hip and knee replacements** (elective cases, orthopaedic departments, replacement procedure codes).

### A leakage story worth telling

My first version reached **0.99 AUROC**. Two problems were responsible:

1. **Target leakage:** the three timestamp columns above were top SHAP features. Removing them dropped AUROC to 0.95.
2. **Mixed-procedure cohort:** minor and major procedures were included, so the procedure code alone separated short from long stays. Restricting to hip and knee replacements brought AUROC to a realistic 0.87-0.88.

## Pipeline

1. **Load and filter:** elective cases, orthopaedic departments (OS, OT), hip and knee replacement codes.
2. **Derive features:** BMI from height and weight; operation duration from operation start and end times.
3. **Define outcome:** `long_stay = 1` if postoperative LOS > 7 days (`LONG_STAY_THRESHOLD` in the notebook).
4. **Preprocess:** median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
5. **Train:** regularised logistic regression (baseline) and XGBoost, with a stratified 70/30 split (`random_state=42`).
6. **Evaluate:** AUROC, AUPRC, confusion matrix, ROC and precision-recall curves, plus 5-fold stratified cross-validation on the training set.
7. **Explain:** global and local SHAP (TreeExplainer).
8. **Screen subgroups:** AUROC and F1 by sex (the dataset contains a single race group, so no race comparison is possible).
9. **Deploy a demo:** Gradio app showing predicted risk and a SHAP waterfall.

## Repository contents

| File | Purpose |
|---|---|
| `Full_Pipeline_For_Prolonged_LOS_leakage_fixed.ipynb` | Full pipeline (Colab-ready) |
| `README.md` | This file |

The INSPIRE data is not included. Obtain it from the source and build the operation-level table `op_level_dataset_postop.csv` expected by the notebook.

## How to run

1. Open the notebook in Google Colab.
2. Run Cell 1 to install the dependencies: `pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib gradio`.
3. Upload `op_level_dataset_postop.csv` when prompted.
4. Run all cells top to bottom. The trained pipeline is saved as `los_xgb_pipeline.joblib` and the feature list as `feature_cols.joblib`.
5. Run the final cell to launch the Gradio demo.

## Limitations

- **Single public dataset, one hospital system:** performance may not transfer elsewhere; no external validation.
- **The 7-day threshold is dataset-specific.** With 77% of patients above it, the model separates unusually long stays within a long-stay setting, not "prolonged" stays against a fast-track norm.
- **Low specificity (about 51%)** at the default threshold: the model flags roughly half of the shorter stays as long. Threshold tuning would be needed for any real use.
- **SHAP explains the model, not causes,** and attributions can be unstable when predictors are correlated.
- **Fairness screening is limited** to sex, using discrimination metrics only (no calibration check).
- **Retrospective only:** no prospective or clinical-utility evaluation.

## Possible next steps

- Sensitivity analysis at other thresholds (for example 5 and 10 days).
- Calibration curves and decision-curve analysis.
- Patient-level splitting if repeated patients are present.
- External validation on another dataset.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · matplotlib · seaborn · Gradio · joblib

## Author

Terrance Boachie · MSc Computer Science, Birmingham City University · [LinkedIn](https://linkedin.com/in/terrance-boachie) · [GitHub](https://github.com/terranceboachie)
