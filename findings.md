# Findings

## Initial State

- Branch: `dev_zsh`.
- Working tree was clean before this task.
- Previous commit: `c80ef30 fix: 修复训练评估复现与WOE数据泄漏`.

## Notes

## Phase 1 Findings

- `run_analysis.py` still computes WOE on `df_fe` before `train_test_split`, which leaks target information into the test features.
- `Telco_Customer_Churn_Analysis.ipynb` has the same old WOE section and outputs.
- `generate_notebook.py` also contains the old WOE logic, so it must be updated with the notebook to keep generated artifacts consistent.
- Notebook and `run_analysis.py` still mention/use SMOTE before cross-validation; that is intentionally deferred to phase 2.

## Phase 2 Findings

- Previous cross-validation used the already oversampled `X_train_smote`, which lets synthetic samples derived from validation-fold neighbors influence validation scores.
- `utils.model_utils.build_smote_pipeline()` now wraps each classifier in an imblearn `Pipeline`, so SMOTE is fitted inside each training fold only.
- `main.py`, `run_analysis.py`, and the generated notebook now save `experiments/cv_results.csv` from fold-internal SMOTE CV.
- Added a Random Forest ablation table comparing Base, Base+RFM, Base+RFM+WOE, and Base+RFM+WOE+SMOTE.
- Current ablation shows SMOTE materially improves recall/F1 (`Recall=0.7086`, `F1=0.6295`) while slightly reducing accuracy and precision versus the no-SMOTE baseline.
- `run_analysis.py` needed UTF-8 stdout/stderr configuration for Windows consoles; otherwise the checkmark output can fail under GBK.

## Phase 3 Findings

- The repository had `train.py` and `evaluate.py`, but no direct batch prediction entry point for new CSV files.
- `predict.py` now loads `models/best_model_with_features.pkl`, aligns features to saved model metadata, and writes `churn_probability` plus `predicted_churn`.
- Prediction preprocessing refits WOE mappings only on the configured training split, then maps incoming rows without requiring their labels.
- Prediction categorical encoding is fitted on the training reference data and reused for incoming rows; unseen categories are encoded as `-1`.
- Tests use Python `unittest` to avoid adding a new test dependency.
- Windows subprocess output needed explicit UTF-8 configuration in `predict.py` for reliable test capture.
