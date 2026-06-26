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
