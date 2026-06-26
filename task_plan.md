# Task Plan

Goal: Improve the course project in four independently committed units.

## Phases

1. [complete] Sync notebook and `run_analysis.py` with the leakage-free `main.py` flow.
2. [in_progress] Move SMOTE into cross-validation pipelines and add ablation experiments.
3. [pending] Add `predict.py` and basic tests.
4. [pending] Update report and PPT so metrics/method descriptions match current code.

## Commit Rules

- Commit after each phase is implemented and verified.
- Use commit messages in the format `类型: 中文描述`.
- Do not rewrite history or revert user changes.

## Verification Log

- Phase 1: `python -m py_compile run_analysis.py generate_notebook.py` passed.
- Phase 1: `python generate_notebook.py` regenerated `Telco_Customer_Churn_Analysis.ipynb`.
- Phase 1: `python -m jupyter nbconvert --to notebook --execute Telco_Customer_Churn_Analysis.ipynb --output Telco_Customer_Churn_Analysis.executed.ipynb --ExecutePreprocessor.timeout=900` passed; temporary executed notebook removed.
