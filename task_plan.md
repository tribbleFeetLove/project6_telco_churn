# Task Plan

Goal: Improve the course project in four independently committed units.

## Phases

1. [complete] Sync notebook and `run_analysis.py` with the leakage-free `main.py` flow.
2. [complete] Move SMOTE into cross-validation pipelines and add ablation experiments.
3. [complete] Add `predict.py` and basic tests.
4. [pending] Update report and PPT so metrics/method descriptions match current code.

## Commit Rules

- Commit after each phase is implemented and verified.
- Use commit messages in the format `类型: 中文描述`.
- Do not rewrite history or revert user changes.

## Verification Log

- Phase 1: `python -m py_compile run_analysis.py generate_notebook.py` passed.
- Phase 1: `python generate_notebook.py` regenerated `Telco_Customer_Churn_Analysis.ipynb`.
- Phase 1: `python -m jupyter nbconvert --to notebook --execute Telco_Customer_Churn_Analysis.ipynb --output Telco_Customer_Churn_Analysis.executed.ipynb --ExecutePreprocessor.timeout=900` passed; temporary executed notebook removed.
- Phase 2: `python -m py_compile main.py run_analysis.py generate_notebook.py utils\model_utils.py train.py` passed.
- Phase 2: `python train.py --config configs/default.yaml` passed and regenerated CV/ablation artifacts.
- Phase 2: `python run_analysis.py` passed after adding UTF-8 stdout/stderr configuration.
- Phase 2: `python -m jupyter nbconvert --to notebook --execute --inplace Telco_Customer_Churn_Analysis.ipynb --ExecutePreprocessor.timeout=900` passed; Windows/joblib emitted non-fatal `resource_tracker` cleanup noise.
- Phase 3: `python -m py_compile predict.py utils\predict_utils.py tests\test_predict.py` passed.
- Phase 3: `python -m unittest discover -s tests` passed with 3 tests.
- Phase 3: `python predict.py --help` passed.
- Phase 3: small-sample `python predict.py --input ... --output ...` passed and wrote prediction probabilities.
