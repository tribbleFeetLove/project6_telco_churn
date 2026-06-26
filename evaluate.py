"""模型评估入口脚本。

加载已保存的最佳模型，并用与训练流程一致的数据切分和WOE映射
在测试集上输出主要分类指标。
"""

import argparse
import os
import sys

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from utils.data_utils import add_woe_features, set_random_seed, split_data


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='评估已训练的客户流失预测模型')
    parser.add_argument('--config', default='configs/default.yaml', help='YAML配置文件路径')
    parser.add_argument(
        '--model-path',
        default='models/best_model_with_features.pkl',
        help='已保存模型路径',
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取YAML配置文件。"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_dataset(cfg: dict) -> pd.DataFrame:
    """按配置加载Telco数据集。"""
    data_cfg = cfg.get('data', {})
    data_file = data_cfg.get('data_file', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data_path = os.path.join(data_cfg.get('data_path', './data/raw'), data_file)

    if os.path.exists(data_path):
        return pd.read_csv(data_path)

    fallback_url = (
        'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'
        'master/data/Telco-Customer-Churn.csv'
    )
    return pd.read_csv(fallback_url)


def prepare_features(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """执行与训练流程一致的基础预处理和RFM特征工程。"""
    df = df_raw.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    missing_strategy = cfg.get('preprocessing', {}).get('missing_strategy', 'median')
    fill_value = df['TotalCharges'].median()
    if missing_strategy == 'mean':
        fill_value = df['TotalCharges'].mean()
    df['TotalCharges'].fillna(fill_value, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df['R_Score'] = 1 / (df['tenure'] + 1)
    service_cols = [
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]
    df['F_Score'] = sum((df[c] > 0).astype(int) for c in service_cols if c in df.columns)
    df['M_Score'] = df['MonthlyCharges']
    df['RFM_Total'] = (
        df['R_Score'] +
        df['F_Score'] / 9 +
        df['M_Score'] / df['M_Score'].max()
    )

    df.fillna(0, inplace=True)
    return df.drop('Churn', axis=1), df['Churn']


def main() -> int:
    """加载模型并输出测试集评估指标。"""
    args = parse_args()
    cfg = load_config(args.config)
    random_state = cfg.get('data', {}).get('random_state', 42)
    set_random_seed(random_state)

    if not os.path.exists(args.model_path):
        print(f'模型文件不存在: {args.model_path}', file=sys.stderr)
        return 1

    df_raw = load_dataset(cfg)
    X, y = prepare_features(df_raw, cfg)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        train_ratio=cfg.get('data', {}).get('train_ratio', 0.8),
        random_state=random_state,
    )

    X_train, X_test = add_woe_features(
        X_train,
        X_test,
        y_train,
        columns=['tenure', 'MonthlyCharges', 'TotalCharges'],
        n_bins=cfg.get('feature_engineering', {}).get('woe_bins', 10),
    )

    model_data = joblib.load(args.model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        feature_names = model_data.get('feature_names')
        if feature_names:
            X_test = X_test[feature_names]
    else:
        model = model_data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('测试集评估结果:')
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
