"""Prediction preprocessing helpers for the Telco churn project."""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.data_utils import add_woe_features


DEFAULT_WOE_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
DEFAULT_DATA_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'


def load_config(config_path: str) -> dict:
    """Load YAML config if it exists."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_model_bundle(model_path: str) -> tuple[Any, Optional[list[str]]]:
    """Load a model saved either directly or with feature metadata."""
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model'], model_data.get('feature_names')
    return model_data, None


def load_training_data(cfg: dict) -> pd.DataFrame:
    """Load the training reference dataset configured for the project."""
    data_cfg = cfg.get('data', {})
    data_file = data_cfg.get('data_file', DEFAULT_DATA_FILE)
    data_path = os.path.join(data_cfg.get('data_path', './data/raw'), data_file)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'训练参考数据不存在: {data_path}。请先放置原始Telco数据或运行训练流程。'
        )
    return pd.read_csv(data_path)


def _transform_categorical_columns(df: pd.DataFrame,
                                   encoders: Optional[dict[str, LabelEncoder]]
                                   ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode object columns with fitted training encoders."""
    fitted_encoders = encoders or {}
    for col in df.select_dtypes(include=['object']).columns:
        values = df[col].astype(str)
        if encoders is None:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(values)
            fitted_encoders[col] = encoder
            continue

        if col not in fitted_encoders:
            df[col] = -1
            continue

        mapping = {
            label: index
            for index, label in enumerate(fitted_encoders[col].classes_)
        }
        df[col] = values.map(mapping).fillna(-1).astype(int)
    return df, fitted_encoders


def _encode_telco_frame(df_raw: pd.DataFrame,
                        cfg: dict,
                        require_target: bool,
                        encoders: Optional[dict[str, LabelEncoder]] = None
                        ) -> tuple[pd.DataFrame, Optional[pd.Series], dict[str, LabelEncoder]]:
    """Encode raw Telco rows and build RFM features."""
    df = df_raw.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    y = None
    if 'Churn' in df.columns:
        y = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
        df.drop('Churn', axis=1, inplace=True)
    elif require_target:
        raise ValueError('训练参考数据必须包含 Churn 列')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_strategy = cfg.get('preprocessing', {}).get('missing_strategy', 'median')
    fill_value = df['TotalCharges'].mean() if missing_strategy == 'mean' else df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(fill_value)

    df, fitted_encoders = _transform_categorical_columns(df, encoders)

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
    return df, y, fitted_encoders


def prepare_prediction_features(input_df: pd.DataFrame,
                                training_df: pd.DataFrame,
                                cfg: dict,
                                feature_names: Optional[list[str]] = None) -> pd.DataFrame:
    """Prepare input rows with the same feature engineering used for training.

    WOE mappings are fitted on the configured training split and then applied to
    the prediction rows, matching the leakage-free evaluation path.
    """
    X_all, y_all, encoders = _encode_telco_frame(training_df, cfg, require_target=True)
    X_input, _, _ = _encode_telco_frame(
        input_df,
        cfg,
        require_target=False,
        encoders=encoders,
    )

    random_state = cfg.get('data', {}).get('random_state', 42)
    train_ratio = cfg.get('data', {}).get('train_ratio', 0.8)
    X_train, _, y_train, _ = train_test_split(
        X_all,
        y_all,
        train_size=train_ratio,
        random_state=random_state,
        stratify=y_all,
    )

    woe_columns = cfg.get('feature_engineering', {}).get('woe_columns', DEFAULT_WOE_COLUMNS)
    X_train_woe, X_input_woe = add_woe_features(
        X_train,
        X_input,
        y_train,
        columns=woe_columns,
        n_bins=cfg.get('feature_engineering', {}).get('woe_bins', 10),
    )

    expected_features = feature_names or X_train_woe.columns.tolist()
    missing = [col for col in expected_features if col not in X_input_woe.columns]
    if missing:
        raise ValueError(f'预测数据缺少模型所需特征: {missing}')
    return X_input_woe[expected_features]


def predict_dataframe(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Return churn labels and probabilities for prepared features."""
    probabilities = model.predict_proba(features)[:, 1]
    labels = model.predict(features)
    return pd.DataFrame({
        'churn_probability': probabilities,
        'predicted_churn': labels,
    })
