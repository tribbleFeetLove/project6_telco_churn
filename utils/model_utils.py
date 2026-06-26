"""
模型工具模块

包含模型训练、评估、超参数调优、实验结果记录等功能。
遵循PEP8规范。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')


def build_model(model_name: str, params: Optional[Dict[str, Any]] = None,
                random_state: int = 42) -> Any:
    """根据名称和参数构建分类模型。

    Parameters
    ----------
    model_name : str
        模型名称，支持 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'
    params : Dict[str, Any], optional
        模型参数
    random_state : int, default=42
        随机种子

    Returns
    -------
    Any
        构建好的分类器实例
    """
    default_params = {
        'logistic_regression': {
            'C': 1.0, 'max_iter': 1000, 'solver': 'liblinear',
            'class_weight': 'balanced', 'random_state': random_state
        },
        'random_forest': {
            'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'class_weight': 'balanced',
            'random_state': random_state, 'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'scale_pos_weight': 3, 'random_state': random_state,
            'use_label_encoder': False, 'eval_metric': 'logloss'
        },
        'lightgbm': {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
            'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'is_unbalance': True, 'random_state': random_state,
            'verbose': -1
        },
    }

    if model_name not in default_params:
        raise ValueError(f"不支持的模型: {model_name}。可选: {list(default_params.keys())}")

    model_params = default_params[model_name].copy()
    if params:
        model_params.update(params)

    if model_name == 'logistic_regression':
        return LogisticRegression(**model_params)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**model_params)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(**model_params)
    elif model_name == 'lightgbm':
        return lgb.LGBMClassifier(**model_params)


def train_and_evaluate(model: Any,
                       X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series) -> Dict[str, Any]:
    """训练模型并进行全面评估。

    Parameters
    ----------
    model : Any
        分类器实例
    X_train : pd.DataFrame
        训练集特征
    X_test : pd.DataFrame
        测试集特征
    y_train : pd.Series
        训练集标签
    y_test : pd.Series
        测试集标签

    Returns
    -------
    Dict[str, Any]
        包含各项评估指标的字典
    """
    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 计算指标
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                                       target_names=['No Churn', 'Churn'],
                                                       output_dict=True),
    }

    # ROC曲线数据
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    # PR曲线数据
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
    results['pr_curve'] = {
        'precision': precision_curve.tolist(),
        'recall': recall_curve.tolist()
    }

    # 特征重要性（如果模型支持）
    if hasattr(model, 'feature_importances_'):
        results['feature_importance'] = model.feature_importances_.tolist()
    elif hasattr(model, 'coef_'):
        results['feature_importance'] = model.coef_[0].tolist()

    # 特征名称
    if isinstance(X_train, pd.DataFrame):
        results['feature_names'] = X_train.columns.tolist()

    return results


def cross_validate_model(model: Any,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv_folds: int = 5,
                         random_state: int = 42) -> Dict[str, Any]:
    """执行分层K折交叉验证。

    Parameters
    ----------
    model : Any
        分类器实例
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    cv_folds : int, default=5
        折数
    random_state : int, default=42
        随机种子

    Returns
    -------
    Dict[str, Any]
        交叉验证结果
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=-1)

    results = {}
    for metric in scoring.keys():
        results[metric] = {
            'mean': cv_results[f'test_{metric}'].mean(),
            'std': cv_results[f'test_{metric}'].std(),
            'scores': cv_results[f'test_{metric}'].tolist(),
        }

    return results


def compare_all_models(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       model_names: Optional[List[str]] = None,
                       random_state: int = 42) -> pd.DataFrame:
    """对比所有模型的性能。

    Parameters
    ----------
    X_train : pd.DataFrame
        训练集特征
    X_test : pd.DataFrame
        测试集特征
    y_train : pd.Series
        训练集标签
    y_test : pd.Series
        测试集标签
    model_names : List[str], optional
        要对比的模型列表
    random_state : int, default=42
        随机种子

    Returns
    -------
    pd.DataFrame
        模型对比结果表
    """
    if model_names is None:
        model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']

    all_results = {}
    best_model = None
    best_auc = 0.0

    for name in model_names:
        print(f"\n{'=' * 50}")
        print(f"训练模型: {name}")
        print('-' * 50)

        model = build_model(name, random_state=random_state)
        results = train_and_evaluate(model, X_train, X_test, y_train, y_test)

        all_results[name] = results
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  PR-AUC:    {results['pr_auc']:.4f}")

        if results['roc_auc'] > best_auc:
            best_auc = results['roc_auc']
            best_model = model
            best_results = results

    # 构建对比表
    comparison_rows = []
    for name, r in all_results.items():
        comparison_rows.append({
            'Model': name,
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'PR-AUC': f"{r['pr_auc']:.4f}",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.set_index('Model')

    return comparison_df, all_results, best_model, best_results


def save_model(model: Any, path: str) -> None:
    """保存训练好的模型。

    Parameters
    ----------
    model : Any
        训练好的模型
    path : str
        保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"模型已保存到: {path}")


def load_model(path: str) -> Any:
    """加载已保存的模型。

    Parameters
    ----------
    path : str
        模型文件路径

    Returns
    -------
    Any
        加载的模型
    """
    return joblib.load(path)


def save_experiment_results(results: Dict[str, Any], exp_dir: str) -> None:
    """保存实验结果到JSON文件。

    Parameters
    ----------
    results : Dict[str, Any]
        实验结果字典
    exp_dir : str
        实验目录路径
    """
    os.makedirs(exp_dir, exist_ok=True)

    # 将numpy数组转为列表
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable[key] = value

    with open(os.path.join(exp_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"实验结果已保存到: {exp_dir}/metrics.json")
