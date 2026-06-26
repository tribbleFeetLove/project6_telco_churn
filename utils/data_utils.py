"""
数据处理工具模块

包含数据加载、清洗、预处理、特征工程等功能函数。
遵循PEP8规范，所有函数包含docstring。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


def set_random_seed(seed: int = 42) -> None:
    """设置全局随机种子以确保可复现性。

    Parameters
    ----------
    seed : int, default=42
        随机种子值
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_telco_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """加载Telco客户流失数据集。

    优先从本地路径加载，若不存在则尝试从网络下载。

    Parameters
    ----------
    data_path : str, optional
        数据文件路径

    Returns
    -------
    pd.DataFrame
        加载后的原始数据框

    Raises
    ------
    FileNotFoundError
        当本地文件不存在且无法从网络获取时抛出
    """
    import os

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # 尝试从网络加载
        url = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
               "master/data/Telco-Customer-Churn.csv")
        df = pd.read_csv(url)

    print(f"数据集加载成功: {df.shape[0]} 条记录, {df.shape[1]} 个特征")
    return df


def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """执行探索性数据分析（EDA）。

    输出数据集的基本统计信息和数据质量报告。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框

    Returns
    -------
    Dict[str, Any]
        包含数据探索结果的字典
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'describe': df.describe().to_dict(),
        'target_distribution': df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else {},
        'target_ratio': (df['Churn'].value_counts(normalize=True).to_dict()
                         if 'Churn' in df.columns else {}),
    }

    # 打印摘要
    print("=" * 60)
    print("数据集探索性分析 (EDA)")
    print("=" * 60)
    print(f"样本数: {info['shape'][0]}, 特征数: {info['shape'][1]}")
    print(f"\n缺失值统计:")
    missing_cols = {k: v for k, v in info['missing'].items() if v > 0}
    if missing_cols:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({info['missing_pct'][col]:.2f}%)")
    else:
        print("  无缺失值")

    if info['target_distribution']:
        print(f"\n目标变量分布:")
        for k, v in info['target_distribution'].items():
            print(f"  {k}: {v} ({info['target_ratio'][k]:.2%})")

    return info


def preprocess_data(df: pd.DataFrame,
                    target_col: str = 'Churn',
                    handle_missing: bool = True,
                    missing_strategy: str = 'median',
                    outlier_method: str = 'iqr',
                    outlier_threshold: float = 1.5,
                    scaling: str = 'standard',
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """执行完整的数据预处理流程。

    包括：缺失值处理、异常值检测、类别编码、特征缩放。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    target_col : str, default='Churn'
        目标变量列名
    handle_missing : bool, default=True
        是否处理缺失值
    missing_strategy : str, default='median'
        缺失值填充策略
    outlier_method : str, default='iqr'
        异常值检测方法
    outlier_threshold : float, default=1.5
        异常值阈值（IQR倍数）
    scaling : str, default='standard'
        特征缩放方法
    random_state : int, default=42
        随机种子

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]
        处理后的特征、目标变量、预处理参数字典
    """
    df_processed = df.copy()
    preprocess_params = {}

    # 1. 移除无关列
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)
        print("已移除无关列: customerID")

    # 2. 处理目标变量编码
    if target_col in df_processed.columns:
        df_processed[target_col] = df_processed[target_col].map({'Yes': 1, 'No': 0})
        print("目标变量 Churn: Yes->1, No->0")

    # 3. 处理TotalCharges（可能包含空格字符串）
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(
            df_processed['TotalCharges'], errors='coerce'
        )

    # 4. 缺失值处理
    if handle_missing:
        numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]

        missing_counts = df_processed[numeric_cols].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) > 0:
            print(f"\n处理缺失值 ({missing_strategy}策略):")
            for col in cols_with_missing.index:
                count = cols_with_missing[col]
                print(f"  {col}: {count} 个缺失值")

            if missing_strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif missing_strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif missing_strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='median')

            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            preprocess_params['imputer'] = imputer
        else:
            print("无缺失值需要处理")

    # 5. 异常值检测（IQR方法）
    if outlier_method == 'iqr':
        numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]
        outlier_counts = {}

        for col in numeric_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            outliers = ((df_processed[col] < lower_bound) |
                        (df_processed[col] > upper_bound))
            outlier_counts[col] = outliers.sum()

        total_outliers = sum(outlier_counts.values())
        preprocess_params['outlier_bounds'] = {
            'method': 'iqr',
            'threshold': outlier_threshold,
        }
        if total_outliers > 0:
            print(f"\n异常值检测 (IQR x {outlier_threshold}): 共 {total_outliers} 个异常值")
            top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for col, count in top_outlier_cols:
                if count > 0:
                    print(f"  {col}: {count} 个")

    # 6. 类别特征编码
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le

    preprocess_params['encoders'] = encoders
    preprocess_params['categorical_cols'] = categorical_cols
    print(f"\n类别特征编码完成: {len(categorical_cols)} 个特征")

    # 7. 特征缩放
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    preprocess_params['scaler'] = scaler
    preprocess_params['numeric_cols'] = numeric_cols
    print(f"特征缩放完成 ({scaling}): {len(numeric_cols)} 个数值特征")

    # 分离特征与目标
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    return X, y, preprocess_params


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """为电信场景构建RFM特征。

    电信场景RFM适配：
    - R (Recency): 使用tenure（在网时长）的倒数表示"最近性"
    - F (Frequency): 使用的服务数量
    - M (Monetary): 月费用与总费用

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框（编码前）

    Returns
    -------
    pd.DataFrame
        添加RFM特征后的数据框
    """
    df_rfm = df.copy()

    # R: 最近性 — tenure越长，越不"近期"，使用负tenure或tenure倒数
    if 'tenure' in df_rfm.columns:
        tenure_col = pd.to_numeric(df_rfm['tenure'], errors='coerce')
        df_rfm['R_score'] = 1 / (tenure_col + 1)  # 新客户R值高

    # F: 频率 — 使用的服务数量
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    available_service_cols = [c for c in service_cols if c in df_rfm.columns]
    if available_service_cols:
        # 将"Yes"/"No"/"DSL"/"Fiber optic"等转为是否使用
        service_count = pd.Series(0, index=df_rfm.index)
        for col in available_service_cols:
            has_service = df_rfm[col].astype(str).str.lower().isin(
                ['yes', 'dsl', 'fiber optic', 'fiber_optic']
            )
            service_count += has_service.astype(int)
        df_rfm['F_score'] = service_count
    else:
        df_rfm['F_score'] = 0

    # M: 货币 — 综合月费用和总费用
    if 'MonthlyCharges' in df_rfm.columns:
        monthly = pd.to_numeric(df_rfm['MonthlyCharges'], errors='coerce')
        df_rfm['M_score'] = monthly

    # RFM综合评分
    rfm_cols = [c for c in ['R_score', 'F_score', 'M_score'] if c in df_rfm.columns]
    if rfm_cols:
        df_rfm['RFM_total'] = df_rfm[rfm_cols].sum(axis=1)

    print(f"RFM特征构建完成: {[c for c in ['R_score', 'F_score', 'M_score', 'RFM_total'] if c in df_rfm.columns]}")
    return df_rfm


def compute_woe_binning(df: pd.DataFrame,
                        feature: str,
                        target: pd.Series,
                        n_bins: int = 10) -> Tuple[pd.Series, Dict[str, Any]]:
    """计算单个特征的WOE（Weight of Evidence）分箱。

    Parameters
    ----------
    df : pd.DataFrame
        特征数据框
    feature : str
        特征列名
    target : pd.Series
        目标变量（0/1编码）
    n_bins : int, default=10
        分箱数量

    Returns
    -------
    Tuple[pd.Series, Dict[str, Any]]
        WOE编码后的特征序列、分箱信息字典
    """
    df_temp = pd.DataFrame({'feature': df[feature], 'target': target})

    # 对数值特征进行等频分箱
    if df_temp['feature'].nunique() > n_bins:
        df_temp['bin'] = pd.qcut(df_temp['feature'], q=n_bins, duplicates='drop')
    else:
        df_temp['bin'] = df_temp['feature']

    # 计算每个分箱的WOE
    grouped = df_temp.groupby('bin').agg(
        good=('target', lambda x: (x == 0).sum()),
        bad=('target', lambda x: (x == 1).sum())
    )

    total_good = (target == 0).sum()
    total_bad = (target == 1).sum()

    # 防止除零
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad

    # WOE = ln(good_pct / bad_pct)，处理除零
    grouped['good_pct'] = grouped['good_pct'].replace(0, 1e-6)
    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 1e-6)
    grouped['WOE'] = np.log(grouped['good_pct'] / grouped['bad_pct'])

    # 映射回原数据
    woe_map = grouped['WOE'].to_dict()
    woe_series = df_temp['bin'].map(woe_map)

    bin_info = {
        'bins': grouped.index.tolist(),
        'woe_values': grouped['WOE'].tolist(),
        'bin_counts': grouped['good'].values + grouped['bad'].values,
    }

    return woe_series.astype(float), bin_info


def split_data(X: pd.DataFrame,
               y: pd.Series,
               train_ratio: float = 0.8,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """划分训练集和测试集。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    train_ratio : float, default=0.8
        训练集比例
    random_state : int, default=42
        随机种子

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=random_state, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    print(f"训练集流失率: {y_train.mean():.2%}, 测试集流失率: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series,
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """使用SMOTE进行过采样以处理类别不平衡。

    Parameters
    ----------
    X_train : pd.DataFrame
        训练集特征
    y_train : pd.Series
        训练集目标变量
    random_state : int, default=42
        随机种子

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        SMOTE过采样后的特征和目标变量
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"SMOTE过采样: {X_train.shape[0]} -> {X_resampled.shape[0]} 样本")
    print(f"过采样后流失率: {y_resampled.mean():.2%}")
    return X_resampled, y_resampled
