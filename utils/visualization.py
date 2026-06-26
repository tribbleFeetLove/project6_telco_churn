"""
可视化工具模块

包含数据可视化、模型评估图表、客户细分可视化、SHAP分析图等功能。
遵循PEP8规范。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Churn',
                             save_path: Optional[str] = None) -> None:
    """绘制目标变量分布图。

    Parameters
    ----------
    df : pd.DataFrame
        数据框
    target_col : str, default='Churn'
        目标变量列名
    save_path : str, optional
        图片保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 计数图
    counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(counts.index.astype(str), counts.values, color=colors, edgecolor='white')
    axes[0].set_title(f'{target_col} Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontsize=12, fontweight='bold')

    # 饼图
    axes[1].pie(counts.values, labels=counts.index.astype(str),
                autopct='%1.1f%%', colors=colors, explode=(0, 0.05),
                shadow=True, startangle=90)
    axes[1].set_title(f'{target_col} Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, cols: Optional[List[str]] = None,
                               n_cols: int = 3, figsize_per_plot: Tuple[int, int] = (5, 4),
                               save_path: Optional[str] = None) -> None:
    """绘制数值特征的分布图。

    Parameters
    ----------
    df : pd.DataFrame
        数据框
    cols : List[str], optional
        要绘制的列名列表
    n_cols : int, default=3
        每行子图数量
    figsize_per_plot : Tuple[int, int], default=(5, 4)
        每个子图的大小
    save_path : str, optional
        图片保存路径
    """
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot[0],
                                                       n_rows * figsize_per_plot[1]))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i, col in enumerate(cols):
        if i < len(axes):
            ax = axes[i]
            data = df[col].dropna()
            ax.hist(data, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.2f}')
            ax.set_title(col, fontsize=11)
            ax.legend(fontsize=8)

    # 隐藏多余子图
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                            figsize: Tuple[int, int] = (16, 14),
                            save_path: Optional[str] = None) -> None:
    """绘制特征相关性热力图。

    Parameters
    ----------
    df : pd.DataFrame
        数据框（仅数值列）
    figsize : Tuple[int, int], default=(16, 14)
        图大小
    save_path : str, optional
        图片保存路径
    """
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame,
                          metrics: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """绘制模型对比柱状图。

    Parameters
    ----------
    comparison_df : pd.DataFrame
        模型对比结果表
    metrics : List[str], optional
        要展示的指标
    save_path : str, optional
        图片保存路径
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

    df_plot = comparison_df[metrics].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.2
    n_models = len(df_plot)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for i, (model_name, row) in enumerate(df_plot.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, row.values, width, label=model_name,
               color=colors[i % len(colors)], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curves(all_results: Dict[str, Dict[str, Any]],
                    save_path: Optional[str] = None) -> None:
    """绘制所有模型的ROC曲线对比。

    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        所有模型的评估结果
    save_path : str, optional
        图片保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (name, results) in enumerate(all_results.items()):
        roc_data = results.get('roc_curve', {})
        fpr = roc_data.get('fpr', [])
        tpr = roc_data.get('tpr', [])
        auc = results.get('roc_auc', 0)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pr_curves(all_results: Dict[str, Dict[str, Any]],
                   save_path: Optional[str] = None) -> None:
    """绘制所有模型的PR曲线对比。

    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        所有模型的评估结果
    save_path : str, optional
        图片保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (name, results) in enumerate(all_results.items()):
        pr_data = results.get('pr_curve', {})
        precision = pr_data.get('precision', [])
        recall = pr_data.get('recall', [])
        pr_auc = results.get('pr_auc', 0)
        ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                label=f'{name} (PR-AUC = {pr_auc:.4f})')

    # 正类基线
    baseline = results.get('baseline_positive_rate', 0.27) if all_results else 0.27
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
               label=f'Baseline (pos rate = {baseline:.2f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrices(all_results: Dict[str, Dict[str, Any]],
                            save_path: Optional[str] = None) -> None:
    """绘制所有模型的混淆矩阵。

    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        所有模型的评估结果
    save_path : str, optional
        图片保存路径
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, all_results.items()):
        cm = np.array(results.get('confusion_matrix', [[0, 0], [0, 0]]))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    ax=ax)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_shap_analysis(model: Any, X: pd.DataFrame,
                       max_display: int = 20,
                       save_path: Optional[str] = None) -> Any:
    """使用SHAP进行模型可解释性分析。

    Parameters
    ----------
    model : Any
        训练好的树模型（RF, XGBoost, LightGBM）
    X : pd.DataFrame
        特征数据
    max_display : int, default=20
        最多显示的特征数
    save_path : str, optional
        图片保存路径

    Returns
    -------
    Any
        SHAP explainer对象
    """
    import shap

    # 创建SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)

    # 对于XGBoost/LightGBM，shap_values可能是list（多类别）
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 取正类（Churn=1）

    # Summary plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    # SHAP summary bar plot
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=max_display, show=False)
    axes[0].set_title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')

    # SHAP summary dot plot
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    axes[1].set_title('SHAP Summary Plot (Impact on Prediction)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return explainer, shap_values


def plot_feature_importance(model: Any, feature_names: List[str],
                            title: str = 'Feature Importance',
                            top_n: int = 15,
                            save_path: Optional[str] = None) -> None:
    """绘制特征重要性排序图。

    Parameters
    ----------
    model : Any
        训练好的模型（需有feature_importances_或coef_属性）
    feature_names : List[str]
        特征名称列表
    title : str, default='Feature Importance'
        图表标题
    top_n : int, default=15
        显示前N个重要特征
    save_path : str, optional
        图片保存路径
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print("该模型不支持特征重要性提取")
        return

    # 排序取前N个
    indices = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[indices]
    top_features = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    ax.barh(range(len(top_features)), top_importance[::-1], color=colors[::-1],
            edgecolor='white')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_customer_segments(df: pd.DataFrame,
                           labels: np.ndarray,
                           pca_features: Optional[np.ndarray] = None,
                           method: str = 'K-Means',
                           save_path: Optional[str] = None) -> None:
    """可视化客户聚类细分结果。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    labels : np.ndarray
        聚类标签
    pca_features : np.ndarray, optional
        PCA降维后的2维特征，用于可视化
    method : str, default='K-Means'
        聚类方法名称
    save_path : str, optional
        图片保存路径
    """
    if pca_features is None:
        from sklearn.decomposition import PCA
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(df[numeric_cols])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 散点图
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(pca_features[mask, 0], pca_features[mask, 1],
                        c=[colors[i]], label=f'Cluster {i}',
                        alpha=0.6, edgecolors='white', s=50)

    # DBSCAN噪声点
    if -1 in labels:
        mask = labels == -1
        axes[0].scatter(pca_features[mask, 0], pca_features[mask, 1],
                        c='black', label='Noise', alpha=0.5, s=30, marker='x')

    axes[0].set_xlabel('PCA Component 1', fontsize=11)
    axes[0].set_ylabel('PCA Component 2', fontsize=11)
    axes[0].set_title(f'{method} Customer Segmentation\n({n_clusters} clusters)',
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)

    # 饼图
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    axes[1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
    axes[1].set_title('Cluster Distribution', fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cluster_profiles(df: pd.DataFrame, labels: np.ndarray,
                          profile_cols: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """绘制客户聚类画像雷达图。

    Parameters
    ----------
    df : pd.DataFrame
        数据框（需包含profile_cols）
    labels : np.ndarray
        聚类标签
    profile_cols : List[str], optional
        用于画像分析的特征列
    save_path : str, optional
        图片保存路径
    """
    if profile_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        profile_cols = numeric_cols[:8]  # 取前8个主要特征

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 计算每个聚类在各特征上的均值
    df_temp = df[profile_cols].copy()
    df_temp['Cluster'] = labels
    df_temp = df_temp[df_temp['Cluster'] != -1]  # 排除噪声

    cluster_means = df_temp.groupby('Cluster')[profile_cols].mean()

    # 标准化到0-1
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(profile_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        values = cluster_means_norm.iloc[i].tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                label=f'Cluster {i}')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(profile_cols, fontsize=9)
    ax.set_title('Customer Cluster Profiles (Radar Chart)', fontsize=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def setup_plot_style() -> None:
    """设置全局绘图样式。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')
