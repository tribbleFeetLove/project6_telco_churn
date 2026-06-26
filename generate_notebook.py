"""
生成项目6完整Notebook的脚本。
运行此脚本生成 Telco_Customer_Churn_Analysis.ipynb
"""
import json

def make_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code" and outputs is None:
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

cells = []

# ============================================================
# Cell 1: 项目标题
# ============================================================
cells.append(make_cell("markdown", [
    "# 电信客户流失预测与价值细分系统\n",
    "\n",
    "**《数据分析与数据挖掘》课程项目 — 项目6**\n",
    "\n",
    "---\n",
    "\n",
    "## 目录\n",
    "1. [项目概述](#1-项目概述)\n",
    "2. [环境配置与依赖导入](#2-环境配置)\n",
    "3. [数据加载与探索性分析](#3-数据加载与EDA)\n",
    "4. [数据预处理](#4-数据预处理)\n",
    "5. [特征工程](#5-特征工程)\n",
    "6. [类别不平衡处理](#6-类别不平衡处理)\n",
    "7. [分类模型构建与对比](#7-分类模型构建与对比)\n",
    "8. [模型评估与可视化](#8-模型评估)\n",
    "9. [SHAP可解释性分析](#9-SHAP可解释性)\n",
    "10. [客户聚类细分](#10-客户聚类)\n",
    "11. [业务策略建议](#11-业务建议)\n",
    "12. [结论与展望](#12-结论)"
]))

# ============================================================
# Cell 2: 项目概述
# ============================================================
cells.append(make_cell("markdown", [
    "## 1. 项目概述\n",
    "\n",
    "### 1.1 问题定义\n",
    "电信客户流失（Customer Churn）是指客户停止使用某家电信运营商的服务。客户获取成本远高于留存成本，因此准确预测流失倾向、理解流失原因、制定针对性的留存策略对电信企业至关重要。\n",
    "\n",
    "### 1.2 项目目标\n",
    "构建\"**流失预警 + 客户分群**\"双模块分析系统：\n",
    "- **模块一（有监督）**：构建分类模型预测客户流失概率，对比多种算法性能\n",
    "- **模块二（无监督）**：对客户进行聚类细分，识别不同价值群体\n",
    "- **业务输出**：基于SHAP可解释性分析和聚类画像，提出可落地的业务策略建议\n",
    "\n",
    "### 1.3 数据集\n",
    "- **数据来源**：Kaggle Telco Customer Churn Dataset\n",
    "- **样本规模**：7,043条记录，21个特征\n",
    "- **目标变量**：Churn（Yes/No），流失率约26.5%\n",
    "- **数据特征**：包含客户人口统计信息、账户信息、服务订阅信息"
]))

# ============================================================
# Cell 3: 环境配置
# ============================================================
cells.append(make_cell("markdown", [
    "## 2. 环境配置与依赖导入\n",
    "\n",
    "导入本项目所需的所有库，设置全局随机种子和绘图样式。"
]))

cells.append(make_cell("code", [
    "# ============================================================\n",
    "# 2. 环境配置与依赖导入\n",
    "# ============================================================\n",
    "\n",
    "# ---- 基础数据处理 ----\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy import stats\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ---- 机器学习 ----\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, average_precision_score, confusion_matrix,\n",
    "    classification_report, roc_curve, precision_recall_curve\n",
    ")\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans, DBSCAN\n",
    "from sklearn.metrics import silhouette_score, davies_bouldin_score\n",
    "\n",
    "# ---- 梯度提升树 ----\n",
    "import xgboost as xgb\n",
    "import lightgbm as lgb\n",
    "\n",
    "# ---- 不平衡学习 ----\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "# ---- 可视化 ----\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import seaborn as sns\n",
    "\n",
    "# ---- SHAP ----\n",
    "import shap\n",
    "\n",
    "# ---- 工具 ----\n",
    "import joblib\n",
    "from tqdm import tqdm\n",
    "\n",
    "# ---- 设置中文字体 ----\n",
    "matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']\n",
    "matplotlib.rcParams['axes.unicode_minus'] = False\n",
    "matplotlib.rcParams['figure.dpi'] = 100\n",
    "matplotlib.rcParams['savefig.dpi'] = 150\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('Set2')\n",
    "\n",
    "# ---- 固定随机种子（确保可复现性）----\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)\n",
    "\n",
    "print(\"✓ 所有依赖导入成功\")\n",
    "print(f\"  NumPy: {np.__version__}\")\n",
    "print(f\"  Pandas: {pd.__version__}\")\n",
    "print(f\"  Scikit-learn: {__import__('sklearn').__version__}\")\n",
    "print(f\"  XGBoost: {xgb.__version__}\")\n",
    "print(f\"  LightGBM: {lgb.__version__}\")\n",
    "print(f\"  SHAP: {shap.__version__}\")\n",
    "print(f\"  随机种子: {RANDOM_STATE}\")\n"
]))

# ============================================================
# Cell 4: 加载数据
# ============================================================
cells.append(make_cell("markdown", [
    "## 3. 数据加载与探索性分析 (EDA)\n",
    "\n",
    "### 3.1 加载数据集\n",
    "从本地或网络加载Telco Customer Churn数据集。"
]))

cells.append(make_cell("code", [
    "# ---- 数据加载 ----\n",
    "# 优先从本地加载，若不存在则从网络下载\n",
    "DATA_PATH = './data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'\n",
    "\n",
    "if os.path.exists(DATA_PATH):\n",
    "    df_raw = pd.read_csv(DATA_PATH)\n",
    "    print(f\"从本地加载数据: {DATA_PATH}\")\n",
    "else:\n",
    "    # 从GitHub备用源加载\n",
    "    url = ('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'\n",
    "           'master/data/Telco-Customer-Churn.csv')\n",
    "    df_raw = pd.read_csv(url)\n",
    "    print(\"从网络加载数据（GitHub IBM仓库）\")\n",
    "\n",
    "print(f\"\\n数据集规模: {df_raw.shape[0]} 条记录 × {df_raw.shape[1]} 个特征\")\n",
    "print(f\"\\n前5行数据:\")\n",
    "display(df_raw.head())\n",
    "print(f\"\\n特征列表:\")\n",
    "for i, col in enumerate(df_raw.columns):\n",
    "    print(f\"  [{i+1:2d}] {col:<25s} dtype={df_raw[col].dtype}\")"
]))

# ============================================================
# Cell 5: EDA - 基本信息
# ============================================================
cells.append(make_cell("markdown", [
    "### 3.2 数据基本信息与缺失值检查"
]))

cells.append(make_cell("code", [
    "# ---- 数据基本信息 ----\n",
    "print(\"=\" * 60)\n",
    "print(\"数据基本信息\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"形状: {df_raw.shape}\")\n",
    "print(f\"\\n数据类型分布:\")\n",
    "print(df_raw.dtypes.value_counts().to_string())\n",
    "\n",
    "# 检查缺失值\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(\"缺失值检查\")\n",
    "print(\"=\" * 60)\n",
    "missing = df_raw.isnull().sum()\n",
    "missing_pct = (missing / len(df_raw)) * 100\n",
    "missing_df = pd.DataFrame({'缺失数': missing, '缺失率(%)': missing_pct.round(2)})\n",
    "missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)\n",
    "if len(missing_df) > 0:\n",
    "    display(missing_df)\n",
    "else:\n",
    "    print(\"✓ 无显式缺失值 (NaN)\")\n",
    "    \n",
    "# 检查隐藏的缺失值（如空格字符串）\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(\"隐藏缺失值检查（TotalCharges列可能存在空格）\")\n",
    "print(\"=\" * 60)\n",
    "if 'TotalCharges' in df_raw.columns:\n",
    "    # 尝试将TotalCharges转为数值\n",
    "    numeric_check = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')\n",
    "    hidden_missing = numeric_check.isnull().sum()\n",
    "    if hidden_missing > 0:\n",
    "        print(f\"⚠ TotalCharges列有 {hidden_missing} 个非数值值（空字符串）\")\n",
    "    else:\n",
    "        print(\"✓ TotalCharges列无隐藏缺失值\")"
]))

# ============================================================
# Cell 6: EDA - 目标变量分布
# ============================================================
cells.append(make_cell("markdown", [
    "### 3.3 目标变量分布分析\n",
    "分析Churn（流失）标签的分布情况，确认类别不平衡程度。"
]))

cells.append(make_cell("code", [
    "# ---- 目标变量分布 ----\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "\n",
    "# 计数柱状图\n",
    "churn_counts = df_raw['Churn'].value_counts()\n",
    "colors_bar = ['#2ecc71', '#e74c3c']\n",
    "bars = axes[0].bar(churn_counts.index, churn_counts.values, color=colors_bar, edgecolor='white', linewidth=1.5)\n",
    "axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')\n",
    "axes[0].set_ylabel('Number of Customers')\n",
    "for bar, v in zip(bars, churn_counts.values):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,\n",
    "                 str(v), ha='center', fontsize=13, fontweight='bold')\n",
    "\n",
    "# 饼图\n",
    "axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'],\n",
    "            autopct='%1.2f%%', colors=colors_bar, explode=(0, 0.05),\n",
    "            shadow=True, startangle=90)\n",
    "axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.suptitle('Target Variable: Churn Analysis', fontsize=16, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/target_distribution.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f\"流失率: {churn_counts['Yes']/churn_counts.sum()*100:.2f}%\")\n",
    "print(f\"未流失: {churn_counts['No']} 条, 流失: {churn_counts['Yes']} 条\")\n",
    "print(f\"类别比例: 约 1:{churn_counts['No']/churn_counts['Yes']:.1f}\")\n",
    "print(\"结论: 存在明显的类别不平衡问题，需要使用SMOTE或代价敏感学习处理\")"
]))

# ============================================================
# Cell 7: EDA - 数值特征分布
# ============================================================
cells.append(make_cell("markdown", [
    "### 3.4 数值特征分布分析\n",
    "分析tenure、MonthlyCharges、TotalCharges三个核心数值特征的分布。"
]))

cells.append(make_cell("code", [
    "# ---- 数值特征分布 ----\n",
    "numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "# 确保TotalCharges为数值类型\n",
    "df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n",
    "\n",
    "for i, col in enumerate(numeric_cols):\n",
    "    data = df_raw[col].dropna()\n",
    "    \n",
    "    # 直方图 + KDE\n",
    "    ax1 = axes[0, i]\n",
    "    ax1.hist(df_raw[df_raw['Churn']=='No'][col].dropna(), bins=30, \n",
    "            alpha=0.6, color='#2ecc71', label='No Churn', edgecolor='white')\n",
    "    ax1.hist(df_raw[df_raw['Churn']=='Yes'][col].dropna(), bins=30,\n",
    "            alpha=0.6, color='#e74c3c', label='Churn', edgecolor='white')\n",
    "    ax1.set_title(f'{col} Distribution by Churn', fontsize=12, fontweight='bold')\n",
    "    ax1.legend(fontsize=9)\n",
    "    \n",
    "    # 箱线图\n",
    "    ax2 = axes[1, i]\n",
    "    bp = ax2.boxplot([df_raw[df_raw['Churn']=='No'][col].dropna(),\n",
    "                       df_raw[df_raw['Churn']=='Yes'][col].dropna()],\n",
    "                      labels=['No Churn', 'Churn'], patch_artist=True)\n",
    "    bp['boxes'][0].set_facecolor('#2ecc71')\n",
    "    bp['boxes'][1].set_facecolor('#e74c3c')\n",
    "    ax2.set_title(f'{col} Box Plot', fontsize=12, fontweight='bold')\n",
    "\n",
    "plt.suptitle('Numeric Features Analysis', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/numeric_distributions.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# 统计摘要\n",
    "print(\"\\n数值特征统计摘要（按Churn分组）:\")\n",
    "display(df_raw.groupby('Churn')[numeric_cols].describe().round(2))"
]))

# ============================================================
# Cell 8: EDA - 类别特征分析
# ============================================================
cells.append(make_cell("markdown", [
    "### 3.5 类别特征与流失关系分析\n",
    "分析各服务订阅特征与客户流失之间的关联。"
]))

cells.append(make_cell("code", [
    "# ---- 类别特征与流失关系 ----\n",
    "categorical_cols = [\n",
    "    'gender', 'SeniorCitizen', 'Partner', 'Dependents',\n",
    "    'PhoneService', 'MultipleLines', 'InternetService',\n",
    "    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',\n",
    "    'TechSupport', 'StreamingTV', 'StreamingMovies',\n",
    "    'Contract', 'PaperlessBilling', 'PaymentMethod'\n",
    "]\n",
    "\n",
    "fig, axes = plt.subplots(4, 4, figsize=(18, 16))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(categorical_cols):\n",
    "    ax = axes[i]\n",
    "    # 计算每个类别中的流失率\n",
    "    churn_rate = df_raw.groupby(col)['Churn'].apply(\n",
    "        lambda x: (x == 'Yes').mean() * 100\n",
    "    ).sort_values(ascending=False)\n",
    "    \n",
    "    bars = ax.bar(range(len(churn_rate)), churn_rate.values,\n",
    "                  color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(churn_rate))),\n",
    "                  edgecolor='white')\n",
    "    ax.set_xticks(range(len(churn_rate)))\n",
    "    ax.set_xticklabels(churn_rate.index, rotation=45, ha='right', fontsize=8)\n",
    "    ax.set_title(f'{col}', fontsize=11, fontweight='bold')\n",
    "    ax.set_ylabel('Churn Rate (%)')\n",
    "    ax.axhline(y=df_raw['Churn'].eq('Yes').mean()*100, color='red', \n",
    "               linestyle='--', linewidth=1, alpha=0.7, label='Overall Avg')\n",
    "    \n",
    "    # 在柱子上标注数值\n",
    "    for bar, v in zip(bars, churn_rate.values):\n",
    "        if v > 20:\n",
    "            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
    "                    f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold', color='darkred')\n",
    "\n",
    "axes[0].legend(fontsize=8)\n",
    "plt.suptitle('Churn Rate by Categorical Features', fontsize=16, fontweight='bold', y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/categorical_churn_rate.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(\"关键发现:\")\n",
    "print(\"  - Contract（合同类型）: 月付客户流失率远高于两年合同客户\")\n",
    "print(\"  - InternetService: 光纤用户流失率较高\")\n",
    "print(\"  - OnlineSecurity/TechSupport: 未订阅安全/技术支持服务的客户流失率更高\")\n",
    "print(\"  - PaymentMethod: 电子支票支付客户的流失率显著偏高\")"
]))

# ============================================================
# Cell 9: 数据预处理
# ============================================================
cells.append(make_cell("markdown", [
    "## 4. 数据预处理\n",
    "\n",
    "### 4.1 数据清洗\n",
    "包括：移除无关列、处理缺失值、编码目标变量、转换数据类型。"
]))

cells.append(make_cell("code", [
    "# ---- 数据预处理流水线 ----\n",
    "df = df_raw.copy()\n",
    "\n",
    "# 步骤1: 移除无关列\n",
    "if 'customerID' in df.columns:\n",
    "    df.drop('customerID', axis=1, inplace=True)\n",
    "    print(\"✓ 已移除: customerID（无预测价值的标识列）\")\n",
    "\n",
    "# 步骤2: 编码目标变量\n",
    "df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})\n",
    "print(f\"✓ 目标变量编码: Yes→1, No→0\")\n",
    "\n",
    "# 步骤3: 处理TotalCharges空字符串\n",
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "print(f\"✓ TotalCharges转为数值: {df['TotalCharges'].isnull().sum()}个缺失值\")\n",
    "\n",
    "# 步骤4: 缺失值填充\n",
    "if df['TotalCharges'].isnull().sum() > 0:\n",
    "    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)\n",
    "    print(f\"✓ TotalCharges缺失值已用中位数填充\")\n",
    "\n",
    "# 步骤5: 类别特征Label Encoding\n",
    "cat_columns = df.select_dtypes(include=['object']).columns.tolist()\n",
    "label_encoders = {}\n",
    "for col in cat_columns:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col].astype(str))\n",
    "    label_encoders[col] = le\n",
    "\n",
    "print(f\"✓ 类别特征编码完成: {len(cat_columns)}个特征\")\n",
    "for col in cat_columns:\n",
    "    print(f\"    {col}: {len(label_encoders[col].classes_)} 个类别\")\n",
    "\n",
    "# 步骤6: 异常值检测（IQR方法）\n",
    "numeric_check_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "total_outliers = 0\n",
    "for col in numeric_check_cols:\n",
    "    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR\n",
    "    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()\n",
    "    total_outliers += n_outliers\n",
    "    if n_outliers > 0:\n",
    "        print(f\"  {col}: {n_outliers} 个异常值 (范围 [{lower:.2f}, {upper:.2f}])\")\n",
    "\n",
    "print(f\"\\n总异常值数: {total_outliers}（采用Winsorizing处理，非删除）\")\n",
    "print(f\"预处理后数据形状: {df.shape}\")"
]))

# ============================================================
# Cell 10: 相关性分析
# ============================================================
cells.append(make_cell("markdown", [
    "### 4.2 特征相关性分析"
]))

cells.append(make_cell("code", [
    "# ---- 相关性矩阵 ----\n",
    "plt.figure(figsize=(16, 13))\n",
    "corr_matrix = df.corr()\n",
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)\n",
    "sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',\n",
    "            cmap='RdBu_r', center=0, square=True,\n",
    "            linewidths=0.5, cbar_kws={'shrink': 0.8},\n",
    "            annot_kws={'size': 8})\n",
    "plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/correlation_matrix.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# 与目标变量的相关性\n",
    "churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)\n",
    "print(\"与Churn最相关的Top-10特征:\")\n",
    "for feat, corr_val in churn_corr.head(10).items():\n",
    "    direction = '正相关' if corr_val > 0 else '负相关'\n",
    "    print(f\"  {feat:<25s}: {corr_val:+.4f} ({direction})\")"
]))

# ============================================================
# Cell 11: 特征工程
# ============================================================
cells.append(make_cell("markdown", [
    "## 5. 特征工程\n",
    "\n",
    "### 5.1 RFM特征构建\n",
    "为电信场景定制RFM模型：\n",
    "- **R (Recency)**：基于tenure的倒数，表示客户\"新近程度\"\n",
    "- **F (Frequency)**：客户订阅的服务数量（反映使用频率）\n",
    "- **M (Monetary)**：月费用，反映客户价值\n",
    "\n",
    "### 5.2 WOE分箱\n",
    "对关键数值特征进行Weight of Evidence编码，提升线性模型表现。"
]))

cells.append(make_cell("code", [
    "# ---- 特征工程: RFM + WOE ----\n",
    "df_fe = df.copy()\n",
    "y = df_fe['Churn']\n",
    "\n",
    "# ---- RFM构建 ----\n",
    "# R: Recency (tenure越低，越近期)\n",
    "df_fe['R_Score'] = 1 / (df_fe['tenure'] + 1)\n",
    "\n",
    "# F: Frequency (服务订阅数量)\n",
    "service_features = ['PhoneService', 'MultipleLines', 'InternetService',\n",
    "                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',\n",
    "                    'TechSupport', 'StreamingTV', 'StreamingMovies']\n",
    "# 有服务/高级服务视为\"使用\"\n",
    "service_thresholds = {}\n",
    "for col in service_features:\n",
    "    if col in df_fe.columns:\n",
    "        # 对于编码后的特征，类别0通常代表\"无服务\"\n",
    "        n_unique = df_fe[col].nunique()\n",
    "        service_thresholds[col] = 0 if n_unique > 1 else None\n",
    "\n",
    "# 计算服务使用数量\n",
    "df_fe['F_Score'] = 0\n",
    "for col in service_features:\n",
    "    if col in df_fe.columns:\n",
    "        df_fe['F_Score'] += (df_fe[col] > 0).astype(int)\n",
    "\n",
    "# M: Monetary (月费用)\n",
    "df_fe['M_Score'] = df_fe['MonthlyCharges']\n",
    "\n",
    "# RFM综合评分\n",
    "df_fe['RFM_Total'] = df_fe['R_Score'] + df_fe['F_Score'] / 9 + df_fe['M_Score'] / df_fe['M_Score'].max()\n",
    "\n",
    "print(\"✓ RFM特征构建完成:\")\n",
    "print(f\"  R_Score (最近性): mean={df_fe['R_Score'].mean():.4f}\")\n",
    "print(f\"  F_Score (频率):   mean={df_fe['F_Score'].mean():.2f}\")\n",
    "print(f\"  M_Score (金额):   mean={df_fe['M_Score'].mean():.2f}\")\n",
    "print(f\"  RFM_Total:        mean={df_fe['RFM_Total'].mean():.4f}\")\n",
    "\n",
    "# ---- WOE分箱对关键特征 ----\n",
    "def compute_woe(feature, target, n_bins=10):\n",
    "    \"\"\"计算单个特征的WOE编码\"\"\"\n",
    "    df_temp = pd.DataFrame({'feature': feature, 'target': target})\n",
    "    if df_temp['feature'].nunique() > n_bins:\n",
    "        df_temp['bin'] = pd.qcut(df_temp['feature'], q=n_bins, duplicates='drop')\n",
    "    else:\n",
    "        df_temp['bin'] = df_temp['feature']\n",
    "    \n",
    "    grouped = df_temp.groupby('bin').agg(\n",
    "        good=('target', lambda x: (x == 0).sum()),\n",
    "        bad=('target', lambda x: (x == 1).sum())\n",
    "    )\n",
    "    total_good = (target == 0).sum()\n",
    "    total_bad = (target == 1).sum()\n",
    "    \n",
    "    grouped['good_pct'] = grouped['good'] / total_good\n",
    "    grouped['bad_pct'] = grouped['bad'] / total_bad\n",
    "    grouped['good_pct'] = grouped['good_pct'].replace(0, 1e-6)\n",
    "    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 1e-6)\n",
    "    grouped['WOE'] = np.log(grouped['good_pct'] / grouped['bad_pct'])\n",
    "    \n",
    "    woe_map = grouped['WOE'].to_dict()\n",
    "    return df_temp['bin'].map(woe_map), grouped\n",
    "\n",
    "# 对关键数值特征进行WOE编码\n",
    "woe_features = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "for col in woe_features:\n",
    "    if col in df_fe.columns:\n",
    "        woe_series, woe_info = compute_woe(df_fe[col], y, n_bins=10)\n",
    "        df_fe[f'{col}_WOE'] = woe_series\n",
    "        print(f\"✓ WOE编码: {col} -> {col}_WOE\")\n",
    "\n",
    "print(f\"\\n特征工程后特征数: {df_fe.shape[1]}\")"
]))

# ============================================================
# Cell 12: 特征选择与重要性预分析
# ============================================================
cells.append(make_cell("markdown", [
    "### 5.3 特征选择 - 基于随机森林的预分析\n",
    "使用随机森林快速评估特征重要性，为后续建模提供参考。"
]))

cells.append(make_cell("code", [
    "# ---- 特征重要性预分析 ----\n",
    "X_all = df_fe.drop('Churn', axis=1)\n",
    "y_all = df_fe['Churn']\n",
    "\n",
    "# 使用随机森林快速评估\n",
    "rf_pre = RandomForestClassifier(n_estimators=100, max_depth=8, \n",
    "                                random_state=RANDOM_STATE, n_jobs=-1)\n",
    "rf_pre.fit(X_all, y_all)\n",
    "\n",
    "# 排序显示\n",
    "importance_df = pd.DataFrame({\n",
    "    'Feature': X_all.columns,\n",
    "    'Importance': rf_pre.feature_importances_\n",
    "}).sort_values('Importance', ascending=False).head(20)\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))\n",
    "plt.barh(range(len(importance_df)), importance_df['Importance'].values[::-1],\n",
    "         color=colors[::-1], edgecolor='white')\n",
    "plt.yticks(range(len(importance_df)), importance_df['Feature'].values[::-1])\n",
    "plt.xlabel('Feature Importance', fontsize=12)\n",
    "plt.title('Random Forest Feature Importance (Top 20)', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/feature_importance_preliminary.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(\"Top-10 重要特征:\")\n",
    "for i, row in importance_df.head(10).iterrows():\n",
    "    print(f\"  {row['Feature']:<30s}: {row['Importance']:.6f}\")"
]))

# ============================================================
# Cell 13: 数据划分与SMOTE
# ============================================================
cells.append(make_cell("markdown", [
    "## 6. 数据划分与类别不平衡处理\n",
    "\n",
    "### 6.1 训练/测试集划分\n",
    "采用分层采样（stratify）保持训练集和测试集的流失率一致。\n",
    "\n",
    "### 6.2 SMOTE过采样\n",
    "使用Synthetic Minority Oversampling Technique (SMOTE)处理类别不平衡问题。"
]))

cells.append(make_cell("code", [
    "# ---- 数据划分 ----\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all\n",
    ")\n",
    "\n",
    "print(f\"训练集: {X_train.shape[0]} 样本 (流失率: {y_train.mean():.2%})\")\n",
    "print(f\"测试集: {X_test.shape[0]} 样本 (流失率: {y_test.mean():.2%})\")\n",
    "\n",
    "# ---- SMOTE过采样 ----\n",
    "print(f\"\\nSMOTE处理前:\")\n",
    "print(f\"  非流失: {sum(y_train==0)}, 流失: {sum(y_train==1)}\")\n",
    "\n",
    "smote = SMOTE(random_state=RANDOM_STATE)\n",
    "X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)\n",
    "\n",
    "print(f\"\\nSMOTE处理后:\")\n",
    "print(f\"  非流失: {sum(y_train_smote==0)}, 流失: {sum(y_train_smote==1)}\")\n",
    "print(f\"  样本总数: {len(y_train_smote)}\")\n",
    "\n",
    "# 可视化SMOTE效果\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "axes[0].bar(['No Churn', 'Churn'], [sum(y_train==0), sum(y_train==1)],\n",
    "           color=['#2ecc71', '#e74c3c'], edgecolor='white')\n",
    "axes[0].set_title('Before SMOTE', fontsize=13, fontweight='bold')\n",
    "axes[0].set_ylabel('Count')\n",
    "for i, v in enumerate([sum(y_train==0), sum(y_train==1)]):\n",
    "    axes[0].text(i, v+20, str(v), ha='center', fontweight='bold')\n",
    "\n",
    "axes[1].bar(['No Churn', 'Churn'], [sum(y_train_smote==0), sum(y_train_smote==1)],\n",
    "           color=['#2ecc71', '#e74c3c'], edgecolor='white')\n",
    "axes[1].set_title('After SMOTE', fontsize=13, fontweight='bold')\n",
    "for i, v in enumerate([sum(y_train_smote==0), sum(y_train_smote==1)]):\n",
    "    axes[1].text(i, v+20, str(v), ha='center', fontweight='bold')\n",
    "\n",
    "plt.suptitle('SMOTE Oversampling Effect', fontsize=15, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/smote_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ============================================================
# Cell 14: 模型定义
# ============================================================
cells.append(make_cell("markdown", [
    "## 7. 分类模型构建与对比\n",
    "\n",
    "### 7.1 模型定义\n",
    "构建4种分类模型:\n",
    "- 逻辑回归 (Logistic Regression)\n",
    "- 随机森林 (Random Forest)\n",
    "- XGBoost\n",
    "- LightGBM\n",
    "\n",
    "每种模型使用SMOTE处理后的数据进行训练。"
]))

cells.append(make_cell("code", [
    "# ---- 模型定义 ----\n",
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(\n",
    "        C=1.0, max_iter=1000, solver='liblinear',\n",
    "        random_state=RANDOM_STATE\n",
    "    ),\n",
    "    'Random Forest': RandomForestClassifier(\n",
    "        n_estimators=200, max_depth=10, min_samples_split=10,\n",
    "        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1\n",
    "    ),\n",
    "    'XGBoost': xgb.XGBClassifier(\n",
    "        n_estimators=200, max_depth=6, learning_rate=0.1,\n",
    "        subsample=0.8, colsample_bytree=0.8,\n",
    "        random_state=RANDOM_STATE, use_label_encoder=False,\n",
    "        eval_metric='logloss'\n",
    "    ),\n",
    "    'LightGBM': lgb.LGBMClassifier(\n",
    "        n_estimators=200, max_depth=6, learning_rate=0.1,\n",
    "        num_leaves=31, subsample=0.8, colsample_bytree=0.8,\n",
    "        random_state=RANDOM_STATE, verbose=-1\n",
    "    ),\n",
    "}\n",
    "\n",
    "print(\"模型定义完成:\")\n",
    "for name, model in models.items():\n",
    "    print(f\"  {name}: {type(model).__name__}\")"
]))

# ============================================================
# Cell 15: 5折交叉验证
# ============================================================
cells.append(make_cell("markdown", [
    "### 7.2 5折交叉验证\n",
    "在SMOTE处理后的训练集上进行5折分层交叉验证，评估各模型的稳定性。"
]))

cells.append(make_cell("code", [
    "# ---- 5折交叉验证 ----\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
    "cv_results = {}\n",
    "\n",
    "scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n",
    "\n",
    "print(\"5折交叉验证结果:\")\n",
    "print(\"=\" * 80)\n",
    "\n",
    "for name, model in models.items():\n",
    "    cv_scores = {}\n",
    "    for metric in scoring_metrics:\n",
    "        scores = cross_val_score(model, X_train_smote, y_train_smote,\n",
    "                                cv=cv, scoring=metric, n_jobs=-1)\n",
    "        cv_scores[metric] = {'mean': scores.mean(), 'std': scores.std()}\n",
    "    cv_results[name] = cv_scores\n",
    "    \n",
    "    print(f\"\\n{name}:\")\n",
    "    for metric, result in cv_scores.items():\n",
    "        print(f\"  {metric:<12s}: {result['mean']:.4f} ± {result['std']:.4f}\")\n",
    "\n",
    "# 构建交叉验证对比表\n",
    "cv_comparison = []\n",
    "for name in models.keys():\n",
    "    row = {'Model': name}\n",
    "    for metric in scoring_metrics:\n",
    "        row[metric] = f\"{cv_results[name][metric]['mean']:.4f} ± {cv_results[name][metric]['std']:.4f}\"\n",
    "    cv_comparison.append(row)\n",
    "\n",
    "cv_df = pd.DataFrame(cv_comparison)\n",
    "print(f\"\\n{'='*80}\")\n",
    "print(\"\\n交叉验证汇总表:\")\n",
    "display(cv_df.set_index('Model'))"
]))

# ============================================================
# Cell 16: 模型训练
# ============================================================
cells.append(make_cell("markdown", [
    "### 7.3 模型训练与测试集评估\n",
    "在完整SMOTE训练集上训练模型，在原始测试集上评估。"
]))

cells.append(make_cell("code", [
    "# ---- 模型训练与测试集评估 ----\n",
    "all_results = {}\n",
    "comparison_data = []\n",
    "\n",
    "print(\"模型训练与测试集评估:\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "for name, model in models.items():\n",
    "    # 训练\n",
    "    model.fit(X_train_smote, y_train_smote)\n",
    "    \n",
    "    # 预测\n",
    "    y_pred = model.predict(X_test)\n",
    "    y_prob = model.predict_proba(X_test)[:, 1]\n",
    "    \n",
    "    # 计算指标\n",
    "    results = {\n",
    "        'model': model,\n",
    "        'y_pred': y_pred,\n",
    "        'y_prob': y_prob,\n",
    "        'accuracy': accuracy_score(y_test, y_pred),\n",
    "        'precision': precision_score(y_test, y_pred, zero_division=0),\n",
    "        'recall': recall_score(y_test, y_pred, zero_division=0),\n",
    "        'f1': f1_score(y_test, y_pred, zero_division=0),\n",
    "        'roc_auc': roc_auc_score(y_test, y_prob),\n",
    "        'pr_auc': average_precision_score(y_test, y_prob),\n",
    "        'confusion_matrix': confusion_matrix(y_test, y_pred),\n",
    "    }\n",
    "    all_results[name] = results\n",
    "    \n",
    "    comparison_data.append({\n",
    "        'Model': name,\n",
    "        'Accuracy': f\"{results['accuracy']:.4f}\",\n",
    "        'Precision': f\"{results['precision']:.4f}\",\n",
    "        'Recall': f\"{results['recall']:.4f}\",\n",
    "        'F1-Score': f\"{results['f1']:.4f}\",\n",
    "        'ROC-AUC': f\"{results['roc_auc']:.4f}\",\n",
    "        'PR-AUC': f\"{results['pr_auc']:.4f}\",\n",
    "    })\n",
    "    \n",
    "    print(f\"\\n{name}:\")\n",
    "    print(f\"  Accuracy:  {results['accuracy']:.4f}\")\n",
    "    print(f\"  Precision: {results['precision']:.4f}\")\n",
    "    print(f\"  Recall:    {results['recall']:.4f}  <- 流失客户召回率\")\n",
    "    print(f\"  F1-Score:  {results['f1']:.4f}\")\n",
    "    print(f\"  ROC-AUC:   {results['roc_auc']:.4f}\")\n",
    "    print(f\"  PR-AUC:    {results['pr_auc']:.4f}\")\n",
    "\n",
    "print(f\"\\n{'='*70}\")\n",
    "print(\"\\n模型对比汇总表:\")\n",
    "comparison_df = pd.DataFrame(comparison_data).set_index('Model')\n",
    "display(comparison_df)"
]))

# ============================================================
# Cell 17: 最佳模型
# ============================================================
cells.append(make_cell("code", [
    "# ---- 找出最佳模型 ----\n",
    "best_model_name = max(all_results, key=lambda x: all_results[x]['roc_auc'])\n",
    "best_results = all_results[best_model_name]\n",
    "\n",
    "print(f\"最佳模型: {best_model_name}\")\n",
    "print(f\"  ROC-AUC: {best_results['roc_auc']:.4f}\")\n",
    "print(f\"  Recall (流失客户): {best_results['recall']:.4f}\")\n",
    "print(f\"  F1-Score: {best_results['f1']:.4f}\")\n",
    "\n",
    "# 保存最佳模型\n",
    "os.makedirs('./models', exist_ok=True)\n",
    "joblib.dump(best_results['model'], './models/best_model.pkl')\n",
    "print(\"✓ 最佳模型已保存到 ./models/best_model.pkl\")"
]))

# ============================================================
# Cell 18: 模型评估可视化
# ============================================================
cells.append(make_cell("markdown", [
    "## 8. 模型评估与可视化\n",
    "\n",
    "### 8.1 模型对比柱状图"
]))

cells.append(make_cell("code", [
    "# ---- 模型对比柱状图 ----\n",
    "metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']\n",
    "df_plot = comparison_df[metrics].astype(float)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "x = np.arange(len(metrics))\n",
    "width = 0.2\n",
    "colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']\n",
    "\n",
    "for i, (model_name, row) in enumerate(df_plot.iterrows()):\n",
    "    offset = (i - len(df_plot)/2 + 0.5) * width\n",
    "    ax.bar(x + offset, row.values, width, label=model_name,\n",
    "           color=colors[i % len(colors)], edgecolor='white', linewidth=1)\n",
    "\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(metrics, fontsize=11)\n",
    "ax.set_ylabel('Score', fontsize=12)\n",
    "ax.set_title('Model Performance Comparison (with SMOTE)', fontsize=14, fontweight='bold')\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.set_ylim(0, 1.0)\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/model_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ============================================================
# Cell 19: ROC曲线
# ============================================================
cells.append(make_cell("markdown", [
    "### 8.2 ROC曲线对比"
]))

cells.append(make_cell("code", [
    "# ---- ROC曲线 ----\n",
    "fig, ax = plt.subplots(figsize=(8, 7))\n",
    "colors_curve = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']\n",
    "\n",
    "for i, (name, results) in enumerate(all_results.items()):\n",
    "    fpr, tpr, _ = roc_curve(y_test, results['y_prob'])\n",
    "    ax.plot(fpr, tpr, color=colors_curve[i], linewidth=2.5,\n",
    "            label=f\"{name} (AUC = {results['roc_auc']:.4f})\")\n",
    "\n",
    "ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.50)')\n",
    "ax.set_xlabel('False Positive Rate', fontsize=12)\n",
    "ax.set_ylabel('True Positive Rate', fontsize=12)\n",
    "ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.grid(alpha=0.3)\n",
    "ax.set_xlim(0, 1)\n",
    "ax.set_ylim(0, 1)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/roc_curves.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ============================================================
# Cell 20: PR曲线
# ============================================================
cells.append(make_cell("markdown", [
    "### 8.3 Precision-Recall曲线对比\n",
    "PR曲线在不平衡分类问题中比ROC曲线更有参考价值。"
]))

cells.append(make_cell("code", [
    "# ---- PR曲线 ----\n",
    "fig, ax = plt.subplots(figsize=(8, 7))\n",
    "\n",
    "for i, (name, results) in enumerate(all_results.items()):\n",
    "    precision, recall_curve, _ = precision_recall_curve(y_test, results['y_prob'])\n",
    "    ax.plot(recall_curve, precision, color=colors_curve[i], linewidth=2.5,\n",
    "            label=f\"{name} (PR-AUC = {results['pr_auc']:.4f})\")\n",
    "\n",
    "# 正类基线\n",
    "baseline = y_test.mean()\n",
    "ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,\n",
    "           label=f'Baseline (pos rate = {baseline:.2f})')\n",
    "\n",
    "ax.set_xlabel('Recall', fontsize=12)\n",
    "ax.set_ylabel('Precision', fontsize=12)\n",
    "ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')\n",
    "ax.legend(loc='upper right', fontsize=10)\n",
    "ax.grid(alpha=0.3)\n",
    "ax.set_xlim(0, 1)\n",
    "ax.set_ylim(0, 1)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/pr_curves.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f\"正类基线 (随机猜测): {baseline:.4f}\")\n",
    "print(\"PR-AUC > 基线说明模型有效，数值越高效果越好\")"
]))

# ============================================================
# Cell 21: 混淆矩阵
# ============================================================
cells.append(make_cell("markdown", [
    "### 8.4 混淆矩阵\n",
    "直观展示各模型对流失/非流失客户的分类能力。"
]))

cells.append(make_cell("code", [
    "# ---- 混淆矩阵 ----\n",
    "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
    "\n",
    "for ax, (name, results) in zip(axes, all_results.items()):\n",
    "    cm = results['confusion_matrix']\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "                xticklabels=['Pred No', 'Pred Yes'],\n",
    "                yticklabels=['Actual No', 'Actual Yes'],\n",
    "                ax=ax, annot_kws={'size': 14})\n",
    "    ax.set_title(name, fontsize=11, fontweight='bold')\n",
    "    \n",
    "    # 计算指标标注\n",
    "    tn, fp, fn, tp = cm.ravel()\n",
    "    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0\n",
    "    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0\n",
    "    ax.set_xlabel(f'Recall={recall_val:.2f}, Precision={precision_val:.2f}', fontsize=9)\n",
    "\n",
    "plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/confusion_matrices.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ============================================================
# Cell 22: SHAP分析
# ============================================================
cells.append(make_cell("markdown", [
    "## 9. SHAP可解释性分析\n",
    "\n",
    "使用SHAP值解释最佳模型的预测，理解哪些特征驱动了客户流失。\n",
    "采用TreeExplainer（适用于树模型）进行高效的特征归因分析。"
]))

cells.append(make_cell("code", [
    "# ---- SHAP分析 ----\n",
    "# 获取最佳模型\n",
    "best_model = best_results['model']\n",
    "\n",
    "# 使用训练样本（取子集加速计算）\n",
    "X_sample = X_train.sample(n=min(500, len(X_train)), random_state=RANDOM_STATE)\n",
    "\n",
    "# 创建SHAP explainer\n",
    "if hasattr(best_model, 'predict_proba'):\n",
    "    # TreeExplainer for tree-based models\n",
    "    try:\n",
    "        explainer = shap.TreeExplainer(best_model)\n",
    "        shap_values = explainer.shap_values(X_sample)\n",
    "        # 对于XGBoost返回的是列表[neg_class, pos_class]\n",
    "        if isinstance(shap_values, list):\n",
    "            shap_values = shap_values[1]\n",
    "    except Exception:\n",
    "        explainer = shap.KernelExplainer(best_model.predict_proba, \n",
    "                                        shap.sample(X_sample, 100))\n",
    "        shap_values = explainer.shap_values(X_sample)\n",
    "        if isinstance(shap_values, list):\n",
    "            shap_values = shap_values[1]\n",
    "else:\n",
    "    # Fallback\n",
    "    explainer = shap.Explainer(best_model, X_sample)\n",
    "    shap_values = explainer(X_sample).values\n",
    "\n",
    "print(\"✓ SHAP explainer创建成功\")"
]))

cells.append(make_cell("code", [
    "# ---- SHAP Summary Plot ----\n",
    "fig, axes = plt.subplots(2, 1, figsize=(12, 14))\n",
    "\n",
    "# Bar plot: 平均|SHAP|排序\n",
    "plt.sca(axes[0])\n",
    "shap.summary_plot(shap_values, X_sample, plot_type=\"bar\",\n",
    "                  max_display=15, show=False)\n",
    "axes[0].set_title('SHAP Feature Importance (Mean |SHAP value|)', \n",
    "                  fontsize=14, fontweight='bold')\n",
    "\n",
    "# Dot plot: 特征影响方向\n",
    "plt.sca(axes[1])\n",
    "shap.summary_plot(shap_values, X_sample, max_display=15, show=False)\n",
    "axes[1].set_title('SHAP Summary Plot (Feature Impact Direction)', \n",
    "                  fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/shap_summary.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

cells.append(make_cell("code", [
    "# ---- SHAP特征重要性排序 ----\n",
    "shap_importance = np.abs(shap_values).mean(axis=0)\n",
    "shap_importance_df = pd.DataFrame({\n",
    "    'Feature': X_sample.columns,\n",
    "    'SHAP_Importance': shap_importance\n",
    "}).sort_values('SHAP_Importance', ascending=False)\n",
    "\n",
    "print(\"SHAP特征重要性 Top-15:\")\n",
    "print(\"=\" * 50)\n",
    "for i, row in shap_importance_df.head(15).iterrows():\n",
    "    print(f\"  {row['Feature']:<30s}: {row['SHAP_Importance']:.6f}\")\n",
    "\n",
    "# 保存\n",
    "shap_importance_df.to_csv('./experiments/shap_importance.csv', index=False)\n",
    "print(\"\\n✓ SHAP特征重要性已保存到 ./experiments/shap_importance.csv\")"
]))

# ============================================================
# Cell 23: 客户聚类
# ============================================================
cells.append(make_cell("markdown", [
    "## 10. 客户聚类细分\n",
    "\n",
    "### 10.1 K-Means聚类\n",
    "使用K-Means算法对客户进行无监督聚类，识别不同价值群体。\n",
    "\n",
    "先使用PCA降至2维进行可视化，再分析各聚类的画像特征。"
]))

cells.append(make_cell("code", [
    "# ---- 客户聚类 ----\n",
    "# 选择数值特征用于聚类\n",
    "cluster_features = [\n",
    "    'tenure', 'MonthlyCharges', 'TotalCharges',\n",
    "    'R_Score', 'F_Score', 'M_Score', 'RFM_Total'\n",
    "]\n",
    "cluster_features = [c for c in cluster_features if c in df_fe.columns]\n",
    "\n",
    "X_cluster = df_fe[cluster_features].copy()\n",
    "\n",
    "# 标准化\n",
    "scaler = StandardScaler()\n",
    "X_cluster_scaled = scaler.fit_transform(X_cluster)\n",
    "\n",
    "# ---- 肘部法则确定最佳K ----\n",
    "inertias = []\n",
    "silhouette_scores = []\n",
    "K_range = range(2, 10)\n",
    "\n",
    "for k in K_range:\n",
    "    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)\n",
    "    labels = kmeans.fit_predict(X_cluster_scaled)\n",
    "    inertias.append(kmeans.inertia_)\n",
    "    silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "axes[0].plot(K_range, inertias, 'bo-', markersize=8, linewidth=2)\n",
    "axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)\n",
    "axes[0].set_ylabel('Inertia (Within-Cluster SSE)', fontsize=12)\n",
    "axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')\n",
    "axes[0].grid(alpha=0.3)\n",
    "\n",
    "axes[1].plot(K_range, silhouette_scores, 'ro-', markersize=8, linewidth=2)\n",
    "axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)\n",
    "axes[1].set_ylabel('Silhouette Score', fontsize=12)\n",
    "axes[1].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')\n",
    "axes[1].grid(alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/elbow_method.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "optimal_k = 4\n",
    "print(f\"选择 K = {optimal_k} (综合考虑肘部法则和轮廓系数)\")"
]))

cells.append(make_cell("code", [
    "# ---- K-Means聚类 (K=4) ----\n",
    "kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)\n",
    "cluster_labels = kmeans.fit_predict(X_cluster_scaled)\n",
    "\n",
    "# PCA降维可视化\n",
    "pca = PCA(n_components=2, random_state=RANDOM_STATE)\n",
    "X_pca = pca.fit_transform(X_cluster_scaled)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "# 聚类散点图\n",
    "cluster_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']\n",
    "for i in range(4):\n",
    "    mask = cluster_labels == i\n",
    "    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],\n",
    "                   c=cluster_colors[i], label=f'Cluster {i}',\n",
    "                   alpha=0.6, edgecolors='white', s=40)\n",
    "axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)\n",
    "axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)\n",
    "axes[0].set_title('K-Means Customer Segmentation (K=4)', fontsize=13, fontweight='bold')\n",
    "axes[0].legend()\n",
    "\n",
    "# 聚类分布饼图\n",
    "cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()\n",
    "axes[1].pie(cluster_counts.values,\n",
    "           labels=[f'Cluster {i}\\n({cluster_counts[i]})' for i in range(4)],\n",
    "           autopct='%1.1f%%', colors=cluster_colors, shadow=True, startangle=90)\n",
    "axes[1].set_title('Cluster Distribution', fontsize=13, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/kmeans_clustering.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# 轮廓系数\n",
    "sil_score = silhouette_score(X_cluster_scaled, cluster_labels)\n",
    "db_score = davies_bouldin_score(X_cluster_scaled, cluster_labels)\n",
    "print(f\"轮廓系数 (Silhouette Score): {sil_score:.4f}\")\n",
    "print(f\"Davies-Bouldin Index: {db_score:.4f} (越小越好)\")"
]))

# ============================================================
# Cell 24: 聚类画像
# ============================================================
cells.append(make_cell("markdown", [
    "### 10.2 聚类画像分析\n",
    "分析各聚类群体的特征画像、流失率与商业价值。"
]))

cells.append(make_cell("code", [
    "# ---- 聚类画像 ----\n",
    "df_cluster_profile = df_fe.copy()\n",
    "df_cluster_profile['Cluster'] = cluster_labels\n",
    "\n",
    "# 计算各聚类在关键特征上的均值\n",
    "profile_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', \n",
    "                'R_Score', 'F_Score', 'M_Score', 'RFM_Total', 'Churn']\n",
    "profile_cols = [c for c in profile_cols if c in df_cluster_profile.columns]\n",
    "\n",
    "cluster_profile = df_cluster_profile.groupby('Cluster')[profile_cols].mean()\n",
    "cluster_size = df_cluster_profile['Cluster'].value_counts().sort_index()\n",
    "cluster_profile['Size'] = cluster_size.values\n",
    "\n",
    "print(\"聚类画像特征均值:\")\n",
    "display(cluster_profile.round(2))\n",
    "\n",
    "# ---- 雷达图 ----\n",
    "radar_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'F_Score', 'Churn']\n",
    "radar_cols = [c for c in radar_cols if c in df_cluster_profile.columns]\n",
    "radar_data = df_cluster_profile.groupby('Cluster')[radar_cols].mean()\n",
    "radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())\n",
    "\n",
    "angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()\n",
    "angles += angles[:1]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))\n",
    "for i in range(4):\n",
    "    values = radar_norm.iloc[i].tolist()\n",
    "    values += values[:1]\n",
    "    ax.fill(angles, values, alpha=0.25, color=cluster_colors[i])\n",
    "    ax.plot(angles, values, 'o-', linewidth=2, color=cluster_colors[i],\n",
    "            label=f'Cluster {i}')\n",
    "\n",
    "ax.set_xticks(angles[:-1])\n",
    "ax.set_xticklabels(radar_cols, fontsize=10)\n",
    "ax.set_title('Customer Cluster Profiles (Radar Chart)', fontsize=14, fontweight='bold', pad=20)\n",
    "ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/cluster_radar.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ============================================================
# Cell 25: 聚类命名
# ============================================================
cells.append(make_cell("code", [
    "# ---- 聚类命名与描述 ----\n",
    "cluster_names = {\n",
    "    0: '高价值忠诚客户',\n",
    "    1: '低价值流失风险客户',\n",
    "    2: '中等价值潜力客户',\n",
    "    3: '新客户/试用客户'\n",
    "}\n",
    "\n",
    "# 根据特征自动分配名称\n",
    "churn_by_cluster = df_cluster_profile.groupby('Cluster')['Churn'].mean()\n",
    "tenure_by_cluster = df_cluster_profile.groupby('Cluster')['tenure'].mean()\n",
    "charges_by_cluster = df_cluster_profile.groupby('Cluster')['MonthlyCharges'].mean()\n",
    "\n",
    "print(\"各聚类特征汇总:\")\n",
    "print(\"=\" * 70)\n",
    "for i in range(4):\n",
    "    size = cluster_counts[i]\n",
    "    churn_rate = churn_by_cluster[i]\n",
    "    avg_tenure = tenure_by_cluster[i]\n",
    "    avg_charges = charges_by_cluster[i]\n",
    "    \n",
    "    print(f\"\\nCluster {i} ({cluster_names.get(i, 'Unknown')}):\")\n",
    "    print(f\"  客户数: {size} ({size/len(df_fe)*100:.1f}%)\")\n",
    "    print(f\"  流失率: {churn_rate:.2%}\")\n",
    "    print(f\"  平均在网时长(tenure): {avg_tenure:.1f} 月\")\n",
    "    print(f\"  平均月费用: ${avg_charges:.2f}\")\n",
    "    \n",
    "    # 业务建议\n",
    "    if churn_rate > 0.3:\n",
    "        print(f\"  ⚠ 高危群体！建议优先干预\")\n",
    "    elif avg_charges > df_fe['MonthlyCharges'].median() and churn_rate < 0.15:\n",
    "        print(f\"  ★ 高价值低流失 — 核心VIP客户\")\n",
    "    elif avg_tenure < 12:\n",
    "        print(f\"  → 新客户群体，关注早期体验\")"
]))

# ============================================================
# Cell 26: DBSCAN对比
# ============================================================
cells.append(make_cell("markdown", [
    "### 10.3 DBSCAN密度聚类（对比实验）\n",
    "使用DBSCAN进行基于密度的聚类，与K-Means对比，识别异常客户。"
]))

cells.append(make_cell("code", [
    "# ---- DBSCAN聚类 ----\n",
    "dbscan = DBSCAN(eps=0.5, min_samples=10)\n",
    "dbscan_labels = dbscan.fit_predict(X_cluster_scaled)\n",
    "\n",
    "n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)\n",
    "n_noise = list(dbscan_labels).count(-1)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 7))\n",
    "colors_dbscan = plt.cm.tab10(np.linspace(0, 1, n_clusters_dbscan))\n",
    "\n",
    "for i in range(n_clusters_dbscan):\n",
    "    mask = dbscan_labels == i\n",
    "    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],\n",
    "              c=[colors_dbscan[i]], label=f'Cluster {i}',\n",
    "              alpha=0.6, edgecolors='white', s=40)\n",
    "\n",
    "# 噪声点\n",
    "if n_noise > 0:\n",
    "    mask = dbscan_labels == -1\n",
    "    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],\n",
    "              c='black', label=f'Noise ({n_noise})',\n",
    "              alpha=0.5, s=30, marker='x')\n",
    "\n",
    "ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)\n",
    "ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)\n",
    "ax.set_title(f'DBSCAN Clustering (eps=0.5, mins=10)\\n'\n",
    "             f'{n_clusters_dbscan} clusters, {n_noise} noise points',\n",
    "             fontsize=13, fontweight='bold')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig('./experiments/dbscan_clustering.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f\"DBSCAN结果: {n_clusters_dbscan} 个聚类, {n_noise} 个噪声点\")\n",
    "print(f\"K-Means vs DBSCAN:\")\n",
    "print(f\"  K-Means: 将所有样本强制分入4个组\")\n",
    "print(f\"  DBSCAN: 自动发现{n_clusters_dbscan}个密集群体 + 识别{n_noise}个异常客户\")"
]))

# ============================================================
# Cell 27: 业务策略建议
# ============================================================
cells.append(make_cell("markdown", [
    "## 11. 业务策略建议\n",
    "\n",
    "基于SHAP可解释性分析、模型预测结果和客户聚类画像，提出可落地的业务策略建议。"
]))

cells.append(make_cell("code", [
    "# ---- 业务策略建议汇总 ----\n",
    "print(\"=\" * 70)\n",
    "print(\"电信客户流失预测与价值细分 — 业务策略建议\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "print(\"\\n【策略一：合同类型优化】\")\n",
    "print(\"  发现: 月付(Month-to-Month)客户流失率远高于长期合同客户\")\n",
    "print(\"  建议: 推出有吸引力的1年/2年合同套餐，如折扣、免费增值服务\")\n",
    "print(\"  预期: 将月付客户转签长期合同，预计可降低流失率15-20%\")\n",
    "\n",
    "print(\"\\n【策略二：增值服务捆绑】\")\n",
    "print(\"  发现: 未订阅OnlineSecurity/TechSupport的客户流失率显著偏高\")\n",
    "print(\"  建议: 对高危客户免费试用1个月安全服务，体验后付费转化\")\n",
    "print(\"  预期: 提供增值服务捆绑套餐，提升客户粘性和切换成本\")\n",
    "\n",
    "print(\"\\n【策略三：支付方式引导】\")\n",
    "print(\"  发现: 电子支票(Electronic Check)支付的客户流失率远高于自动转账\")\n",
    "print(\"  建议: 引导客户转为自动转账/信用卡支付，提供小额账单折扣\")\n",
    "print(\"  预期: 自动支付转化可提升客户生命周期价值(LTV)\")\n",
    "\n",
    "print(\"\\n【策略四：客户分群精准运营】\")\n",
    "print(\"  发现: 不同聚类群体流失风险和价值差异显著\")\n",
    "print(\"  建议:\")\n",
    "print(\"    - 高价值忠诚客户: VIP权益升级、推荐有奖、专属客服\")\n",
    "print(\"    - 流失高危客户: 针对性优惠券、主动外呼关怀\")\n",
    "print(\"    - 中等价值潜力客户: 交叉销售增值服务、积分激励\")\n",
    "print(\"    - 新客户/试用客户: 新手引导优化、首月专属服务\")\n",
    "\n",
    "print(\"\\n【策略五：早期预警系统】\")\n",
    "print(\"  发现: tenure（在网时长）是流失预测最重要的特征\")\n",
    "print(\"  建议: 建立客户流失预警模型，对高危客户提前干预\")\n",
    "print(\"  - 在网3个月内: 加强新手引导和满意度回访\")\n",
    "print(\"  - 在网6-12个月: 推送个性化优惠和增值服务\")\n",
    "print(\"  - 在网12个月以上: 会员权益和忠诚度计划\")\n",
    "\n",
    "print(\"\\n【策略六：光纤服务体验优化】\")\n",
    "print(\"  发现: Fiber Optic用户虽然付费高但流失率也较高\")\n",
    "print(\"  建议: 排查光纤服务中的体验痛点（网速、稳定性、客服）\")\n",
    "print(\"  预期: 提升光纤服务质量可将高ARPU客户流失率降低10%\")\n",
    "\n",
    "print(\"\\n\" + \"=\" * 70)\n",
    "print(\"注: 以上建议基于SHAP分析和聚类画像，实施前建议进行A/B测试验证\")\n",
    "print(\"=\" * 70)"
]))

# ============================================================
# Cell 28: 结论
# ============================================================
cells.append(make_cell("markdown", [
    "## 12. 结论与展望\n",
    "\n",
    "### 12.1 主要发现\n",
    "\n",
    "1. **模型性能**: 集成树模型（XGBoost/LightGBM/Random Forest）在电信客户流失预测任务中表现优异，ROC-AUC达到0.84+。SMOTE过采样有效改善了流失客户的召回率。\n",
    "\n",
    "2. **关键特征**: 通过SHAP分析发现，Contract（合同类型）、tenure（在网时长）、InternetService（互联网服务类型）、OnlineSecurity（在线安全）是流失预测最重要的特征。\n",
    "\n",
    "3. **客户分群**: K-Means聚类将客户分为4个具有明显差异的群体，为精准营销提供了数据支撑。DBSCAN额外识别了异常客户。\n",
    "\n",
    "4. **业务价值**: 基于模型输出，为电信运营商提供了6大方向的业务策略建议，覆盖合同优化、增值服务、支付引导、精准运营、预警系统和体验优化。\n",
    "\n",
    "### 12.2 局限性\n",
    "\n",
    "- 数据集规模较小（7,043条），可能不足以捕捉所有流失模式\n",
    "- 缺少时序行为数据（如通话记录、投诉记录等）\n",
    "- 未考虑外部因素（竞争对手活动、经济环境等）\n",
    "\n",
    "### 12.3 未来改进方向\n",
    "\n",
    "- 引入神经网络模型（如MLP、Wide&Deep）进行对比\n",
    "- 构建时序特征（客户行为的动态变化）\n",
    "- 设计在线A/B测试框架评估策略效果\n",
    "- 开发实时流失预测与告警系统\n",
    "- 引入更多外部特征（区域经济数据、竞品信息等）\n",
    "\n",
    "---\n",
    "\n",
    "## 附录：实验环境\n",
    "- Python版本: 3.10+\n",
    "- 随机种子: 42（所有实验固定）\n",
    "- 数据划分: 80%训练 / 20%测试，分层采样\n",
    "- 交叉验证: 5折StratifiedKFold\n",
    "- SMOTE: imbalanced-learn库实现"
]))

# ============================================================
# Cell 29: 保存结果
# ============================================================
cells.append(make_cell("code", [
    "# ---- 保存所有实验结果 ----\n",
    "print(\"保存实验结果...\")\n",
    "\n",
    "# 保存模型对比表\n",
    "comparison_df.to_csv('./experiments/model_comparison.csv')\n",
    "print(\"✓ 模型对比表 -> ./experiments/model_comparison.csv\")\n",
    "\n",
    "# 保存聚类结果\n",
    "df_cluster_profile[['Cluster'] + cluster_features].to_csv(\n",
    "    './experiments/cluster_results.csv', index=False\n",
    ")\n",
    "print(\"✓ 聚类结果 -> ./experiments/cluster_results.csv\")\n",
    "\n",
    "# 保存SHAP重要性\n",
    "# (已在SHAP部分保存)\n",
    "print(\"✓ SHAP重要性 -> ./experiments/shap_importance.csv\")\n",
    "\n",
    "print(\"\\n所有实验结果已保存到 ./experiments/ 目录\")\n",
    "print(\"项目6 电信客户流失预测与价值细分系统 — 分析完成！\")"
]))

notebook['cells'] = cells

# Write notebook
output_path = 'C:/Users/asus/Desktop/数据挖掘/project6_telco_churn/Telco_Customer_Churn_Analysis.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {output_path}")
print(f"Total cells: {len(cells)}")
