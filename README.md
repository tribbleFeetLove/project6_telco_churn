# 电信客户流失预测与价值细分系统

## 项目描述

本项目构建"流失预警 + 客户分群"双模块分析系统，基于电信客户行为数据，完整经历数据清洗、特征工程、模型构建、结果解释的全流程，最终输出可落地的业务策略建议。

## 数据集

使用 Kaggle **Telco Customer Churn** 数据集：
- 7,043 条记录，21 个特征
- 目标变量：`Churn`（是否流失）
- 数据来源：https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

1. 从 Kaggle 下载数据集：`WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. 将数据文件放入 `data/raw/` 目录
3. 运行 Jupyter Notebook 即可自动加载

或者直接在 Notebook 中运行，数据将自动从网络加载。

## 运行说明

### 完整分析流程

```bash
jupyter notebook Telco_Customer_Churn_Analysis.ipynb
```

按顺序执行所有单元格即可获得：
1. 数据探索性分析（EDA）
2. 数据预处理与特征工程
3. 分类模型训练与对比
4. 模型评估与可解释性分析
5. 客户聚类细分
6. 业务策略建议

### 单独运行模块

```bash
# 数据预处理
python -m utils.data_utils

# 模型训练
python train.py --config configs/default.yaml

# 模型评估
python evaluate.py --model_path models/best_model.pth
```

## 项目结构

```
project6_telco_churn/
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后的数据
├── models/                  # 模型定义
│   ├── __init__.py
│   └── base_model.py
├── utils/                   # 工具函数
│   ├── data_utils.py        # 数据处理工具
│   ├── model_utils.py       # 模型工具
│   └── visualization.py     # 可视化工具
├── configs/                 # 配置文件
│   └── default.yaml
├── experiments/             # 实验记录
├── Telco_Customer_Churn_Analysis.ipynb  # 主分析流程
├── requirements.txt
└── README.md
```

## 核心技术栈

- **数据预处理**: 缺失值处理、异常检测、SMOTE过采样
- **特征工程**: RFM模型构建、WOE分箱、特征重要性分析
- **分类模型**: 逻辑回归、随机森林、XGBoost、LightGBM
- **无监督挖掘**: K-Means / DBSCAN 客户细分
- **模型评估**: ROC-AUC、Precision-Recall曲线、SHAP值解释

## 实验结果

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |
