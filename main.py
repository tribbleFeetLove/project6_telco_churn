"""
电信客户流失预测与价值细分系统 — 完整分析脚本
可直接运行: python main.py
"""
import os, sys, warnings, json, yaml
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib

# ===== 加载配置 =====
config_path = './configs/default.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f"已加载配置文件: {config_path}")
else:
    cfg = {}
    print("配置文件不存在，使用默认参数")

RANDOM_STATE = cfg.get('data', {}).get('random_state', 42)
np.random.seed(RANDOM_STATE)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
EXPERIMENTS_DIR = cfg.get('logging', {}).get('log_dir', './experiments')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./data/raw', exist_ok=True)


def safe_fillna(df):
    """安全填充NaN：将categorical列转为float再填充"""
    for col in df.columns:
        if df[col].isnull().any():
            try:
                df[col] = df[col].astype(float).fillna(0)
            except Exception:
                df[col] = df[col].fillna(0)
    return df


def load_config_models():
    """从配置中读取模型参数并构建模型字典"""
    train_cfg = cfg.get('training', {})
    model_names = cfg.get('model', {}).get('models',
        ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'])

    model_map = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
    }

    models = {}
    for m in model_names:
        params = train_cfg.get(m, {})
        display_name = model_map.get(m, m)
        if m == 'logistic_regression':
            models[display_name] = LogisticRegression(
                C=params.get('C', 1.0), max_iter=params.get('max_iter', 1000),
                solver=params.get('solver', 'liblinear'), random_state=RANDOM_STATE)
        elif m == 'random_forest':
            models[display_name] = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 200), max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 10),
                min_samples_leaf=params.get('min_samples_leaf', 5),
                random_state=RANDOM_STATE, n_jobs=-1)
        elif m == 'xgboost':
            models[display_name] = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 200), max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1), subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=RANDOM_STATE, eval_metric='logloss')
        elif m == 'lightgbm':
            models[display_name] = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 200), max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1), num_leaves=params.get('num_leaves', 31),
                subsample=params.get('subsample', 0.8), colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=RANDOM_STATE, verbose=-1)
    return models


print("=" * 70)
print("电信客户流失预测与价值细分系统 — 项目6")
print("=" * 70)

# ===== 1. 数据加载 =====
print("\n[1/10] 数据加载...")
data_cfg = cfg.get('data', {})
data_file = data_cfg.get('data_file', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
data_path = os.path.join(data_cfg.get('data_path', './data/raw'), data_file)
if os.path.exists(data_path):
    df_raw = pd.read_csv(data_path)
else:
    fallback_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df_raw = pd.read_csv(fallback_url)
print(f"  数据集: {df_raw.shape[0]} 条 × {df_raw.shape[1]} 列, 流失率: {df_raw['Churn'].eq('Yes').mean():.2%}")

# ===== 2. 数据预处理 =====
print("\n[2/10] 数据预处理...")
df = df_raw.copy()
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_strategy = cfg.get('preprocessing', {}).get('missing_strategy', 'median')
if missing_strategy == 'median':
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
else:
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ===== 3. 特征工程 =====
print("\n[3/10] 特征工程 (RFM + WOE)...")
# RFM
df['R_Score'] = 1 / (df['tenure'] + 1)
svc_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['F_Score'] = sum((df[c] > 0).astype(int) for c in svc_cols if c in df.columns)
df['M_Score'] = df['MonthlyCharges']
df['RFM_Total'] = df['R_Score'] + df['F_Score'] / 9 + df['M_Score'] / df['M_Score'].max()

# WOE
def compute_woe(feature, target, n_bins=10):
    df_t = pd.DataFrame({'f': feature, 't': target})
    if df_t['f'].nunique() > n_bins:
        df_t['bin'] = pd.qcut(df_t['f'], q=n_bins, duplicates='drop')
    else:
        df_t['bin'] = df_t['f']
    g = df_t.groupby('bin').agg(good=('t', lambda x: (x==0).sum()), bad=('t', lambda x: (x==1).sum()))
    tg, tb = (target==0).sum(), (target==1).sum()
    g['woe'] = np.log((g['good']/tg + 1e-6) / (g['bad']/tb + 1e-6))
    return df_t['bin'].map(g['woe'])

woe_bins = cfg.get('feature_engineering', {}).get('woe_bins', 10)
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    df[f'{col}_WOE'] = compute_woe(df[col], df['Churn'], n_bins=woe_bins)

for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].astype(float)
df.fillna(0, inplace=True)
print(f"  特征工程后: {df.shape[1]} 个特征")

# ===== 4. 数据划分与SMOTE =====
print("\n[4/10] 数据划分与SMOTE过采样...")
X = df.drop('Churn', axis=1)
y = df['Churn']
train_ratio = cfg.get('data', {}).get('train_ratio', 0.8)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1-train_ratio, random_state=RANDOM_STATE, stratify=y)
print(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote = safe_fillna(X_train_smote)
print(f"  SMOTE后: {X_train_smote.shape[0]} ({y_train_smote.mean():.1%} 流失)")

# ===== 5. 模型定义 =====
print("\n[5/10] 构建4种分类模型...")
models = load_config_models()

# ===== 6. 交叉验证 =====
print("\n[6/10] 5折交叉验证...")
cv_folds = cfg.get('model', {}).get('cv_folds', 5)
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    row = {'Model': name}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring=metric, n_jobs=-1)
        row[metric] = f"{scores.mean():.4f}±{scores.std():.4f}"
        print(f"  {name:<22s} {metric}: {scores.mean():.4f} ± {scores.std():.4f}")
    cv_rows.append(row)

# ===== 7. 训练与评估 =====
print("\n[7/10] 训练模型与测试集评估...")
all_results = {}
comp_rows = []
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'cm': confusion_matrix(y_test, y_pred),
        'y_prob': y_prob, 'y_pred': y_pred, 'model': model,
    }
    all_results[name] = results
    comp_rows.append({'Model': name, 'Accuracy': results['accuracy'], 'Precision': results['precision'],
        'Recall': results['recall'], 'F1': results['f1'], 'ROC_AUC': results['roc_auc'], 'PR_AUC': results['pr_auc']})
    print(f"  {name:<22s} ROC-AUC={results['roc_auc']:.4f}  Recall={results['recall']:.4f}  F1={results['f1']:.4f}")

comp_df = pd.DataFrame(comp_rows).set_index('Model')

# ===== 8. 可视化 =====
print("\n[8/10] 生成评估图表...")

fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']
x = np.arange(len(metrics))
width = 0.2
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, (name, row) in enumerate(comp_df.iterrows()):
    offset = (i - len(comp_df)/2 + 0.5) * width
    ax.bar(x + offset, [row[m] for m in metrics], width, label=name, color=colors[i % 4], edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.0)
ax.set_title('Model Performance Comparison'); ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/model_comparison.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
for i, (name, r) in enumerate(all_results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=colors[i], linewidth=2.5, label=f"{name} (AUC={r['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/roc_curves.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
for i, (name, r) in enumerate(all_results.items()):
    p, rec, _ = precision_recall_curve(y_test, r['y_prob'])
    ax.plot(rec, p, color=colors[i], linewidth=2.5, label=f"{name} (PR-AUC={r['pr_auc']:.4f})")
ax.axhline(y=y_test.mean(), color='gray', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curves'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/pr_curves.png', dpi=150); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, r) in zip(axes, all_results.items()):
    cm = r['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred No', 'Pred Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax, annot_kws={'size': 14})
    ax.set_title(name, fontweight='bold'); ax.set_xlabel(f'Recall={cm[1,1]/(cm[1,1]+cm[1,0]):.2f}')
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/confusion_matrices.png', dpi=150); plt.close()

best_name = max(all_results, key=lambda x: all_results[x]['roc_auc'])
best_model = all_results[best_name]['model']
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(15), imp[idx][::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1], edgecolor='white')
    ax.set_yticks(range(15)); ax.set_yticklabels(X.columns[idx][::-1])
    ax.set_title(f'Feature Importance ({best_name})'); ax.set_xlabel('Importance')
    plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/feature_importance.png', dpi=150); plt.close()

print(f"  最佳模型: {best_name} (ROC-AUC={all_results[best_name]['roc_auc']:.4f})")
joblib.dump(best_model, './models/best_model.pkl')

# ===== 9. SHAP分析 =====
print("\n[9/10] SHAP可解释性分析...")
X_sample = X_train.sample(n=min(300, len(X_train)), random_state=RANDOM_STATE)
try:
    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
except Exception:
    explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_sample, 50))
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

fig, axes = plt.subplots(2, 1, figsize=(12, 14))
plt.sca(axes[0]); shap.summary_plot(shap_vals, X_sample, plot_type="bar", max_display=15, show=False)
axes[0].set_title('SHAP Feature Importance')
plt.sca(axes[1]); shap.summary_plot(shap_vals, X_sample, max_display=15, show=False)
axes[1].set_title('SHAP Summary Plot')
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/shap_summary.png', dpi=150); plt.close()

# ===== 10. 客户聚类 =====
print("\n[10/10] 客户聚类细分...")
cluster_cfg = cfg.get('clustering', {})
n_clusters = cluster_cfg.get('n_clusters_kmeans', 4)
dbscan_eps = cluster_cfg.get('dbscan_eps', 0.5)
dbscan_min_samples = cluster_cfg.get('dbscan_min_samples', 10)

cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'R_Score', 'F_Score', 'M_Score', 'RFM_Total']
X_cl = df[cluster_features]
X_cl_scaled = StandardScaler().fit_transform(X_cl)

kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
cl_labels = kmeans.fit_predict(X_cl_scaled)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_cl_scaled)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cl_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i in range(n_clusters):
    mask = cl_labels == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=cl_colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='white', s=30)
axes[0].set_title('K-Means Customer Segmentation'); axes[0].legend()
cl_counts = pd.Series(cl_labels).value_counts().sort_index()
axes[1].pie(cl_counts.values, labels=[f'C{i} ({cl_counts[i]})' for i in range(n_clusters)], autopct='%1.1f%%', colors=cl_colors)
axes[1].set_title('Cluster Distribution')
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/kmeans_clustering.png', dpi=150); plt.close()

dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
db_labels = dbscan.fit_predict(X_cl_scaled)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(n_clusters_db):
    ax.scatter(X_pca[db_labels==i, 0], X_pca[db_labels==i, 1], label=f'C{i}', alpha=0.6, s=30)
if n_noise > 0:
    ax.scatter(X_pca[db_labels==-1, 0], X_pca[db_labels==-1, 1], c='black', label=f'Noise({n_noise})', alpha=0.5, s=20, marker='x')
ax.set_title(f'DBSCAN: {n_clusters_db} clusters, {n_noise} noise'); ax.legend()
plt.tight_layout(); plt.savefig(f'{EXPERIMENTS_DIR}/dbscan_clustering.png', dpi=150); plt.close()

df['Cluster'] = cl_labels
cluster_profile = df.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'F_Score']].mean()
cluster_profile['Size'] = cl_counts.values

comp_df.to_csv(f'{EXPERIMENTS_DIR}/model_comparison.csv')
df[['Cluster'] + cluster_features].to_csv(f'{EXPERIMENTS_DIR}/cluster_results.csv', index=False)
shap_mean = np.abs(shap_vals).mean(axis=0)
if shap_mean.ndim > 1:
    shap_mean = shap_mean.flatten()
n_show = min(len(X_sample.columns), len(shap_mean))
shap_imp = pd.DataFrame({
    'Feature': X_sample.columns[:n_show],
    'SHAP': shap_mean[:n_show]
}).sort_values('SHAP', ascending=False)
shap_imp.to_csv(f'{EXPERIMENTS_DIR}/shap_importance.csv', index=False)

print("\n" + "=" * 70)
print("分析完成！结果汇总")
print("=" * 70)
print(f"\n最佳模型: {best_name}")
print(f"  ROC-AUC:  {all_results[best_name]['roc_auc']:.4f}")
print(f"  Recall:   {all_results[best_name]['recall']:.4f}")
print(f"  Precision:{all_results[best_name]['precision']:.4f}")
print(f"  F1-Score: {all_results[best_name]['f1']:.4f}")
print(f"  PR-AUC:   {all_results[best_name]['pr_auc']:.4f}")

print(f"\n聚类分析:")
for i in range(n_clusters):
    churn_r = cluster_profile.loc[i, 'Churn']
    print(f"  Cluster {i}: {cluster_profile.loc[i, 'Size']:.0f}人, 流失率={churn_r:.2%}, 月费=${cluster_profile.loc[i, 'MonthlyCharges']:.1f}")

print(f"\n模型对比表:")
print(comp_df.to_string())
print(f"\n✓ 最佳模型: ./models/best_model.pkl")
print(f"✓ 实验结果: {EXPERIMENTS_DIR}/")
print(f"  - model_comparison.csv")
print(f"  - shap_importance.csv")
print(f"  - cluster_results.csv")
print(f"  - *.png (可视化图表)")
print("\nDone!")
