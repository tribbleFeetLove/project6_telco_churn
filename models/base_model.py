"""
模型基类模块

提供所有模型的抽象基类，定义统一的训练、预测、保存、加载接口。
遵循PEP8规范。
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import joblib
import os


class BaseModel(ABC):
    """所有模型的抽象基类。

    定义了分类模型的标准接口，所有自定义模型需继承此类。

    Attributes
    ----------
    model : Any
        底层的sklearn-compatible模型实例
    config : Dict[str, Any]
        模型配置参数字典
    feature_names : List[str]
        训练时使用的特征名称
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化基类。

        Parameters
        ----------
        config : Dict[str, Any], optional
            模型配置参数
        """
        self.config = config or {}
        self.model = None
        self.feature_names = []

    @abstractmethod
    def build(self) -> None:
        """构建底层模型实例。子类必须实现此方法。"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """训练模型。

        Parameters
        ----------
        X : pd.DataFrame
            训练特征
        y : pd.Series
            训练标签

        Returns
        -------
        BaseModel
            自身实例，支持链式调用
        """
        if self.model is None:
            self.build()

        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else []
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别标签。

        Parameters
        ----------
        X : pd.DataFrame
            特征数据

        Returns
        -------
        np.ndarray
            预测类别
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练或构建，请先调用 fit() 或 build()")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别概率。

        Parameters
        ----------
        X : pd.DataFrame
            特征数据

        Returns
        -------
        np.ndarray
            预测概率，shape为 (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练或构建，请先调用 fit() 或 build()")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """保存模型到磁盘。

        Parameters
        ----------
        path : str
            保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
        }
        joblib.dump(save_data, path)
        print(f"模型已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """从磁盘加载模型。

        Parameters
        ----------
        path : str
            模型文件路径

        Returns
        -------
        BaseModel
            加载后的模型实例
        """
        save_data = joblib.load(path)
        instance = cls(config=save_data['config'])
        instance.model = save_data['model']
        instance.feature_names = save_data['feature_names']
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性。

        Returns
        -------
        Optional[Dict[str, float]]
            特征名称到重要性的映射字典，若不支持则返回None
        """
        if self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return None

        if self.feature_names and len(self.feature_names) == len(importance):
            return dict(zip(self.feature_names, importance))
        return None
