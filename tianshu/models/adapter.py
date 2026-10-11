"""L2 · 适配层 —— 把世界顶级开源实现接进天枢契约。

这里**没有任何一行自研算法数学**。每个家族都委托给：
    scikit-learn（RF / ET / HistGB / SVM / KNN / MLP / Ridge / GP …）
    XGBoost（Chen & Guestrin 2016，梯度提升的 SOTA 实现）
    LightGBM（Ke et al. 2017，leaf-wise + GOSS + EFB）
适配器只负责三件事：统一 fit/predict 签名、暴露搜索空间、声明代价先验。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.contracts import SearchSpace, Task


class SkAdapter:
    """把任意 sklearn-API 估计器包装为天枢 Estimator。

    关键点：
        - 分类任务同时暴露 label / proba / score 三种输出，
          因为不同度量需要不同输出形态（accuracy 要 label，AUC 要 score）。
        - 训练/推理计时内建，供元路由做预算感知与门禁做代价审计。
    """

    def __init__(self, name: str, task: Task, builder, params: dict[str, Any] | None = None):
        self.name = name
        self.task = task
        self._builder = builder
        self.params: dict[str, Any] = dict(params or {})
        self.model: Any = None
        self.fit_sec = 0.0
        self.n_features_: int | None = None

    # ---- 契约 -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SkAdapter:
        X = np.asarray(X)
        t0 = time.perf_counter()
        self.model = self._build()
        if y is None:
            self.model.fit(X)
        else:
            self.model.fit(X, np.asarray(y))
        self.fit_sec = time.perf_counter() - t0
        self.n_features_ = X.shape[1]
        return self

    def _build(self):
        return self._builder(**self.params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        # 无监督变换器（PCA 等）没有 predict，退到 transform；
        # 纯聚类器再退到 fit_predict。保证「所有注册能力都可 predict」这一契约。
        if hasattr(self.model, "transform"):
            return np.asarray(self.model.transform(X))
        if hasattr(self.model, "fit_predict"):
            return np.asarray(self.model.fit_predict(X))
        raise AttributeError(f"{type(self.model).__name__} exposes neither predict/transform/fit_predict")

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            try:
                return np.asarray(self.model.predict_proba(np.asarray(X)))
            except Exception:
                return None
        return None

    def decision(self, X: np.ndarray) -> np.ndarray:
        """连续分数：回归=预测值；分类=正类概率(二分类)/最大类概率(多分类)/decision_function。"""
        if self.task == "regression":
            return self.predict(X).astype(float)
        p = self.predict_proba(X)
        if p is not None:
            return p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
        if hasattr(self.model, "decision_function"):
            d = np.asarray(self.model.decision_function(np.asarray(X)))
            return d.ravel() if d.ndim == 1 else d.max(axis=1)
        return self.predict(X).astype(float, copy=False)

    def space(self) -> SearchSpace:
        return SearchSpace()

    def __repr__(self) -> str:
        return f"<{self.name} task={self.task} fit={self.fit_sec:.3f}s>"


def _n_classes(y: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(y))))
