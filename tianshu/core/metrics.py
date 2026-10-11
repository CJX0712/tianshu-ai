"""L0 · 度量内核 —— 系统只会优化它能衡量的东西。

零重写原则：分类/回归的标量度量直接委托 scikit-learn（久经审计的实现），
本模块只做三件 sklearn 没做好/没做的事：
    1. 统一「越小越好」的 loss 语义（优化器与门禁只需要一种方向）。
    2. 提供 sklearn 缺失的排序度量 nDCG@k / MRR / Recall@k（L7 检索依赖）。
    3. 提供带置信区间的 Bootstrap 评估（门禁判定「是否真的提升了」必须有它）。

可验证不变量：
    I1  loss 方向一致：所有 metric 的 .loss 属性越大越差。
    I2  完美预测下 accuracy=1 / rmse=0 / ndcg=1。
    I3  Bootstrap 均值落在点估计 ±3·se 内（大数定律）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

__all__ = [
    "Metric",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cross_entropy",
    "roc_auc",
    "average_precision",
    "rmse",
    "mae",
    "r2",
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "bootstrap_ci",
    "get_metric",
    "TASK_METRICS",
]


@dataclass(frozen=True)
class Metric:
    """度量的统一契约。

    name     唯一标识（注册表键）
    fn       (y_true, y_score) -> float，越大越好
    loss     越大越差的等价量（= -fn 或 fn 本身，取决于方向）
    higher   fn 是否越大越好
    task     'classification' | 'regression' | 'ranking'
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    higher: bool
    task: str

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(self.fn(np.asarray(y_true), np.asarray(y_score)))

    def loss(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """统一为「越小越好」，供 HPO / 集成选择 / 门禁使用。"""
        v = self(y_true, y_score)
        return -v if self.higher else v


def _safe(fn: Callable, default: float = 0.0) -> Callable:
    """包裹：退化输入（单类、全同预测）下返回 default 而非抛异常。

    这是集成搜索里的实战刚需——弱模型的候选预测可能只落一个类，
    roc_auc_score 会直接 raise，导致整个搜索崩掉。
    """

    def wrapped(y_true, y_score):
        try:
            v = fn(y_true, y_score)
            return float(v) if np.isfinite(v) else default
        except Exception:
            return default

    return wrapped


accuracy = Metric("accuracy", _safe(accuracy_score), True, "classification")
balanced_accuracy = Metric("balanced_accuracy", _safe(balanced_accuracy_score), True, "classification")
macro_f1 = Metric(
    "macro_f1", _safe(lambda a, b: f1_score(a, b, average="macro", zero_division=0)), True, "classification"
)
cross_entropy = Metric(
    "cross_entropy", _safe(lambda a, b: log_loss(a, b, labels=np.unique(a))), False, "classification"
)
roc_auc = Metric(
    "roc_auc", _safe(lambda a, b: roc_auc_score(a, b, multi_class="ovr") if len(np.unique(a)) > 1 else 0.5),
    True, "classification",
)
average_precision = Metric("average_precision", _safe(average_precision_score), True, "classification")
rmse = Metric("rmse", _safe(lambda a, b: float(np.sqrt(mean_squared_error(a, b)))), False, "regression")
mae = Metric("mae", _safe(mean_absolute_error), False, "regression")
r2 = Metric("r2", _safe(r2_score, default=-1.0), True, "regression")


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """归一化折损累计增益 —— 排序与检索的黄金指标。

    零重写：采用标准 DCG 折损 1/log2(i+1) 与理想排序归一化（Järvelin & Kekäläinen 2002）。
    支持 graded relevance（y_true 可为任意非负实值，非仅 0/1）。
    """
    y_true = np.asarray(y_true, float).ravel()
    y_score = np.asarray(y_score, float).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0
    k = int(min(k, n))
    order = np.argsort(-y_score, kind="stable")[:k]
    gains = y_true[order]
    # (2^rel - 1) 是标准指数增益；rel<=0 时贡献 0
    num = np.sum((np.power(2.0, gains) - 1.0) / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y_true)[::-1][:k]
    den = np.sum((np.power(2.0, ideal) - 1.0) / np.log2(np.arange(2, k + 2)))
    return float(num / den) if den > 0 else 0.0


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """首个相关结果的排名倒数均值。y_true>0 视为相关。"""
    y_true = np.asarray(y_true, float).ravel()
    y_score = np.asarray(y_score, float).ravel()
    if not np.any(y_true > 0):
        return 0.0
    order = np.argsort(-y_score, kind="stable")
    rank = int(np.argmax(y_true[order] > 0)) + 1
    return 1.0 / rank


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_score = np.asarray(y_score, float).ravel()
    total = np.sum(y_true > 0)
    if total == 0:
        return 0.0
    k = int(min(k, len(y_true)))
    top = np.argsort(-y_score, kind="stable")[:k]
    return float(np.sum(y_true[top] > 0) / total)


_NDCG10 = Metric("ndcg@10", lambda a, b: ndcg_at_k(a, b, 10), True, "ranking")

_REGISTRY: dict[str, Metric] = {
    m.name: m
    for m in (
        accuracy, balanced_accuracy, macro_f1, cross_entropy,
        roc_auc, average_precision, rmse, mae, r2, _NDCG10,
    )
}

TASK_METRICS: dict[str, tuple[str, ...]] = {
    "classification": ("accuracy", "balanced_accuracy", "macro_f1", "cross_entropy", "roc_auc"),
    "regression": ("rmse", "mae", "r2"),
    "ranking": ("ndcg@10",),
}

DEFAULT_METRIC = {"classification": "accuracy", "regression": "rmse", "ranking": "ndcg@10"}


def get_metric(name: str) -> Metric:
    if name not in _REGISTRY:
        raise KeyError(f"unknown metric {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """度量的 Bootstrap 置信区间 → (点估计, 下界, 上界)。

    门禁为什么必须有它：单次 holdout 上的 +0.3% 差异极可能是噪声。
    只有当「挑战者的 CI 下界 >  incumbents 上界」或至少点估计提升超过
    标准误的倍数时，才允许宣称改进——这是可复现研究的底线。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    point = metric(y_true, y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2 and metric.task == "classification":
            continue
        vals.append(metric(y_true[idx], y_score[idx]))
    if not vals:
        return point, point, point
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
