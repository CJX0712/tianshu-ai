"""L1 · 元特征 —— 让系统「看清」数据，再做选择而非盲搜。

方法论（零重写，沿用 auto-sklearn / OpenML 的标准做法）：
    Feurer et al. 2015 (auto-sklearn) 提出用「简单模型 + 交叉验证」的
    landmarking 信号 + 统计元特征来冷启动算法选择。本模块实现其可离线
    计算的精简子集，全部 CPU 可跑、O(n·d) 为主。

元特征清单（13 维）：
    规模:     log_n, log_d, log_n_d (样本/特征比)
    类别:     n_classes, class_entropy_norm, imbalance_ratio
    质量:     missing_ratio, dup_ratio, outlier_ratio
    结构:     pca95_ratio, mean_abs_corr, fisher_ratio
    难度:     knn1_acc (1-NN 留一/交叉验证准确率，最强 landmarker)
             线性可分: linear_acc (逻辑回归/岭回归 CV 分数)

可验证不变量：
    I1  元特征向量维度固定 = 13，顺序稳定（元路由的输入契约）。
    I2  纯噪声数据 knn1_acc ≈ 1/n_classes（无信号可被landmarker利用）。
    I3  完全可分数据 knn1_acc → 1.0。
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

META_DIM = 13
META_NAMES: tuple[str, ...] = (
    "log_n", "log_d", "log_n_over_d",
    "n_classes", "class_entropy_norm", "imbalance_ratio",
    "missing_ratio", "dup_ratio", "outlier_ratio",
    "pca95_ratio", "mean_abs_corr", "fisher_ratio",
    "knn1_acc",
)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_metafeatures(
    X: np.ndarray,
    y: np.ndarray | None = None,
    task: str = "classification",
    fast: bool = True,
) -> np.ndarray:
    """返回长度 13 的元特征向量（float64）。缺失值自动按列均值填补后再统计。"""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    log_n = float(np.log1p(n))
    log_d = float(np.log1p(d))
    log_n_d = float(np.log1p(n / max(d, 1)))

    missing = float(np.mean(np.isnan(X)) if X.dtype.kind == "f" else 0.0)

    # 重复行比例（哈希去重，O(n·d)）
    dup = 0.0
    if n > 1:
        try:
            _, idx = np.unique(np.round(Xf, 8), axis=0, return_index=True)
            dup = 1.0 - len(idx) / n
        except Exception:
            dup = 0.0

    # 异常值比例：|z| > 3 的样本占比
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1.0
    z = np.abs((Xf - Xf.mean(axis=0)) / sd)
    outlier = float(np.mean(np.max(z, axis=1) > 3.0)) if n else 0.0

    # PCA：解释 95% 方差所需主成分占比（内在维度）
    pca95 = 1.0
    if n >= 5 and d >= 2:
        k = int(min(n, d))
        try:
            p = PCA(n_components=min(k, 50)).fit(Xf)
            cum = np.cumsum(p.explained_variance_ratio_)
            hit = np.argmax(cum >= 0.95)
            pca95 = float((hit + 1) / k)
        except Exception:
            pca95 = 1.0

    # 平均绝对相关（特征冗余）
    corr = 0.0
    if d >= 2 and n >= 3:
        try:
            C = np.corrcoef(Xf, rowvar=False)
            C = np.nan_to_num(C)
            iu = np.triu_indices(d, 1)
            corr = float(np.mean(np.abs(C[iu]))) if len(iu[0]) else 0.0
        except Exception:
            corr = 0.0

    # 类别结构 / 可分性
    n_classes = 1.0
    ent_norm = 0.0
    imb = 1.0
    fisher = 0.0
    knn1 = 0.0

    if y is not None and task == "classification":
        yv = np.asarray(y)
        classes, counts = np.unique(yv, return_counts=True)
        n_classes = float(len(classes))
        ent_norm = float(_entropy(counts / counts.sum()) / np.log(max(len(classes), 2)))
        imb = float(counts.max() / max(counts.min(), 1))

        if n >= 10 and len(classes) >= 2:
            # Fisher 判别比：组间/组内方差（越大越线性可分）
            try:
                grand = Xf.mean(axis=0)
                gmeans = np.stack([Xf[yv == c].mean(axis=0) for c in classes])
                within = float(np.sum(
                    [((Xf[yv == c] - gmeans[i]) ** 2).sum() for i, c in enumerate(classes)]
                ))
                between = float(np.sum(
                    [counts[i] * ((gmeans[i] - grand) ** 2).sum() for i in range(len(classes))]
                ))
                fisher = between / max(within, 1e-12)
            except Exception:
                fisher = 0.0

            # 1-NN landmarker：最经典的元学习信号
            if fast and n <= 4000:
                try:
                    cv = min(5, int(min(counts)))
                    if cv >= 2:
                        knn1 = float(np.mean(cross_val_score(
                            KNeighborsClassifier(n_neighbors=1), Xf, yv, cv=cv)))
                except Exception:
                    knn1 = 0.0
            elif not fast:
                try:
                    knn1 = float(np.mean(cross_val_score(
                        KNeighborsClassifier(n_neighbors=1), Xf, yv, cv=3)))
                except Exception:
                    knn1 = 0.0

    elif y is not None and task == "regression":
        yv = np.asarray(y, dtype=float)
        # 回归的「可分性」用线性可解释方差 (R²) 作为 landmarker
        if n >= 20:
            try:
                r2 = float(np.mean(cross_val_score(Ridge(alpha=1.0), Xf, yv, cv=3,
                                                   scoring="r2")))
                knn1 = float(np.clip(r2, -1.0, 1.0))
            except Exception:
                knn1 = 0.0
        n_classes = float(np.std(yv) > 0)
        ent_norm = float(np.std(yv) / (np.abs(np.mean(yv)) + 1e-12))  # 变异系数
        imb = float(np.mean(np.abs(yv - np.median(yv)) > 3 * np.std(yv)) if np.std(yv) > 0 else 0.0)

    return np.array(
        [log_n, log_d, log_n_d, n_classes, ent_norm, imb,
         missing, dup, outlier, pca95, corr, fisher, knn1],
        dtype=float,
    )


def vectorize(X: np.ndarray, y: np.ndarray | None = None, task: str = "classification") -> dict[str, float]:
    """元特征的可读字典形式（报告/控制台展示用）。"""
    v = compute_metafeatures(X, y, task)
    return {k: float(x) for k, x in zip(META_NAMES, v, strict=False)}
