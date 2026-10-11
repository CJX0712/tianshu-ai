"""L5 · 元路由 —— 让系统「先选对，再调优」，而不是对全部算法盲搜。

两层机制（零重写，沿用 AutoML 元学习主流范式）：
    A. 冷启动先验（规则表）
       把 AutoML 社区公认的经验编码成「元特征 → 家族适配度」的解析函数：
       Brazdil et al. 2008《Metalearning》与 auto-sklearn (Feurer 2015)
       的元特征-算法相关结论（样本量/维度/可分性/不平衡度如何左右家族选择）。
    B. 在线局部加权记忆 + UCB 探索
       每观测一次 (数据集元特征, 算法, 分数) 就写入记忆；
       查询时用高斯核做**局部加权回归**（相似数据集上的表现更有参考价值），
       并叠加 UCB1 探索项（Auer et al. 2002）保证新算法不被永久饿死。

为什么不用「离线元模型」：
   离线元模型需要成千上万个数据集的元特征-性能矩阵才能训练，
   单机无法获得。局部加权记忆 + UCB 是**冷启动即可用、随使用变强**的工程解。

可验证不变量：
    I1  rank() 输出的分数有限且非 NaN（元特征含 NaN 时也要能排）。
    I2  观测越多，同数据集上的排名趋于稳定（收敛性）。
    I3  记忆为空时退化为纯先验（不崩溃）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.metafeatures import compute_metafeatures

# 元特征索引（与 metafeatures.META_NAMES 严格对应）
I_LOG_N, I_LOG_D, I_LOG_ND, I_NCLS, I_ENT, I_IMB = 0, 1, 2, 3, 4, 5
I_MISS, I_DUP, I_OUT, I_PCA95, I_CORR, I_FISHER, I_KNN1 = 6, 7, 8, 9, 10, 11, 12


def _s(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))  # logistic 压到 (0,1)


def prior_family(family: str, m: np.ndarray, task: str) -> float:
    """家族在给定元特征下的适配度（0~1）。规则来自 AutoML 实证共识。"""
    n = float(np.expm1(m[I_LOG_N])) if m[I_LOG_N] > 0 else 1.0
    d = float(np.expm1(m[I_LOG_D])) if m[I_LOG_D] > 0 else 1.0
    knn1 = float(m[I_KNN1])
    fisher = float(m[I_FISHER])
    pca95 = float(m[I_PCA95])
    imb = float(m[I_IMB])
    corr = float(m[I_CORR])
    outl = float(m[I_OUT])

    high_dim = d / max(n, 1.0) > 0.1          # 特征相对样本过多
    small = n < 2000
    large = n > 20000
    noisy = knn1 < 0.55 if task == "classification" else knn1 < 0.1

    if family == "linear":
        # 高维稀疏 / 噪声大 / 强线性可分 → 线性族吃香（正则化抗噪）
        return float(np.clip(
            0.45 + 0.35 * _s((fisher - 0.1) * 10) + 0.25 * float(high_dim)
            + 0.2 * float(noisy) - 0.15 * float(large and d > 100), 0, 1))

    if family == "tree":
        # 默认强者：样本足、非线性（fisher 低但 knn1 高）、含异常值
        nonlin = knn1 > 0.7 and fisher < 0.5 if task == "classification" else False
        return float(np.clip(
            0.62 + 0.2 * float(not small) + 0.15 * float(nonlin)
            + 0.1 * float(outl > 0.05) - 0.2 * float(high_dim), 0, 1))

    if family == "kernel":
        # 中小样本、中低维时 SVM/GP 强；大样本 O(n²~n³) 不可行
        return float(np.clip(
            0.55 + 0.25 * float(small and d < 200) - 0.4 * float(large)
            - 0.15 * float(high_dim), 0, 1))

    if family == "instance":
        # kNN：knn1 landmarker 直接就是它的成绩单
        return float(np.clip(0.3 + 0.5 * knn1 + 0.1 * float(small) - 0.2 * float(d > 100), 0, 1))

    if family == "probabilistic":
        return float(np.clip(0.4 + 0.2 * float(small) + 0.15 * float(high_dim)
                             - 0.1 * float(large), 0, 1))

    if family == "neural":
        # 需要数据量；高维表示学习占优
        return float(np.clip(0.35 + 0.3 * float(not small) + 0.2 * float(d > 50 or pca95 > 0.5)
                             - 0.2 * float(n < 500), 0, 1))

    if family == "cluster" or family == "manifold" or family == "anomaly":
        return float(np.clip(0.4 + 0.2 * float(corr > 0.3 or pca95 < 0.5) + 0.1 * float(imb > 3), 0, 1))

    return 0.5


@dataclass
class Memory:
    """局部加权记忆：[(元特征, 算法名, 归一化分数)]"""

    meta: list[np.ndarray] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    bandwidth: float = 0.75

    def add(self, m: np.ndarray, name: str, score: float) -> None:
        self.meta.append(np.asarray(m, dtype=float))
        self.name.append(name)
        self.score.append(float(score))

    def query(self, m: np.ndarray, name: str) -> tuple[float, int]:
        """返回 (核加权平均分数, 有效样本数)。无记忆 → (nan, 0)。"""
        m = np.asarray(m, dtype=float)
        idx = [i for i, nm in enumerate(self.name) if nm == name]
        if not idx:
            return float("nan"), 0
        M = np.stack([self.meta[i] for i in idx])
        s = np.array([self.score[i] for i in idx])
        dist2 = np.sum((M - m) ** 2, axis=1)
        w = np.exp(-dist2 / (2.0 * self.bandwidth**2))
        tot = float(w.sum())
        if tot <= 1e-12:
            return float(np.mean(s)), len(idx)
        return float(np.dot(w, s) / tot), int(max(1.0, np.sum(w > 0.05)))

    def __len__(self) -> int:
        return len(self.name)

    def to_json(self) -> str:
        return json.dumps({
            "meta": [m.tolist() for m in self.meta],
            "name": self.name,
            "score": self.score,
            "bandwidth": self.bandwidth,
        })

    @staticmethod
    def from_json(s: str) -> Memory:
        d = json.loads(s)
        mem = Memory(bandwidth=d.get("bandwidth", 0.75))
        mem.meta = [np.asarray(x) for x in d.get("meta", [])]
        mem.name = list(d.get("name", []))
        mem.score = list(d.get("score", []))
        return mem


class MetaRouter:
    """元路由器：先验 + 记忆 + UCB 探索 → 候选排序。"""

    def __init__(self, seed: int = 0, beta: float = 1.0, prior_w: float = 0.6,
                 cost_w: float = 0.15, memory: Memory | None = None):
        self.seed = seed
        self.beta = beta          # UCB 探索强度
        self.prior_w = prior_w    # 先验权重（其余给在线估计）
        self.cost_w = cost_w      # 代价惩罚权重（预算感知）
        self.memory = memory or Memory()

    def score(self, meta: np.ndarray, name: str, family: str, task: str, cost: float) -> float:
        prior = prior_family(family, meta, task)
        mu, n_eff = self.memory.query(meta, name)
        if np.isnan(mu):
            # 无观测：给最大探索 bonus，保证每个家族至少被试一次
            online, bonus = 0.5, 0.5
        else:
            online = mu
            bonus = self.beta * np.sqrt(np.log(len(self.memory) + 2.0) / (n_eff + 1.0))
        mem_part = online + bonus
        s = self.prior_w * prior + (1.0 - self.prior_w) * mem_part
        return float(s - self.cost_w * np.log1p(max(cost, 1e-6)))

    def rank(self, meta: np.ndarray, caps: list, task: str) -> list[tuple[str, float]]:
        """返回 [(capability_name, score)] 降序。"""
        scored = []
        for c in caps:
            m = np.nan_to_num(np.asarray(meta, dtype=float), nan=0.0)
            scored.append((c.name, self.score(m, c.name, c.family, task, c.cost)))
        return sorted(scored, key=lambda kv: -kv[1])

    def observe(self, meta: np.ndarray, name: str, score: float) -> None:
        """记录一次真实评测结果（score 建议已归一化到 [0,1]）。"""
        self.memory.add(np.nan_to_num(np.asarray(meta, dtype=float), nan=0.0), name, float(score))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.memory.to_json(), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.memory = Memory.from_json(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def metafeatures(X, y=None, task: str = "classification") -> np.ndarray:
        return compute_metafeatures(X, y, task)
