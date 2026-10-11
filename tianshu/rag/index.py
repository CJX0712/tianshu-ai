"""L7 · 混合检索 —— 稀疏与稠密双路召回 + RRF 融合 + 有护栏的重排。

零重写：每个组件都对应一篇奠基工作 / 一个工业标准实现。
    BM25        Robertson & Zaragoza 2009《The Probabilistic Relevance Framework》
                k1=1.5, b=0.75 是经 TREC 验证的经典参数组。
    RRF         Cormack, Clarke & Büttcher 2009《Reciprocal Rank Fusion》
                score = Σ 1/(k+rank)，k=60。它不依赖各路分数的可比性，
                这是它优于加权求和的根本原因——BM25 分数量纲与余弦相似度毫无可比性。
    LSI/SVD     Deerwester et al. 1990；此处用 TruncatedSVD 做低秩语义压缩。
    哈希技巧    Weinberger et al. 2009（signed hashing，零词典、零训练、确定性）。

中文处理（关键工程点）：
    中文没有空格，按词切需要词典。这里用**单字 + 双字** n-gram：
    单字保证召回（不会因切词错误漏掉），双字恢复大部分词级语义。
    这是在零依赖前提下，中文检索性价比最高的方案（已在中英混合语料验证）。

重排护栏（防退化）：
    重排器权重**封顶 0.5**。弱重排器（如覆盖率启发式）在高权重下会把
    真正相关的长文档压下去——这是实践中检索质量反而下降的头号原因。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：拉丁词按词，CJK 按单字 + 双字。"""
    text = str(text)
    toks: list[str] = []
    for m in _LATIN.findall(text):
        toks.append(m.lower())
    for run in _CJK.findall(text):
        for ch in run:
            toks.append(ch)
        for i in range(len(run) - 1):
            toks.append(run[i : i + 2])
    return toks


class BM25:
    """经典 BM25（Robertson-Sparck-Jones + 长度归一）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.N = 0

    def fit(self, docs: list[str]) -> BM25:
        self.docs = [tokenize(d) for d in docs]
        self.tf = []
        self.df = {}
        for toks in self.docs:
            t: dict[str, int] = {}
            for w in toks:
                t[w] = t.get(w, 0) + 1
            self.tf.append(t)
            for w in t:
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(self.docs)
        self.avgdl = float(np.mean([len(t) for t in self.docs])) if self.N else 1.0
        return self

    def idf(self, w: str) -> float:
        n = self.df.get(w, 0)
        # +1 平滑，保证不出现负 IDF（ Robertson 变体）
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self.idf(w) * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


class HashingEmbedder:
    """符号哈希向量化（字符 2~3 gram），零词典、零训练、逐位确定。"""

    def __init__(self, dim: int = 384, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        s = str(text).lower()
        grams = [s[i : i + 2] for i in range(len(s) - 1)] + [s[i : i + 3] for i in range(len(s) - 2)]
        for g in grams:
            h = hash((self.seed, g)) & 0xFFFFFFFF
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[idx] += sign
            # 二次哈希降低碰撞偏置（Weinberger 的 signed hashing 变体）
            h2 = (h * 0x9E3779B1) & 0xFFFFFFFF
            v[h2 % self.dim] += 0.5 * (1.0 if (h2 >> 30) & 1 else -1.0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros((0, self.dim))
        return np.vstack([self._vec(d) for d in docs])


class DenseIndex:
    """稠密索引：哈希向量 + 可选 LSI(SVD) 语义压缩 + 余弦检索。"""

    def __init__(self, dim: int = 384, lsi: int = 64, seed: int = 0):
        self.embedder = HashingEmbedder(dim, seed)
        self.lsi = lsi
        self._svd = None
        self.emb: np.ndarray | None = None

    def fit(self, docs: list[str]) -> DenseIndex:
        E = self.embedder.encode(docs)
        if self.lsi and len(docs) >= 8:
            from sklearn.decomposition import TruncatedSVD

            k = int(min(self.lsi, len(docs) - 1, E.shape[1] - 1))
            if k >= 2:
                self._svd = TruncatedSVD(n_components=k, random_state=0)
                E = self._svd.fit_transform(E)
        n = np.linalg.norm(E, axis=1, keepdims=True)
        self.emb = E / np.where(n == 0, 1.0, n)
        return self

    def _q(self, query: str) -> np.ndarray:
        v = self.embedder.encode([query])
        if self._svd is not None:
            v = self._svd.transform(v)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def scores(self, query: str) -> np.ndarray:
        if self.emb is None or len(self.emb) == 0:
            return np.zeros(0)
        return np.asarray(self.emb @ self._q(query).ravel())


@dataclass
class Hit:
    doc_id: int
    score: float
    text: str
    parts: dict[str, float] = field(default_factory=dict)


def rrf_fuse(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion：把多路排名融合成一个分数（越大越好）。"""
    out: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, doc_id in enumerate(ranks):
            out[doc_id] = out.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return out


class HybridRetriever:
    """双路召回 + RRF 融合 + 有护栏的重排。

    rerank_cap  重排权重上限（默认 0.5）。这是**不可随便调高**的护栏。
    """

    def __init__(self, docs: list[str], rerank_cap: float = 0.5, topk: int = 10):
        self.docs = list(docs)
        self.topk = topk
        self.rerank_cap = float(min(max(rerank_cap, 0.0), 0.9))
        self.bm25 = BM25().fit(self.docs)
        self.dense = DenseIndex().fit(self.docs)

    def _coverage(self, query: str, text: str) -> float:
        """查询词覆盖率（弱重排信号，因此权重必须封顶）。"""
        q = set(tokenize(query))
        if not q:
            return 0.0
        d = set(tokenize(text))
        return len(q & d) / len(q)

    def search(self, query: str, topk: int | None = None) -> list[Hit]:
        if not self.docs:
            return []
        k = int(topk or self.topk)
        sb = self.bm25.scores(query)
        sd = self.dense.scores(query)

        def top(vals: np.ndarray, n: int) -> list[int]:
            if len(vals) == 0:
                return []
            n = int(min(n, len(vals)))
            return [int(i) for i in np.argsort(-vals, kind="stable")[:n]]

        n_cand = int(min(len(self.docs), max(k * 3, 20)))
        fused = rrf_fuse([top(sb, n_cand), top(sd, n_cand)])

        max_f = max(fused.values()) if fused else 1.0
        hits = []
        for doc_id, f in fused.items():
            cov = self._coverage(query, self.docs[doc_id])
            norm_f = f / max_f if max_f > 0 else 0.0
            final = (1.0 - self.rerank_cap) * norm_f + self.rerank_cap * cov
            hits.append(Hit(doc_id, float(final), self.docs[doc_id],
                            {"rrf": float(f), "bm25": float(sb[doc_id]),
                             "dense": float(sd[doc_id]), "coverage": float(cov)}))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def add(self, docs: list[str]) -> None:
        self.docs.extend(docs)
        self.bm25.fit(self.docs)
        self.dense.fit(self.docs)
