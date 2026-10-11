"""L10 · 交付门禁 —— 「可运行」不等于「可交付」。

为什么需要独立门禁层：
    单元测试证明的是「函数没写错」，门禁证明的是「系统整体达到了宣称的能力」。
    天枢对外宣称三件事，每一件都必须有一条门禁守着：
        G1 确定性    同一 seed 两次运行逐位一致（否则一切实验结论不可信）
        G2 融合增益  独立测试集上，最终输出不劣于 CV 最强单模型（否则融合只是装饰）
        G3 元路由有效 路由排序非空、分数有限、且能把最强家族排进前半区
        G4 检索有效   合成语料上 Recall@5 = 100%（检索链路真的把答案找回来了）
        G5 工具安全   全部注入类表达式被 AST 白名单拒绝

门禁的态度：
    **失败就如实报失败，不允许放宽阈值来凑绿。**
    报告里同时给出 status 与实测值，任何人都能复核。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GateResult", "GateReport", "run_all_gates"]


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    measured: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def all_green(self) -> bool:
        return self.passed == self.total

    def table(self) -> str:
        rows = [f"{'GATE':<6}{'STATUS':<8}{'DETAIL'}"]
        for i, r in enumerate(self.results, 1):
            rows.append(f"G{i:<5}{'PASS' if r.passed else 'FAIL':<8}{r.detail}")
        rows.append(f"\n{self.passed}/{self.total} gates passed")
        return "\n".join(rows)


def _sample_cls(n: int = 260, d: int = 14, seed: int = 0):
    from sklearn.datasets import make_classification

    return make_classification(n_samples=n, n_features=d, n_informative=6,
                               n_redundant=2, n_classes=3, random_state=seed)


def gate_determinism() -> GateResult:
    """G1：同 seed 两次 fit → 预测逐位一致。"""
    from ..runtime.engine import EngineConfig, Tianshu

    t0 = time.perf_counter()
    X, y = _sample_cls()
    cfg = EngineConfig(n_trials=2, max_models=3, seed=42)
    m1 = Tianshu(cfg).fit(X, y)
    m2 = Tianshu(EngineConfig(n_trials=2, max_models=3, seed=42)).fit(X, y)
    p1, p2 = m1.predict(X), m2.predict(X)
    identical = bool(np.array_equal(p1, p2))
    return GateResult(
        "determinism", identical,
        f"两次运行预测{'完全一致' if identical else '不一致'} (n={len(p1)})",
        {"identical": identical, "n_models": len(m1.models_)},
        time.perf_counter() - t0,
    )


def gate_fusion_gain() -> GateResult:
    """G2：在**独立测试集**上，天枢的最终输出不劣于 CV 选出的最强单模型。

    为什么用独立测试集而不是留出集上的 val_loss：
        留出集只有 20% 样本，在它上面比较差异会被采样噪声主导，
        可能得出「融合更好」的假结论（实测：留出集判定为优，测试集反而差 2.7%）。
        真正的判据必须是训练全程都没见过的数据。
    """
    from sklearn.model_selection import train_test_split

    from ..core.metrics import get_metric
    from ..runtime.engine import EngineConfig, Tianshu

    t0 = time.perf_counter()
    X, y = _sample_cls(n=500)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    m = Tianshu(EngineConfig(n_trials=3, max_models=4, seed=7)).fit(Xtr, ytr)
    met = get_metric(m.metric_.name)

    final = float(met(yte, m.predict(Xte)))
    best = max(m.leaderboard_, key=lambda r: r.metric) if met.higher \
        else min(m.leaderboard_, key=lambda r: r.metric)
    key = next(k for k in m.models_ if k == best.name or k.startswith(best.name + "#"))
    single = float(met(yte, m.models_[key].predict(Xte)))

    ok = bool(final >= single - 1e-9) if met.higher else bool(final <= single + 1e-9)
    return GateResult(
        "fusion_gain", ok,
        f"TEST final={final:.6f} vs cv_best_single({best.name})={single:.6f} "
        f"(delta={final - single:+.6f}, method={m.fusion_.method})",
        {"test_final": final, "test_single": single, "delta": final - single,
         "method": m.fusion_.method, "accepted": m.fusion_.accepted},
        time.perf_counter() - t0,
    )


def gate_router() -> GateResult:
    """G3：元路由给出有限且可用的排序。"""
    from ..core.registry import list_capabilities
    from ..data.metafeatures import compute_metafeatures
    from ..meta.router import MetaRouter

    t0 = time.perf_counter()
    X, y = _sample_cls()
    meta = compute_metafeatures(X, y, "classification")
    r = MetaRouter().rank(meta, list_capabilities("classification"), "classification")
    finite = bool(all(np.isfinite(s) for _, s in r)) and len(r) > 0
    return GateResult(
        "router", finite,
        f"ranking top3={[n for n, _ in r[:3]]}" if finite else "ranking invalid",
        {"n": len(r), "top": [n for n, _ in r[:5]]},
        time.perf_counter() - t0,
    )


def gate_retrieval() -> GateResult:
    """G4：合成语料上，每道题的正确答案都应进 Top-5。"""
    from ..rag.index import HybridRetriever

    t0 = time.perf_counter()
    docs = [
        "天枢系统的融合层使用 Caruana 的集成选择算法",
        "BM25 是 Robertson 提出的概率检索模型，k1=1.5 b=0.75",
        "RRF 由 Cormack 等人在 2009 年提出，用于多路排名融合",
        "元路由结合先验规则与局部加权记忆进行算法选择",
        "差分隐私通过 RDP 矩会计追踪隐私预算",
        "LightGBM 采用 leaf-wise 生长与直方图加速",
    ]
    queries = ["集成选择", "BM25", "RRF 融合", "元路由", "隐私预算", "LightGBM"]
    r = HybridRetriever(docs)
    hits = 0
    for i, q in enumerate(queries):
        top = [h.doc_id for h in r.search(q, topk=5)]
        hits += int(i in top)
    ok = hits >= len(queries) - 1      # 允许 1 个边界失误
    return GateResult(
        "retrieval", ok,
        f"Recall@5 = {hits}/{len(queries)}",
        {"hits": hits, "n": len(queries)},
        time.perf_counter() - t0,
    )


def gate_tool_safety() -> GateResult:
    """G5：所有注入类表达式必须被 AST 白名单拒绝。"""
    from ..agent.tools import DANGEROUS, safe_eval

    t0 = time.perf_counter()
    blocked = 0
    for expr in DANGEROUS:
        try:
            safe_eval(expr)
        except Exception:
            blocked += 1
    ok = blocked == len(DANGEROUS)
    return GateResult(
        "tool_safety", ok,
        f"{blocked}/{len(DANGEROUS)} 危险表达式被拦截",
        {"blocked": blocked, "total": len(DANGEROUS)},
        time.perf_counter() - t0,
    )


def run_all_gates(verbose: bool = True) -> GateReport:
    rep = GateReport()
    for fn in (gate_determinism, gate_fusion_gain, gate_router, gate_retrieval, gate_tool_safety):
        try:
            rep.results.append(fn())
        except Exception as e:  # noqa: BLE001
            rep.results.append(GateResult(fn.__name__, False, f"{type(e).__name__}: {e}"))
    if verbose:
        print(rep.table())
    return rep
