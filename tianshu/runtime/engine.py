"""L6 · 天枢引擎 —— 端到端 AutoML 流水线，一次 fit 拿到「超越单模型」的最终系统。

流水线（十层中的 L1→L6 全链路）：
    元特征 → 元路由排序 → 逐模型 HPO(交叉验证) → OOF 预测矩阵
    → 融合旗舰选权 → 增益门控 → 全量重拟合 → 可预测对象

设计上的三个硬决定（每个都有代价，也都值得）：
    1. HPO 的评估与 OOF 收集**同一次 CV 完成**，不为最佳参数重跑 CV。
       代价：OOF 带了「被选中」的乐观偏差。
       补偿：融合层设有增益门控 + 独立的 holdout 门禁（L10），
       偏差若存在会在门禁处被拦下并回退单模型。
    2. 每个 trial 用 spawn_seed 派生独立流，CV 折切分固定。
       保证同一 seed 下整条流水线逐位可复现。
    3. 时间预算（time_budget）是硬约束：到期即停，已完成的候选照常融合。
       宁可少搜几个模型，也不能让调用方无上限等待。

零重写声明：本文件不含任何自研算法，只做调度与治理。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ..core.contracts import FitResult
from ..core.metrics import DEFAULT_METRIC, Metric, get_metric
from ..core.registry import list_capabilities
from ..core.seed import spawn_seed
from ..data.metafeatures import compute_metafeatures
from ..fusion.fusion_core import FusionCore, FusionResult
from ..hpo.search import make_searcher
from ..meta.router import MetaRouter

__all__ = ["EngineConfig", "Tianshu"]


@dataclass
class EngineConfig:
    task: str | None = None              # None = 自动推断
    metric: str | None = None            # None = 任务默认
    cv: int | None = None                # None = 自适应（小数据用 3 折以控时）
    # 三个参数共同决定总代价 ≈ max_models × n_trials × (cv 折数 + 2)。
    # CPU-only 机器上的默认值按「开箱即用 < 2 分钟」标定，不是按最优质量标定。
    n_trials: int = 6                    # 每个候选模型的 HPO trial 数
    max_models: int = 4                  # 最多尝试多少个算法家族
    # 每个家族保留几个 trial 进入融合池。None = 全部（库越大集成越强，但代价线性增长）。
    # 注意这**不会**引入选择偏差——trial 的筛选发生在训练集的 CV 上，
    # 集成选择在留出集上进行，两者已解耦。
    pool_per_model: int | None = 2
    holdout: float = 0.2                 # 集成选择专用留出集比例（0 = 禁用，退回 OOF）
    min_holdout: int = 120               # 样本数低于此值不切留出集
    time_budget: float | None = None     # 秒；None = 不限
    hpo: str = "auto"                    # 'auto' | 'tpe' | 'random'
    seed: int = 20261011
    fusion_budget: int = 30
    fusion_bagged: int = 4
    fusion_margin: float = 0.0           # 相对增益门槛（0 = 不劣即采纳）
    router: MetaRouter | None = None
    verbose: bool = False


@dataclass
class Tianshu:
    """天枢主引擎。用法与 sklearn 一致：fit → predict。"""

    config: EngineConfig = field(default_factory=EngineConfig)

    # ---- 运行时状态 ----
    task_: str = field(default="classification", init=False)
    metric_: Metric = field(init=False, repr=False)
    models_: dict[str, Any] = field(default_factory=dict, init=False)
    params_: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    leaderboard_: list[FitResult] = field(default_factory=list, init=False)
    fusion_: FusionResult | None = field(default=None, init=False)
    classes_: np.ndarray | None = field(default=None, init=False)
    holdout_: bool = field(default=False, init=False)
    meta_: np.ndarray | None = field(default=None, init=False)
    ranking_: list[tuple[str, float]] = field(default_factory=list, init=False)
    elapsed_: float = field(default=0.0, init=False)

    # =====================================================================
    # fit
    # =====================================================================
    def fit(self, X, y) -> Tianshu:
        X = np.asarray(X)
        y = np.asarray(y)
        cfg = self.config
        t_start = time.perf_counter()

        self.task_ = cfg.task or self._infer_task(y)
        self.metric_ = get_metric(cfg.metric or DEFAULT_METRIC[self.task_])

        self.meta_ = compute_metafeatures(X, y, self.task_)
        router = cfg.router or MetaRouter(seed=cfg.seed)

        caps = list_capabilities(self.task_)
        if not caps:
            raise RuntimeError(f"no capability registered for task={self.task_!r}")
        self.ranking_ = router.rank(self.meta_, caps, self.task_)
        chosen = [n for n, _ in self.ranking_][: cfg.max_models]

        # ---------------------------------------------------------------
        # 留出集切分 —— 这一处决策决定了融合是否真的有效
        #
        # 若集成选择与超参选择共用同一份 OOF，会产生**选择偏差**：
        # 池里那个"最强单模型"是从 n_trials×max_models 个候选中挑出来的，
        # 它的 OOF 分数已经被选中过程抬高了，融合几乎不可能超过它
        # （实测 3 个数据集 gain 恒为 0）。
        # auto-sklearn (Feurer 2015) 的做法正是把 ensemble 构建放在
        # 与 HPO **解耦**的留出集上。这里沿用：
        #   · HPO / CV   → 在 Xtr 上做
        #   · 集成选择   → 在从未参与训练的 Xva 上做
        #   · 最终模型   → 用全量 X 重拟合
        # ---------------------------------------------------------------
        self.classes_ = np.unique(y) if self.task_ == "classification" else None
        classes = self.classes_
        use_holdout = cfg.holdout > 0 and len(y) >= cfg.min_holdout
        if use_holdout:
            strat = y if self.task_ == "classification" else None
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=cfg.holdout, random_state=cfg.seed, stratify=strat
            )
        else:
            Xtr, Xva, ytr, yva = X, X, y, y

        pool: list[tuple[str, str, dict[str, Any], np.ndarray]] = []
        for name in chosen:
            if cfg.time_budget and (time.perf_counter() - t_start) > cfg.time_budget:
                break
            cap = self._cap(name)
            cands, secs = self._optimize(cap, Xtr, ytr, classes, t_start)
            if not cands:
                continue
            best_params, best_oof, best_score = cands[0]
            self.params_[name] = best_params
            self.leaderboard_.append(
                FitResult(name=name, params=best_params, metric=best_score, fit_sec=secs,
                          diagnostics={"n_trials_ok": len(cands)})
            )
            router.observe(self.meta_, name, self._normalize(best_score))
            if cfg.verbose:
                print(f"[tianshu] {name:<20} {self.metric_.name}={best_score:.5f}  ({secs:.2f}s)")

            keep = cands if cfg.pool_per_model is None else cands[: max(1, cfg.pool_per_model)]
            for rank, (params, oof, _score) in enumerate(keep):
                key = name if len(keep) == 1 else f"{name}#{rank}"
                if use_holdout:
                    est = cap.factory(params)
                    est.fit(Xtr, ytr)
                    col = self._matrix(est, Xva, classes)
                else:
                    col = oof          # 数据太少，退回 OOF（偏差已知且已在文档说明）
                pool.append((key, name, params, col))

        if not pool:
            raise RuntimeError("all candidates failed; check data and dependencies")

        P = np.stack([p[3] for p in pool], axis=0)      # (M, N_va) 或 (M, N_va, C)
        fc = FusionCore(
            task=self.task_, metric=self.metric_, budget=cfg.fusion_budget,
            bagged=cfg.fusion_bagged, margin=cfg.fusion_margin, seed=cfg.seed,
        )
        self.fusion_ = fc.fit(P, yva)
        self.holdout_ = bool(use_holdout)

        # 池中每个候选都用其超参在**全量**数据上重拟合（供 predict 使用）
        for key, name, params, _col in pool:
            est = self._cap(name).factory(params)
            est.fit(X, y)
            self.models_[key] = est

        # 未被采纳时，回退对象必须选对：
        #   留出集上的最优 ≠ CV 上的最优。留出集通常只有 20% 样本，
        #   用它做**模型选择**方差很大（实测比 CV 选择差 2.7 个点的准确率）。
        #   CV 用了全部训练样本的折外预测，是更可靠的选择信号。
        #   因此：融合赌的是「组合增益」，赌输了就退回最稳的 CV 冠军，
        #   而不是退回留出集上的临时冠军。
        if not self.fusion_.accepted and self.leaderboard_:
            best = (max if self.metric_.higher else min)(
                self.leaderboard_, key=lambda r: r.metric
            )
            key = next((k for k in self.models_
                        if k == best.name or k.startswith(best.name + "#")), None)
            if key is not None:
                idx = list(self.models_).index(key)
                w = np.zeros(len(self.models_))
                w[idx] = 1.0
                self.fusion_.weights = w
                self.fusion_.method = "single(cv-best)"
                self.fusion_.diagnostics["fallback"] = {
                    "reason": "fusion not accepted on holdout",
                    "cv_best": best.name,
                    "cv_metric": round(float(best.metric), 6),
                }

        self.elapsed_ = time.perf_counter() - t_start
        return self

    def _matrix(self, est, X, classes: np.ndarray | None) -> np.ndarray:
        """取单个估计器在 X 上的预测矩阵：分类为概率 (n,C)，回归为 (n,)。"""
        if self.task_ == "classification":
            C = len(classes) if classes is not None else 0
            p = est.predict_proba(X)
            if p is None or (C and p.shape[1] != C):
                labs = est.predict(X)
                idx = np.clip(np.searchsorted(classes, labs), 0, max(C - 1, 0))
                return np.eye(C)[idx]
            return np.asarray(p)
        return np.asarray(est.predict(X), dtype=float)

    # =====================================================================
    # predict
    # =====================================================================
    def predict(self, X) -> np.ndarray:
        if not self.models_ or self.fusion_ is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X)
        parts = []
        order = list(self.models_)
        C = len(self.classes_) if self.classes_ is not None else 0
        for name in order:
            est = self.models_[name]
            if self.task_ == "classification":
                p = est.predict_proba(X)
                # 类别数必须取自**训练集**（self.classes_）。
                # 若从预测结果推断，退化模型（预测全落一类）会得到 C=1，
                # 与其余候选的列数不一致，np.stack 直接崩——这是实战踩过的坑。
                if p is None or p.shape[1] != C:
                    labs = est.predict(X)
                    idx = np.clip(np.searchsorted(self.classes_, labs), 0, C - 1)
                    p = np.eye(C)[idx]
                parts.append(np.asarray(p))
            else:
                parts.append(np.asarray(est.predict(X), dtype=float))
        P = np.stack(parts, axis=0)
        w = self.fusion_.weights
        s = w.sum()
        w = w / s if s > 0 else np.ones(len(order)) / len(order)
        if P.ndim == 3:
            avg = np.tensordot(w, P, axes=(0, 0))
            return np.asarray(np.argmax(avg, axis=1))
        return np.asarray(np.tensordot(w, P, axes=(0, 0)))

    def predict_proba(self, X) -> np.ndarray | None:
        if self.task_ != "classification" or not self.models_:
            return None
        X = np.asarray(X)
        parts, ws = [], []
        for i, name in enumerate(self.models_):
            p = self.models_[name].predict_proba(X)
            if p is None:
                continue
            parts.append(p)
            ws.append(self.fusion_.weights[i])
        if not parts:
            return None
        parts = [p for p in parts]
        C = min(p.shape[1] for p in parts)
        parts = [p[:, :C] for p in parts]
        w = np.array(ws) / max(sum(ws), 1e-12)
        return np.tensordot(w, np.stack(parts, axis=0), axes=(0, 0))

    # =====================================================================
    # 报告
    # =====================================================================
    def report(self) -> dict[str, Any]:
        best_single = max(self.leaderboard_, key=lambda r: r.metric) if self.metric_.higher \
            else min(self.leaderboard_, key=lambda r: r.metric)
        return {
            "task": self.task_,
            "metric": self.metric_.name,
            "n_models": len(self.models_),
            "elapsed_sec": round(self.elapsed_, 3),
            "ranking": [(n, round(s, 4)) for n, s in self.ranking_],
            "leaderboard": [r.as_row() for r in self.leaderboard_],
            "best_single": {"name": best_single.name, "metric": round(best_single.metric, 6)},
            "fusion": {
                "method": self.fusion_.method if self.fusion_ else None,
                "accepted": self.fusion_.accepted if self.fusion_ else None,
                "gain": round(self.fusion_.gain, 6) if self.fusion_ else None,
                "val_loss": round(self.fusion_.val_loss, 6) if self.fusion_ else None,
                "weights": {n: round(float(w), 4) for n, w in
                            zip(self.models_, self.fusion_.weights, strict=False)}
                if self.fusion_ else {},
            },
            "metafeatures": self.meta_.round(6).tolist() if self.meta_ is not None else [],
        }

    def summary(self) -> str:
        r = self.report()
        lines = [
            f"TianShu · task={r['task']} · metric={r['metric']} · {r['n_models']} models · "
            f"{r['elapsed_sec']}s",
            f"  元路由 Top3: {', '.join(n for n, _ in r['ranking'][:3])}",
            f"  最强单模型: {r['best_single']['name']} = {r['best_single']['metric']:.6f}",
        ]
        if self.fusion_:
            lines.append("  " + self.fusion_.report())
        return "\n".join(lines)

    # =====================================================================
    # 内部
    # =====================================================================
    def _cap(self, name: str):
        from ..core.registry import get

        return get(name)

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        y = np.asarray(y)
        if y.dtype.kind in "fc" and len(np.unique(y)) > 20:
            return "regression"
        return "classification"

    def _normalize(self, score: float) -> float:
        """把度量压到 [0,1]（元路由记忆需要可比分数）。"""
        v = float(score)
        if not np.isfinite(v):
            return 0.0
        if self.metric_.higher:
            return float(np.clip(v, 0.0, 1.0)) if self.task_ == "classification" \
                else float(np.clip(v, -1.0, 1.0) / 2 + 0.5)
        # loss 型：用 log 压缩到 (0,1]
        return float(1.0 / (1.0 + max(v, 0.0)))

    def _cv_folds(self, y: np.ndarray):
        cfg = self.config
        # 自适应折数：小样本 3 折既稳又快（5 折在 n<400 时方差优势有限，
        # 却要多付 67% 的训练时间——GBM 族在大 trial 数下这点非常明显）。
        base = cfg.cv if cfg.cv is not None else (3 if len(y) < 400 else 5)
        cv = max(2, min(base, int(np.min(np.bincount(
            np.unique(y, return_inverse=True)[1])) if self.task_ == "classification" else len(y))))
        if self.task_ == "classification":
            return list(StratifiedKFold(n_splits=cv, shuffle=True,
                                        random_state=cfg.seed).split(np.zeros(len(y)), y))
        return list(KFold(n_splits=cv, shuffle=True, random_state=cfg.seed).split(np.zeros(len(y))))

    def _oof(self, builder, X, y, folds, classes):
        """交叉验证收集 OOF 预测 + 分数。返回 (oof, score)。"""
        n = len(y)
        if self.task_ == "classification":
            C = len(classes) if classes is not None else len(np.unique(y))
            oof = np.zeros((n, C))
            for tr, te in folds:
                m = builder()
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[te])
                if p is None or p.shape[1] != C:
                    labs = m.predict(X[te])
                    idx = np.clip(np.searchsorted(classes if classes is not None
                                                  else np.unique(y), labs), 0, C - 1)
                    p = np.eye(C)[idx]
                oof[te] = p
            pred = np.argmax(oof, axis=1)
            if classes is not None:
                pred = classes[np.clip(pred, 0, len(classes) - 1)]
            return oof, self.metric_(y, pred)
        oof = np.zeros(n)
        for tr, te in folds:
            m = builder()
            m.fit(X[tr], y[tr])
            oof[te] = np.asarray(m.predict(X[te]), dtype=float)
        return oof, self.metric_(y, oof)

    def _optimize(self, cap, X, y, classes, t_start):
        cfg = self.config
        folds = self._cv_folds(y)
        space = cap.space
        searcher = make_searcher(space, seed=spawn_seed(cfg.seed, cap.name),
                                 maximize=self.metric_.higher, strategy=cfg.hpo)
        n_trials = 1 if len(space) == 0 else cfg.n_trials

        results: list[tuple[dict[str, Any], np.ndarray, float]] = []
        t0 = time.perf_counter()
        for _ in range(n_trials):
            if cfg.time_budget and (time.perf_counter() - t_start) > cfg.time_budget:
                break
            params = searcher.ask()
            oof = None
            try:
                oof, score = self._oof(lambda p=params: cap.factory(p), X, y, folds, classes)
            except Exception:
                score = -np.inf if self.metric_.higher else np.inf
            searcher.tell(params, score)
            if oof is not None and np.isfinite(score):
                results.append((params, oof, float(score)))
        secs = time.perf_counter() - t0
        if not results:
            return [], secs
        # 最好的排前面；保留全部（调用方按 pool_per_model 截取组成融合池）
        results.sort(key=lambda r: r[2], reverse=bool(self.metric_.higher))
        return results, secs
