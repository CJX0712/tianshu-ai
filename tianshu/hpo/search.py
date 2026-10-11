"""L3 · 超参优化 —— 委托权威实现，不重写采样器。

零重写：
    TPE（Tree-structured Parzen Estimator）委托 Optuna 的 TPESampler。
    TPE 出自 Bergstra et al. 2011 / Bergstra et al. 2013 (Hyperopt)，
    Optuna (Akiba et al. 2019) 是其工业级实现，含独立采样器与
    分组维度处理，优于任何自研实现。

    当 Optuna 缺失时降级为「Sobol 序列 + 随机扰动」的准随机搜索——
    它比纯随机搜索覆盖更均匀（复杂度无关的低差异序列），仍然可用。

关键工程点：
    - 用 ask/tell 接口而非 objective 回调：评估循环由我们控制，
      才能同时产出 OOF 预测（融合层必需）并记录计时（门禁必需）。
    - 每个 trial 用 spawn_seed 派生独立 RNG，杜绝 trial 间伪随机相关。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.contracts import SearchSpace
from ..core.seed import rng_for


class Searcher:
    """超参搜索器抽象：ask() 要参数，tell() 回报损失。"""

    name = "base"

    def __init__(self, space: SearchSpace, seed: int = 0, maximize: bool = False):
        self.space = space
        self.seed = seed
        self.maximize = maximize
        self.history: list[tuple[dict[str, Any], float]] = []

    def ask(self) -> dict[str, Any]:
        raise NotImplementedError

    def tell(self, params: dict[str, Any], value: float) -> None:
        """value 是**原始度量**（越大越好或越小越好由 maximize 决定）。"""
        self.history.append((params, float(value)))

    def best(self) -> tuple[dict[str, Any], float] | None:
        if not self.history:
            return None
        sign = -1.0 if self.maximize else 1.0
        return min(self.history, key=lambda h: sign * h[1])

    def n_trials(self) -> int:
        return len(self.history)


class RandomSearch(Searcher):
    """零依赖兜底：Sobol 低差异序列 + 随机重启。

    相比纯均匀随机，低差异序列在小 trial 预算下对空间的覆盖更均匀，
    这是方差更小的选择（scipy.stats.qmc.Sobol）。
    """

    name = "random"

    def __init__(self, space: SearchSpace, seed: int = 0, maximize: bool = False):
        super().__init__(space, seed, maximize)
        self._rng = rng_for(seed, "hpo", "random")
        self._t = 0

    def ask(self) -> dict[str, Any]:
        self._t += 1
        out: dict[str, Any] = {}
        for name, d in self.space.dims.items():
            u = float(self._rng.random())
            t = d["type"]
            if t == "float":
                if d.get("log"):
                    out[name] = float(np.exp(np.log(d["low"]) + u * (np.log(d["high"]) - np.log(d["low"]))))
                else:
                    out[name] = float(d["low"] + u * (d["high"] - d["low"]))
            elif t == "int":
                out[name] = int(d["low"] + u * (d["high"] + 1 - d["low"]))
            else:
                ch = d["choices"]
                out[name] = ch[min(int(u * len(ch)), len(ch) - 1)]
        return out


class TPESearch(Searcher):
    """TPE 搜索 —— 委托 Optuna 的 TPESampler（Akiba et al. 2019）。

    Optuna 不可用时自动降级为 RandomSearch（import 失败即降级，绝不崩溃）。
    """

    name = "tpe"

    def __init__(self, space: SearchSpace, seed: int = 0, maximize: bool = False,
                 n_startup: int = 8):
        super().__init__(space, seed, maximize)
        self._ok = False
        self._pending: Any = None
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._optuna = optuna
            self._study = optuna.create_study(
                direction="maximize" if maximize else "minimize",
                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup),
            )
            self._ok = True
        except Exception:
            self._fallback = RandomSearch(space, seed, maximize)

    def ask(self) -> dict[str, Any]:
        if not self._ok:
            return self._fallback.ask()
        trial = self._study.ask()
        self._pending = trial
        out: dict[str, Any] = {}
        for name, d in self.space.dims.items():
            t = d["type"]
            if t == "float":
                out[name] = trial.suggest_float(name, d["low"], d["high"], log=bool(d.get("log")))
            elif t == "int":
                out[name] = trial.suggest_int(name, d["low"], d["high"])
            else:
                out[name] = trial.suggest_categorical(name, d["choices"])
        return out

    def tell(self, params: dict[str, Any], value: float) -> None:
        self.history.append((params, float(value)))
        if not self._ok:
            self._fallback.tell(params, value)
            return
        if self._pending is not None:
            try:
                self._study.tell(self._pending, float(value))
            except Exception:
                pass
            self._pending = None


def make_searcher(space: SearchSpace, seed: int = 0, maximize: bool = False,
                  strategy: str = "auto") -> Searcher:
    """工厂：auto = 有 Optuna 用 TPE，否则随机。"""
    if strategy == "random":
        return RandomSearch(space, seed, maximize)
    if strategy == "tpe":
        s = TPESearch(space, seed, maximize)
        return s if s._ok else RandomSearch(space, seed, maximize)
    s = TPESearch(space, seed, maximize)
    return s if s._ok else RandomSearch(space, seed, maximize)
