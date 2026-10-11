"""L0 · 契约层 —— 系统内一切可插拔能力的抽象边界。

设计哲学（借鉴 scikit-learn 的 fit/predict 契约 + PyTorch nn.Module 的
参数自省 + Ray Tune 的 search space 描述）：
    一个「能力」只需满足 Estimator 协议即可接入元路由、HPO、融合与门禁，
    无需继承任何基类，也无需知道系统其余部分的存在。

这带来的直接收益：
    - 从 sklearn / xgboost / lightgbm / 自研 forge 家族接入一个新模型，
      只需写一个薄 adapter，不动系统任何一行。
    - 测试可以用 MockEstimator 完全离线验证（无网络、无 GPU、无密钥）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

Task = str  # 'classification' | 'regression' | 'ranking' | 'unsupervised' | 'retrieval'


@dataclass
class SearchSpace:
    """超参搜索空间声明 —— HPO 层据此采样，元路由层据此估算代价。

    每个维度: name -> {'type': 'float'|'int'|'categorical', ...}
    float:   low, high, log (是否对数均匀, 学习率/正则必须 True)
    int:     low, high
    categorical: choices
    """

    dims: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_float(self, name: str, low: float, high: float, log: bool = False) -> SearchSpace:
        self.dims[name] = {"type": "float", "low": low, "high": high, "log": log}
        return self

    def add_int(self, name: str, low: int, high: int) -> SearchSpace:
        self.dims[name] = {"type": "int", "low": low, "high": high}
        return self

    def add_cat(self, name: str, choices: list[Any]) -> SearchSpace:
        self.dims[name] = {"type": "categorical", "choices": list(choices)}
        return self

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, d in self.dims.items():
            t = d["type"]
            if t == "float":
                if d.get("log"):
                    out[name] = float(np.exp(rng.uniform(np.log(d["low"]), np.log(d["high"]))))
                else:
                    out[name] = float(rng.uniform(d["low"], d["high"]))
            elif t == "int":
                out[name] = int(rng.integers(d["low"], d["high"] + 1))
            else:
                ch = d["choices"]
                out[name] = ch[int(rng.integers(0, len(ch)))]
        return out

    def defaults(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, d in self.dims.items():
            t = d["type"]
            if t == "float":
                lo, hi = d["low"], d["high"]
                out[name] = float(np.sqrt(max(lo, 1e-12) * hi)) if d.get("log") else float((lo + hi) / 2)
            elif t == "int":
                # 取下四分位而非中位数：整数维的上界常是「极端配置」
                # （如 PCA n_components），中位数容易越界或过慢。
                out[name] = int(d["low"] + (d["high"] - d["low"]) * 0.25)
            else:
                out[name] = d["choices"][0]
        return out

    def __len__(self) -> int:
        return len(self.dims)


@runtime_checkable
class Estimator(Protocol):
    """可接入系统的最小能力面。"""

    name: str
    task: Task

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Estimator: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def space(self) -> SearchSpace: ...


@dataclass
class Capability:
    """能力卡：注册表里的一行。含代价模型所需的先验（供元路由做预算感知）。"""

    name: str
    task: Task
    factory: Any  # Callable[..., Estimator]
    space: SearchSpace
    family: str = "general"
    cost: float = 1.0          # 相对训练代价先验（1.0 = 基准）
    requires: tuple[str, ...] = ()   # 可选依赖包名，缺失则自动跳过
    tags: tuple[str, ...] = ()


@dataclass
class FitResult:
    """一次拟合的完整可审计记录（门禁与报告消费它）。"""

    name: str
    params: dict[str, Any]
    metric: float
    fit_sec: float
    predict_sec: float = 0.0
    oof: np.ndarray | None = None      # out-of-fold 预测，融合层的输入
    test: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": round(self.metric, 6),
            "fit_sec": round(self.fit_sec, 4),
            "params": {k: v for k, v in self.params.items()},
        }
