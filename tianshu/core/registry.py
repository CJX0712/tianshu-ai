"""L0 · 能力注册表 —— 系统可扩展性的唯一入口。

为什么不是硬编码字典：
    AutoML 系统的价值与「可接入算法数」超线性相关。注册表把「有哪些算法」
    从控制流中剥离，使得新增算法 = 注册一行，而非改十处分支。
    这是 scikit-learn / HuggingFace / MLflow 共同验证过的架构决策。

内置能力全部是**世界顶级开源实现的适配层**（零重写）：
    sklearn 的 HistGradientBoosting / RandomForest / SVM / GP / MLP …
    xgboost / lightgbm（若已安装）自动启用，缺失则静默降级。
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator

from .contracts import Capability, SearchSpace, Task

_LOCK = threading.RLock()
_REGISTRY: dict[str, Capability] = {}


def register(
    name: str,
    task: Task,
    factory: Callable,
    space: SearchSpace | None = None,
    family: str = "general",
    cost: float = 1.0,
    requires: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> Capability:
    """注册一项能力。重名覆盖（幂等重载，便于交互式迭代）。"""
    cap = Capability(
        name=name,
        task=task,
        factory=factory,
        space=space or SearchSpace(),
        family=family,
        cost=cost,
        requires=requires,
        tags=tags,
    )
    with _LOCK:
        _REGISTRY[name] = cap
    return cap


def available(name: str) -> bool:
    """依赖是否齐备（缺 xgboost 时不应崩溃，应优雅缺席）。"""
    cap = _REGISTRY.get(name)
    if cap is None:
        return False
    return all(_has(pkg) for pkg in cap.requires)


def _has(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


def get(name: str) -> Capability:
    with _LOCK:
        if name not in _REGISTRY:
            raise KeyError(f"capability {name!r} not registered; have {sorted(_REGISTRY)[:20]}...")
        return _REGISTRY[name]


def list_capabilities(task: Task | None = None, only_available: bool = True) -> list[Capability]:
    with _LOCK:
        caps = list(_REGISTRY.values())
    if task is not None:
        caps = [c for c in caps if c.task == task]
    if only_available:
        caps = [c for c in caps if all(_has(p) for p in c.requires)]
    return sorted(caps, key=lambda c: (c.family, c.name))


def iter_names(task: Task | None = None) -> Iterator[str]:
    for c in list_capabilities(task):
        yield c.name


def unregister(name: str) -> None:
    with _LOCK:
        _REGISTRY.pop(name, None)


def clear() -> None:
    with _LOCK:
        _REGISTRY.clear()
