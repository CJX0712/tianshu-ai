"""天枢 TianShu — 世界顶级 AI 系统（作者：晨星）

十层架构：
    L0 core      确定性 / 度量 / 契约 / 注册表
    L1 data      元特征（让系统看清数据）
    L2 models    算法家族（全部适配世界顶级开源实现）
    L3 hpo       超参优化（TPE，委托 Optuna）
    L4 fusion    融合旗舰 FusionCore（Caruana 集成选择 + 堆叠 + 门控）
    L5 meta      元路由（先验 + 局部加权记忆 + UCB）
    L6 runtime   引擎（端到端 AutoML）
    L7 rag       混合检索（BM25 + 稠密向量 + RRF 融合）
    L8 agent     ReAct 智能体（工具调用 + 记忆）
    L9 serve     服务与观测（零依赖 HTTP）
    L10 verify   不变量与交付门禁

快速开始：
    >>> from tianshu import Tianshu
    >>> m = Tianshu().fit(X, y)
    >>> m.predict(X_test)
"""

from __future__ import annotations

from .core.metrics import get_metric
from .core.registry import list_capabilities, register
from .fusion.fusion_core import FusionCore
from .meta.router import MetaRouter
from .models.families import install_builtins
from .runtime.engine import EngineConfig, Tianshu

__version__ = "1.0.0"
__author__ = "晨星"

__all__ = [
    "Tianshu", "EngineConfig", "FusionCore", "MetaRouter",
    "get_metric", "list_capabilities", "register", "install_builtins",
    "__version__", "__author__",
]

# 导入即注册内置算法家族（幂等）
try:
    install_builtins()
except Exception:  # pragma: no cover - 依赖缺失时不阻断 import
    pass
