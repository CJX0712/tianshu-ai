"""L8 · 工具层 —— 智能体的「手」，每一只都必须安全可审计。

安全设计（这是工具调用最大的风险面）：
    1. 计算器走 **AST 白名单求值**，绝不 eval()。
       只允许数字常量与四则/幂/取模运算符，任何 Name/Call/Attribute 节点
       一律拒绝——这挡住了 `__import__('os').system(...)` 之类的注入。
    2. 每个工具有 name/description/args_schema，供 LLM 选择与参数校验。
    3. 工具执行结果统一包成 ToolResult(ok, output, error)，
       失败不上抛，交给智能体去推理（ReAct 的核心：把错误也当成观察）。
"""

from __future__ import annotations

import ast
import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "pow": math.pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    """AST 白名单求值器。只接受算术表达式。"""
    tree = ast.parse(str(expr).strip(), mode="eval")

    def _n(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float):
                return float(node.value)
            raise ValueError("only numeric constants allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](_n(node.left), _n(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](_n(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError(f"function not allowed: {getattr(node.func, 'id', '?')}")
            return _FUNCS[node.func.id](*[_n(a) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ValueError(f"name not allowed: {node.id}")
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    return float(_n(tree.body))


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]
    args: str = "expr: str"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __call__(self, inp: str) -> ToolResult:
        try:
            return ToolResult(True, str(self.fn(inp)))
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, "", f"{type(e).__name__}: {e}")


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}({t.args}): {t.description}" for t in self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def default_tools(retriever: Any | None = None) -> ToolRegistry:
    """默认工具集。含计算器、检索（可选）、时间与回显。"""
    reg = ToolRegistry()
    reg.add(Tool(
        "calculator", "计算数学表达式，支持 abs/sqrt/log/exp/sin/cos/floor/ceil/pow 与常量 pi/e",
        lambda s: str(safe_eval(s)), tags=("math",),
    ))
    if retriever is not None:
        reg.add(Tool(
            "search", "在本地知识库中检索与查询相关的文档片段，返回编号与内容",
            lambda q: "\n".join(
                f"[{i}] {h.text[:300]}" for i, h in enumerate(retriever.search(q, topk=3))
            ) or "(无结果)",
            args="query: str", tags=("retrieval",),
        ))
    reg.add(Tool(
        "now", "返回当前时间（ISO 格式）",
        lambda _s: time.strftime("%Y-%m-%d %H:%M:%S"), args="", tags=("system",),
    ))
    reg.add(Tool(
        "echo", "原样返回输入，用于自检链路是否连通",
        lambda s: s, tags=("debug",),
    ))
    return reg


# ---- 安全门禁用例（测试引用） ----
DANGEROUS = [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd').read()",
    "(1).__class__.__bases__[0].__subclasses__()",
    "eval('1+1')",
]
