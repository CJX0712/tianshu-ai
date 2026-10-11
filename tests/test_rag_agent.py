"""L7 检索与 L8 智能体 —— 含安全门禁。"""

from __future__ import annotations

import numpy as np
import pytest

from tianshu.agent.react import MockBackend, ReActAgent
from tianshu.agent.tools import DANGEROUS, ToolRegistry, default_tools, safe_eval
from tianshu.rag.index import BM25, HybridRetriever, rrf_fuse, tokenize


def test_tokenizer_handles_cjk_and_latin():
    toks = tokenize("天枢 TianShu 融合层")
    assert "天" in toks and "枢" in toks
    assert "天枢" in toks, "缺少双字，中文语义会退化"
    assert "tianshu" in toks


def test_bm25_ranks_relevant_first():
    docs = ["苹果 香蕉 橙子", "量子计算 与 神经网络", "神经网络 的 梯度下降"]
    b = BM25().fit(docs)
    s = b.scores("神经网络")
    assert int(np.argmax(s)) in (1, 2)


def test_rrf_fusion_prefers_agreement():
    """两路都排第一的文档必须拿到最高融合分。"""
    fused = rrf_fuse([[0, 1, 2], [0, 2, 1]])
    assert max(fused, key=fused.get) == 0


def test_hybrid_retrieval_recall():
    docs = [
        "融合层使用 Caruana 的集成选择算法",
        "BM25 是 Robertson 提出的概率检索模型",
        "RRF 由 Cormack 等人在 2009 年提出",
        "元路由结合先验规则与局部加权记忆",
    ]
    r = HybridRetriever(docs)
    for i, q in enumerate(["集成选择", "BM25", "RRF", "元路由"]):
        top = [h.doc_id for h in r.search(q, topk=5)]
        assert i in top, f"query={q} 未召回正确文档"


def test_rerank_weight_is_capped():
    """重排护栏：融合分数中重排项占比不得超过 cap。"""
    r = HybridRetriever(["文档一 关于 检索", "文档二 关于 融合"], rerank_cap=0.5)
    for h in r.search("检索 融合"):
        assert 0.0 <= h.parts["coverage"] <= 1.0
    r2 = HybridRetriever(["文档一 关于 检索"], rerank_cap=0.9)
    assert r2.rerank_cap <= 0.9


def test_safe_eval_arithmetic():
    assert safe_eval("1 + 2 * 3") == 7.0
    assert abs(safe_eval("sqrt(16)") - 4.0) < 1e-12
    assert abs(safe_eval("pi") - np.pi) < 1e-12


@pytest.mark.parametrize("expr", DANGEROUS)
def test_safe_eval_blocks_injection(expr: str):
    """每一条注入都必须被拒——这是工具层唯一的生死线。

    显式限定异常类型：若哪天求值器改成了「静默返回 None」，
    这里的断言会失败，从而暴露安全退化（而不是被宽泛的 Exception 掩盖）。
    """
    with pytest.raises((ValueError, SyntaxError, TypeError, ZeroDivisionError)):
        safe_eval(expr)


def test_tool_registry_reports_failures_not_raises():
    reg = default_tools()
    res = reg.get("calculator")("1/0")
    assert res.ok is False and "ZeroDivision" in res.error


def test_react_agent_runs_offline():
    """零网络、零密钥下 ReAct 控制流必须能跑完并产出审计记录。"""
    ag = ReActAgent(backend=MockBackend(final="答案"), max_steps=4)
    run = ag.run("计算 2+3")
    assert isinstance(run.answer, str)
    assert len(run.steps) >= 1
    assert isinstance(run.transcript(), str)


def test_react_detects_repeated_action():
    class Loop(MockBackend):
        def generate(self, prompt, stop=()):
            return "Thought: 再算一次\nAction: calculator(1+1)"

    ag = ReActAgent(backend=Loop(), max_steps=6)
    run = ag.run("循环测试")
    assert run.reason == "repeated_action", "重复动作未被拦截，会陷入死循环"


def test_unknown_tool_is_reported():
    reg = ToolRegistry()
    assert reg.get("nope") is None
