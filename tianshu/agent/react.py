"""L8 · ReAct 智能体 —— 推理与行动交织的闭环（Yao et al. 2022）。

零重写：ReAct 范式出自 Yao et al. 2022《ReAct: Synergizing Reasoning and
Acting in Language Models》。本模块实现其**控制流**，并做三处工程加固：

    1. LLM 后端抽象（Protocol）
       默认 MockBackend：确定性、零网络、零密钥 —— 这让整条 Agent 链路
       在完全离线的机器上也能端到端验证（CI 友好）。
       生产可切成 OpenAI 兼容后端 / 本地 llama.cpp(GGUF)，无需改控制流。
    2. 步数上限 + 重复动作检测
       防止模型陷入「同一动作反复调用」的死循环（LLM Agent 最常见的失控）。
    3. 每一步都是可审计事件（Thought/Action/Observation 全记录），
       便于回放与归因，而不是只给一个最终答案。

输出契约：
    AgentRun(answer, steps, tools_used, success, reason)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import ToolRegistry, default_tools


class LLMBackend(Protocol):
    def generate(self, prompt: str, stop: tuple[str, ...] = ()) -> str: ...


class MockBackend:
    """确定性后端：能识别 `Action: calculator(表达式)` 并真实调用，
    其余情况给一个可预测的收尾。用于离线验证控制流，不用于真实问答。"""

    name = "mock"

    def __init__(self, final: str = "已基于工具结果给出答案。"):
        self.final = final

    def generate(self, prompt: str, stop: tuple[str, ...] = ()) -> str:
        # 若上一步已给出 Observation，则收尾
        if "Observation:" in prompt.split("Thought:")[-1]:
            return f"Thought: 信息已足够。\nFinal Answer: {self.final}"
        m = re.search(r"[Cc]alc\w*\s*(.+?)(?:$|\n)", prompt)
        if m:
            return f"Thought: 需要计算。\nAction: calculator({m.group(1).strip()})"
        return f"Thought: 无工具可用。\nFinal Answer: {self.final}"


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    observation: str = ""
    ok: bool = True


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    success: bool = False
    reason: str = ""

    def transcript(self) -> str:
        out = [f"Q: {self.question}"]
        for i, s in enumerate(self.steps, 1):
            out.append(f"[{i}] Thought: {s.thought}")
            if s.action:
                out.append(f"    Action: {s.action}")
                out.append(f"    Observation: {s.observation[:200]}")
        out.append(f"A: {self.answer}")
        return "\n".join(out)


SYSTEM = """你是一个可以使用工具的智能体。严格按以下格式循环输出：

Thought: <你的推理>
Action: <tool_name>(<参数>)
Observation: <工具返回，系统填入>

当你有足够信息时输出：
Thought: <总结>
Final Answer: <最终答案>

可用工具：
{tools}

规则：一次只调用一个工具；不要重复调用完全相同的动作。
"""


class ReActAgent:
    """ReAct 循环控制器。"""

    def __init__(self, backend: LLMBackend | None = None, tools: ToolRegistry | None = None,
                 max_steps: int = 6):
        self.backend = backend or MockBackend()
        self.tools = tools or default_tools()
        self.max_steps = max_steps

    def run(self, question: str) -> AgentRun:
        prompt = SYSTEM.format(tools=self.tools.describe()) + f"\n\nQ: {question}\nThought:"
        run = AgentRun(question=question, answer="")
        seen: set[str] = set()

        for _ in range(self.max_steps):
            out = self.backend.generate(prompt, stop=("Observation:",))
            step = Step()
            m_th = re.search(r"Thought:\s*(.*)", out)
            step.thought = m_th.group(1).strip() if m_th else ""

            m_fa = re.search(r"Final Answer:\s*(.*)", out, re.S)
            if m_fa:
                run.answer = m_fa.group(1).strip()
                run.steps.append(step)
                run.success = True
                run.reason = "final"
                return run

            m_ac = re.search(r"Action:\s*(\w+)\((.*)\)\s*$", out, re.M | re.S)
            if not m_ac:
                run.steps.append(step)
                run.success = False
                run.reason = "no_action_parsed"
                run.answer = "未能解析出动作，终止。"
                return run

            name, arg = m_ac.group(1), m_ac.group(2).strip().strip('"').strip("'")
            key = f"{name}({arg})"
            if key in seen:
                run.steps.append(step)
                run.success = False
                run.reason = "repeated_action"
                run.answer = "检测到重复动作，终止以避免死循环。"
                return run
            seen.add(key)

            tool = self.tools.get(name)
            if tool is None:
                step.action = key
                step.observation = f"未知工具: {name}"
                step.ok = False
            else:
                res = tool(arg)
                step.action = key
                step.observation = res.output if res.ok else f"ERROR: {res.error}"
                step.ok = res.ok
                run.tools_used.append(name)
            run.steps.append(step)
            prompt += f" {step.thought}\nAction: {key}\nObservation: {step.observation}\nThought:"

        run.success = False
        run.reason = "max_steps"
        run.answer = "达到步数上限。"
        return run


class OpenAICompatBackend:
    """OpenAI 兼容后端（含 vLLM / Ollama / llama.cpp server 的 /v1 接口）。

    需要 base_url 与可选 api_key；不传 api_key 时按无鉴权本地服务处理。
    未安装 requests 时不阻塞导入，调用时报错。
    """

    name = "openai"

    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5:0.5b", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, stop: tuple[str, ...] = ()) -> str:
        import json
        import urllib.request

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = list(stop)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/completions", data=data,
            headers={"Content-Type": "application/json"}
            | ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        return str(body.get("choices", [{}])[0].get("text", ""))


def make_agent(retriever: Any | None = None, backend: LLMBackend | None = None,
               max_steps: int = 6) -> ReActAgent:
    """便捷构造：带检索工具 + 指定后端的智能体。"""
    return ReActAgent(backend=backend, tools=default_tools(retriever), max_steps=max_steps)
