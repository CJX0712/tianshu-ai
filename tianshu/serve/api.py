"""L9 · 服务层 —— 零第三方依赖的 HTTP 网关（仅用标准库）。

为什么不用 FastAPI/Flask：
    本系统的定位是「可在无 GPU、无网络、无密钥的机器上完整跑通并验证」。
    把服务层压到标准库 http.server，意味着 `pip install tianshu-ai` 之后
    不需要再拉任何 web 依赖就能起服务。生产环境要 ASGI/uvicorn 时，
    把同样的 handler 挂上去即可，路由逻辑零改动。

端点：
    GET  /health                存活 + 后端可用性
    POST /automl/fit            {"X": [[...]], "y": [...]} → 训练并返回报告
    POST /automl/predict        {"X": [[...]]} → 预测
    POST /rag/search            {"docs": [...], "query": "...", "topk": 5}
    POST /agent/run             {"question": "...", "docs": [...]}
    GET  /                      单文件控制台（若存在 console.html）
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


class _Handler(BaseHTTPRequestHandler):
    server_ref: Any = None     # 由 TianshuServer 注入
    console_path: Path | None = None

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server_ref, "verbose", False):
            print(f"[tianshu.http] {self.address_string()} {fmt % args}")

    # ---- helpers ----
    def _send(self, code: int, payload: dict | bytes, ctype: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- routes ----
    def do_GET(self) -> None:  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/", "/console", "/index.html"):
            if self.console_path and self.console_path.exists():
                self._send(200, self.console_path.read_bytes(), "text/html")
            else:
                self._send(404, {"error": "console.html not found"})
            return
        if p == "/health":
            self._send(200, self._health())
            return
        if p == "/capabilities":
            from ..core.registry import list_capabilities

            self._send(200, {
                "classification": [c.name for c in list_capabilities("classification")],
                "regression": [c.name for c in list_capabilities("regression")],
                "unsupervised": [c.name for c in list_capabilities("unsupervised")],
            })
            return
        self._send(404, {"error": f"no route {p}"})

    def do_POST(self) -> None:  # noqa: N802
        p = self.path.split("?")[0]
        body = self._json_body()
        try:
            if p == "/automl/fit":
                self._send(200, self._fit(body))
            elif p == "/automl/predict":
                self._send(200, self._predict(body))
            elif p == "/rag/search":
                self._send(200, self._rag(body))
            elif p == "/agent/run":
                self._send(200, self._agent(body))
            else:
                self._send(404, {"error": f"no route {p}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    # ---- handlers ----
    def _health(self) -> dict:
        srv = self.server_ref
        return {
            "status": "ok",
            "version": getattr(srv, "version", "1.0.0"),
            "model_fitted": bool(getattr(srv, "model", None)),
            "backends": {
                "xgboost": _has("xgboost"),
                "lightgbm": _has("lightgbm"),
                "optuna": _has("optuna"),
            },
            "docs_loaded": getattr(srv, "n_docs", 0),
        }

    def _fit(self, body: dict) -> dict:
        from ..runtime.engine import EngineConfig, Tianshu

        X = np.asarray(body.get("X", []), dtype=float)
        y = np.asarray(body.get("y", []))
        cfg = {k: v for k, v in (body.get("config") or {}).items()
               if k in EngineConfig.__dataclass_fields__}
        m = Tianshu(EngineConfig(**cfg))
        m.fit(X, y)
        self.server_ref.model = m
        return {"ok": True, "report": m.report(), "summary": m.summary()}

    def _predict(self, body: dict) -> dict:
        m = self.server_ref.model
        if m is None:
            return {"ok": False, "error": "model not fitted; POST /automl/fit first"}
        X = np.asarray(body.get("X", []), dtype=float)
        pred = m.predict(X)
        out: dict[str, Any] = {"ok": True, "prediction": np.asarray(pred).tolist()}
        p = m.predict_proba(X)
        if p is not None:
            out["proba"] = np.asarray(p).tolist()
        return out

    def _rag(self, body: dict) -> dict:
        from ..rag.index import HybridRetriever

        docs = body.get("docs")
        srv = self.server_ref
        if docs is not None:
            srv.retriever = HybridRetriever(list(docs))
            srv.n_docs = len(docs)
        r = getattr(srv, "retriever", None)
        if r is None:
            return {"ok": False, "error": "no docs; provide 'docs'"}
        q = str(body.get("query", ""))
        topk = int(body.get("topk", 5))
        return {"ok": True, "query": q,
                "hits": [{"doc_id": h.doc_id, "score": round(h.score, 6), "text": h.text,
                          "parts": {k: round(v, 6) for k, v in h.parts.items()}}
                         for h in r.search(q, topk)]}

    def _agent(self, body: dict) -> dict:
        from ..agent.react import make_agent

        srv = self.server_ref
        docs = body.get("docs")
        if docs is not None:
            from ..rag.index import HybridRetriever

            srv.retriever = HybridRetriever(list(docs))
            srv.n_docs = len(docs)
        ag = make_agent(getattr(srv, "retriever", None))
        run = ag.run(str(body.get("question", "")))
        return {"ok": True, "answer": run.answer, "success": run.success,
                "reason": run.reason, "tools_used": run.tools_used,
                "transcript": run.transcript()}


def _has(pkg: str) -> bool:
    import importlib

    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


class TianshuServer:
    """天枢服务。serve() 阻塞；start_background() 返回线程。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8801, verbose: bool = False,
                 console: str | None = None):
        self.host, self.port = host, port
        self.verbose = verbose
        self.version = "1.0.0"
        self.model = None
        self.retriever = None
        self.n_docs = 0
        self.console = Path(console) if console else Path(__file__).resolve().parents[2] / "console.html"

    def _make(self) -> ThreadingHTTPServer:
        ns = type("H", (_Handler,), {"server_ref": self, "console_path": self.console})
        return ThreadingHTTPServer((self.host, self.port), ns)

    def serve(self) -> None:
        srv = self._make()
        print(f"TianShu serving on http://{self.host}:{self.port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()

    def start_background(self) -> tuple[ThreadingHTTPServer, threading.Thread]:
        srv = self._make()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, t
