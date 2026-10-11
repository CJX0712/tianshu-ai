"""L9 服务层 —— 用真实 HTTP 请求验证（不起 mock，跑真端口）。"""

from __future__ import annotations

import json
import urllib.request

import pytest
from sklearn.datasets import make_classification

from tianshu.serve.api import TianshuServer


@pytest.fixture(scope="module")
def server():
    srv = TianshuServer(host="127.0.0.1", port=0)      # port=0 → 系统分配空闲端口
    httpd, _t = srv.start_background()
    srv.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield srv
    httpd.shutdown()
    httpd.server_close()


def _get(srv, path):
    with urllib.request.urlopen(srv.base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def _post(srv, path, payload):
    req = urllib.request.Request(
        srv.base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def test_health(server):
    h = _get(server, "/health")
    assert h["status"] == "ok"
    assert "backends" in h and "xgboost" in h["backends"]


def test_capabilities(server):
    c = _get(server, "/capabilities")
    assert "classification" in c and len(c["classification"]) > 0


def test_automl_fit_predict_roundtrip(server):
    X, y = make_classification(n_samples=150, n_features=8, n_classes=2, random_state=0)
    fit = _post(server, "/automl/fit", {
        "X": X.tolist(), "y": y.tolist(),
        "config": {"n_trials": 2, "max_models": 3},
    })
    assert fit["ok"] is True
    assert "leaderboard" in fit["report"]
    pred = _post(server, "/automl/predict", {"X": X[:5].tolist()})
    assert pred["ok"] is True and len(pred["prediction"]) == 5


def test_rag_search_endpoint(server):
    r = _post(server, "/rag/search", {
        "docs": ["BM25 概率检索模型", "RRF 多路排名融合", "元路由 算法选择"],
        "query": "RRF 融合", "topk": 2,
    })
    assert r["ok"] is True and len(r["hits"]) <= 2
    assert any("RRF" in h["text"] for h in r["hits"])


def test_agent_endpoint_offline(server):
    r = _post(server, "/agent/run", {
        "docs": ["天枢是集大成 AI 系统"],
        "question": "天枢是什么",
    })
    assert r["ok"] is True and isinstance(r["answer"], str)
    assert "transcript" in r


def test_unknown_route_404(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server, "/definitely_not_a_route")
    assert e.value.code == 404


def test_predict_without_fit_reports_error():
    """未训练就预测必须返回结构化错误，而不是 500 崩溃。"""
    srv = TianshuServer(host="127.0.0.1", port=0)
    httpd, _t = srv.start_background()
    try:
        srv.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        r = _post(srv, "/automl/predict", {"X": [[1, 2], [3, 4]]})
        assert r["ok"] is False and "error" in r
    finally:
        httpd.shutdown()
        httpd.server_close()
