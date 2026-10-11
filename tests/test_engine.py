"""L6 引擎端到端 —— 用户实际接触到的那一条链路。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tianshu.runtime.engine import EngineConfig, Tianshu

SMALL = EngineConfig(n_trials=2, max_models=3, seed=5)


def test_classification_end_to_end():
    X, y = make_classification(n_samples=240, n_features=10, n_informative=5,
                               n_classes=3, random_state=0)
    m = Tianshu(SMALL).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(y),)
    assert set(np.unique(pred)) <= set(np.unique(y))
    assert (pred == y).mean() > 0.5, "拟合质量异常，可能链路有问题"


def test_regression_end_to_end():
    X, y = make_regression(n_samples=240, n_features=10, noise=0.2, random_state=0)
    m = Tianshu(EngineConfig(task="regression", n_trials=2, max_models=3, seed=5)).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    assert m.task_ == "regression"


def test_determinism_same_seed():
    X, y = make_classification(n_samples=260, n_features=10, n_informative=5,
                               n_classes=3, random_state=1)
    a = Tianshu(SMALL).fit(X, y).predict(X)
    b = Tianshu(SMALL).fit(X, y).predict(X)
    assert np.array_equal(a, b), "同 seed 两次运行不一致"


def test_final_model_never_worse_than_cv_best():
    """引擎对外承诺：拿到的模型不会比 CV 冠军更差。"""
    from sklearn.model_selection import train_test_split

    from tianshu.core.metrics import get_metric

    X, y = make_classification(n_samples=400, n_features=12, n_informative=6,
                               n_classes=2, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    m = Tianshu(EngineConfig(n_trials=3, max_models=4, seed=7)).fit(Xtr, ytr)
    met = get_metric(m.metric_.name)
    final = float(met(yte, m.predict(Xte)))
    best = max(m.leaderboard_, key=lambda r: r.metric)
    key = next(k for k in m.models_ if k == best.name or k.startswith(best.name + "#"))
    single = float(met(yte, m.models_[key].predict(Xte)))
    assert final >= single - 1e-9, f"final={final} < cv_best={single}"


def test_report_structure_complete():
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2, random_state=0)
    r = Tianshu(SMALL).fit(X, y).report()
    for k in ("task", "metric", "leaderboard", "fusion", "ranking", "metafeatures"):
        assert k in r, f"报告缺少字段 {k}"
    assert len(r["metafeatures"]) == 13, "元特征维度必须是 13"


def test_time_budget_is_respected():
    X, y = make_classification(n_samples=600, n_features=14, n_informative=5,
                               n_classes=3, random_state=0)
    m = Tianshu(EngineConfig(n_trials=30, max_models=6, time_budget=2.0, seed=1)).fit(X, y)
    assert m.elapsed_ < 60, f"时间预算未生效: {m.elapsed_}s"


def test_predict_proba_shape():
    X, y = make_classification(n_samples=200, n_features=8, n_informative=5,
                               n_classes=3, random_state=0)
    m = Tianshu(SMALL).fit(X, y)
    p = m.predict_proba(X)
    if p is not None:
        assert p.shape == (len(y), 3)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_fit_before_predict_raises():
    with pytest.raises(RuntimeError):
        Tianshu().predict(np.zeros((3, 2)))


def test_unsupervised_capability_available():
    from tianshu.core.registry import get

    X, _ = make_classification(n_samples=120, n_features=6, random_state=0)
    est = get("kmeans").factory({"n_clusters": 3})
    est.fit(X)
    assert len(np.unique(est.predict(X))) <= 3
