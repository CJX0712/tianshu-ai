"""L0 内核不变量 —— 确定性是天枢的第一性约束。"""

from __future__ import annotations

import numpy as np
import pytest

from tianshu.core.metrics import (
    accuracy,
    bootstrap_ci,
    cross_entropy,
    macro_f1,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rmse,
)
from tianshu.core.seed import deterministic, rng_for, spawn_seed, stable_hash


def test_spawn_seed_streams_are_independent():
    """派生的子流必须互不相关——否则 HPO 各 trial 会共享伪随机结构。"""
    r1 = rng_for(0, "a")
    r2 = rng_for(0, "b")
    a, b = r1.normal(size=20000), r2.normal(size=20000)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.03, f"子流相关过高: {corr}"


def test_spawn_seed_deterministic_and_distinct():
    assert spawn_seed(7, "x", 1) == spawn_seed(7, "x", 1)
    assert spawn_seed(7, "x", 1) != spawn_seed(7, "x", 2)


def test_seed_zero_is_valid():
    """0 是合法种子，绝不能因为 falsy 被短路（常见坑）。"""
    with deterministic(0) as g:
        a = g.normal(size=5)
    with deterministic(0) as g:
        b = g.normal(size=5)
    assert np.array_equal(a, b)


def test_stable_hash_platform_independent():
    """稳定哈希不能依赖 PYTHONHASHSEED（内置 hash() 会随机化，绝不可用）。"""
    assert stable_hash("天枢") == stable_hash("天枢")
    assert stable_hash("a") != stable_hash("b")
    assert len(stable_hash("a")) == 16


def test_metrics_at_perfect_prediction():
    y = np.array([0, 1, 1, 0, 2])
    assert accuracy(y, y) == pytest.approx(1.0)
    assert macro_f1(y, y) == pytest.approx(1.0)
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_loss_direction_is_uniformly_minimize():
    """所有度量的 .loss 必须是「越小越好」，优化器只认一个方向。"""
    y = np.array([0, 1, 1, 0])
    good, bad = np.array([0, 1, 1, 0]), np.array([1, 0, 0, 1])
    for m in (accuracy, macro_f1, cross_entropy):
        assert m.loss(y, good) < m.loss(y, bad), f"{m.name} 方向错误"
    yr = np.array([1.0, 2.0, 3.0])
    assert rmse.loss(yr, np.array([1.0, 2.0, 3.0])) < rmse.loss(yr, np.array([3.0, 1.0, 9.0]))


def test_degenerate_input_never_raises():
    """单类/全同预测下不得抛异常——集成搜索里这是常态。"""
    y = np.array([1, 1, 1])
    for m in (accuracy, macro_f1, cross_entropy):
        assert np.isfinite(m(y, np.array([1, 1, 1])))


def test_ranking_metrics():
    y = np.array([0.0, 1.0, 0.0, 3.0])
    perfect = np.array([0.1, 1.0, 0.2, 3.0])
    assert ndcg_at_k(y, perfect, k=4) == pytest.approx(1.0)
    assert ndcg_at_k(y, -perfect, k=4) < 1.0
    # perfect 把 y=3 的样本排在第 1 位 → MRR = 1
    assert mrr(y, perfect) == pytest.approx(1.0)
    # 相关项被压到第 2 位 → MRR = 1/2
    assert mrr(np.array([0.0, 3.0, 0.0]), np.array([0.9, 0.5, 0.1])) == pytest.approx(0.5)
    # 相关项被压到最后 → MRR = 1/3
    assert mrr(np.array([0.0, 0.0, 3.0]), np.array([0.5, 0.9, 0.1])) == pytest.approx(1 / 3)
    assert recall_at_k(y, perfect, k=1) == pytest.approx(0.5)
    assert recall_at_k(y, perfect, k=4) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y + rng.normal(0, 0.3, 300), 0, 1).round()
    point, lo, hi = bootstrap_ci(y, p, accuracy, n_boot=200, seed=1)
    assert lo <= point <= hi
