"""L4 融合旗舰不变量 —— 融合层的四条硬约束。"""

from __future__ import annotations

import numpy as np
import pytest

from tianshu.core.metrics import accuracy, rmse
from tianshu.fusion.fusion_core import FusionCore, greedy_ensemble_selection


def _probs(rng, n=200, C=3, noise=0.5, seed=0):
    """构造带噪声的概率立方体候选集 (M, N, C)。"""
    y = rng.integers(0, C, n)
    P = []
    for i in range(5):
        r = np.random.default_rng(seed + i)
        p = np.full((n, C), 0.05)
        p[np.arange(n), y] = 1.0 - noise * (0.4 + 0.15 * i)
        p += r.normal(0, 0.05, p.shape)
        p = np.clip(p, 1e-6, None)
        P.append(p / p.sum(axis=1, keepdims=True))
    return np.stack(P), y


def test_weights_on_simplex():
    rng = np.random.default_rng(0)
    P, y = _probs(rng)
    for bagged in (1, 4):
        w = greedy_ensemble_selection(P, y, "classification", accuracy, bagged=bagged, seed=3)
        assert np.all(w >= -1e-12), "权重出现负值"
        assert abs(w.sum() - 1.0) < 1e-9, f"权重和 = {w.sum()}"


def test_gate_never_claims_false_improvement():
    """accepted=True 必须意味着真的比最强单模型好（以目标度量计）。"""
    rng = np.random.default_rng(1)
    P, y = _probs(rng, noise=0.9)
    fc = FusionCore(task="classification", metric=accuracy, seed=5)
    r = fc.fit(P, y)
    if r.accepted:
        assert r.val_loss < r.best_single_loss - 1e-12, "宣称增益但未超过最强单模型"
    else:
        assert r.method == "single(cv-best)" or r.method == "single", r.method


def test_degenerate_candidates_do_not_degrade():
    """所有候选完全相同时，融合必须等于该候选，绝不能变坏。"""
    rng = np.random.default_rng(2)
    n, C = 150, 2
    y = rng.integers(0, C, n)
    p = np.full((n, C), 0.1)
    p[np.arange(n), y] = 0.9
    P = np.stack([p.copy() for _ in range(4)])
    fc = FusionCore(task="classification", metric=accuracy, seed=1)
    r = fc.fit(P, y)
    single = accuracy(y, np.argmax(p, axis=1))
    assert abs(abs(r.val_loss) - single) < 1e-9


def test_fusion_is_deterministic():
    rng = np.random.default_rng(3)
    P, y = _probs(rng)
    w1 = FusionCore(metric=accuracy, seed=9).fit(P, y).weights
    w2 = FusionCore(metric=accuracy, seed=9).fit(P, y).weights
    assert np.array_equal(w1, w2), "同 seed 融合权重不一致"


def test_regression_fusion_runs():
    rng = np.random.default_rng(4)
    n = 200
    y = rng.normal(size=n)
    P = np.stack([y + rng.normal(0, 0.5 + 0.3 * i, n) for i in range(4)])
    r = FusionCore(task="regression", metric=rmse, seed=2).fit(P, y)
    assert np.all(r.weights >= -1e-12) and abs(r.weights.sum() - 1) < 1e-9
    assert r.val_loss <= r.best_single_loss + 1e-12


def test_single_candidate_short_circuit():
    rng = np.random.default_rng(5)
    P, y = _probs(rng)
    r = FusionCore(metric=accuracy).fit(P[:1], y)
    assert r.method == "single" and len(r.weights) == 1


@pytest.mark.parametrize("bagged", [1, 3, 6])
def test_bagged_variants_all_valid(bagged: int):
    rng = np.random.default_rng(6)
    P, y = _probs(rng)
    w = greedy_ensemble_selection(P, y, "classification", accuracy, bagged=bagged, seed=0)
    assert abs(w.sum() - 1.0) < 1e-9 and np.all(w >= -1e-12)
