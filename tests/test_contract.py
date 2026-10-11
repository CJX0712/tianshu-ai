"""契约门禁 —— 系统的第一道防线。

这条门禁存在的理由（血的教训）：
    搜索空间声明 `min_samples_leaf` 为 float，而 sklearn 要求 int，
    结果整个树家族在 HPO 阶段**静默全灭**（异常被 except 吞掉），
    系统只用了 4 个模型还在报告里宣称「最优」。这是最危险的一类 bug：
    不报错、不崩溃、只是悄悄变弱。

    因此这里的强制要求是：每一个注册的算法家族，都必须能用
    「默认参数」和「随机采样参数」成功 fit + predict 小数据。
    任何参数类型/取值域与底层实现的不匹配都会在此被拦下。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tianshu.core.registry import list_capabilities
from tianshu.core.seed import rng_for

N, D = 120, 12


def _data(task: str):
    rng = np.random.default_rng(0)
    if task == "classification":
        X, y = make_classification(n_samples=N, n_features=D, n_informative=5,
                                   n_classes=3, random_state=0)
    elif task == "regression":
        X, y = make_regression(n_samples=N, n_features=D, noise=0.1, random_state=0)
    else:
        X = rng.normal(size=(N, D))
        y = None
    return X, y


@pytest.mark.parametrize("task", ["classification", "regression", "unsupervised"])
def test_all_families_fit_and_predict(task: str):
    caps = list_capabilities(task)
    assert caps, f"no capability registered for {task}"
    X, y = _data(task)
    rng = rng_for(7, "contract", task)

    failures = []
    for cap in caps:
        # 1) 默认参数
        trials = [cap.space.defaults()]
        # 2) 随机采样参数（覆盖 log 维、categorical 维）
        for _ in range(3):
            trials.append(cap.space.sample(rng))
        for params in trials:
            try:
                est = cap.factory(params)
                est.fit(X, y)
                out = est.predict(X)
                assert out is not None and len(out) == N, f"{cap.name} bad output shape"
            except Exception as e:  # noqa: BLE001 - 门禁就是要收集全部失败
                failures.append(f"{cap.name} / {params} -> {type(e).__name__}: {e}")
    assert not failures, "契约违约:\n" + "\n".join(failures)


def test_space_types_match_estimator_signature():
    """整数维必须真的产出 Python int（采样器绝不能返回 float 冒充 int）。"""
    rng = rng_for(11, "contract", "types")
    for cap in list_capabilities("classification"):
        p = cap.space.sample(rng)
        for k, d in cap.space.dims.items():
            if d["type"] == "int":
                assert isinstance(p[k], int), f"{cap.name}.{k} sampled {type(p[k])} not int"
                assert d["low"] <= p[k] <= d["high"], f"{cap.name}.{k}={p[k]} out of range"
            elif d["type"] == "float":
                assert d["low"] <= p[k] <= d["high"], f"{cap.name}.{k}={p[k]} out of range"


def test_unknown_metric_raises():
    from tianshu.core.metrics import get_metric

    with pytest.raises(KeyError):
        get_metric("not_a_metric")


def test_registry_unknown_capability_raises():
    from tianshu.core.registry import get

    with pytest.raises(KeyError):
        get("definitely_not_registered")
