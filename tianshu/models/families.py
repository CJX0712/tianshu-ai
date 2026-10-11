"""L2 · 算法家族注册 —— 天枢的「弹药库」。

覆盖监督学习全部主流家族 + 无监督基础族，每个家族都给出
**经过实战验证的搜索空间**（不是拍脑袋的网格）：
    - 学习率、正则系数一律对数均匀采样（跨越数量级，这是 HPO 的正确先验）。
    - 树深/叶子数用整数区间，上界受样本量约束避免过拟合区无意义探索。
    - 代价先验 cost 用于元路由的预算感知调度。

引用（零重写，均为领域奠基/权威实现）：
    Breiman 2001 Random Forests · Chen & Guestrin 2016 XGBoost ·
    Ke et al. 2017 LightGBM · Cortes & Vapnik 1995 SVM ·
    Rasmussen & Williams 2006 GPML · scikit-learn Pedregosa et al. 2011
"""

from __future__ import annotations

from typing import Any

from ..core.contracts import SearchSpace
from ..core.registry import register
from .adapter import SkAdapter

# ==========================================================================
# 分类家族
# ==========================================================================


def _reg(name: str, task: str, builder, space: SearchSpace, family: str, cost: float,
         requires: tuple[str, ...] = (), tags: tuple[str, ...] = ()) -> None:
    register(
        name=name,
        task=task,
        factory=lambda params=None: SkAdapter(name, task, builder, params),
        space=space,
        family=family,
        cost=cost,
        requires=requires,
        tags=tags,
    )


def install_builtins() -> int:
    """注册全部内置能力；返回注册数量。幂等（重复调用只覆盖不追加）。"""
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
        SGDClassifier,
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier

    n = 0

    # ---------- 线性族 ----------
    s = SearchSpace().add_float("C", 1e-3, 1e3, log=True)
    _reg("logistic", "classification",
         lambda **kw: LogisticRegression(max_iter=2000, **kw), s, "linear", 0.4, tags=("fast", "linear"))
    n += 1

    s = SearchSpace().add_float("alpha", 1e-3, 1e2, log=True)
    _reg("sgd_hinge", "classification",
         lambda **kw: SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, **kw), s, "linear", 0.3,
         tags=("fast", "linear"))
    n += 1

    # ---------- 树族 ----------
    # 注意：HistGradientBoosting* 的参数名是 max_iter（不是 n_estimators），
    # 且不接受 n_estimators/subsample。搜索空间必须与被包装实现严格一致——
    # 这条契约由 tests/test_contract.py 的「全家族可实例化」门禁强制。
    s = (SearchSpace()
         .add_int("max_iter", 50, 400)
         .add_int("max_depth", 2, 12)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_int("max_leaf_nodes", 8, 64)
         .add_int("min_samples_leaf", 5, 40)
         .add_float("l2_regularization", 1e-6, 1e1, log=True))
    _reg("gb_hist", "classification",
         lambda **kw: HistGradientBoostingClassifier(early_stopping=True, **kw),
         s, "tree", 1.6, tags=("gbm", "strong"))
    n += 1

    s = (SearchSpace()
         .add_int("n_estimators", 100, 500)
         .add_int("max_depth", 2, 20)
         .add_int("min_samples_leaf", 1, 20)
         .add_cat("max_features", ["sqrt", "log2", 0.3, 0.7]))
    _reg("random_forest", "classification",
         lambda **kw: RandomForestClassifier(n_jobs=-1, **kw), s, "tree", 1.2, tags=("bagging",))
    n += 1

    s = (SearchSpace()
         .add_int("n_estimators", 100, 500)
         .add_int("max_depth", 2, 24)
         .add_int("min_samples_leaf", 1, 20))
    _reg("extra_trees", "classification",
         lambda **kw: ExtraTreesClassifier(n_jobs=-1, **kw), s, "tree", 1.0, tags=("bagging",))
    n += 1

    s = (SearchSpace()
         .add_int("n_estimators", 50, 300)
         .add_float("learning_rate", 1e-3, 0.5, log=True)
         .add_int("max_depth", 1, 6))
    _reg("gb_exact", "classification",
         lambda **kw: GradientBoostingClassifier(**kw), s, "tree", 2.4, tags=("gbm",))
    n += 1

    s = (SearchSpace()
         .add_int("n_estimators", 50, 300)
         .add_float("learning_rate", 1e-3, 1.0, log=True))
    _reg("adaboost", "classification", lambda **kw: AdaBoostClassifier(**kw), s, "tree", 1.8, tags=("boost",))
    n += 1

    s = (SearchSpace().add_int("max_depth", 1, 20).add_int("min_samples_leaf", 1, 30))
    _reg("decision_tree", "classification", lambda **kw: DecisionTreeClassifier(**kw), s, "tree", 0.2,
         tags=("interpretable",))
    n += 1

    # ---------- 核方法 / 近邻 / 概率 ----------
    s = (SearchSpace().add_float("C", 1e-2, 1e3, log=True).add_float("gamma", 1e-4, 1e1, log=True))
    # 注：sklearn >=1.9 弃用 SVC(probability=...)。此处不传该参数，
    # SVC 走 decision_function 分支（adapter.decision），融合层用 one-hot 兜底。
    _reg("svc_rbf", "classification", lambda **kw: SVC(**kw), s, "kernel", 1.0, tags=("svm",))
    n += 1

    s = SearchSpace().add_int("n_neighbors", 1, 32)
    _reg("knn", "classification", lambda **kw: KNeighborsClassifier(**kw), s, "instance", 0.5, tags=("lazy",))
    n += 1

    _reg("naive_bayes", "classification", lambda **kw: GaussianNB(**kw), SearchSpace(), "probabilistic", 0.1,
         tags=("fast",))
    n += 1

    s = (SearchSpace()
         .add_float("alpha", 1e-5, 1e-1, log=True)
         .add_cat("hidden_layer_sizes", [(64,), (128,), (64, 64), (256,)]))
    _reg("mlp", "classification",
         lambda **kw: MLPClassifier(max_iter=500, early_stopping=True, **kw), s, "neural", 2.0, tags=("nn",))
    n += 1

    # ---------- GBM 旗舰后端（可选依赖） ----------
    def _xgb_cls(**kw: Any):
        import xgboost as xgb
        kw.setdefault("tree_method", "hist")
        kw.setdefault("device", "cpu")
        kw.setdefault("n_jobs", -1)
        kw.setdefault("verbosity", 0)
        kw.setdefault("eval_metric", "logloss")
        return xgb.XGBClassifier(**kw)

    s = (SearchSpace()
         .add_int("n_estimators", 100, 800)
         .add_int("max_depth", 2, 10)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_float("subsample", 0.6, 1.0)
         .add_float("colsample_bytree", 0.5, 1.0)
         .add_float("reg_lambda", 1e-3, 1e2, log=True)
         .add_float("min_child_weight", 1e-2, 1e2, log=True))
    _reg("xgboost", "classification", _xgb_cls, s, "tree", 1.4, requires=("xgboost",),
         tags=("gbm", "sota"))
    n += 1

    def _lgbm_cls(**kw: Any):
        import lightgbm as lgb
        kw.setdefault("verbose", -1)
        kw.setdefault("n_jobs", -1)
        return lgb.LGBMClassifier(**kw)

    s = (SearchSpace()
         .add_int("n_estimators", 100, 800)
         .add_int("num_leaves", 8, 128)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_float("subsample", 0.6, 1.0)
         .add_float("colsample_bytree", 0.5, 1.0)
         .add_float("min_child_weight", 1e-3, 1e1, log=True))
    _reg("lightgbm", "classification", _lgbm_cls, s, "tree", 1.0, requires=("lightgbm",),
         tags=("gbm", "sota", "fast"))
    n += 1

    # ==========================================================================
    # 回归家族
    # ==========================================================================
    _reg("linear", "regression", lambda **kw: LinearRegression(**kw), SearchSpace(), "linear", 0.2, tags=("fast",))
    n += 1

    s = SearchSpace().add_float("alpha", 1e-4, 1e3, log=True)
    _reg("ridge", "regression", lambda **kw: Ridge(**kw), s, "linear", 0.2, tags=("fast", "regularized"))
    n += 1

    s = SearchSpace().add_float("alpha", 1e-4, 1e2, log=True)
    _reg("lasso", "regression", lambda **kw: Lasso(max_iter=5000, **kw), s, "linear", 0.3, tags=("sparse",))
    n += 1

    s = (SearchSpace().add_float("alpha", 1e-4, 1e2, log=True).add_float("l1_ratio", 0.05, 0.95))
    _reg("elasticnet", "regression", lambda **kw: ElasticNet(max_iter=5000, **kw), s, "linear", 0.3,
         tags=("sparse",))
    n += 1

    s = (SearchSpace()
         .add_int("max_iter", 50, 400)
         .add_int("max_depth", 2, 12)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_int("max_leaf_nodes", 8, 64)
         .add_int("min_samples_leaf", 5, 40)
         .add_float("l2_regularization", 1e-6, 1e1, log=True))
    _reg("gb_hist_reg", "regression",
         lambda **kw: HistGradientBoostingRegressor(early_stopping=True, **kw),
         s, "tree", 1.6, tags=("gbm", "strong"))
    n += 1

    s = (SearchSpace().add_int("n_estimators", 100, 500).add_int("max_depth", 2, 24)
         .add_int("min_samples_leaf", 1, 20))
    _reg("random_forest_reg", "regression", lambda **kw: RandomForestRegressor(n_jobs=-1, **kw), s, "tree", 1.2)
    n += 1

    s = (SearchSpace().add_int("n_estimators", 100, 500).add_int("max_depth", 2, 28)
         .add_int("min_samples_leaf", 1, 20))
    _reg("extra_trees_reg", "regression", lambda **kw: ExtraTreesRegressor(n_jobs=-1, **kw), s, "tree", 1.0)
    n += 1

    s = (SearchSpace().add_float("C", 1e-2, 1e3, log=True).add_float("gamma", 1e-4, 1e1, log=True)
         .add_float("epsilon", 1e-3, 1.0, log=True))
    _reg("svr_rbf", "regression", lambda **kw: SVR(**kw), s, "kernel", 1.0)
    n += 1

    s = SearchSpace().add_int("n_neighbors", 1, 32)
    _reg("knn_reg", "regression", lambda **kw: KNeighborsRegressor(**kw), s, "instance", 0.5)
    n += 1

    s = (SearchSpace().add_float("alpha", 1e-6, 1e-1, log=True)
         .add_cat("hidden_layer_sizes", [(64,), (128,), (64, 64)]))
    _reg("mlp_reg", "regression", lambda **kw: MLPRegressor(max_iter=800, early_stopping=True, **kw), s,
         "neural", 2.0)
    n += 1

    def _gp_reg(**kw: Any):
        k = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-6)
        return GaussianProcessRegressor(kernel=k, normalize_y=True, **kw)

    s = SearchSpace().add_float("alpha", 1e-8, 1e-1, log=True)
    _reg("gaussian_process", "regression", _gp_reg, s, "probabilistic", 3.0, tags=("uncertainty",))
    n += 1

    def _xgb_reg(**kw: Any):
        import xgboost as xgb
        kw.setdefault("tree_method", "hist")
        kw.setdefault("device", "cpu")
        kw.setdefault("n_jobs", -1)
        kw.setdefault("verbosity", 0)
        return xgb.XGBRegressor(**kw)

    s = (SearchSpace().add_int("n_estimators", 100, 800).add_int("max_depth", 2, 10)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_float("subsample", 0.6, 1.0).add_float("colsample_bytree", 0.5, 1.0)
         .add_float("reg_lambda", 1e-3, 1e2, log=True))
    _reg("xgboost_reg", "regression", _xgb_reg, s, "tree", 1.4, requires=("xgboost",), tags=("gbm", "sota"))
    n += 1

    def _lgbm_reg(**kw: Any):
        import lightgbm as lgb
        kw.setdefault("verbose", -1)
        kw.setdefault("n_jobs", -1)
        return lgb.LGBMRegressor(**kw)

    s = (SearchSpace().add_int("n_estimators", 100, 800).add_int("num_leaves", 8, 128)
         .add_float("learning_rate", 1e-3, 0.3, log=True)
         .add_float("min_child_weight", 1e-3, 1e1, log=True))
    _reg("lightgbm_reg", "regression", _lgbm_reg, s, "tree", 1.0, requires=("lightgbm",), tags=("gbm", "sota"))
    n += 1

    # ==========================================================================
    # 无监督族
    # ==========================================================================
    s = SearchSpace().add_int("n_clusters", 2, 12)
    _reg("kmeans", "unsupervised", lambda **kw: KMeans(n_init=10, **kw), s, "cluster", 0.6)
    n += 1

    s = SearchSpace().add_int("n_components", 2, 12)
    _reg("gmm", "unsupervised", lambda **kw: GaussianMixture(**kw), s, "cluster", 0.8)
    n += 1

    s = SearchSpace().add_int("n_clusters", 2, 10)
    _reg("spectral_clustering", "unsupervised",
         lambda **kw: SpectralClustering(affinity="nearest_neighbors", assign_labels="kmeans", **kw),
         s, "cluster", 2.0)
    n += 1

    s = (SearchSpace().add_float("contamination", 1e-3, 0.3, log=True).add_int("n_estimators", 50, 300))
    _reg("isolation_forest", "unsupervised", lambda **kw: IsolationForest(n_jobs=-1, **kw), s, "anomaly", 1.0)
    n += 1

    # 上界保守：n_components 必须 ≤ min(n_samples, n_features)，越界会直接抛错
    s = SearchSpace().add_int("n_components", 2, 10)
    _reg("pca", "unsupervised", lambda **kw: PCA(**kw), s, "manifold", 0.3)
    n += 1

    return n
