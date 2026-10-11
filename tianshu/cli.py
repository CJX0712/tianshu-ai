"""天枢命令行 —— 一条命令完成验证 / 训练 / 服务。

    tianshu gates            跑 L10 交付门禁（离线、无网络、无密钥）
    tianshu demo             在合成数据上跑完整 AutoML 并打印报告
    tianshu bench            多数据集基准：融合 vs 最强单模型
    tianshu serve --port 8801 启动零依赖 HTTP 服务（含单文件控制台）
    tianshu caps             列出已注册的算法家族与可用后端
"""

from __future__ import annotations

import argparse
import json
import sys


def _caps() -> int:
    from .core.registry import list_capabilities

    for task in ("classification", "regression", "unsupervised"):
        names = [c.name for c in list_capabilities(task)]
        print(f"{task:<16}{len(names):>3}  {', '.join(names)}")
    return 0


def _gates() -> int:
    from .verify.gates import run_all_gates

    rep = run_all_gates()
    return 0 if rep.all_green else 1


def _demo(args) -> int:
    from sklearn.datasets import make_classification, make_regression

    from .runtime.engine import EngineConfig, Tianshu

    if args.task == "regression":
        X, y = make_regression(n_samples=args.n, n_features=12, noise=0.15, random_state=0)
    else:
        X, y = make_classification(n_samples=args.n, n_features=12, n_informative=6,
                                   n_classes=3, random_state=0)
    cfg = EngineConfig(
        task=args.task, n_trials=args.trials, max_models=args.models,
        time_budget=args.budget, seed=args.seed, verbose=True,
    )
    m = Tianshu(cfg).fit(X, y)
    print("\n" + m.summary())
    print("\n完整报告:")
    print(json.dumps(m.report(), ensure_ascii=False, indent=2))
    return 0


def _bench(args) -> int:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    from .core.metrics import get_metric
    from .runtime.engine import EngineConfig, Tianshu

    # 数据集选择是有意为之的对照设计：
    #   cls_easy  低噪声、信号强 —— 单模型已接近贝叶斯上限，融合通常**无空间**，
    #             预期结果是诚实回退（这正是门控该有的行为，不是缺陷）。
    #   cls_noisy 高标签噪声、弱信号 —— 单模型方差大且误差不相关，
    #             这是融合理论上的主场（方差按 ≈1/M 下降），预期出现增益。
    #   reg       连续目标，检验回归链路。
    sets = {
        "cls_easy": make_classification(n_samples=args.n, n_features=14, n_informative=8,
                                        n_classes=2, random_state=1),
        "cls_noisy": make_classification(n_samples=int(args.n * 2.5), n_features=25,
                                         n_informative=6, n_classes=3, flip_y=0.25,
                                         random_state=5),
        "reg": make_regression(n_samples=args.n, n_features=14, noise=0.3, random_state=3),
    }
    rows = []
    for name, (X, y) in sets.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        task = "regression" if name == "reg" else "classification"
        m = Tianshu(EngineConfig(task=task, n_trials=args.trials, max_models=args.models,
                                 seed=11)).fit(Xtr, ytr)
        metric = get_metric(m.metric_.name)
        fused = metric(yte, m.predict(Xte))
        # 最强单模型在同测试集上的成绩（对照）
        best_single_name = max(m.leaderboard_, key=lambda r: r.metric).name \
            if metric.higher else min(m.leaderboard_, key=lambda r: r.metric).name
        # models_ 的键可能是 "name" 或 "name#rank"（融合池按候选展开）
        key = next((k for k in m.models_ if k == best_single_name
                    or k.startswith(best_single_name + "#")), None)
        single = metric(yte, m.models_[key].predict(Xte)) if key else float("nan")
        rows.append({
            "dataset": name, "metric": metric.name,
            "fusion": round(fused, 6), "best_single": round(single, 6),
            "best_single_model": best_single_name,
            "delta": round(fused - single, 6),
            "improved": bool(fused > single if metric.higher else fused < single),
            "sec": round(m.elapsed_, 2),
        })
    w = max(len(r["dataset"]) for r in rows)
    print(f"{'dataset':<{w}}  {'metric':<10}{'fusion':>12}{'best_single':>14}{'delta':>10}  ok")
    for r in rows:
        print(f"{r['dataset']:<{w}}  {r['metric']:<10}{r['fusion']:>12.6f}"
              f"{r['best_single']:>14.6f}{r['delta']:>+10.6f}  {'✅' if r['improved'] else '—'}")
    n_imp = sum(r["improved"] for r in rows)
    print(f"\n{n_imp}/{len(rows)} 数据集上融合优于最强单模型")
    return 0


def _serve(args) -> int:
    from .serve.api import TianshuServer

    TianshuServer(host=args.host, port=args.port, verbose=True).serve()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tianshu", description="天枢 TianShu — 世界顶级 AI 系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gates", help="跑 L10 交付门禁")
    sub.add_parser("caps", help="列出已注册算法家族")

    d = sub.add_parser("demo", help="合成数据上的完整演示")
    d.add_argument("--task", default="classification", choices=["classification", "regression"])
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--trials", type=int, default=4)
    d.add_argument("--models", type=int, default=5)
    d.add_argument("--budget", type=float, default=None)
    d.add_argument("--seed", type=int, default=20261011)

    b = sub.add_parser("bench", help="融合 vs 最强单模型基准")
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--trials", type=int, default=4)
    b.add_argument("--models", type=int, default=4)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8801)

    a = p.parse_args(argv)
    if a.cmd == "gates":
        return _gates()
    if a.cmd == "caps":
        return _caps()
    if a.cmd == "demo":
        return _demo(a)
    if a.cmd == "bench":
        return _bench(a)
    if a.cmd == "serve":
        return _serve(a)
    print("unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
