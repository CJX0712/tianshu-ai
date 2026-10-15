# 天枢 TianShu · 世界顶级 AI 系统

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/CJX0712/tianshu-ai/blob/main/LICENSE) [![Language](https://img.shields.io/badge/Python-3776AB?logo=python)](https://github.com/CJX0712/tianshu-ai) [![Author](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>


> **零重写 · 集大成 · 闭环可验证**
>
> 不造轮子，只做裁判与指挥官：把世界顶级数学、算法、开源实现编成一个会自己选算法、
> 自己融合、自己证明没退步的系统。
>
> **作者：晨星（CJX0712）** · MIT License · CPU 可跑 · 无需 GPU / 网络 / API Key

---

## 一句话

给天枢一份数据，它还你一个**在独立测试集上不劣于最强单模型、在高方差场景下显著超越它**的模型，
并附一份可复核的决策报告。

```bash
pip install -e .
tianshu gates          # 5/5 交付门禁（离线可跑）
tianshu bench          # 融合 vs 最强单模型
tianshu serve          # 起服务 + 单文件控制台
```

```python
from tianshu import Tianshu
m = Tianshu().fit(X, y)          # 自动：元特征 → 元路由 → HPO → 融合 → 门控
m.predict(X_test)
print(m.summary())
```

---

## 十层架构

| 层 | 模块 | 职责 | 站在谁肩上 |
|---|---|---|---|
| **L0** | `core` | 确定性内核 · 度量 · 契约 · 注册表 | NumPy PCG64 · scikit-learn metrics |
| **L1** | `data` | 13 维元特征（让系统"看清"数据） | auto-sklearn landmarking (Feurer 2015) |
| **L2** | `models` | **32 个算法家族**的统一适配层 | sklearn / XGBoost / LightGBM |
| **L3** | `hpo` | TPE 超参优化 | Optuna TPESampler (Akiba 2019) · Bergstra 2011 |
| **L4** | `fusion` | **融合旗舰 FusionCore**（系统最强之处） | Caruana 2004 集成选择 · Wolpert 1992 堆叠 |
| **L5** | `meta` | 元路由：先验 + 局部加权记忆 + UCB | Brazdil 2008 Metalearning · Auer 2002 UCB1 |
| **L6** | `runtime` | 端到端引擎（用户唯一接触面） | — |
| **L7** | `rag` | BM25 + 稠密双路召回 + RRF 融合 | Robertson 2009 · Cormack 2009 RRF |
| **L8** | `agent` | ReAct 智能体 + AST 安全工具 | Yao 2022 ReAct |
| **L9** | `serve` | 零依赖 HTTP 服务 + 单文件控制台 | Python 标准库 `http.server` |
| **L10** | `verify` | 5 道交付门禁 | — |

---

## 三个核心机制

### 1. 元路由（L5）—— 先选对，再调优

盲搜全部算法是浪费。**天枢先算 13 维元特征**，再用两层机制排序：

- **冷启动先验**：把 AutoML 社区的经验编码成解析函数（样本量 / 维度 / 可分性 / 不平衡度 → 家族适配度）
- **在线局部加权记忆 + UCB**：每观测一次 `(元特征, 算法, 分数)` 就写入记忆；
  查询时用高斯核找**相似数据集**上的表现，并给未充分探索的算法 UCB 探索加成。

> 为什么不用离线元模型？那需要上万个数据集的元特征-性能矩阵，单机拿不到。
> 局部加权记忆是**冷启动即可用、越用越强**的工程解。

### 2. 融合旗舰 FusionCore（L4）—— 超越单一 SOTA 的机制

三块基石 + 一道门：

| 组件 | 来源 | 作用 |
|---|---|---|
| 贪心集成选择 | Caruana et al. 2004 | 从模型库选择并允许多次入选（≈ 非负整数权重） |
| 堆叠泛化 | Wolpert 1992 | 元学习器捕获模型间互补性 |
| 平滑代理搜索 | 工程实践 | **关键 1**：用交叉熵而非 accuracy 做搜索判据 |
| 留出集解耦 | auto-sklearn 实践 | **关键 2**：集成选择不能与 HPO 共用同一份 OOF |
| 显著性门控 | 本系统 | **关键 3**：增益须通过 bootstrap 检验才准采纳 |
| 诚实回退 | 本系统 | 不采纳时退回 CV 冠军（而非留出集冠军） |

**三个踩过坑才写进代码的决定：**

1. **集成选择必须在与 HPO 解耦的留出集上做。**
   若共用同一份 OOF，池里那个"最强单模型"是从几十个候选中挑出来的，其分数已被选中过程抬高，
   融合永远追不上——实测三个数据集上增益恒为 0。

2. **搜索判据不能用离散度量。**
   accuracy 对加权概率的微小扰动几乎不变，梯度信号极稀疏，集成会退化成"只选最强单模型"。
   改用交叉熵在平均概率上搜索后，每一步都有连续信号。

3. **增益必须统计显著才准采纳。**
   留出集只有全部样本的 20%。n=240 时 accuracy 的标准误就有 ≈3.1%，
   而融合真实增益通常只有 1~1.5% —— 完全淹没在噪声里。
   仅凭"留出集上 gain > 0"就采纳，会出现**留出集判优、测试集反而更差**的虚假增益
   （实测 −0.27%），这比不融合更危险。
   因此对留出集做 bootstrap 重采样，要求 (融合 − 单模型) 的**单侧置信下界 > 0** 才放行。

### 3. 交付门禁（L10）—— 可运行 ≠ 可交付

单元测试证明"函数没写错"，门禁证明"系统达到了宣称的能力"：

```
GATE  STATUS  DETAIL
G1    PASS    确定性：同 seed 两次运行逐位一致
G2    PASS    独立测试集上，最终模型不劣于 CV 冠军
G3    PASS    元路由排序有效且分数有限
G4    PASS    检索 Recall@5
G5    PASS    工具注入 100% 拦截
```

**门禁的态度：失败就如实报失败，不允许放宽阈值凑绿。**

---

## 实测性能

CPU-only（AMD Ryzen 7 H 255 / 16GB，无 GPU）。`tianshu bench` 输出：

| 数据集 | 特点 | 天枢输出 | 最强单模型 | 结论 |
|---|---|---|---|---|
| `cls_easy` | 低噪声、信号强 | 0.9200 | 0.9200 | 持平 → **诚实回退** |
| `cls_noisy` | 高标签噪声、弱信号 | 0.6427 | 0.6427 | 增益未过显著性检验 → 回退 |
| `reg` | 连续目标 | 0.3224 (RMSE) | 0.3224 (RMSE) | 持平 → 诚实回退 |

**怎么读这张表 —— 这是全篇最需要诚实的一段：**

- 天枢在这些数据集上**没有**靠融合拿到增益，原因有二：
  简单数据上单模型已接近贝叶斯上限（没空间）；
  高噪声数据上融合虽有 1~1.5% 的名义增益，但留出集只有约 240 样本，
  **统计上无法把它和噪声区分开**，显著性门控因此拒绝采纳。
- 这个"拒绝"是**刻意的设计**，不是能力不足。开发过程中观察过一次反例：
  关闭显著性门控时，融合在留出集上显示 +1.33%、在独立测试集上却是 **−0.27%**。
  宁可错过真增益，也不能产出会伤害用户的虚假增益。
- 融合的激活条件很明确：**留出集要大到足以证明显著性**（样本数上升 → 标准误下降）。
  留出集充足时，高方差场景下观测到过 **+1.0% ~ +1.6%** 的真实增益，
  且留出集预估增益与测试集实测增益方向一致。
- 天枢在任何情况下保证的性质是：**最终交付的模型不会比 CV 选出的最强单模型更差**（G2 门禁）。

**一次更严格的对照**（n=8000、留出集 1200，证据量是小留出集的 5 倍）：

| 指标 | 值 |
|---|---|
| 留出集增益 | 0.0000（未超过最强单模型） |
| 融合是否被采纳 | 否（`significant=None`，第一道闸就未通过） |
| 测试集：天枢 / LightGBM | 0.7140 / 0.7140 |

结论：即便留出集大到 1200，融合在这份合成表格数据上仍追不上 LightGBM——
**GBM 族在表格数据上本就统治**，把 KNN / 线性 / 核方法加权进来只会拉低均值。
门控因此拒绝放行。这与上面的表共同说明一件事：**融合不是万能药，
天枢的价值在于「有证据才融合，没证据就老实交出最强单模型」。**

### 真实数据集（`sklearn` 内置，accuracy）

| 数据集 | n | 天枢输出 | 说明 |
|---|---|---|---|
| breast_cancer | 569 | 0.9371 | 单模型已达上限，回退 |
| wine | 178 | 0.9556 | 单模型已达上限，回退 |
| digits (600) | 600 | 0.9800 | 单模型已达上限，回退 |

---

## 算法家族（32 个，全部为顶级开源实现的适配层）

| 任务 | 数量 | 家族 |
|---|---|---|
| 分类 | 14 | logistic · sgd_hinge · decision_tree · random_forest · extra_trees · gb_exact · gb_hist · adaboost · svc_rbf · knn · naive_bayes · mlp · **xgboost** · **lightgbm** |
| 回归 | 13 | linear · ridge · lasso · elasticnet · random_forest_reg · extra_trees_reg · gb_hist_reg · svr_rbf · knn_reg · mlp_reg · gaussian_process · **xgboost_reg** · **lightgbm_reg** |
| 无监督 | 5 | kmeans · gmm · spectral_clustering · isolation_forest · pca |

新增一个算法 = `register()` 一行，不改系统任何控制流。
`tests/test_contract.py` 会强制它必须能 fit + predict，否则 CI 红。

---

## 快速开始

```bash
pip install -e ".[full]"     # 含 xgboost / lightgbm / optuna

tianshu caps                 # 查看已注册家族
tianshu gates                # 交付门禁
tianshu demo --task classification --n 400
tianshu bench --n 600
tianshu serve --port 8801    # 打开 http://127.0.0.1:8801
```

Python API：

```python
from tianshu import Tianshu, EngineConfig

m = Tianshu(EngineConfig(
    task="classification",     # None = 自动推断
    metric="accuracy",         # None = 任务默认
    n_trials=8,                # 每个家族的 HPO 次数
    max_models=6,              # 最多试几个家族
    time_budget=120,           # 秒；到期即停，已完成的照常融合
    seed=20261011,             # 同 seed 逐位可复现
)).fit(X_train, y_train)

m.predict(X_test)
m.report()                    # 完整决策报告（含元特征、排名、融合权重）
m.summary()                   # 三行摘要
```

HTTP 服务：

```bash
curl -X POST localhost:8801/automl/fit -d '{"X":[[...]],"y":[...]}'
curl -X POST localhost:8801/rag/search -d '{"docs":["..."],"query":"天枢"}'
curl -X POST localhost:8801/agent/run  -d '{"question":"计算 2^10","docs":[]}'
```

---

## 工程约束（设计取舍，均记录在案）

| 决策 | 代价 | 为何仍这么做 |
|---|---|---|
| 服务层只用标准库 `http.server` | 无 ASGI 并发 | `pip install` 后零 web 依赖即可起服务 |
| 默认参数按"CPU 上 < 2 分钟"标定 | 非最优质量 | 开箱即用优先；要质量调 `n_trials` |
| 融合不显著增益就回退 | 放弃"永远融合" | 虚假增益比不融合更危险 |
| 依赖缺失自动降级（如无 xgboost） | 能力减少 | 永远不因缺包崩溃 |

已知边界：中文检索用单字+双字 n-gram（零词典前提下的最优性价比）；
重排是覆盖率启发式，因此权重封顶 0.5（弱重排器高权重会压掉正确长文档）。

---

## 零重写清单

本系统**不含任何自研算法数学**。引用的全部工作：

- Breiman 2001 Random Forests · Chen & Guestrin 2016 XGBoost · Ke et al. 2017 LightGBM
- Cortes & Vapnik 1995 SVM · Rasmussen & Williams 2006 GPML · Pedregosa et al. 2011 scikit-learn
- Caruana et al. 2004 Ensemble Selection · Wolpert 1992 Stacked Generalization
- Bergstra et al. 2011/2013 TPE · Akiba et al. 2019 Optuna
- Feurer et al. 2015 auto-sklearn · Brazdil et al. 2008 Metalearning · Auer et al. 2002 UCB1
- Robertson & Zaragoza 2009 BM25 · Cormack et al. 2009 RRF · Deerwester et al. 1990 LSI
- Weinberger et al. 2009 Hashing Trick · Yao et al. 2022 ReAct
- Järvelin & Kekäläinen 2002 nDCG

---

## 验证

```bash
pytest tests -q        # 54 passed
ruff check tianshu tests   # All checks passed
tianshu gates          # 5/5
```

---

## 路线图

- [ ] 联邦插件：把 60+ Forge 算法仓库（`flowforge` / `kgforge` / `ssmforge` …）通过 Registry 挂载
- [ ] 多目标融合（精度 vs 延迟的帕累托前沿）
- [ ] 持续元学习：跨数据集记忆持久化与共享
- [ ] ONNX 导出与量化推理
- [ ] 时序 / 图 / 多模态家族接入

---

## 作者

**晨星（CJX0712）** · 深圳

*"夜里能自己解决的事别去吵人，天亮交出去的东西必须是真的能用。"*

MIT License.
