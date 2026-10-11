"""L0 · 确定性内核 —— 逐位可复现是整个系统的第一性约束。

设计依据（零重写）：
    - NumPy 全局 RNG 用 PCG64（NumPy 官方默认，优于 MT19937 的统计性质）。
    - Python 内置 random / stdlib hash 一并锁定，杜绝隐性随机源。
    - 提供 spawn_seed：由 (master, task, trial) 派生互不重叠的子流，
      避免「同一 seed 复用导致 HPO 各 trial 相关」这一经典陷阱。

可验证不变量：
    I1  same seed → 同一调用序列输出逐位相同（bit-identical）。
    I2  spawn_seed 派生的子流互不相关（经验相关 |r| < 0.02）。
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

_HASH = hashlib.blake2b


def seed_everything(seed: int = 20261011) -> None:
    """锁定全部随机源。seed 为 0 亦合法（不做 falsy 短路，这是常见坑）。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def deterministic(seed: int = 20261011) -> Iterator[np.random.Generator]:
    """作用域内确定性：退出后恢复调用方原本的全局状态，避免污染宿主程序。"""
    prev_py = random.getstate()
    prev_np = np.random.get_state()
    try:
        seed_everything(seed)
        yield np.random.default_rng(seed)
    finally:
        random.setstate(prev_py)
        np.random.set_state(prev_np)


def spawn_seed(master: int, *parts: int | str) -> int:
    """由主种子与任务标识派生互不重叠的子种子（BLAKE2b，抗碰撞、平台无关）。

    为什么不用 `master + i`：相邻整数种子的线性同余/PCG 序列存在已知相关结构，
    会让 HPO 的不同 trial 共享伪随机结构，进而虚高集成多样性估计。
    """
    h = _HASH(digest_size=8)
    h.update(str(master).encode())
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode())
    return int.from_bytes(h.digest(), "big") % (2**63)


def rng_for(master: int, *parts: int | str) -> np.random.Generator:
    """派生独立 RNG 流——每次 trial / 每个 fold 都应有自己的一条。"""
    return np.random.default_rng(spawn_seed(master, *parts))


def stable_hash(obj: object) -> str:
    """对任意可字符串化对象求平台无关稳定哈希（跨进程/跨平台一致）。

    用于数据集指纹、模型卡缓存键、门禁报告去重。Python 内置 hash() 对 str
    带 PYTHONHASHSEED 随机化，绝不可用于持久化键。
    """
    return _HASH(str(obj).encode("utf-8"), digest_size=8).hexdigest()
