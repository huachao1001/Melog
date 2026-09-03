"""内置标量指标：平均 / 求和 / 最近值 / 计数。

标量累积型指标的状态是一组命名数值，跨 GPU 合并方式由 _ops 声明，
子类只需实现 feed()，无需接触任何分布式代码。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .base import Metric, _to_float

__all__ = ["ScalarMetric", "Mean", "Sum", "Last", "Count"]


class ScalarMetric(Metric):
    """命名标量累积型指标基类。

    子类声明：
        - _ops: 状态名 -> 跨 rank 合并方式（"sum" / "max" / "min" / "first"）
        - _init: 各状态初始值（缺省 0.0，max/min 用 ±inf）

    并实现 feed()；合并结果可经 _finalize() 加工（如 sum/count）。
    """

    _ops: Dict[str, str] = {}
    _init: Dict[str, float] = {}

    def __init__(self) -> None:
        self._acc: Dict[str, float] = {k: self._init.get(k, 0.0) for k in self._ops}

    def _zero(self) -> None:
        for k in self._acc:
            self._acc[k] = self._init.get(k, 0.0)

    def state(self) -> Dict[str, float]:
        return dict(self._acc)

    def merge_states(self, states: List[Dict[str, float]]) -> Any:
        merged: Dict[str, float] = {}
        for name, op in self._ops.items():
            vals = [_to_float(s[name]) for s in states]
            if op == "sum":
                merged[name] = math.fsum(vals)
            elif op == "max":
                merged[name] = max(vals)
            elif op == "min":
                merged[name] = min(vals)
            elif op == "first":  # gather 按 rank 顺序，first 即 rank0 的值
                merged[name] = vals[0]
            else:
                raise ValueError(f"未知的合并方式: {op}")
        return self._finalize(merged)

    def _finalize(self, merged: Dict[str, float]) -> Any:
        return merged

    def reset(self) -> None:
        self._zero()


class Mean(ScalarMetric):
    """按观测数加权的平均。feed(value, count=1.0)：结果 = sum(value * count) / sum(count)。

    常规训练无需传 count（各 batch 等权平均）；各 batch 样本数不均、
    需要按样本/token 精确平均时才传 count（经 MetricGroup 喂入时
    传元组 (值, count)，如 loss=(loss, token_num)）。

    多 GPU 下自动按各 rank 的 count 之和合并，等价于全局样本加权平均，
    而非"各卡平均值的平均"。
    """

    _ops = {"sum": "sum", "count": "sum"}

    def feed(self, value: Any, count: Any = 1.0) -> None:
        self._acc["sum"] += _to_float(value) * _to_float(count)
        self._acc["count"] += _to_float(count)

    def _finalize(self, merged: Dict[str, float]) -> float:
        count = merged["count"]
        return merged["sum"] / count if count else float("nan")


class Sum(ScalarMetric):
    """累加求和。feed(value)。"""

    _ops = {"total": "sum"}

    def feed(self, value: Any) -> None:
        self._acc["total"] += _to_float(value)

    def _finalize(self, merged: Dict[str, float]) -> float:
        return merged["total"]


class Last(ScalarMetric):
    """最近一次观测值；多 GPU 下取 rank0 的最新值（如当前学习率）。"""

    _ops = {"v": "first"}

    def feed(self, value: Any) -> None:
        self._acc["v"] = _to_float(value)

    def _finalize(self, merged: Dict[str, float]) -> float:
        return merged["v"]


class Count(ScalarMetric):
    """观测次数（样本数），常与 Sum 配合实现自定义平均。"""

    _ops = {"n": "sum"}

    def feed(self, n: Any = 1.0) -> None:
        self._acc["n"] += _to_float(n)

    def _finalize(self, merged: Dict[str, float]) -> float:
        return merged["n"]
