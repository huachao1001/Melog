"""内置标量指标：平均 / 求和 / 最近值 / 计数。

除 Last（"最新值"无法用求和/拼接合并）外：
- Mean 为实时指标（只实现 compute，逐 batch 出值、按观测数加权）；
- Sum / Count 需要跨 batch 累加总量，用 prepare 返回增量、compute 出结果。
"""

from __future__ import annotations

from typing import Any, List

from .base import Metric, _to_float

__all__ = ["Mean", "Sum", "Last", "Count"]


class Mean(Metric):
    """按观测数加权的平均。feed(value, count=1.0)。

    实时指标：compute 返回 (本批值, 观测数)，框架自动按观测数加权出
    全局值。配合 StepsBar(metrics=...) 时 count 自动取识别到的批次样
    本数，常规训练直接 feed(value) 即可；识别失败（如迭代 range）等权
    平均。手动指定时经 MetricGroup 喂元组 (值, 观测数)，如
    loss=(loss, token_num)。

    多 GPU 下合并为全局样本加权平均，而非"各卡平均值的平均"。
    """

    def compute(self, value: Any, count: Any = 1.0) -> "tuple[float, float]":
        return _to_float(value), _to_float(count)


class Sum(Metric):
    """累加求和。feed(value)。"""

    def prepare(self, value: Any) -> float:
        return _to_float(value)

    def compute(self, total: float) -> float:
        return total


class Last:
    """最近一次观测值；多 GPU 下取 rank0 的最新值（如当前学习率）。

    "最新值"无法用求和 / 拼接表达，故不继承 Metric 的自动合并，
    跨 GPU 直接取 rank0 的状态。
    """

    def __init__(self) -> None:
        self._v: Any = None

    def feed(self, value: Any) -> None:
        self._v = _to_float(value)

    def state(self) -> Any:
        return self._v

    def merge_states(self, states: List[Any]) -> float:
        for s in states:
            if s is not None:
                return _to_float(s)
        return float("nan")

    def reset(self) -> None:
        self._v = None

    def result(self) -> float:
        return self.merge_states([self.state()])


class Count(Metric):
    """观测次数（样本数），常与 Sum 配合实现自定义平均。"""

    def prepare(self, n: Any = 1.0) -> float:
        return _to_float(n)

    def compute(self, n: float) -> float:
        return n
