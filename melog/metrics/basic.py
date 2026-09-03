"""内置标量指标：平均 / 求和 / 最近值 / 计数。

除 Last（"最新值"无法用求和/拼接合并）外，全部基于 Metric 的两个
纯函数：update() 返回本批次增量，compute() 由合并后的总量算出结果，
跨 GPU 合并由框架自动完成。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Metric, _to_float

__all__ = ["Mean", "Sum", "Last", "Count"]


class Mean(Metric):
    """按观测数加权的平均。feed(value, count=1.0)：结果 = sum(value*count)/sum(count)。

    配合 StepsBar(metrics=...) 时 count 自动取识别到的批次样本数，
    常规训练直接 feed(value) 即可；识别失败（如迭代 range）等权平均。
    手动指定时传 count：单独使用直接 feed(value, count)，经 MetricGroup
    喂入时传元组 (值, 观测数)，如 loss=(loss, token_num)。

    多 GPU 下自动按各 rank 的 count 之和合并，等价于全局样本加权平均，
    而非"各卡平均值的平均"。
    """

    def update(self, value: Any, count: Any = 1.0) -> Dict[str, float]:
        return {"total": _to_float(value) * _to_float(count), "count": _to_float(count)}

    def compute(self, total: float, count: float) -> float:
        return total / count if count else float("nan")


class Sum(Metric):
    """累加求和。feed(value)。"""

    def update(self, value: Any) -> float:
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

    def update(self, n: Any = 1.0) -> float:
        return _to_float(n)

    def compute(self, n: float) -> float:
        return n
