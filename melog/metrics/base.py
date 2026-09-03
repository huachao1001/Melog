"""指标基类。

- Metric：完整契约（feed / state / merge_states / reset），适合 epoch 级
  或需要自定义合并方式的指标，跨 GPU 的状态收集（all_gather_object）与
  单进程直通由 compute() 统一完成。
- BatchMetric：单批次指标，子类只需实现 compute_batch() 一个函数，
  累积、合并、reset、分布式同步全部由框架完成。

同步约定：compute() 是集合操作，所有 rank 必须以相同顺序调用
（通常在每个 epoch 或验证轮结束时统一调用），各 rank 返回一致的全局结果。
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..utils.distributed import gather_object

__all__ = ["Metric", "BatchMetric"]


def _to_float(value: Any) -> float:
    # 兼容 0 维 torch tensor / numpy 标量
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class Metric(ABC):
    """训练指标基类。

    生命周期（典型：一个 epoch 一轮）::

        metric.feed(...)            # 每个 batch 累积本地观测
        value = metric.compute()    # epoch 末：跨 GPU 同步并计算全局结果
        metric.reset()              # 开启下一轮统计

    子类契约：
        - feed(*args, **kwargs): 累积一个批次的本地观测
        - state(): 导出本地累积状态（任意可 pickle 对象，如数值、列表、字典）
        - merge_states(states): 把各 rank 的状态列表（按 rank 顺序）合并为
          最终结果，返回数值或数值字典
        - reset(): 清空状态，开启新一轮统计
    """

    def compute(self) -> Any:
        """同步所有 rank 的状态并计算全局指标值。"""
        return self.merge_states(gather_object(self.state()))

    @abstractmethod
    def feed(self, *args: Any, **kwargs: Any) -> None:
        """累积一个批次的本地观测。"""

    @abstractmethod
    def state(self) -> Any:
        """导出本地累积状态（需可 pickle）。"""

    @abstractmethod
    def merge_states(self, states: List[Any]) -> Any:
        """合并所有 rank 的状态列表（按 rank 顺序）为最终结果。"""

    @abstractmethod
    def reset(self) -> None:
        """清空累积状态。"""


class BatchMetric(Metric):
    """单批次指标基类：子类只实现 compute_batch()，其余全部交给框架。

    用户只需提供"单 batch 怎么算"。框架在 feed() 时按 compute_batch
    声明的形参名从观测中自动取值回调；形参名与个数完全由用户自定义
    （logits / labels 仅为常见示例）::

        class MaskedAcc(BatchMetric):
            def compute_batch(self, logits, labels, mask):
                ...  # 单 batch 计算，需要几个参数就声明几个

        metric.feed(logits=..., labels=..., mask=...)  # 具名喂入
        metric.feed(logits, labels, mask)              # 位置喂入亦可

    compute_batch 返回 float：各 batch 等权平均；返回 (value, count)
    元组：按 count（观测数，如样本数）加权平均（各 batch 样本数不同时
    推荐）。多 GPU 下自动合并为全局结果（而非各卡平均值的平均）。

    需要全局计数或排序的指标（如 macro F1、AUC），请改用 Metric 完整
    契约（参考内置分类指标 _CountMetric 的 _consume 写法）。
    """

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0.0

    @abstractmethod
    def compute_batch(self, *args: Any, **kwargs: Any) -> Any:
        """计算单个 batch 的指标值：返回 float，或 (value, count) 元组。"""

    # ------------------------------------------------------------ 参数分发
    def _param_specs(self) -> List[Tuple[str, Any, Any]]:
        # compute_batch 的 (形参名, kind, 默认值) 列表；绑定方法不含 self，按实例缓存
        specs = self.__dict__.get("_cb_specs")
        if specs is None:
            specs = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(self.compute_batch).parameters.values()
            ]
            self._cb_specs = specs
        return specs

    def _build_kwargs(self, args: Tuple[Any, ...], batch: Dict[str, Any]) -> Dict[str, Any]:
        """把位置/具名观测按 compute_batch 的形参名组装成调用参数。"""
        specs = self._param_specs()
        named = [
            s
            for s in specs
            if s[1] not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        var_kw = any(s[1] == inspect.Parameter.VAR_KEYWORD for s in specs)
        if len(args) > len(named) and not any(
            s[1] == inspect.Parameter.VAR_POSITIONAL for s in specs
        ):
            raise TypeError(
                f"{type(self).__name__}.compute_batch 至多接受 {len(named)} 个位置参数，"
                f"收到 {len(args)} 个"
            )
        given = {**dict(zip([s[0] for s in named], args)), **batch}
        kwargs: Dict[str, Any] = {}
        for name, _kind, default in named:
            if name in given:
                kwargs[name] = given[name]
            elif default is inspect.Parameter.empty:
                raise KeyError(
                    f"{type(self).__name__} 缺少观测值: {name}；已提供: {sorted(given) or '无'}"
                )
        if var_kw:  # **kwargs 形参收下全部剩余观测
            kwargs.update({k: v for k, v in given.items() if k not in kwargs})
        extra = args[len(named):]
        if extra:  # 多出的位置参数交给 *args 形参
            var_pos = next(s[0] for s in specs if s[1] == inspect.Parameter.VAR_POSITIONAL)
            kwargs[var_pos] = extra
        return kwargs

    # ------------------------------------------------------------ 生命周期
    def feed(self, *args: Any, **batch: Any) -> None:
        """喂入一个 batch 的观测；框架按形参名取值并调用 compute_batch。"""
        self._consume(self.compute_batch(**self._build_kwargs(args, batch)))

    def _consume(self, out: Any) -> None:
        """把 compute_batch 的返回值累积进状态；自定义累积方式时重写此处。"""
        if isinstance(out, tuple):
            value, count = _to_float(out[0]), _to_float(out[1])
        else:
            value, count = _to_float(out), 1.0
        self._sum += value * count
        self._count += count

    def state(self) -> List[float]:
        return [self._sum, self._count]

    def merge_states(self, states: List[List[float]]) -> float:
        total = math.fsum(s[0] for s in states)
        count = math.fsum(s[1] for s in states)
        return total / count if count else float("nan")

    def reset(self) -> None:
        self._sum = self._count = 0.0
