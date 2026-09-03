"""MetricGroup：具名指标集合，一次同步合并全部指标。"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterator, Optional, Tuple

from ..distributed import gather_object
from .base import BatchMetric, Metric

__all__ = ["MetricGroup"]


class MetricGroup:
    """把一批指标组织在一起统一使用。

    用法::

        metrics = MetricGroup({"loss": Mean(), "acc": Accuracy()})

        # 每个 batch（所有 rank 都执行）：按形参名/注册名自动分发
        metrics.feed(logits=logits, labels=labels, loss=loss, n=1)

        # epoch 末：一次同步合并全部，返回全局结果，可直接记录
        logger.scalar(metrics.compute())
        metrics.reset()   # 开启新一轮统计
    """

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self._metrics: Dict[str, Metric] = dict(metrics or {})
        self._weight_ok: Dict[str, bool] = {}  # 指标名 -> update() 是否有 weight 形参

    def add(self, name: str, metric: Metric) -> "MetricGroup":
        if name in self._metrics:
            raise KeyError(f"指标重复注册: {name}")
        self._metrics[name] = metric
        return self

    def update(self, weight: Any = None, **kwargs: Any) -> None:
        """按名字分发观测值。

        Args:
            weight: 组级默认权重——所有 update() 带 weight 形参的指标
                （如 Mean）自动以它加权，batch_size 传一次即可；带权重的
                指标传元组可按指标覆盖，如 loss=(3.2, n)。
            **kwargs: 指标名 -> 观测值。

        组内注册了名为 weight 的指标时，本参数退回普通分发（向后兼容）。
        """
        if "weight" in self._metrics:
            if weight is not None:
                kwargs["weight"] = weight
            weight = None  # 仅作为该指标的观测值，不再作组级权重
        for name, value in kwargs.items():
            metric = self._metrics.get(name)
            if metric is None:
                raise KeyError(f"未注册的指标: {name}")
            if isinstance(value, tuple):
                metric.update(*value)
            elif weight is not None and self._accepts_weight(name, metric):
                metric.update(value, weight)
            else:
                metric.update(value)

    def feed(self, **batch: Any) -> None:
        """把一个 batch 的全部观测喂给整组指标，分发由框架完成。

        - BatchMetric（分类指标及同型自定义指标）：按各指标 compute_batch
          声明的形参名自动取值，形参名任意（logits / labels 仅为示例），
          多余的键自动忽略；形参里声明 weight 即可接收组级权重
        - 其余指标：按注册名取 batch 中的同名观测（带权重传元组，
          如 loss=(3.2, batch_size)；或传 batch 级 weight=... 统一加权）；
          batch 中没有的名字本 batch 不累积
        """
        weight = batch.get("weight")
        for name, metric in self._metrics.items():
            if isinstance(metric, BatchMetric):
                metric.update(**batch)
            elif name in batch:
                value = batch[name]
                if isinstance(value, tuple):
                    metric.update(*value)
                elif weight is not None and name != "weight" and self._accepts_weight(name, metric):
                    metric.update(value, weight)
                else:
                    metric.update(value)

    def _accepts_weight(self, name: str, metric: Metric) -> bool:
        """指标 update() 是否带 weight 形参（按名缓存，避免热路径反复自省）。"""
        if name not in self._weight_ok:
            try:
                params = inspect.signature(metric.update).parameters
                self._weight_ok[name] = "weight" in params
            except (TypeError, ValueError):
                self._weight_ok[name] = False
        return self._weight_ok[name]

    def compute(self) -> Dict[str, Any]:
        """同步合并组内全部指标并返回全局结果。

        所有 rank 必须以相同顺序调用（一次 all_gather 完成全部同步），
        返回值在各 rank 上一致，可直接交给 logger.scalar()。
        """
        names = list(self._metrics)
        states = gather_object([self._metrics[name].state() for name in names])
        return {
            name: self._metrics[name].merge_states([state[i] for state in states])
            for i, name in enumerate(names)
        }

    def reset(self) -> None:
        """重置组内全部指标，开启新一轮统计。"""
        for metric in self._metrics.values():
            metric.reset()

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self) -> Iterator[Tuple[str, Metric]]:
        return iter(self._metrics.items())
