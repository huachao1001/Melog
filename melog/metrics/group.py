"""MetricGroup：具名指标集合，一次同步合并全部指标。"""

from __future__ import annotations

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
        metrics.reset()   # 开启下一轮统计
    """

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self._metrics: Dict[str, Metric] = dict(metrics or {})

    def add(self, name: str, metric: Metric) -> "MetricGroup":
        if name in self._metrics:
            raise KeyError(f"指标重复注册: {name}")
        self._metrics[name] = metric
        return self

    def update(self, **kwargs: Any) -> None:
        """按名字分发观测值；带权重的指标传元组，如 loss=(3.2, batch_size)。"""
        for name, value in kwargs.items():
            metric = self._metrics.get(name)
            if metric is None:
                raise KeyError(f"未注册的指标: {name}")
            if isinstance(value, tuple):
                metric.update(*value)
            else:
                metric.update(value)

    def feed(self, **batch: Any) -> None:
        """把一个 batch 的全部观测喂给整组指标，分发由框架完成。

        - BatchMetric（分类指标及同型自定义指标）：按各指标 compute_batch
          声明的形参名自动取值，形参名任意（logits / labels 仅为示例），
          多余的键自动忽略
        - 其余指标：按注册名取 batch 中的同名观测（带权重传元组，
          如 loss=(3.2, batch_size)）；batch 中没有的名字本 batch 不累积
        """
        for name, metric in self._metrics.items():
            if isinstance(metric, BatchMetric):
                metric.update(**batch)
            elif name in batch:
                value = batch[name]
                if isinstance(value, tuple):
                    metric.update(*value)
                else:
                    metric.update(value)

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
