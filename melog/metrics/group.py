"""MetricGroup：具名指标集合，一次同步合并全部指标。"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from ..utils.distributed import gather_object
from .base import Metric, _param_specs
from .basic import Mean

__all__ = ["MetricGroup"]


class MetricGroup:
    """把一批指标组织在一起统一使用。

    用法::

        metrics = MetricGroup({"loss": Mean(), "acc": Accuracy()})

        # 每个 batch（所有 rank 都执行）：标量按注册名喂入，
        # 观测型指标的观测单独放进 args（元组按位置 / 字典按键名）
        metrics.feed(args={"logits": logits, "labels": labels}, loss=loss, n=1)

        # epoch 末：交给 StepsBar 的 metrics=... 自动合并记录并重置；
        # 不用 StepsBar 时手动落盘一次 + 开启新一轮统计
        melog.scalar(metrics)
        metrics.reset()

    配合 StepsBar(metrics=...) 使用时，StepsBar 每次迭代自动从批次数据
    识别样本数（批次大小），Mean 按它精确平均，feed 无需传元组；识别
    失败（如迭代 range）回退等权平均并警告一次。显式传 (值, 观测数)
    元组时以显式值优先。
    """

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self._metrics: Dict[str, Metric] = dict(metrics or {})
        # 每次 feed() 后触发的回调（由 StepsBar 挂载，用于进度条实时显示本地值）
        self._on_feed: Optional[Callable[[bool], None]] = None
        # StepsBar 每次迭代自动注入的当前批次样本数（None = 未知，等权）
        self._batch_count: Optional[float] = None
        self._count_warned = False  # 识别失败仅警告一次

    def add(self, name: str, metric: Metric) -> "MetricGroup":
        if name in self._metrics:
            raise KeyError(f"指标重复注册: {name}")
        self._metrics[name] = metric
        return self

    def feed(self, args: Optional[Union[Tuple, Dict]] = None, *, write: bool = True,
             **batch: Any) -> None:
        """把一个 batch 的全部观测喂给整组指标，分发由框架完成。

        挂接 StepsBar(metrics=...) 时，feed 后默认（write=True）即自动
        把本卡本地值写入日志/面板（零通信，仅 rank0 落盘），无需再手动
        scalar；epoch 末由 StepsBar 跨 GPU 合并记录并 reset（write=False
        时同样自动）。write=False 适合不想逐 batch 写曲线的场景（如验
        证集）。未挂接 StepsBar 的组不自动记录，手动 melog.scalar(metrics)
        落盘。

        Args:
            args: 观测型指标（如分类指标的 logits/labels）的观测，单独
                成组传入：字典按键名对应指标 compute / prepare 的形参名
                （推荐，形参多时更可读），自动分发给形参名匹配的指标，
                多余的键忽略；元组按位置喂给未被注册名喂入的指标。
            write: 是否实时写入日志/面板（挂接 StepsBar 时生效，默认
                True 即 feed 即记录）。
            **batch: 按注册名喂入的指标观测（如 loss / lr），同名键取值；
                Mean 的观测数自动取 StepsBar 识别的批次样本数（识别失败
                等权）；需手动指定时传元组 (值, 观测数)，如
                loss=(3.2, batch_size)。没有同名键就跳过（不累积也不报错）。
        """
        count = self._batch_count
        named: Dict[str, Any] = {}
        for name, metric in self._metrics.items():
            if name in batch:
                named[name] = batch[name]
                value = batch[name]
                if isinstance(value, tuple):
                    metric.feed(*value)
                elif isinstance(metric, Mean) and count:
                    metric.feed(value, count)
                else:
                    metric.feed(value)
        if args is not None:
            if isinstance(args, dict):
                for name, metric in self._metrics.items():
                    if name in named:
                        continue  # 已按注册名喂入，不重复
                    if self._accepts(metric, args):
                        metric.feed(**args)
            else:
                for name, metric in self._metrics.items():
                    if name not in named and self._dispatchable(metric):
                        metric.feed(*args)
        if self._on_feed is not None:
            self._on_feed(write)

    @staticmethod
    def _dispatchable(metric: Any) -> bool:
        """是否为具备 _entry 协议的 Metric（Last 等鸭子类型只按注册名喂）。"""
        return hasattr(metric, "_entry")

    @staticmethod
    def _accepts(metric: Any, args: Dict[str, Any]) -> bool:
        """指标的 compute / prepare 形参名是否与 args 的键匹配。"""
        if not MetricGroup._dispatchable(metric):
            return False
        specs = _param_specs(metric._entry())
        return bool(args) and (
            any(name in args for name, _k, _d in specs)
            or any(kind is inspect.Parameter.VAR_KEYWORD for _n, kind, _d in specs)
        )

    def local(self) -> Dict[str, Any]:
        """当前 rank 的本地指标值（零通信，不触发跨 rank 收集），供实时显示。

        等价于把本 rank 状态单方面合并：无观测的指标为 NaN；返回矩阵的
        指标（如 ConfusionMatrix）原样返回，调用方可按需过滤。
        """
        return {name: m.merge_states([m.state()]) for name, m in self._metrics.items()}

    def _compute(self) -> Dict[str, Any]:
        """同步合并组内全部指标并返回全局结果（内部方法，由 scalar 调用）。

        所有 rank 必须以相同顺序调用（一次 all_gather 完成全部同步），
        返回值在各 rank 上一致，可直接交给 melog.scalar()。
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
        self._batch_count = None  # 丢弃上一轮遗留的批次样本数

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self) -> Iterator[Tuple[str, Metric]]:
        return iter(self._metrics.items())
