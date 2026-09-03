"""指标基类：一套流程统一"实时指标"与"epoch 级指标"，多卡合并由框架完成。

自定义指标继承 Metric，只关注两点：

- compute()：怎么算。实时指标只实现它——每次 feed 立即用本批观测算出
  指标值；epoch 级指标则在 epoch 末用 prepare() 准备好的总量算出全局值。
- prepare()（可选）：epoch 级指标才需要。每次 feed 接收同样的观测，
  返回本批次贡献的增量（如 tp/fp 计数、逐样本得分对），框架自动累积
  并跨 GPU 合并，epoch 末把合并后的总量交给 compute()。

合并规则由框架自动完成，用户无需感知多卡：实时值按观测数加权平均；
增量数值求和、字典按键递归合并、列表拼接。
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..utils.distributed import gather_object

__all__ = ["Metric"]


def _to_float(value: Any) -> float:
    # 兼容 0 维 torch tensor / numpy 标量
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _merge(a: Any, b: Any) -> Any:
    """合并两份增量 / 状态：数值求和、字典按键递归合并、序列拼接。

    该规则即 prepare() 增量观测的全部合并语义，与合并次序无关
    （求和与拼接均可结合），因此跨 rank 合并与逐 batch 累积共用。
    """
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) + list(b)
    return a + b


def _param_specs(fn: Any) -> List[Tuple[str, Any, Any]]:
    """fn 的 (形参名, kind, 默认值) 列表（按底层函数缓存）。"""
    target = getattr(fn, "__func__", fn)
    specs = target.__dict__.get("_melog_specs")
    if specs is None:
        specs = [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
        ]
        target._melog_specs = specs
    return specs


def _build_kwargs(fn: Any, args: Tuple[Any, ...], batch: Dict[str, Any]) -> Dict[str, Any]:
    """把位置 / 具名观测按 fn 的形参名组装成调用参数。"""
    specs = _param_specs(fn)
    named = [
        s for s in specs
        if s[1] not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    var_kw = any(s[1] == inspect.Parameter.VAR_KEYWORD for s in specs)
    if len(args) > len(named) and not any(
        s[1] == inspect.Parameter.VAR_POSITIONAL for s in specs
    ):
        raise TypeError(
            f"{target.__qualname__} 至多接受 {len(named)} 个位置参数，收到 {len(args)} 个"
        )
    given = {**dict(zip([s[0] for s in named], args)), **batch}
    kwargs: Dict[str, Any] = {}
    for name, _kind, default in named:
        if name in given:
            kwargs[name] = given[name]
        elif default is inspect.Parameter.empty:
            qualname = getattr(fn, "__qualname__", fn)
            raise KeyError(f"{qualname} 缺少观测值: {name}；已提供: {sorted(given) or '无'}")
    if var_kw:  # **kwargs 形参收下全部剩余观测
        kwargs.update({k: v for k, v in given.items() if k not in kwargs})
    extra = args[len(named):]
    if extra:  # 多出的位置参数交给 *args 形参
        var_pos = next(s[0] for s in specs if s[1] == inspect.Parameter.VAR_POSITIONAL)
        kwargs[var_pos] = extra
    return kwargs


class Metric(ABC):
    """自定义指标唯一基类：实现 compute()；epoch 末才能算的再加 prepare()。

    实时指标——只实现 compute()，每次喂入立即计算::

        class Acc(Metric):
            def compute(self, logits, labels):
                ...
                return (hits / n, n)   # (值, 观测数)：按样本数加权出全局值
                                       # 只返回 float 时各 batch 等权

    epoch 级指标——再实现 prepare()：每次喂入先"备料"（返回本批次增量），
    epoch 末由合并后的总量调用 compute()::

        class F1(Metric):
            def prepare(self, tp, fp, fn):      # 本批次贡献的计数
                return {"tp": tp, "fp": fp, "fn": fn}

            def compute(self, tp, fp, fn):      # 由总量算全局值
                return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")

    feed() 位置 / 具名喂入均可，框架按 compute / prepare 的形参名组装。
    合并规则（框架自动，无需感知多卡）：实时值按观测数加权平均（而非
    "各卡平均值的平均"）；增量数值求和、字典按键递归合并、列表拼接。

    epoch 末取全局值用 result()（跨 GPU 合并；通常经 MetricGroup /
    StepsBar 自动调用），重置用 reset()。
    """

    _sum: float = 0.0
    _count: float = 0.0
    _acc: Any = None

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """计算指标：实时指标用本批观测，epoch 级指标用合并后的总量。"""

    # ------------------------------------------------------------ 框架流程
    def _has_prepare(self) -> bool:
        """子类是否定义了 prepare（决定实时 / epoch 级两种流程）。"""
        cls = type(self)
        cached = cls.__dict__.get("_melog_prepare")
        if cached is None:
            cached = any("prepare" in c.__dict__ for c in cls.__mro__)
            cls._melog_prepare = cached
        return cached

    def _entry(self):
        """feed 的入口函数：有 prepare 用 prepare（epoch 级），否则 compute（实时）。"""
        return self.prepare if self._has_prepare() else self.compute

    def feed(self, *args: Any, **kwargs: Any) -> None:
        """喂入一个 batch 的观测（位置 / 具名均可，按形参名组装）。"""
        params = _build_kwargs(self._entry(), args, kwargs)
        if self._has_prepare():
            self._accumulate(self.prepare(**params))
        else:
            out = self.compute(**params)
            if isinstance(out, tuple):
                value, count = _to_float(out[0]), _to_float(out[1])
            else:
                value, count = _to_float(out), 1.0
            if count:  # 观测数为 0 的批次不参与统计（如空 batch）
                self._sum += value * count
                self._count += count

    def _accumulate(self, inc: Any) -> None:
        """累积一份 prepare 增量进本地状态。"""
        if self._acc is None:
            self._acc = inc
        elif isinstance(self._acc, list) and isinstance(inc, list):
            self._acc.extend(inc)  # 原地拼接：逐 batch 列表增量避免 O(n²) 拷贝
        else:
            self._acc = _merge(self._acc, inc)

    def state(self) -> Any:
        """导出本地状态（可 pickle）；框架内部使用。"""
        return self._acc if self._has_prepare() else [self._sum, self._count]

    def merge_states(self, states: List[Any]) -> Any:
        """合并各 rank 的状态并计算全局结果；框架内部使用。"""
        if not self._has_prepare():
            total = math.fsum(s[0] for s in states)
            count = math.fsum(s[1] for s in states)
            return total / count if count else float("nan")
        acc = None
        for s in states:
            acc = _merge(acc, s)
        return self._finalize(acc)

    def result(self) -> Any:
        """跨 GPU 合并并返回当前全局结果（epoch 末调用；单进程直通）。"""
        return self.merge_states(gather_object(self.state()))

    def reset(self) -> None:
        """清空累积状态，开启新一轮统计。"""
        self._sum = self._count = 0.0
        self._acc = None

    def _finalize(self, acc: Any) -> Any:
        if acc is None:
            return float("nan")
        if isinstance(acc, dict) and acc and all(isinstance(k, str) for k in acc):
            params = inspect.signature(self.compute).parameters
            if any(k in params for k in acc) or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                return self.compute(**acc)
        return self.compute(acc)
