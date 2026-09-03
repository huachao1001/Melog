"""指标基类：用户只写计算方法，累积与多卡合并由框架完成。

- Metric：epoch 级指标基类，只实现 update()（本批次贡献的增量）与
  compute()（由总量算出全局结果）两个纯函数；增量观测的累积、导出、
  跨 GPU 合并（数值求和、字典按键合并、列表拼接）全部由框架自动完成，
  用户无需感知多卡。
- BatchMetric：单批次指标基类，子类只实现 compute_batch() 一个函数，
  适合"每个 batch 都能算出一个值"的指标（如准确率）；累积、合并、
  reset、分布式同步同样全部由框架完成。

同步约定：result() / MetricGroup 的合并是集合操作，所有 rank 必须
以相同顺序调用（通常在每个 epoch 或验证轮结束时统一调用），各 rank
返回一致的全局结果；单进程自动直通。
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


def _merge(a: Any, b: Any) -> Any:
    """合并两份增量 / 状态：数值求和、字典按键递归合并、序列拼接。

    该规则即 Metric 用户增量观测的全部合并语义，与合并次序无关
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


def _build_kwargs(fn: Any, args: Tuple[Any, ...], batch: Dict[str, Any]) -> Dict[str, Any]:
    """把位置 / 具名观测按 fn 的形参名组装成调用参数。"""
    # 缓存挂在底层函数上（bound method 每次访问都是新对象）；签名取 fn
    # 本身：绑定方法自动跳过 self，普通函数形参原样
    target = getattr(fn, "__func__", fn)
    specs = target.__dict__.get("_melog_specs")
    if specs is None:
        specs = [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
        ]
        target._melog_specs = specs
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
            raise KeyError(f"{target.__qualname__} 缺少观测值: {name}；已提供: {sorted(given) or '无'}")
    if var_kw:  # **kwargs 形参收下全部剩余观测
        kwargs.update({k: v for k, v in given.items() if k not in kwargs})
    extra = args[len(named):]
    if extra:  # 多出的位置参数交给 *args 形参
        var_pos = next(s[0] for s in specs if s[1] == inspect.Parameter.VAR_POSITIONAL)
        kwargs[var_pos] = extra
    return kwargs


class Metric(ABC):
    """epoch 级指标基类：只实现 update() 与 compute()，多卡合并无需感知。

    生命周期（典型：一个 epoch 一轮）::

        class F1(Metric):
            def update(self, tp, fp, fn):       # 每个 batch：本批次贡献的计数
                return {"tp": tp, "fp": fp, "fn": fn}

            def compute(self, tp, fp, fn):      # epoch 末：由总量算出全局值
                return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")

        metric.feed(tp=1, fp=0, fn=2)   # 逐 batch 喂入（位置 / 具名均可）
        metric.result()                  # 跨 GPU 合并并计算全局值

    增量观测的合并规则由框架自动完成，用户无需感知多卡：

    - 数值：求和（如混淆计数）
    - 字典：按键递归合并（嵌套数值求和、列表拼接）
    - 列表 / 元组：拼接（如逐样本 ``(得分, 标签)`` 对）

    compute() 收到合并后的总量：总量是字符串键字典且形参名与之匹配时
    按关键字传入，否则作为单个位置参数传入；缺省实现原样返回总量
    （纯计数指标无需重写）。
    """

    def __init__(self) -> None:
        self._acc: Any = None

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """累积一个 batch：返回本批次的增量观测（数值 / 字典 / 列表）。"""

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """由合并后的总量计算全局结果；缺省原样返回总量。"""
        if args and not kwargs:
            return args[0] if len(args) == 1 else args
        if kwargs and not args:
            return kwargs
        return None

    # ------------------------------------------------------------ 框架流程
    def feed(self, *args: Any, **kwargs: Any) -> None:
        """喂入一个 batch 的观测（位置 / 具名均可，按 update 形参名组装）。"""
        inc = self.update(**_build_kwargs(self.update, args, kwargs))
        if self._acc is None:
            self._acc = inc
        elif isinstance(self._acc, list) and isinstance(inc, list):
            self._acc.extend(inc)  # 原地拼接：逐 batch 的列表增量避免 O(n²) 拷贝
        else:
            self._acc = _merge(self._acc, inc)

    def state(self) -> Any:
        """导出本地累积状态（可 pickle）；框架内部使用。"""
        return self._acc

    def merge_states(self, states: List[Any]) -> Any:
        """合并各 rank 的状态并计算全局结果；框架内部使用。"""
        acc = None
        for s in states:
            acc = _merge(acc, s)
        return self._finalize(acc)

    def result(self) -> Any:
        """跨 GPU 合并并返回当前全局结果（epoch 末调用；单进程直通）。"""
        return self.merge_states(gather_object(self.state()))

    def reset(self) -> None:
        """清空累积状态，开启新一轮统计。"""
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


class BatchMetric(Metric):
    """单批次指标基类：子类只实现 compute_batch()，其余全部交给框架。

    用户只需提供"单 batch 怎么算"。框架在 feed() 时按 compute_batch
    的形参名自动从观测中取值回调；各 batch 结果按观测数（样本数）加权
    累积，epoch 末合并出全局值。

    用法::

        class MaskedAcc(BatchMetric):
            def compute_batch(self, logits, labels, mask):
                ...  # 单 batch 计算，需要几个参数就声明几个

        metric.feed(logits=..., labels=..., mask=...)  # 具名喂入
        metric.feed(logits, labels, mask)              # 位置喂入亦可

    compute_batch 返回 float：各 batch 等权平均；返回 (value, count)
    元组：按 count（观测数，如样本数）加权平均（各 batch 样本数不同时
    推荐）。多 GPU 下自动合并为全局结果（而非各卡平均值的平均）。

    需要全局计数或排序的指标（如 macro F1、AUC），请改用 Metric
    （update 返回增量、compute 由总量算结果），同样无需感知多卡。
    """

    def __init__(self) -> None:
        super().__init__()
        self._sum = 0.0
        self._count = 0.0

    @abstractmethod
    def compute_batch(self, *args: Any, **kwargs: Any) -> Any:
        """计算单个 batch 的指标值：返回 float，或 (value, count) 元组。"""

    def update(self, *args: Any, **kwargs: Any) -> Any:
        # Metric 的增量累积不适用本基类（走 compute_batch + 加权累积）
        raise NotImplementedError("BatchMetric 子类请实现 compute_batch()")

    # ------------------------------------------------------------ 生命周期
    def feed(self, *args: Any, **batch: Any) -> None:
        """喂入一个 batch 的观测；框架按形参名取值并调用 compute_batch。"""
        self._consume(self.compute_batch(**_build_kwargs(self.compute_batch, args, batch)))

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
