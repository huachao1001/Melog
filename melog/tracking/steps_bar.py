"""StepsBar：tqdm 风格训练进度条（epoch 绑定 + 指标实时显示 + 末尾自动记录）。

用法（绑定全局活动实例，无需持有 Melog 对象）::

    from melog import StepsBar

    for step in StepsBar(loader, epoch=epoch, metrics=metrics, reset=True):
        metrics.feed(loss=loss)
        melog.scalar({"loss": loss})

等价的模块级写法：``melog.stepsbar(loader, epoch=epoch, ...)``。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, Optional

from ..metrics import MetricGroup
from ..utils.tqdm import tqdm
from ..utils.epoch_end_iterable import EpochEndIterable

if TYPE_CHECKING:
    from .core import Melog

__all__ = ["StepsBar"]


def _progress_disabled() -> bool:
    # CI 日志里进度条噪音大，保留开关
    return os.environ.get("MELOG_DISABLE_PROGRESS", "0") == "1"


class StepsBar(tqdm):
    """tqdm 风格训练进度条：直接包裹可迭代对象，迭代时自动推进，无需手动 update。

    本库按 epoch 组织训练记录：**每个 epoch 的循环必须用 StepsBar
    包裹**并传入 epoch，坐标（epoch/step）由它统一管理——scalar() /
    log_group() / image() / audio() 都没有坐标参数，记录自动依附
    当前 epoch 与下一个空槽；不用 StepsBar 包裹的记录退化为全局
    自增 x、无 epoch 分界。

    用法与 tqdm.tqdm 一致::

        from melog import StepsBar

        for batch in StepsBar(loader):
            melog.scalar({"loss": loss})   # 指标实时显示在进度条上

    传入 epoch 时，进入进度条即绑定该 epoch（epoch 内步数清零、全局
    x 从上一位置接续），bar 结束后沿用，直至下一个 epoch；行首描述
    自动标为 "epoch N"（需自定义时透传 tqdm 的 desc=...）::

        for epoch in range(epochs):
            for _ in StepsBar(loader, epoch=epoch):
                melog.scalar({"loss": loss})   # 坐标自动依附 epoch

    传入 metrics（MetricGroup）时，进度条实时显示本卡本地值——每次
    feed() 后零通信刷新 postfix（实际渲染的只有 rank0，即主卡本地
    值；无观测的指标与非数值结果自动跳过）。迭代自然结束即 gather
    所有 rank 的状态、log_group 全局值一次（reset=True 则记录后重
    置组内指标），曲线上得到跨 GPU 精确合并的结果::

        for _ in StepsBar(loader, epoch=e, metrics=metrics, reset=True):
            metrics.feed(...)

    自动记录仅在循环自然跑完时触发：提前 break / 抛异常不会记录
    （此时各 rank 的进度可能不一致，自动 compute() 的 all_gather 会
    互相等待甚至挂死；需要中途落盘请显式调用 scalar() / log_group()）。
    所有 rank 都会触发回调，compute() 在各 rank 同一位置执行，落盘
    仅 rank0。

    total 缺省时自动取 len(iterable)。进度条实时渲染到控制台，并经
    Mirror 同步进 console.log；非 rank0 或设置 MELOG_DISABLE_PROGRESS=1
    时静默。迭代自然结束后自动出栈，可再次调用（如每个 epoch 一条
    进度条）。

    允许嵌套（如训练 bar 内嵌验证 bar）：内部以栈管理，current_bar()
    返回栈顶即当前环境；scalar() / log_group() 的 postfix 与 advance
    自动作用于栈顶，下层 bar 暂停渲染（计数与 postfix 照常更新），
    栈顶关闭后自动恢复下层渲染。提前 break 的 bar 请 close()（或用
    with 包裹），否则会一直留在栈中占位。
    """

    def __init__(
        self,
        iterable: Iterable,
        total: Optional[float] = None,
        epoch: Optional[int] = None,
        metrics: Optional[MetricGroup] = None,
        reset: bool = False,
        **kwargs: Any,
    ):
        """绑定全局活动实例（melog.init 创建的）并打开进度条。

        Args:
            iterable: 可迭代对象（训练/验证循环）。
            total: 总步数；缺省时自动取 len(iterable)。
            epoch: 绑定该 epoch（epoch 内步数清零、全局 x 接续、行首
                自动标注 "epoch N"）。
            metrics: MetricGroup；bar 实时显示本卡本地值，迭代自然结束
                自动 log_group 全局值。
            reset: 自动记录后是否重置组内指标。
            **kwargs: 其余参数透传 tqdm（desc / leave / mininterval 等）。
        """
        from ..core import current  # 延迟导入：core 也引用本模块，避免循环

        host: "Melog" = current()
        if metrics is not None and not isinstance(metrics, MetricGroup):
            raise TypeError(f"metrics 须为 MetricGroup，收到 {type(metrics).__name__}")
        if epoch is not None:
            with host._lock:
                host._axis.bind_epoch(epoch)
        if metrics is not None:
            iterable = EpochEndIterable(
                iterable, lambda: host.log_group(metrics, reset=reset)
            )
        disable = (not host._is_primary) or (not host._enable_progress) or _progress_disabled()
        if epoch is not None and "desc" not in kwargs:
            kwargs["desc"] = f"epoch {epoch}"
        super().__init__(iterable=iterable, total=total, disable=disable, **kwargs)
        if metrics is not None:
            self._hook_metrics(metrics)
        host._bars.push(self, metrics)
        self.on_close = lambda: host._bars.forget(self)

    def _hook_metrics(self, metrics: MetricGroup) -> None:
        """挂载实时刷新钩子：feed() 后把本卡本地数值刷进自己的 postfix。

        NaN 与非数值（如混淆矩阵）不上 postfix；即使本条被上层 bar
        覆盖，postfix 数据照常更新，恢复渲染时可见。
        """

        def _display_local() -> None:
            snap = {
                k: v
                for k, v in metrics.local().items()
                if isinstance(v, (int, float)) and v == v
            }
            self.set_postfix(snap)

        metrics._on_feed = _display_local
