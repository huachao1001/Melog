"""StepsBar：tqdm 风格训练进度条（epoch 绑定 + 指标实时显示 + 末尾自动记录）。

用法（绑定全局活动实例，无需持有 Melog 对象）::

    from melog import StepsBar

    for step in StepsBar(loader, epoch=epoch, metrics=metrics):
        metrics.feed(loss=loss)
        melog.scalar({"loss": loss})

等价的模块级写法：``melog.stepsbar(loader, epoch=epoch, ...)``。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

from ..metrics import MetricGroup
from ..utils.tqdm import tqdm
from ..utils.epoch_end_iterable import EpochEndIterable

if TYPE_CHECKING:
    from .core import Melog

__all__ = ["StepsBar"]


def _progress_disabled() -> bool:
    # CI 日志里进度条噪音大，保留开关
    return os.environ.get("MELOG_DISABLE_PROGRESS", "0") == "1"


def _detect_count(batch: Any) -> Optional[float]:
    """从批次数据自动识别观测数（样本数）；识别失败返回 None。

    识别规则（尽力而为，覆盖常见格式）：
    - 带形状的批次对象（torch tensor / numpy）：shape[0]
    - 字典：递归取第一个能识别的值
    - 元组：一批观测的多个部分（如 (images, labels)），递归取第一个
    - 列表：样本列表（元素为单条样本或标量），取长度
    """
    shape = getattr(batch, "shape", None)
    if shape is not None and len(shape) >= 1:
        return float(shape[0]) or None  # 空批次按未知处理
    if isinstance(batch, dict):
        for value in batch.values():
            n = _detect_count(value)
            if n is not None:
                return n
        return None
    if isinstance(batch, tuple) and batch:
        return _detect_count(batch[0])
    if isinstance(batch, list) and batch:
        return float(len(batch))
    return None


class StepsBar(tqdm):
    """tqdm 风格训练进度条：直接包裹可迭代对象，迭代时自动推进，无需手动 update。

    本库按 epoch 组织训练记录：**每个 epoch 的循环必须用 StepsBar
    包裹**并传入 epoch，坐标（epoch/step）由它统一管理——scalar() /
    image() / audio() 都没有坐标参数，记录自动依附
    当前 epoch 与下一个空槽；不用 StepsBar 包裹的记录退化为全局
    自增 x、无 epoch 分界。

    用法与 tqdm.tqdm 一致::

        from melog import StepsBar

        for batch in StepsBar(loader):
            melog.scalar({"loss": loss})   # 指标实时显示在进度条上

    传入 epoch 时，进入进度条即绑定该 epoch（epoch 内步数清零、全局
    x 从上一位置接续），bar 结束后沿用，直至下一个 epoch；行首描述
    自动标为 "epoch=N"（需自定义时透传 tqdm 的 desc=...）::

        for epoch in range(epochs):
            for _ in StepsBar(loader, epoch=epoch):
                melog.scalar({"loss": loss})   # 坐标自动依附 epoch

    传入 metrics（MetricGroup）时，每次 feed() 默认（write=True）即自动
    记录本卡本地值进日志/面板（零通信，仅 rank0 落盘；write=False 则
    只累积内存，如验证集场景不想逐 batch 写曲线），同时刷新 postfix
    实时显示（NaN 与非数值结果自动跳过）；并自动从批次数据识别样本数
    注入指标组，Mean 按它精确平均（feed 无需传元组），识别失败（如迭
    代 range）回退等权平均并警告一次。迭代自然结束再 gather 所有 rank
    的状态、合并记录全局值一次并重置组内指标（开启下一轮统计）——
    write=False 时也自动执行，无需手动 scalar。曲线上 epoch 内是主卡
    实时值、epoch 末是跨 GPU 精确合并的结果。on_end=... 可在 epoch 末
    自动记录之后收到合并后的指标字典（如按验证指标保存 checkpoint）::

        for _ in StepsBar(loader, epoch=e, metrics=metrics):
            metrics.feed(...)

    自动记录仅在循环自然跑完时触发：提前 break / 抛异常不会记录
    （此时各 rank 的进度可能不一致，自动 compute() 的 all_gather 会
    互相等待甚至挂死；需要中途落盘请显式调用 scalar()）。
    所有 rank 都会触发回调，compute() 在各 rank 同一位置执行，落盘
    仅 rank0。

    total 缺省时自动取 len(iterable)。进度条实时渲染到控制台，并经
    Mirror 同步进 console.log；非 rank0 或设置 MELOG_DISABLE_PROGRESS=1
    时静默。迭代自然结束后自动出栈，可再次调用（如每个 epoch 一条
    进度条）。

    允许嵌套（如训练 bar 内嵌验证 bar）：内部以栈管理，current_bar()
    返回栈顶即当前环境；scalar() 的 postfix 与 advance
    自动作用于栈顶，下层 bar 暂停渲染（计数与 postfix 照常更新），
    栈顶关闭后自动恢复下层渲染。提前 break / 抛异常时 bar 自动出栈
    （迭代器释放时定稿，绑定名字的变量存续期间由 GC 兜底；如需立即
    释放可显式 close() 或用 with 包裹）。
    """

    def __init__(
        self,
        iterable: Iterable,
        total: Optional[float] = None,
        epoch: Optional[int] = None,
        metrics: Optional[MetricGroup] = None,
        on_end: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ):
        """绑定全局活动实例（melog.init 创建的）并打开进度条。

        Args:
            iterable: 可迭代对象（训练/验证循环）。
            total: 总步数；缺省时自动取 len(iterable)。
            epoch: 绑定该 epoch（epoch 内步数清零、全局 x 接续、行首
                固定标注 "epoch=N"，用户 desc 拼在其后）。
            metrics: MetricGroup；每次 feed 自动记录本卡本地值（实时
                曲线 + bar 显示），自动从批次识别样本数供 Mean 精确
                平均，迭代自然结束自动合并记录全局值并重置组内指标。
            on_end: 迭代自然结束（epoch 末）时的回调，参数为跨 GPU
                合并后的指标字典（与 scalar() 落盘的值一致；未观测到
                数据的指标为 NaN）。需配合 metrics 使用；所有 rank 都
                会执行（合并是集合操作），各卡收到的值一致，仅想主卡
                执行时在回调内自行判断 rank。提前 break / 抛异常不触发。
            **kwargs: 其余参数透传 tqdm（desc / leave / mininterval 等）。
        """
        from ..core import current  # 延迟导入：core 也引用本模块，避免循环

        host: "Melog" = current()
        if metrics is not None and not isinstance(metrics, MetricGroup):
            raise TypeError(f"metrics 须为 MetricGroup，收到 {type(metrics).__name__}")
        if on_end is not None and metrics is None:
            raise ValueError("on_end 需配合 metrics 使用（回调参数为合并后的指标）")
        if metrics is not None and metrics._category:
            host._announce_category(metrics._category)  # 大类别声明（幂等）
        if epoch is not None:
            host._bind_epoch(epoch)  # 续训时该 epoch 已有记录会先截断重叠区
        if metrics is not None:

            def _on_item(item: Any) -> None:
                # 每次迭代把识别到的批次样本数注入指标组（feed 前生效）
                n = _detect_count(item)
                if n is not None:
                    metrics._batch_count = n
                elif not metrics._count_warned:
                    metrics._count_warned = True
                    host.warn(
                        "无法从批次自动识别样本数，Mean 类指标按等权平均；"
                        "需精确加权时在 feed 中传 (值, 观测数) 元组"
                    )

            def _on_epoch_end() -> None:
                result = host._log_group(metrics, reset=True)
                if on_end is not None:
                    on_end(result)

            iterable = EpochEndIterable(iterable, _on_epoch_end, on_item=_on_item)
        disable = (not host._is_primary) or (not host._enable_progress) or _progress_disabled()
        if not disable:
            # 嵌套时本条将覆盖栈顶：先擦掉栈顶在屏幕上的行，首帧在干净行上渲染
            host._bars.cover_top()
        if epoch is not None:
            # 行首固定标注 "epoch=N"；用户传了 desc 时拼在 epoch 之后，
            # 无需自己写 desc=f"epoch={epoch}"
            desc = kwargs.pop("desc", None)
            kwargs["desc"] = f"epoch={epoch} {desc}" if desc else f"epoch={epoch}"
        super().__init__(iterable=iterable, total=total, disable=disable, **kwargs)
        if metrics is not None:
            self._hook_metrics(metrics, host)
        host._bars.push(self, metrics)
        self.on_close = lambda: host._bars.forget(self)

    def _hook_metrics(self, metrics: MetricGroup, host: "Melog") -> None:
        """挂载实时钩子：feed() 后把本卡本地值刷进 postfix 并写日志/面板。

        NaN 与非数值（如混淆矩阵）跳过；即使本条被上层 bar 覆盖，
        postfix 数据照常更新，恢复渲染时可见。postfix 显示**用户注册名**
        （不带 category 前缀）；记录走 _record_local：零通信、仅 rank0
        落盘，落盘名带 category 前缀（与 epoch 末合并记录一致，面板据此
        分区）；feed(write=False) 时只刷 postfix、不写日志。
        """

        def _on_feed(write: bool = True) -> None:
            snap = {
                k: v
                for k, v in metrics.local().items()
                if isinstance(v, (int, float)) and v == v
            }
            if not snap:
                return
            self.set_postfix(snap)
            if write:
                cat = metrics._category
                if cat:
                    snap = {f"{cat}/{k}": v for k, v in snap.items()}
                host._record_local(snap)

        metrics._on_feed = _on_feed
