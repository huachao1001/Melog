"""可迭代包装：逐项回调 + 仅在自然耗尽时触发一次结束回调。"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

__all__ = ["EpochEndIterable"]


class EpochEndIterable:
    """包装可迭代对象：逐项触发 on_item，自然耗尽时触发一次 on_end。

    供 StepsBar(..., metrics=...) 实现：on_item 在每个元素交给用户
    代码前触发（StepsBar 据此识别批次样本数并注入指标组，feed 前生效）；
    on_end 在最后一个元素之后、StopIteration 传给进度条之前触发，此时
    进度条尚未关闭，记录的指标值能渲染进 postfix。提前 break / 异常
    不触发 on_end（此时各 rank 的进度可能不一致，自动合并的 all_gather
    会互相等待甚至挂死；需要中途落盘请显式调用 scalar()）。所有 rank
    都会执行回调（合并是集合操作，各 rank 必须在同一位置调用；落盘与
    展示由 scalar() 内部仅 rank0 处理）。
    """

    def __init__(
        self,
        iterable: Iterable,
        on_end: Callable[[], None],
        on_item: Optional[Callable[[Any], None]] = None,
    ):
        self._src = iterable
        self._it = iter(iterable)
        self._on_end = on_end
        self._on_item = on_item
        self._fired = False

    def __iter__(self) -> "EpochEndIterable":
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._it)
        except StopIteration:
            if not self._fired:
                self._fired = True
                self._on_end()
            raise
        if self._on_item is not None:
            self._on_item(item)
        return item

    def __len__(self) -> int:
        # 透传 len()，tqdm 才能自动取 total（无 len 时 TypeError 由 tqdm 捕获）
        return len(self._src)  # type: ignore[arg-type]
