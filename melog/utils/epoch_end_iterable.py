"""可迭代包装：仅在自然耗尽时触发一次回调（提前 break / 异常不触发）。"""

from __future__ import annotations

from typing import Any, Callable, Iterable

__all__ = ["EpochEndIterable"]


class EpochEndIterable:
    """包装可迭代对象：仅在自然耗尽时触发一次回调（提前 break / 异常不触发）。

    供 StepsBar(..., metrics=...) 实现 epoch 末自动记录。回调发生在最后
    一个元素之后、StopIteration 传给进度条之前，此时进度条尚未关闭，
    记录的指标值能渲染进 postfix。所有 rank 都会执行回调（compute() 是
    集合操作，各 rank 必须在同一位置调用；落盘与展示由 log_group 内部
    仅 rank0 处理）。
    """

    def __init__(self, iterable: Iterable, on_end: Callable[[], None]):
        self._src = iterable
        self._it = iter(iterable)
        self._on_end = on_end
        self._fired = False

    def __iter__(self) -> "EpochEndIterable":
        return self

    def __next__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            if not self._fired:
                self._fired = True
                self._on_end()
            raise

    def __len__(self) -> int:
        # 透传 len()，tqdm 才能自动取 total（无 len 时 TypeError 由 tqdm 捕获）
        return len(self._src)  # type: ignore[arg-type]
