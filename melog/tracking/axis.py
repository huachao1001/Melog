"""训练坐标轴：全局 x 与 epoch 内步数的唯一裁决者。"""

from __future__ import annotations

from typing import Dict, Optional

__all__ = ["Axis"]


class Axis:
    """训练坐标轴：全局 x 与 epoch 内步数的唯一裁决者。

    坐标不接受手动指定，完全由 StepsBar 驱动（scalar 写入与媒体定位
    共用同一实现，避免规则漂移）：
    - 未绑定 epoch（没用 StepsBar(epoch=...)）：x 即全局提交计数
    - epoch 模式：x = 该 epoch 的全局基准 + epoch 内已提交步数；
      步数每次提交自增，全局 x 跨 epoch 连续接续

    写入型记录（scalar）走 resolve_commit + commit，会推进计数器；
    附着型记录（媒体）走 resolve_attach，只读、绝不推进任何计数。
    """

    def __init__(self) -> None:
        self.step = 0  # 下一个全局 x（= 已提交记录数）
        self.epoch: Optional[int] = None  # 当前绑定 epoch（粘滞，幂等切换）
        self.epoch_step = 0  # 当前 epoch 内已提交步数
        self.bases: Dict[int, int] = {}  # 各 epoch 的全局 x 基准
        self.last_x = 0  # 最近一次记录的全局 x
        self.last_epoch: Optional[int] = None  # 最近一次记录的 epoch

    @property
    def base(self) -> int:
        """当前 epoch 的全局 x 基准（未绑定 epoch 时为 0）。"""
        return self.bases.get(self.epoch, 0) if self.epoch is not None else 0

    def bind_epoch(self, epoch: int) -> None:
        """进入 epoch（幂等）：epoch 内步数清零，全局 x 从上一位置接续。"""
        if epoch != self.epoch:
            self.epoch = epoch
            self.epoch_step = 0
            self.bases[epoch] = self.step

    def resolve_commit(self) -> "tuple[int, Optional[int]]":
        """scalar 写入位置：下一个空槽（epoch 模式为 epoch 内步数）。"""
        if self.epoch is None:
            return self.step, None
        return self.base + self.epoch_step, self.epoch

    def resolve_attach(self) -> "tuple[int, Optional[int]]":
        """媒体附着位置：最近一次 scalar() 的提交位置（只读，不推进计数）。"""
        return self.last_x, self.last_epoch

    def commit(self, x: int, epoch: Optional[int]) -> None:
        """scalar 记录后推进计数器（附着型记录不调用）。"""
        self.last_x, self.last_epoch = x, epoch
        if epoch is not None:
            self.epoch_step += 1
        self.step = x + 1

    # ------------------------------------------------------------ 断点续训
    def absorb(self, x: int, epoch: Optional[int]) -> None:
        """从历史日志重建状态：吸收一条记录（只推进，不产生写入）。"""
        if epoch is not None and epoch not in self.bases:
            self.bases[epoch] = x  # 该 epoch 的首条记录即其全局基准
        self.step = x + 1
        self.last_x, self.last_epoch = x, epoch

    def cut_on_rebind(self, epoch: int) -> Optional[int]:
        """重新绑定历史 epoch 时应截断的重叠起点；无需截断返回 None。

        续训后再次进入某个已写过记录的 epoch（中断残留），该 epoch 及
        其后的一切都作废，从它的全局基准处截断。当前已绑定的 epoch 重入
        （如嵌套进度条）不算——bind_epoch 对同 epoch 幂等。
        """
        base = self.bases.get(epoch)
        if epoch != self.epoch and base is not None and base < self.step:
            return base
        return None

    def rollback(self, cut: int, last: "tuple[Optional[int], Optional[int]]") -> None:
        """回滚到截断点：丢弃 cut 起的坐标状态（配合文件截断使用）。

        last 为截断后最后一条保留记录的 (step, epoch)；全文件无保留
        记录时 last 为 (None, None)。
        """
        self.step = cut
        for e in [e for e, b in self.bases.items() if b >= cut]:
            del self.bases[e]
        self.last_x, self.last_epoch = (cut - 1, None) if last[0] is None else last
