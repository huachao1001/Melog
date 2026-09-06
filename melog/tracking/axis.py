"""训练坐标轴：按分区（tab）隔离的全局 x 与 epoch 内步数的唯一裁决者。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

__all__ = ["Axis", "row_section"]


def row_section(names: Iterable[str], tabs) -> Optional[str]:
    """指标记录行的分区归属：命中首个已声明 tab 前缀的名字（如
    train/loss → train）；未命中为默认序列（None）。

    与面板的分区规则一致（charts.js）：只有前缀命中**显式声明的 tab**
    才算分区，指标名本身的层级命名（recall/class_0 式）不算。历史日志
    的回灌与截断据此判断记录属于哪个分区序列。
    """
    for name in names:
        i = name.find("/")
        if i > 0 and name[:i] in tabs:
            return name[:i]
    return None


class _Section:
    """单个分区序列的坐标状态：全局 x、绑定 epoch 与各 epoch 全局基准。

    __slots__ + 轻量：每个 tab（及无 tab 的默认序列）各一份，互不影响。
    """

    __slots__ = ("step", "epoch", "epoch_step", "bases")

    def __init__(self) -> None:
        self.step = 0  # 该序列下一个全局 x（= 已提交记录数）
        self.epoch: Optional[int] = None  # 当前绑定 epoch（粘滞，幂等切换）
        self.epoch_step = 0  # 当前 epoch 内已提交步数
        self.bases: Dict[int, int] = {}  # 各 epoch 的全局 x 基准


class Axis:
    """训练坐标轴：按分区（tab）隔离的全局 x 与 epoch 内步数的唯一裁决者。

    坐标不接受手动指定，完全由 StepsBar 驱动（scalar 写入与媒体定位
    共用同一实现，避免规则漂移）：

    - 分区规则：scalar / _record_local 携带的 tab 决定提交进哪个序列，
      **各序列的 x 独立计数、互不影响**（train 的 40 步用 train 序列的
      x 0..39，val 的记录从 val 序列的 x 0 起，写 val 不推进 train 的
      步数）；无 tab 的记录进默认序列（section=None）。StepsBar 声明了
      tab=... 时，bar 内不带 tab 的 scalar 记录也归入该分区
    - epoch 规则：StepsBar(epoch=..., tab=...) 绑定**该分区序列**的
      epoch（epoch 内步数清零、该序列 x 从自身基准接续）；每 (分区，
      epoch) 各有全局基准，同一分区跨 epoch 连续接续

    写入型记录（scalar）走 resolve_commit + commit，会推进计数器；
    附着型记录（媒体）走 resolve_attach，只读、绝不推进任何计数
    （媒体附着跨分区取最近一次提交的位置，记录携带所属分区供截断）。
    """

    def __init__(self) -> None:
        self._sections: Dict[Optional[str], _Section] = {}
        self.last_x = 0  # 最近一次记录的全局 x（媒体附着用，跨分区）
        self.last_epoch: Optional[int] = None  # 最近一次记录的 epoch
        self.last_section: Optional[str] = None  # 最近一次记录的分区

    # ------------------------------------------------------------ 提交
    def section(self, section: Optional[str] = None) -> _Section:
        """返回分区序列的状态（缺省 = 默认序列；首次访问自动创建）。"""
        s = self._sections.get(section)
        if s is None:
            s = self._sections[section] = _Section()
        return s

    def bind_epoch(self, epoch: int, section: Optional[str] = None) -> None:
        """进入 epoch（幂等）：该序列 epoch 内步数清零，x 从自身位置接续。"""
        s = self.section(section)
        if epoch != s.epoch:
            s.epoch = epoch
            s.epoch_step = 0
            s.bases[epoch] = s.step

    def resolve_commit(self, section: Optional[str] = None) -> "tuple[int, Optional[int]]":
        """scalar 写入位置：该序列下一个空槽（epoch 模式为 epoch 内步数）。"""
        s = self.section(section)
        if s.epoch is None:
            return s.step, None
        return s.bases[s.epoch] + s.epoch_step, s.epoch

    def resolve_attach(self) -> "tuple[int, Optional[int], Optional[str]]":
        """媒体附着位置：最近一次 scalar() 的提交位置（只读，不推进计数）。

        附着跨分区全局取最近一次提交（含所属分区，媒体记录据此携带
        分区，供续训截断按分区判断去留）。
        """
        return self.last_x, self.last_epoch, self.last_section

    def commit(self, x: int, epoch: Optional[int], section: Optional[str] = None) -> None:
        """scalar 记录后推进该序列计数器（附着型记录不调用）。"""
        s = self.section(section)
        self.last_x, self.last_epoch, self.last_section = x, epoch, section
        if epoch is not None:
            s.epoch_step += 1
        s.step = x + 1

    # ------------------------------------------------------------ 断点续训
    def absorb(self, x: int, epoch: Optional[int], section: Optional[str] = None) -> None:
        """从历史日志重建状态：吸收一条记录（只推进，不产生写入）。"""
        s = self.section(section)
        if epoch is not None and epoch not in s.bases:
            s.bases[epoch] = x  # 该分区该 epoch 的首条记录即其全局基准
        s.step = x + 1
        self.last_x, self.last_epoch = x, epoch

    def cut_on_rebind(self, epoch: int, section: Optional[str] = None) -> Optional[int]:
        """重新绑定历史 epoch 时应截断的重叠起点；无需截断返回 None。

        续训后再次进入该分区序列中已写过记录的 epoch（中断残留），该
        epoch 及其后的一切都作废，从它的全局基准处截断（**只截该分区
        序列**，其他分区的记录不受影响）。当前已绑定的 epoch 重入
        （如嵌套进度条）不算——bind_epoch 对同 epoch 幂等。
        """
        s = self.section(section)
        base = s.bases.get(epoch)
        if epoch != s.epoch and base is not None and base < s.step:
            return base
        return None

    def rollback(self, cut: int, last: "tuple[Optional[int], Optional[int]]",
                 section: Optional[str] = None) -> None:
        """回滚该分区序列到截断点：丢弃 cut 起的坐标状态（配合文件截断）。

        last 为截断后最后一条保留记录的 (step, epoch)（跨分区全局，仅
        用于更新媒体附着位置）；全文件无保留记录时为 (None, None)。
        """
        s = self.section(section)
        s.step = cut
        for e in [e for e, b in s.bases.items() if b >= cut]:
            del s.bases[e]
        self.last_x, self.last_epoch = (cut - 1, None) if last[0] is None else last
