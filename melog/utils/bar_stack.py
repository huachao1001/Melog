"""进度条栈帧管理：打开中的进度条以栈组织（栈顶 = 当前环境）。"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from ..metrics import MetricGroup
from .tqdm import tqdm

__all__ = ["BarFrame", "BarStack"]


class BarFrame:
    """进度条栈帧：一条打开的进度条及其挂载的指标组。"""

    __slots__ = ("bar", "metrics")

    def __init__(self, bar: tqdm, metrics: Optional[MetricGroup]):
        self.bar = bar
        self.metrics = metrics


class BarStack:
    """打开中的进度条栈（栈顶 = 当前环境），由 Melog 实例持有。

    允许嵌套（如训练 bar 内嵌验证 bar）：
    - push：压栈登记，并覆盖下层渲染（只有栈顶渲染，下层计数与
      postfix 照常更新）
    - forget：bar 关闭时出栈，解除其指标组的实时刷新钩子，并恢复
      新栈顶的渲染
    - update_top / advance_top：scalar() 的 postfix 与 advance 自动
      作用于栈顶
    - close_all：自栈顶往下逐条定稿（收尾用；每条 close 触发自身
      on_close 出栈，逐层恢复渲染）
    """

    def __init__(self) -> None:
        self._frames: List[BarFrame] = []
        self._lock = threading.Lock()

    def cover_top(self) -> None:
        """擦除当前栈顶在终端上占用的行（嵌套子条首帧渲染前调用）。

        子条即将覆盖栈顶：先擦掉栈顶整行，避免子条首帧只覆盖行首部分、
        栈顶尾部以残迹形式留在屏幕上。仅 TTY 生效。
        """
        with self._lock:
            if self._frames:
                self._frames[-1].bar.clear_line()

    def push(self, bar: tqdm, metrics: Optional[MetricGroup] = None) -> None:
        """压栈登记一条进度条（实时刷新钩子由 StepsBar 自行挂载）。"""
        with self._lock:
            for frame in self._frames:  # 覆盖下层：只有栈顶渲染
                frame.bar.covered = True
            self._frames.append(BarFrame(bar, metrics))

    def forget(self, bar: tqdm) -> None:
        """bar 关闭：出栈并解除其指标组钩子；恢复新栈顶的渲染。"""
        with self._lock:
            for i, frame in enumerate(self._frames):
                if frame.bar is bar:
                    del self._frames[i]
                    if frame.metrics is not None:
                        frame.metrics._on_feed = None  # bar 已关闭，停止实时刷新
                    break
            if self._frames:
                top = self._frames[-1].bar
                top.covered = False
                # 强制恢复：子条的覆盖/清除会抹掉栈顶在屏幕与文件中的行
                top.refresh(force=True)

    def top(self) -> Optional[tqdm]:
        """当前栈顶进度条（无打开的 bar 时为 None）。"""
        with self._lock:
            return self._frames[-1].bar if self._frames else None

    def update_top(self, metrics: Dict[str, float]) -> None:
        """把一批指标刷进栈顶 postfix（scalar 用）。"""
        top = self.top()
        if top is not None:
            top.set_postfix(metrics)

    def advance_top(self, n: int) -> None:
        """额外推进栈顶 n 步（scalar(advance=N) 用）。"""
        if n:
            top = self.top()
            if top is not None:
                top.update(n)

    def close_all(self) -> None:
        """定稿所有打开中的进度条（栈顶先关，逐层恢复渲染）。"""
        while True:
            bar = self.top()
            if bar is None:
                return
            bar.close()  # 触发自身 on_close -> forget() 出栈
