"""控制台训练进度条：基于 rich，实时显示最新指标。"""

from __future__ import annotations

from typing import Dict, Optional

from rich.console import Group, RenderableType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

__all__ = ["TrainProgress"]

# 进度条上最多展示的指标个数，避免过长换行
_MAX_METRICS_SHOWN = 8


class MetricsColumn(ProgressColumn):
    """在进度条行尾渲染最新指标值，如 loss=0.123 lr=1e-3。"""

    def render(self, task) -> RenderableType:
        text: str = task.fields.get("metrics_text", "")
        return Text(text, style="bold cyan")


def _fmt(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 1e5 or (abs_v < 1e-3 and abs_v > 0):
        return f"{value:.3e}"
    return f"{value:.4f}"


class TrainProgress:
    """训练进度条上下文管理器。

    用法：
        with logger.train(total=steps) as bar:
            for step in range(steps):
                ...
                logger.log({"loss": loss})
                bar.advance(1)
    """

    def __init__(self, total: int, description: str = "train"):
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            MetricsColumn(),
            transient=False,
        )
        self._task_id = self._progress.add_task(description, total=total, metrics_text="")
        self._progress.start()

    # ------------------------------------------------------------------ 展示
    def show_metrics(self, metrics: Dict[str, float]) -> None:
        """将最新指标合并进进度条尾部文本并刷新。"""
        items = list(metrics.items())[:_MAX_METRICS_SHOWN]
        text = "  ".join(f"{k}={_fmt(v)}" for k, v in items)
        self._progress.update(self._task_id, metrics_text=text, refresh=True)

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)

    def set_description(self, description: str) -> None:
        self._progress.update(self._task_id, description=description)

    @property
    def completed(self) -> float:
        return self._progress.tasks[self._task_id].completed

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        self._progress.stop()

    def __enter__(self) -> "TrainProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
