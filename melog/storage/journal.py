"""二进制日志落盘：指标批量写入、媒体记录即时追加、重叠区截断。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..web.store import MetricStore
from .melog_file import MelogFile

__all__ = ["Journal"]


class Journal:
    """metrics.melog 日志文件写入器。

    指标记录先进入 MetricStore 暂存，每 flush_every 次提交批量落盘
    （减少写盘次数）；媒体等非指标记录即时追加。文件为 melog 二进制
    容器（见 storage.melog_file），同一 run 目录的每次启动各写一个带
    时间戳的会话文件，互不覆盖。
    """

    def __init__(self, path: Path, store: MetricStore, flush_every: int = 1):
        self._path = Path(path)
        self._store = store
        self._file = MelogFile(path)
        self._every = max(1, flush_every)
        self._pending = 0

    @property
    def path(self) -> Path:
        return self._path

    def add(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        """暂存一批指标并按 flush_every 节奏落盘（调用方需持有宿主锁）。"""
        self._store.add(step, metrics, epoch)
        self._pending += 1
        if self._pending >= self._every:
            self.flush()
            self._pending = 0

    def append(self, record: Dict[str, Any]) -> None:
        """即时追加一条原始记录（媒体等非指标记录）。"""
        self._file.append_media(record)

    def flush(self) -> None:
        """把暂存的指标记录写盘（收尾时调用，幂等）。"""
        records = self._store.drain()
        self._pending = 0
        if records:
            self._file.add_batch(records)

    def truncate_from(self, cut_step: int) -> Tuple[Optional[int], Optional[int]]:
        """截断本会话文件中 step >= cut_step 的记录，返回最后保留记录的
        (step, epoch)（续训清除重叠区用；调用方需先 flush）。"""
        return self._file.truncate_from(cut_step)

    def close(self) -> None:
        self._file.close()
