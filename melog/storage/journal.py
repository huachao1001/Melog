"""JSONL 日志落盘：指标批量写入、媒体记录即时追加。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..web.store import MetricStore

__all__ = ["Journal"]


class Journal:
    """metrics.melog 日志文件写入器。

    指标记录先进入 MetricStore 暂存，每 flush_every 次提交批量落盘
    （减少写盘次数）；媒体等非指标记录即时追加。行格式为 JSONL，
    ensure_ascii=False 保留中文。
    """

    def __init__(self, path: Path, store: MetricStore, flush_every: int = 1):
        self._path = path
        self._store = store
        self._every = max(1, flush_every)
        self._pending = 0

    def add(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        """暂存一批指标并按 flush_every 节奏落盘（调用方需持有宿主锁）。"""
        self._store.add(step, metrics, epoch)
        self._pending += 1
        if self._pending >= self._every:
            self.flush()
            self._pending = 0

    def append(self, record: Dict[str, Any]) -> None:
        """即时追加一条原始记录（媒体等非指标记录）。"""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        """把暂存的指标记录写盘（收尾时调用，幂等）。"""
        records = self._store.drain()
        if not records:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
