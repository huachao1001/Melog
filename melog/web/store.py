"""内存指标存储：实时历史 + 待落盘队列，线程安全。"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List

from ..downsample import downsample


class MetricStore:
    """按指标名维护 (step, value) 历史；drain 取出未落盘记录。"""

    def __init__(self):
        self._data: Dict[str, List[tuple]] = defaultdict(list)
        self._pending: List[Dict] = []  # 待落盘队列
        self._lock = threading.Lock()

    def add(self, step: int, metrics: Dict[str, float]) -> None:
        with self._lock:
            for name, value in metrics.items():
                self._data[name].append((step, float(value)))
                self._pending.append({"metric": name, "step": step, "value": float(value)})

    def snapshot(self, max_points: int | None = None) -> Dict[str, List[Dict]]:
        """全量历史的展示快照，可选降采样。"""
        with self._lock:
            return {
                name: [{"step": s, "value": v} for s, v in downsample(points, max_points)]
                for name, points in self._data.items()
            }

    def drain(self) -> List[Dict]:
        """取出尚未持久化的记录（用于 JSONL 落盘），取出后清空。"""
        with self._lock:
            drained, self._pending = self._pending, []
            return drained
