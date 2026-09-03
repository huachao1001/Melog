"""内存指标存储：实时历史 + 待落盘队列，线程安全。"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Optional

from ..utils.downsample import downsample


class MetricStore:
    """按指标名维护 (step, value, epoch) 历史；drain 取出未落盘记录。"""

    def __init__(self):
        self._data: Dict[str, List[tuple]] = defaultdict(list)
        self._pending: List[Dict] = []  # 待落盘队列
        self._lock = threading.Lock()

    def add(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None,
            persist: bool = True) -> None:
        """记录一批指标；persist=False 仅入展示历史（如续训回灌，已落盘）。"""
        with self._lock:
            for name, value in metrics.items():
                self._data[name].append((step, float(value), epoch))
                if not persist:
                    continue
                rec = {"metric": name, "step": step, "value": float(value)}
                if epoch is not None:
                    rec["epoch"] = epoch
                self._pending.append(rec)

    def snapshot(self, max_points: int | None = None) -> Dict[str, List[Dict]]:
        """全量历史的展示快照，可选降采样；epoch 仅在启用时输出。"""
        with self._lock:
            return {
                name: [
                    {"step": s, "value": v, **({"epoch": e} if e is not None else {})}
                    for s, v, e in downsample(points, max_points)
                ]
                for name, points in self._data.items()
            }

    def drain(self) -> List[Dict]:
        """取出尚未持久化的记录（用于落盘），取出后清空。"""
        with self._lock:
            drained, self._pending = self._pending, []
            return drained

    def truncate(self, cut_step: int) -> None:
        """丢弃 step >= cut_step 的历史与未落盘记录（续训清除重叠区）。"""
        with self._lock:
            for name, points in self._data.items():
                self._data[name] = [p for p in points if p[0] < cut_step]
            self._pending = [r for r in self._pending if r["step"] < cut_step]
