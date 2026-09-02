"""展示视图：实时指标与手动加载的历史日志之间的切换，统一降采样输出。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..downsample import downsample
from .store import MetricStore


class MetricView:
    """当前 Web 端看到的指标视图：历史日志优先，否则实时指标。"""

    def __init__(self, store: MetricStore, max_points: int = 2000):
        self._store = store
        self._max_points = max_points
        self._loaded: Optional[Dict[str, List[Tuple[int, float]]]] = None

    @property
    def max_points(self) -> int:
        return self._max_points

    @property
    def has_loaded(self) -> bool:
        return self._loaded is not None

    def set_loaded(self, series: Dict[str, List[Tuple[int, float]]]) -> None:
        """切换到历史日志视图。"""
        self._loaded = series

    def clear_loaded(self) -> None:
        """切回实时视图。"""
        self._loaded = None

    def snapshot(self) -> Dict[str, List[Dict]]:
        """当前视图的展示快照（均降采样）。"""
        if self._loaded is not None:
            return {
                name: [{"step": s, "value": v} for s, v in downsample(pts, self._max_points)]
                for name, pts in self._loaded.items()
            }
        return self._store.snapshot(self._max_points)
