"""展示视图：实时指标与手动加载的历史日志之间的切换，统一降采样输出。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..utils.downsample import downsample
from .store import MetricStore

Point = Tuple[int, float, Optional[int]]


class MetricView:
    """当前 Web 端看到的指标视图：历史日志优先，否则实时指标。"""

    def __init__(self, store: MetricStore, max_points: int = 2000):
        self._store = store
        self._max_points = max_points
        self._loaded: Optional[Dict[str, List[Point]]] = None
        self._colors: Dict[str, str] = {}  # 实时指标的用户指定颜色
        self._loaded_colors: Optional[Dict[str, str]] = None  # 历史日志自带的颜色
        self._categories: set = set()  # 实时 run 的大类别（train/val/test）
        self._loaded_categories: Optional[set] = None  # 历史日志自带的大类别

    @property
    def max_points(self) -> int:
        return self._max_points

    @property
    def has_loaded(self) -> bool:
        return self._loaded is not None

    def add_categories(self, categories) -> None:
        """登记实时 run 的大类别（去重）。"""
        self._categories.update(categories)

    def set_categories(self, categories) -> None:
        """整体替换实时 run 的大类别集合（历史恢复时用）。"""
        self._categories = set(categories)

    @property
    def categories(self) -> set:
        """当前视图的大类别集合（历史日志视图优先）。"""
        if self._loaded is not None:
            return self._loaded_categories or set()
        return set(self._categories)

    def set_colors(self, colors: Dict[str, str]) -> None:
        """设置实时运行的用户指定颜色（指标名 -> CSS 颜色）。"""
        self._colors = dict(colors)

    @property
    def colors(self) -> Dict[str, str]:
        """当前视图应使用的颜色映射（历史日志视图优先）。"""
        if self._loaded is not None:
            return self._loaded_colors or {}
        return self._colors

    def set_loaded(self, series: Dict[str, List[Point]], colors: Optional[Dict[str, str]] = None,
                   categories: Optional[set] = None) -> None:
        """切换到历史日志视图，可附带该日志的颜色配置与大类别。"""
        self._loaded = series
        self._loaded_colors = dict(colors or {})
        self._loaded_categories = set(categories or set())

    def clear_loaded(self) -> None:
        """切回实时视图。"""
        self._loaded = None
        self._loaded_colors = None
        self._loaded_categories = None

    def snapshot(self) -> Dict[str, List[Dict]]:
        """当前视图的展示快照（均降采样）；epoch 仅在启用时输出。"""
        if self._loaded is not None:
            return {
                name: [
                    {"step": s, "value": v, **({"epoch": e} if e is not None else {})}
                    for s, v, e in downsample(pts, self._max_points)
                ]
                for name, pts in self._loaded.items()
            }
        return self._store.snapshot(self._max_points)
