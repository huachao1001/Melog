"""指标序列降采样：等宽分桶取均值，保留首尾点。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["downsample"]

Point = Tuple[int, float]


def downsample(points: Sequence[Point], max_points: Optional[int]) -> List[Point]:
    """点数超过 max_points 时按等宽分桶取均值降采样。

    首尾原始点强制保留，避免曲线起止失真。
    """
    n = len(points)
    if max_points is None or max_points < 2 or n <= max_points:
        return list(points)

    bucket = n / max_points
    out: List[Point] = []
    for i in range(max_points):
        start = int(i * bucket)
        end = max(start + 1, int((i + 1) * bucket))
        chunk = points[start:end]
        step = sum(p[0] for p in chunk) / len(chunk)
        value = sum(p[1] for p in chunk) / len(chunk)
        out.append((step, value))
    out[0] = (float(points[0][0]), float(points[0][1]))
    out[-1] = (float(points[-1][0]), float(points[-1][1]))
    return out
