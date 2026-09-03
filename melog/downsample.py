"""指标序列降采样：等宽分桶取均值，保留首尾点。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["downsample"]

Point = Tuple[int, float]
PointWithEpoch = Tuple[int, float, Optional[int]]


def downsample(points: Sequence[Point], max_points: Optional[int]) -> List[Point]:
    """点数超过 max_points 时按等宽分桶取均值降采样。

    首尾原始点强制保留，避免曲线起止失真。点为 (step, value) 或
    (step, value, epoch) 三元组：分桶均值取 step/value，epoch 取桶内
    最后一个非空值（跨 epoch 分界时归入新 epoch），无 epoch 则原样输出
    二元组。
    """
    n = len(points)
    if max_points is None or max_points < 2 or n <= max_points:
        return list(points)

    has_epoch = len(points[0]) > 2
    bucket = n / max_points
    out: List[Point] = []
    for i in range(max_points):
        start = int(i * bucket)
        end = max(start + 1, int((i + 1) * bucket))
        chunk = points[start:end]
        step = sum(p[0] for p in chunk) / len(chunk)
        value = sum(p[1] for p in chunk) / len(chunk)
        if has_epoch:
            epoch = None
            for p in chunk:
                if p[2] is not None:
                    epoch = p[2]
            out.append((step, value, epoch))
        else:
            out.append((step, value))
    out[0] = points[0]  # type: ignore[assignment]
    out[-1] = points[-1]  # type: ignore[assignment]
    return out
