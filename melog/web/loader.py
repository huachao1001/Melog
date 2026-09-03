"""日志文件解析：metrics.melog → 指标时间序列。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, float, Optional[int]]


class LogLoader:
    """把 JSONL 日志解析为 {metric: [(step, value, epoch), ...]}，坏行自动跳过。

    epoch 为可选字段：记录里没有或不合法时该点 epoch 记为 None。
    """

    @staticmethod
    def parse(path: Path) -> Dict[str, List[Point]]:
        series: Dict[str, List[Point]] = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "metric" in rec and "step" in rec and "value" in rec:
                    epoch = rec.get("epoch")
                    if not isinstance(epoch, (int, float)):
                        epoch = None
                    series[rec["metric"]].append((rec["step"], float(rec["value"]), epoch))
        for pts in series.values():
            pts.sort(key=lambda p: p[0])
        return dict(series)
