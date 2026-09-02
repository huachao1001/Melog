"""日志文件解析：metrics.jsonl → 指标时间序列。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class LogLoader:
    """把 JSONL 日志解析为 {metric: [(step, value), ...]}，坏行自动跳过。"""

    @staticmethod
    def parse(path: Path) -> Dict[str, List[Tuple[int, float]]]:
        series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
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
                    series[rec["metric"]].append((rec["step"], float(rec["value"])))
        for pts in series.values():
            pts.sort(key=lambda p: p[0])
        return dict(series)
