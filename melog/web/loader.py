"""日志文件解析：metrics.melog → 指标时间序列 / 媒体索引。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, float, Optional[int]]


class LogLoader:
    """把 JSONL 日志解析为 {metric: [(step, value, epoch), ...]}，坏行自动跳过。

    epoch 为可选字段：记录里没有或不合法时该点 epoch 记为 None。
    媒体记录（含 type 字段）没有 value，自动跳过，由 MediaLoader 解析。
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


class MediaLoader:
    """把 JSONL 日志中的媒体记录解析为 {kind: {name: [entry, ...]}}。

    记录格式：{"type": "image"|"audio", "metric": name, "step": n,
    "epoch": 可选, "file": 相对日志目录的媒体文件路径, "sr": 音频可选}。
    entry 为 {step, epoch?, file, sr?}，按 step 升序。
    """

    @staticmethod
    def parse(path: Path, max_per_name: int = 1000) -> Dict[str, Dict[str, List[Dict]]]:
        items: Dict[str, Dict[str, List[Dict]]] = {"image": {}, "audio": {}}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("type")
                if kind not in items:
                    continue
                name, step, file = rec.get("metric"), rec.get("step"), rec.get("file")
                if not isinstance(name, str) or not isinstance(step, int) or not isinstance(file, str):
                    continue
                entry: Dict[str, object] = {"step": step, "file": file}
                if isinstance(rec.get("epoch"), (int, float)):
                    entry["epoch"] = rec["epoch"]
                if isinstance(rec.get("sr"), int):
                    entry["sr"] = rec["sr"]
                if isinstance(rec.get("caption"), str) and rec["caption"]:
                    entry["caption"] = rec["caption"]
                items[kind].setdefault(name, []).append(entry)
        for kind in items:
            for name, entries in items[kind].items():
                entries.sort(key=lambda e: e["step"])
                if len(entries) > max_per_name:
                    del entries[: len(entries) - max_per_name]
        return items
