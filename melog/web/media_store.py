"""内存媒体索引：图像 / 音频条目（实时），线程安全。

条目按 step 去重（同 step 重复记录视为覆盖），超出上限丢弃最旧；
完整历史始终在日志文件里，由 MediaLoader 按需加载。
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

MAX_PER_NAME = 200  # 单个媒体名在内存中保留的最近条目数

_KINDS = ("image", "audio")
Entry = Dict[str, object]  # {step, epoch?, file}


class MediaStore:
    """按 (kind, name) 维护最近条目列表；drain 不需要——元数据即时落盘。"""

    def __init__(self, max_per_name: int = MAX_PER_NAME):
        self._max = max(1, max_per_name)
        self._items: Dict[str, Dict[str, List[Entry]]] = {k: {} for k in _KINDS}
        self._lock = threading.Lock()

    def add(self, kind: str, name: str, step: int, file: str, epoch: Optional[int] = None,
            sr: Optional[int] = None, caption: Optional[str] = None) -> None:
        entry: Entry = {"step": int(step), "file": file}
        if epoch is not None:
            entry["epoch"] = epoch
        if sr is not None:
            entry["sr"] = sr
        if caption:
            entry["caption"] = caption
        with self._lock:
            by_name = self._items.setdefault(kind, {})
            entries = by_name.setdefault(name, [])
            for i, old in enumerate(entries):  # 同 step 覆盖
                if old["step"] == entry["step"]:
                    entries[i] = entry
                    return
            entries.append(entry)
            entries.sort(key=lambda e: e["step"])
            if len(entries) > self._max:
                del entries[: len(entries) - self._max]

    def snapshot(self) -> Dict[str, Dict[str, List[Entry]]]:
        """全量条目快照（深拷贝，避免并发修改）。"""
        with self._lock:
            return {
                kind: {name: [dict(e) for e in entries] for name, entries in by_name.items()}
                for kind, by_name in self._items.items()
            }
