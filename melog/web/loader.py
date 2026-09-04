"""日志文件解析：metrics-*.melog 二进制会话文件 → 指标时间序列 / 媒体索引。

同一 run 目录下的多个会话文件（每次启动一个，带时间戳；已有会话时
文件名加序号前缀 2.、3.……）在加载时按文件名顺序合并成完整时间线。
"""

from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from ..storage.melog_file import MelogFileReader

Point = Tuple[int, float, Optional[int]]
_PATHS = Union[str, Path, Iterable[Union[str, Path]]]

# 会话文件名：[<序号>.]metrics-<YYYYmmdd_HHMMSS>[-<同秒序号>].melog
# （首个会话无序号前缀，后续会话带 2.、3.…… 前缀）
_SESSION_RE = re.compile(r"^(?:(\d+)\.)?metrics-(\d{8}_\d{6})(?:-(\d+))?\.melog$")


def _session_sort_key(path: Path) -> Tuple[str, int, int]:
    """按启动时间排序（会话序号、同秒序号依次次之）；不符合命名的文件排最前。"""
    m = _SESSION_RE.match(path.name)
    if m:
        return (m.group(2), int(m.group(1) or 1), int(m.group(3) or 0))
    return ("", 0, 0)


class LogLoader:
    """把 melog 二进制日志解析为 {metric: [(step, value, epoch), ...]}。"""

    @staticmethod
    def session_files(target: Union[str, Path]) -> List[Path]:
        """把文件 / 目录参数展开为要合并的会话文件列表（按时间戳序）。

        - 目录：取其中 metrics 会话文件（含序号前缀命名，无则回退
          *.melog；再无则递归扫描并选中最近写入的那个 run 目录，兼容
          旧版时间戳子目录布局）
        - metrics* 文件：合并其所在目录的全部会话文件（一次训练的完整曲线）
        - 其他文件：单独解析
        """
        p = Path(target)
        if p.is_dir():
            files = sorted(p.glob("*metrics-*.melog"), key=_session_sort_key) \
                or sorted(p.glob("*.melog"))
            if files:
                return files
            nested = sorted(p.rglob("*.melog"), key=lambda f: f.stat().st_mtime)
            return sorted(nested[-1].parent.glob("*.melog")) if nested else []
        if (_SESSION_RE.match(p.name) or p.name.startswith("metrics")) \
                and p.parent.is_dir():
            files = sorted(p.parent.glob("*metrics-*.melog"), key=_session_sort_key)
            return files or [p]
        return [p]

    @staticmethod
    def _paths(target: _PATHS) -> List[Path]:
        """文件 / 目录 / 多路径参数统一展开为会话文件列表。"""
        if isinstance(target, (str, Path)):
            return LogLoader.session_files(target)
        return [Path(t) for t in target]

    @staticmethod
    def parse(target: _PATHS) -> Dict[str, List[Point]]:
        """解析一个或多个会话文件为指标时间序列；坏文件自动跳过。"""
        series: Dict[str, List[Point]] = defaultdict(list)
        for path in LogLoader._paths(target):
            for step, epoch, values in MelogFileReader(path).records():
                for name, value in values.items():
                    series[name].append((step, float(value), epoch))
        for pts in series.values():
            pts.sort(key=lambda p: p[0])
        return dict(series)


class MediaLoader:
    """把 melog 二进制日志中的媒体记录解析为 {kind: {name: [entry, ...]}}。

    entry 为 {step, epoch?, file, sr?, caption?}，按 step 升序。
    """

    @staticmethod
    def parse(target: _PATHS, max_per_name: int = 1000) -> Dict[str, Dict[str, List[Dict]]]:
        items: Dict[str, Dict[str, List[Dict]]] = {"image": {}, "audio": {}}
        for path in LogLoader._paths(target):
            for rec in MelogFileReader(path).media():
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
