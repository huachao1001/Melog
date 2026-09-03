"""媒体展示视图：实时媒体与手动加载的历史日志之间的切换。

条目里的 file 是相对媒体根目录的路径；URL 由 media_url 构造，服务端
经 resolve() 白名单校验后回源——前端永远接触不到任意路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from ..media import media_url
from .media_store import MediaStore

MediaIndex = Dict[str, Dict[str, list]]  # {kind: {name: [entry, ...]}}


class MediaView:
    """当前 Web 端看到的媒体视图：历史日志优先，否则实时媒体。"""

    def __init__(self, store: MediaStore):
        self._store = store
        self._base: Optional[Path] = None  # 实时媒体根目录（run_dir/media）
        self._loaded: Optional[MediaIndex] = None
        self._loaded_base: Optional[Path] = None

    def set_base(self, base: Union[str, Path, None]) -> None:
        """设置实时媒体根目录（媒体文件按 <root>/<kind>/<name>/… 存放）。"""
        self._base = Path(base) if base else None

    def set_loaded(self, series: MediaIndex, base: Union[str, Path, None]) -> None:
        """切换到历史日志视图，base 为该日志的媒体根目录。"""
        self._loaded = series
        self._loaded_base = Path(base) if base else None

    def clear_loaded(self) -> None:
        """切回实时视图。"""
        self._loaded = None
        self._loaded_base = None

    def snapshot(self) -> MediaIndex:
        """当前视图的条目快照：entry 携带可直接访问的 url。"""
        if self._loaded is not None:
            return self._with_urls(self._loaded)
        return self._with_urls(self._store.snapshot())

    def resolve(self, relpath: str) -> Optional[Path]:
        """把相对路径解析为当前视图媒体根目录内的真实文件（防穿越）。

        仅放行 media/ 前缀的路径（日志记录中的媒体相对路径约定），
        且解析后必须位于根目录内。
        """
        base = self._loaded_base if self._loaded is not None else self._base
        norm = (relpath or "").replace("\\", "/")
        if not base or not norm.startswith("media/"):
            return None
        try:
            root = base.resolve()
            path = (root / norm).resolve()
            path.relative_to(root)  # 越出媒体根目录即拒绝
        except (ValueError, OSError):
            return None
        return path if path.is_file() else None

    @staticmethod
    def _with_urls(data: MediaIndex) -> MediaIndex:
        out: MediaIndex = {}
        for kind, by_name in data.items():
            out[kind] = {
                name: [
                    {**{k: v for k, v in e.items() if k != "file"}, "url": media_url(e["file"])}
                    for e in entries
                ]
                for name, entries in by_name.items()
            }
        return out
