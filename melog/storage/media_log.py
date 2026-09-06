"""媒体记录流程：定位 -> 落盘 -> 索引 -> 日志 -> 推送 Web。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from .media import sanitize_name, save_audio, save_image

if TYPE_CHECKING:
    from ..core import Melog

__all__ = ["MediaLog"]


class MediaLog:
    """图像 / 音频的记录流程，由 Melog 组合持有。

    多 GPU 下仅 rank0 落盘与展示，其余 rank 直接返回；位置自动附着
    最近一次 scalar() 的记录处（Axis.resolve_attach，
    只读、不推进计数）。
    """

    def __init__(self, host: "Melog"):
        self._host = host

    def image(self, name: str, data: Any, caption: Optional[str] = None) -> None:
        """记录一帧图像：文件路径 / PIL.Image / numpy / torch 张量。"""
        self._log("image", name, data, caption=caption,
                  save=lambda out_dir, stem: save_image(data, out_dir, stem))

    def audio(self, name: str, data: Any, sr: int = 22050,
              caption: Optional[str] = None) -> None:
        """记录一段音频：文件路径（按原格式复制）/ numpy / torch 波形。"""
        self._log("audio", name, data, sr=sr, caption=caption,
                  save=lambda out_dir, stem: save_audio(data, out_dir, stem, sr))

    def _log(self, kind: str, name: str, data: Any, save: Callable,
             sr: Optional[int] = None, caption: Optional[str] = None) -> None:
        """单条媒体记录公共流程；文件名以附着位置的全局 x 编号（000000000.png）。

        各分区序列的 x 独立计数可能重复（train 与 val 都有 x=0）：同名
        媒体编号碰撞时顺延 stem，避免互相覆盖。记录携带所属分区
        （rec["tab"]，附着位置所在序列），供续训截断按分区判断去留。
        """
        host = self._host
        if not host._is_primary:
            return
        safe = sanitize_name(name)
        with host._lock:
            x, e, sec = host._axis.resolve_attach()
            out_dir = host._run_dir / "media" / kind / safe
            stem = f"{int(x):09d}"
            while any(out_dir.glob(f"{stem}.*")):  # 同名媒体编号碰撞：顺延
                stem = f"{int(stem) + 1:09d}"
            rel = f"media/{kind}/{safe}/{save(out_dir, stem)}"
            record: Dict[str, Any] = {"type": kind, "metric": name, "step": int(x), "file": rel}
            if sec is not None:
                record["tab"] = sec
            if e is not None:
                record["epoch"] = e
            if sr is not None:
                record["sr"] = sr
            if caption:
                record["caption"] = caption
            host.media.add(kind, name, x, rel, e, sr=sr, caption=caption)
            host._journal.append(record)
            if host._web is not None:
                host._web.publish_media(kind, name, x, e, rel, sr=sr, caption=caption)
