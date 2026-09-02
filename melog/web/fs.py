"""文件浏览：盘符列举、目录内容列举、日志文件定位。"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Dict, List


class FileBrowser:
    """供 Web 文件浏览器使用的文件系统只读操作。"""

    SCAN_LIMIT = 500  # 递归扫描日志的上限，防止在大目录（如盘符根）上卡死
    LOG_EXT = ".melog"  # 日志文件扩展名

    @staticmethod
    def list_roots() -> List[str]:
        """Windows 盘符列表。"""
        roots: List[str] = []
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:/")
            try:
                if drive.exists():
                    roots.append(str(drive))
            except OSError:
                continue
        return roots

    @staticmethod
    def list_dir(path: str) -> Dict:
        """列出目录内容：子目录 + .melog 日志文件（其他文件不显示）。

        Raises:
            FileNotFoundError: 路径不存在。
            NotADirectoryError: 路径是文件。
            PermissionError: 目录不可读。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if p.is_file():
            return {"file": str(p)}

        dirs: List[str] = []
        files: List[str] = []
        for entry in p.iterdir():
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.is_file() and entry.suffix == FileBrowser.LOG_EXT:
                    files.append(entry.name)
            except OSError:
                continue  # 跳过无权限项
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)
        parent = str(p.parent) if p.parent != p else None
        return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}

    @staticmethod
    def find_latest_log(directory: Path) -> Path:
        """定位目录下最新的 .melog 日志（递归，限量扫描）。"""
        candidates: List[Path] = []
        for i, pth in enumerate(directory.rglob(f"*{FileBrowser.LOG_EXT}")):
            candidates.append(pth)
            if i >= FileBrowser.SCAN_LIMIT:
                break
        if not candidates:
            raise FileNotFoundError(f"目录下未找到 {FileBrowser.LOG_EXT} 日志: {directory}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
