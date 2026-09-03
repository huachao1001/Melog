"""控制台消息：print 风格接口（log / success / error / warn）+ 官方 print 拦截。

由 Melog 组合使用；消息渲染到当前 stdout，console.log 镜像由 Mirror
（接管 stdout）完成，本模块不关心落盘。每条消息自动带 ``[HH:MM:SS]``
时间戳前缀（空内容不加，保持空行语义）。
"""

from __future__ import annotations

import builtins
import sys
import time
from typing import Any

from ..utils.tqdm import _is_tty

__all__ = ["Console"]

# 控制台消息色（SGR 标准色，随终端主题）；log 走终端默认色（黑字）
_GREEN, _RED, _YELLOW = "\x1b[32m", "\x1b[31m", "\x1b[33m"
_RESET = "\x1b[0m"

# 被 print 拦截顶掉的原生 print，以及当前已拦截 print 的实例栈（close 时还原）
_ORIG_PRINT = None
_PRINT_PATCHED: list = []


class Console:
    """控制台消息统一出口：多参数转 str、图标前缀、TTY 着色、print 拦截。"""

    def log(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """普通控制台输出，签名对齐 print；终端默认色（黑字），无图标前缀，自动带 [HH:MM:SS] 时间戳前缀。"""
        self._emit("", "", values, sep, end, flush)

    def success(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """绿色文字 + ✔ 前缀，自动带 [HH:MM:SS] 时间戳前缀。"""
        self._emit("✔", _GREEN, values, sep, end, flush)

    def error(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """红色文字 + ✘ 前缀，自动带 [HH:MM:SS] 时间戳前缀。"""
        self._emit("✘", _RED, values, sep, end, flush)

    def warn(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """黄色文字 + ⚠ 前缀，自动带 [HH:MM:SS] 时间戳前缀。"""
        self._emit("⚠", _YELLOW, values, sep, end, flush)

    def _emit(self, icon: str, color: str, values: tuple, sep: str,
              end: str, flush: bool) -> None:
        """控制台消息统一出口：时间戳、转 str、拼图标、按 TTY 着色，写入当前 stdout。"""
        text = sep.join(str(v) for v in values)
        stream = sys.stdout
        if not text:  # 空内容不加时间戳（print() / print(end="") 的空行语义）
            stream.write(end)
            if flush:
                stream.flush()
            return
        line = f"{time.strftime('[%H:%M:%S]')} {icon} {text}" if icon \
            else f"{time.strftime('[%H:%M:%S]')} {text}"
        if color and _is_tty(stream):
            line = f"{color}{line}{_RESET}"
        stream.write(line + end)
        if flush:
            stream.flush()

    # ------------------------------------------------------------ print 拦截
    def patch_print(self) -> None:
        """拦截官方 print：用户代码的 print(...) 内部改走 self.log()。"""
        global _ORIG_PRINT
        if not _PRINT_PATCHED:
            _ORIG_PRINT = builtins.print
        _PRINT_PATCHED.append(self)
        builtins.print = self._print_proxy

    def unpatch_print(self) -> None:
        global _ORIG_PRINT
        if self in _PRINT_PATCHED:
            _PRINT_PATCHED.remove(self)
        if not _PRINT_PATCHED and _ORIG_PRINT is not None:
            builtins.print = _ORIG_PRINT
            _ORIG_PRINT = None

    def _print_proxy(self, *values: Any, sep: str = " ", end: str = "\n",
                     flush: bool = False, file: Any = None) -> None:
        """print 替身：file 显式指定时走原生 print，否则改道 log()。"""
        if file is not None:
            _ORIG_PRINT(*values, sep=sep, end=end, flush=flush, file=file)
            return
        self.log(*values, sep=sep, end=end, flush=flush)
