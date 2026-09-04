"""控制台镜像：把控制台输出（含进度条）同步到日志文件。

写入协议（与终端显示约定一致）：
- 以 ``\\r`` 结尾的片段是进度条行——文件中只保留最后一行，就地刷新
  （seek 回行首、截断、重写），默认实时（throttle=0）；行内容与终端
  渲染一致（仅剥离颜色码），并把细线进度条字符替换为加高的块状字符
  （``━``→``█``、``─``→``░``，文件里进度条更醒目）。用支持自动刷新
  的编辑器打开即可看到动的进度条
- 以 ``\\n`` 结尾的是普通行——先把进度条行定稿为完整行，再追加该行；
  进度条由下一次渲染在消息下方重新开始一条新的就地刷新行
- hook_stdio() 把 sys.stdout / sys.stderr 替换为分流器：控制台（如有）
  与日志文件内容一致；unhook_stdio() 还原

打开已有文件时，若最后一行以 ``\\r`` 结尾，则视为一条未完的进度条行，
后续刷新继续覆盖该行（支持进程重启后续写）。注意：进度条行的就地
重写对 ``tail -f`` 不可见（无新增字节），查看进度条请用编辑器实时刷新。
"""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

__all__ = ["Mirror"]

_TAIL_LIMIT = 8192  # 打开时回读的最大字节数（只用于判断最后一行）

# ANSI 转义序列（颜色/光标控制等）：日志文件保持纯文本，写入前统一剥离
_ANSI_RE = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]")

# 文件侧进度条加高：细线字符替换为块状字符（终端保持细线样式不变）
_BAR_TALL_MAP = str.maketrans({"━": "█", "─": "░"})


class _Tee:
    """分流器：同时写控制台与镜像，其余属性全部代理给控制台流。"""

    def __init__(self, console, mirror: "Mirror"):
        self._console = console
        self._mirror = mirror

    def write(self, s: str) -> int:
        self._console.write(s)
        self._mirror.write(s)
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        try:
            return self._console.isatty()
        except (ValueError, OSError):
            return False

    def __getattr__(self, name):
        return getattr(self._console, name)


class Mirror:
    """日志文件镜像：进度条行就地刷新（文件里始终一行在动的进度条），普通行直接落盘，并可接管标准输入输出。"""

    def __init__(
        self,
        path: Union[str, Path],
        throttle: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            path: 日志文件路径。
            throttle: 进度条行就地刷新的最小间隔（秒），0 为实时；定稿
                时不受限。
            clock: 时间源，测试可注入假时钟。
        """
        self._path = Path(path)
        self._throttle = max(0.0, throttle)
        self._clock = clock
        self._lock = threading.Lock()
        self._buf = ""  # 尚未遇到 \r / \n 的不完整输出
        self._bar_content: Optional[str] = None  # 当前进度条内容（不含 \r；None 表示无）
        self._bar_in_file = False  # 文件最后一行是否为进度条行
        self._bar_start = 0  # 进度条行起始字节偏移
        self._last_write = 0.0
        self._saved: Tuple = ()
        self._stdout_tee = None
        self._hooked = False
        self._file = self._open()

    @property
    def path(self) -> Path:
        """日志文件路径。"""
        return self._path

    # ------------------------------------------------------------------ 写入
    def write(self, text: str) -> None:
        """接收一段控制台输出（原样，含 \\r / \\n），按协议写入文件。

        ANSI 颜色/控制码在落盘前剥离：终端保持主题色，日志文件纯文本。
        多实例场景下本镜像可能已随 close 关闭（stdio 链仍串着旧分流器），
        此时静默丢弃，不再向已关闭文件写入。
        """
        if self._file.closed:
            return
        text = _ANSI_RE.sub("", text)
        if not text:
            return
        with self._lock:
            self._buf += text
            while True:
                i_r, i_n = self._buf.find("\r"), self._buf.find("\n")
                if i_r == -1 and i_n == -1:
                    break
                if i_r != -1 and (i_n == -1 or i_r < i_n):
                    seg, self._buf = self._buf[:i_r], self._buf[i_r + 1:]
                    self._on_bar(seg)
                else:
                    seg, self._buf = self._buf[:i_n], self._buf[i_n + 1:]
                    self._on_line(seg)

    def flush(self) -> None:
        """落盘文件缓冲（不完整行保持缓冲，等终止符到达后成行）。"""
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    def close(self) -> None:
        """收尾：进度条定稿为最新内容，剩余缓冲按普通行落盘，关闭文件。"""
        with self._lock:
            if self._file.closed:
                return
            if self._buf:
                seg, self._buf = self._buf, ""
                self._on_line(seg)
            elif self._bar_content is not None or self._bar_in_file:
                self._finalize_bar()
            self._file.close()

    # ------------------------------------------------------------------ 内部
    def _open(self):
        p = self._path
        if p.exists():
            f = open(p, "r+b")
            f.seek(0, 2)
            size = f.tell()
            if size:
                f.seek(max(0, size - _TAIL_LIMIT))
                tail = f.read()
                f.seek(0, 2)
                last = tail[tail.rfind(b"\n") + 1:]
                if last.endswith(b"\r"):  # 未完的进度条行：后续刷新继续覆盖
                    self._bar_in_file = True
                    self._bar_start = size - len(last)
                    self._bar_content = None
            return f
        p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, "w+b")

    def _on_bar(self, seg: str) -> None:
        if not seg.strip():
            # 行清除序列（\r 后用空格覆盖残留）的空产物，非进度条内容：忽略
            return
        seg = seg.translate(_BAR_TALL_MAP)  # 文件侧进度条加高
        self._bar_content = seg
        now = self._clock()
        if self._bar_in_file and now - self._last_write < self._throttle:
            return  # 节流：文件暂不刷新，定稿时会写最新内容
        self._write_bar(seg)

    def _write_bar(self, seg: str) -> None:
        data = seg.encode("utf-8") + b"\r"
        if self._bar_in_file:
            self._file.seek(self._bar_start)
            self._file.truncate()
        else:
            self._bar_start = self._file.tell()
            self._bar_in_file = True
        self._file.write(data)
        self._file.flush()
        self._last_write = self._clock()

    def _on_line(self, seg: str) -> None:
        finalized = self._bar_content is not None or self._bar_in_file
        if finalized:
            self._finalize_bar()
        # 空片段仅在"没有进度条可定稿"时才落盘（print("\n") 的空行）；
        # 刚定稿完进度条的空片段是 bar 行自己的换行，不再补一行空行
        if seg or not finalized:
            self._file.write(seg.encode("utf-8") + b"\n")
            self._file.flush()

    def _finalize_bar(self) -> None:
        """把进度条行以最新内容定稿为完整行（绕过节流）。"""
        content = self._bar_content
        if self._bar_in_file:
            self._file.seek(self._bar_start)
            self._file.truncate()
            if content:
                self._file.write(content.encode("utf-8") + b"\n")
            else:
                self._file.write(b"\n")  # 恢复自旧会话的未完行，无最新内容：仅补换行
        self._file.flush()
        self._bar_in_file = False
        self._bar_content = None

    # ------------------------------------------------------------------ stdio
    def hook_stdio(self) -> None:
        """接管 sys.stdout / sys.stderr：控制台与日志文件同步写入。"""
        if self._hooked:
            return
        self._saved = (sys.stdout, sys.stderr)
        self._stdout_tee = _Tee(sys.stdout, self)
        sys.stdout = self._stdout_tee
        sys.stderr = _Tee(sys.stderr, self)
        self._hooked = True

    def unhook_stdio(self) -> None:
        """还原接管前的标准输入输出。

        多实例嵌套接管时（新实例的分流器包在本实例之外），stdout 已不归
        本实例所有，此时只解除自身标记、不动 sys.stdout，交给最外层实例
        统一还原。
        """
        if not self._hooked:
            return
        if sys.stdout is self._stdout_tee:
            sys.stdout, sys.stderr = self._saved
        self._saved = ()
        self._stdout_tee = None
        self._hooked = False
