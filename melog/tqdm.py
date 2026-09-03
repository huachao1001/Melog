"""tqdm 兼容进度条：用法与 tqdm.tqdm 一致，样式重设计，并同步到日志镜像。

用法（与 tqdm.tqdm 相同）::

    from melog import tqdm

    for x in tqdm(items, desc="处理"):
        ...

    with tqdm(total=100, desc="train") as bar:
        ...
        bar.update(10)
        bar.set_postfix(loss=0.21)

样式（新设计，终端下与 Melog 主题色一致：紫色进度条 + 灰色次要信息）::

    train ████████████░░░░░░░░░░  55.0% 110/200 0:03<0:03 33.3it/s loss=0.215

每帧以 ``\\r`` 结尾：终端原地重绘；若标准输入输出已被 Mirror 接管，
日志文件里也保持同一行进度条（就地刷新、节流落盘，见 melog.mirror）。
颜色只在输出流为终端（TTY）时启用，重定向 / 管道 / 日志文件始终纯文本。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable, List, Optional, Tuple

__all__ = ["tqdm"]

BAR_WIDTH = 24  # 进度条字符宽度
_FILL, _EMPTY = "█", "░"

# Melog 主题色（与 Web 面板 accent 一致的紫色）+ 次要信息灰
_ACCENT = "\x1b[38;2;168;85;247m"
_ACCENT_BOLD = "\x1b[1;38;2;168;85;247m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[38;2;128;134;145m"
_RESET = "\x1b[0m"


def _fmt_clock(seconds: float) -> str:
    """秒 -> m:ss / h:mm:ss。"""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_value(value: Any) -> str:
    """postfix 指标值：浮点 4 位有效数字，其余原样。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class tqdm:
    """进度条：迭代器 / 手动 update 两种用法，接口对齐 tqdm.tqdm。

    Args:
        iterable: 可迭代对象（提供时支持 for 直接迭代）。
        total: 总步数；缺省时尝试取 len(iterable)。
        desc: 进度条前缀描述。
        leave: 结束后是否保留进度条行（False 为清除该行）。
        file: 输出流，缺省每次渲染时取 sys.stdout（被 Mirror 接管后
            自动同步进日志文件）。
        disable: True 时完全静默（非主进程等场景）。
        mininterval: 终端重绘的最小间隔（秒）；日志文件侧的节流由
            Mirror 负责（默认 2 秒）。
        unit: 计数单位名，默认 "it"。
    """

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[float] = None,
        desc: Optional[str] = None,
        leave: bool = True,
        file: Any = None,
        disable: bool = False,
        mininterval: float = 0.1,
        unit: str = "it",
        colour: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            iterable: 可迭代对象（提供时支持 for 直接迭代）。
            total: 总步数；缺省时尝试取 len(iterable)。
            desc: 进度条前缀描述。
            leave: 结束后是否保留进度条行（False 为清除该行）。
            file: 输出流，缺省每次渲染时取 sys.stdout（被 Mirror 接管后
                自动同步进日志文件）。
            disable: True 时完全静默（非主进程等场景）。
            mininterval: 终端重绘的最小间隔（秒）；日志文件侧的节流由
                Mirror 负责（默认 2 秒）。
            unit: 计数单位名，默认 "it"。
            colour: 是否使用主题色；None 时自动检测（终端 TTY 启用）。
        """
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                pass
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.leave = leave
        self.file = file
        self.disable = disable
        self.mininterval = mininterval
        self.unit = unit
        self.colour = colour
        self.n = 0
        self.postfix: dict = {}

        self._t0 = time.monotonic()
        self._last_render = 0.0
        self._rendered_len = 0
        self._closed = False
        if not disable:
            self.render()

    # ------------------------------------------------------------ tqdm 兼容
    def update(self, n: int = 1) -> None:
        """推进 n 步并按 mininterval 重绘。"""
        if self.disable or self._closed or not n:
            return
        self.n += n
        now = time.monotonic()
        if now - self._last_render >= self.mininterval:
            self.render()

    advance = update  # 兼容 Melog 旧版进度条 API

    def set_description(self, desc: Optional[str] = None) -> None:
        """更新前缀描述并立即重绘。"""
        self.desc = desc
        if not self.disable and not self._closed:
            self.render()

    def set_postfix(self, ordered_dict: Optional[dict] = None, **kwargs: Any) -> None:
        """更新行尾指标（如 loss=0.21），键相同时覆盖。"""
        if ordered_dict:
            self.postfix.update(ordered_dict)
        if kwargs:
            self.postfix.update(kwargs)
        if not self.disable and not self._closed:
            self.render()

    @classmethod
    def write(cls, s: Any, file: Any = None) -> None:
        """打印一行普通输出（不破坏进度条；被接管时同步进日志文件）。"""
        fp = file if file is not None else sys.stdout
        fp.write(f"{s}\n")
        fp.flush()

    def refresh(self) -> None:
        """立即重绘（绕过 mininterval）。"""
        if not self.disable and not self._closed:
            self.render()

    def close(self) -> None:
        """以最新状态定稿：重绘一行后换行（leave=False 则清除该行）。"""
        if self._closed:
            return
        self._closed = True
        if self.disable:
            return
        stream = self._stream()
        self.render()
        if self.leave:
            stream.write("\n")
        else:
            stream.write("\r" + " " * self._rendered_len + "\r")
        stream.flush()

    def __enter__(self) -> "tqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self):
        if self.disable:
            yield from self.iterable
            return
        for obj in self.iterable:
            yield obj
            self.update(1)
        self.close()  # 自然迭代完成即定稿（提前 break 时不关闭，与 tqdm 一致）

    # ------------------------------------------------------------ 渲染
    def _stream(self):
        return self.file if self.file is not None else sys.stdout

    def _use_color(self, stream) -> bool:
        """主题色开关：显式指定优先，否则仅终端 TTY 启用（重定向为纯文本）。"""
        if self.colour is not None:
            return self.colour
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def render(self) -> None:
        """把当前状态渲染为一行并以 \\r 结尾输出（终端原地重绘）。

        终端下带 Melog 主题色；pad 计算按可见宽度（ANSI 码零显示宽度）。
        """
        stream = self._stream()
        line, plain = self._format(self._use_color(stream))
        # 行变短时用空格覆盖残留（文件侧由 Mirror 截断重写并剥离颜色码）
        pad = " " * max(0, self._rendered_len - len(plain))
        stream.write(line + pad + "\r")
        stream.flush()
        self._rendered_len = len(plain) + len(pad)
        self._last_render = time.monotonic()

    def _format(self, use_color: bool) -> Tuple[str, str]:
        """渲染当前状态，返回 (终端行[含颜色码], 纯文本行)。

        进度条的填充/剩余两段直接拼接为一段（中间无空格），仅颜色不同。
        """
        def seg(text: str, code: str = "") -> Tuple[str, str]:
            if not text:
                return ("", "")
            if use_color and code:
                return (f"{code}{text}{_RESET}", text)
            return (text, text)

        elapsed = time.monotonic() - self._t0
        parts: List[Tuple[str, str]] = []
        if self.desc:
            parts.append(seg(str(self.desc), _BOLD))

        rate = self.n / elapsed if elapsed > 0 and self.n > 0 else 0.0
        if self.total:
            frac = min(max(self.n / self.total, 0.0), 1.0)
            filled = int(frac * BAR_WIDTH)
            bar = seg(_FILL * filled, _ACCENT)[0] + seg(_EMPTY * (BAR_WIDTH - filled), _DIM)[0]
            parts.append((bar, _FILL * filled + _EMPTY * (BAR_WIDTH - filled)))
            parts.append(seg(f"{100 * frac:.1f}%", _ACCENT_BOLD))
            parts.append(seg(f"{self.n}/{self.total}"))
            if rate > 0:
                parts.append(seg(f"{_fmt_clock(elapsed)}<{_fmt_clock((self.total - self.n) / rate)}", _DIM))
            else:
                parts.append(seg(_fmt_clock(elapsed), _DIM))
        else:
            parts.append(seg(f"{self.n}{self.unit}"))
            parts.append(seg(_fmt_clock(elapsed), _DIM))

        if rate >= 1:
            parts.append(seg(f"{rate:.1f}{self.unit}/s", _DIM))
        elif rate > 0:
            parts.append(seg(f"{1 / rate:.1f}s/{self.unit}", _DIM))

        if self.postfix:
            parts.append(seg(" ".join(f"{k}={_fmt_value(v)}" for k, v in self.postfix.items())))
        drop_empty = lambda p: bool(p[0])  # noqa: E731  # 空片段不占位，避免多余空格
        kept = [p for p in parts if drop_empty(p)]
        return " ".join(p[0] for p in kept), " ".join(p[1] for p in kept)
