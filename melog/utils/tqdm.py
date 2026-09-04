"""tqdm 兼容进度条：用法与 tqdm.tqdm 一致，样式重设计，并同步到日志镜像。

用法（与 tqdm.tqdm 相同）::

    from melog import tqdm

    for x in tqdm(items, desc="处理"):
        ...

    with tqdm(total=100, desc="train") as bar:
        ...
        bar.update(10)
        bar.set_postfix(loss=0.21)

样式（新设计，终端下与 Melog 主题色一致：紫→粉逐格渐变进度条 + 青色计数
+ 白色读数 + 黄色速率 + 灰色次要信息）::

    [110/200] loss=0.2153 ━━━━━━━━━━━━────────────  55.0% [0:03<0:03 33.3it/s]

    train loss=0.2153 ━━━━━━━━━━━━────────────  55.0% [110/200] [0:03<0:03 33.3it/s]

各段定宽右对齐：数值位数变化不改变行宽，尾部（条形图/百分比/耗时）
逐帧位置稳定不抖动。

每帧以 ``\\r`` 结尾：终端原地重绘；若标准输入输出已被 Mirror 接管，
日志文件里也保持同一行进度条（就地刷新、节流落盘，见 melog.storage.mirror）。
颜色只在输出流为终端（TTY）时启用，重定向 / 管道 / 日志文件始终纯文本。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__all__ = ["tqdm"]

BAR_WIDTH = 24  # 进度条字符宽度
_FILL, _EMPTY = "━", "─"  # 横线字符：垂直居中、细条、相邻格无缝拼接

# Melog 主题色（与 Web 面板 accent 一致的紫色）+ 分段点缀色
_ACCENT = "\x1b[38;2;168;85;247m"        # 紫色：百分比
_GRAD_FROM = (168, 85, 247)              # 渐变起点：主题紫
_GRAD_TO = (236, 72, 153)                # 渐变末端：粉
_ACCENT_BOLD = "\x1b[1;38;2;168;85;247m"
_BOLD_WHITE = "\x1b[1;38;2;230;233;238m"
_WHITE = "\x1b[38;2;230;233;238m"
_CYAN = "\x1b[38;2;97;214;214m"          # [n/total] 计数
_YELLOW = "\x1b[38;2;250;204;21m"        # 速率
_DIM = "\x1b[38;2;128;134;145m"
_BAR_EMPTY = "\x1b[38;2;70;74;80m"
_RESET = "\x1b[0m"


def _is_tty(stream) -> bool:
    """输出流是否为终端（颜色 / 光标控制只在真实终端启用）。"""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _fmt_clock(seconds: float) -> str:
    """秒 -> m:ss / h:mm:ss。"""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_value(value: Any) -> str:
    """postfix 指标值：常规范围浮点固定 4 位小数（宽度恒定、小数点对齐），
    过大 / 过小退化为 4 位有效数字，其余原样。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if 1e-2 <= abs(value) < 1e4:
            return f"{value:.4f}"
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
        on_close: Optional[Callable[[], None]] = None,
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
            on_close: 进度条关闭（close / 自然迭代结束）后的回调，
                恰好执行一次；调用方（如 Melog）据此解除登记。
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
        self.on_close = on_close
        self.n = 0
        self.postfix: dict = {}
        self.covered = False  # 被上层 bar 覆盖时暂停渲染（计数与 postfix 照常更新）
        self._value_w: Dict[str, int] = {}  # 各指标值的历史最宽字符数（定宽右对齐用）

        self._t0 = time.monotonic()
        self._last_render = 0.0
        self._rendered_len = 0
        self._last_plain: Optional[str] = None  # 上次渲染的纯文本（内容去重用）
        self._closed = False
        self._cursor_hidden = False
        if not disable:
            # 终端块状光标会压在行首字符上（\r 后光标停在 0 列），先隐藏，
            # close 时恢复（与 tqdm 行为一致）
            stream = self._stream()
            if _is_tty(stream):
                stream.write("\x1b[?25l")
                stream.flush()
                self._cursor_hidden = True
            self.render()

    # ------------------------------------------------------------ tqdm 兼容
    def update(self, n: int = 1) -> None:
        """推进 n 步并按 mininterval 重绘。"""
        if self.disable or self._closed or not n:
            return
        self.n += n
        if self.covered:
            return
        now = time.monotonic()
        if now - self._last_render >= self.mininterval:
            self.render()

    def set_description(self, desc: Optional[str] = None) -> None:
        """更新前缀描述并立即重绘。"""
        self.desc = desc
        if not self.disable and not self._closed:
            self.render()

    def set_postfix(self, ordered_dict: Optional[dict] = None, **kwargs: Any) -> None:
        """更新指标（如 loss=0.21），键相同时覆盖。"""
        if ordered_dict:
            self.postfix.update(ordered_dict)
        if kwargs:
            self.postfix.update(kwargs)
        if not self.disable and not self._closed and not self.covered:
            self.render()

    @classmethod
    def write(cls, s: Any, file: Any = None) -> None:
        """打印一行普通输出（不破坏进度条；被接管时同步进日志文件）。"""
        fp = file if file is not None else sys.stdout
        fp.write(f"{s}\n")
        fp.flush()

    def refresh(self, force: bool = False) -> None:
        """立即重绘（绕过 mininterval）。

        force=True 时内容未变也重绘：消息擦除、子条覆盖后恢复等场景
        需要修复性重绘；缺省只在实际内容变化时输出。
        """
        if not self.disable and not self._closed and not self.covered:
            self.render(force=force)

    def clear_line(self) -> None:
        """清除本条进度条当前占用的终端行（消息打印 / 嵌套覆盖前调用）。

        渲染总以 ``\\r`` 结尾（光标停在行首），``\\x1b[2K`` 擦除整行即可；
        仅 TTY 生效（重定向时无操作，文件侧由 Mirror 协议处理）。
        """
        if self.disable or self._closed:
            return
        stream = self._stream()
        if _is_tty(stream):
            stream.write("\x1b[2K")
            stream.flush()

    def close(self) -> None:
        """以最新状态定稿：重绘一行后换行（leave=False 则清除该行）。

        定稿为强制重绘：屏幕行可能刚被子条覆盖/清除，文件侧也保证
        最终状态成行（内容未变时由 Mirror 相邻去重，不重复落盘）。
        """
        if self._closed:
            return
        self._closed = True
        if not self.disable:
            stream = self._stream()
            self.render(force=True)
            if self.leave:
                stream.write("\n")
            else:
                stream.write("\r" + " " * self._rendered_len + "\r")
            if self._cursor_hidden:
                self._cursor_hidden = False
                stream.write("\x1b[?25h")
            stream.flush()
        if self.on_close is not None:
            self.on_close()

    def __enter__(self) -> "tqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self):
        try:
            if self.disable:
                yield from self.iterable
            else:
                for obj in self.iterable:
                    yield obj
                    self.update(1)
        finally:
            # 自然耗尽、提前 break、异常传播、生成器被 GC（GeneratorExit）
            # 均定稿并出栈（on_close 回调解除栈登记）；自动记录不受影响，
            # 仅自然耗尽触发（见 EpochEndIterable）
            self.close()

    def __del__(self):
        # 兜底：绑定名字后未手动 close 的 bar（如提前 break 后变量仍存活），
        # 引用释放时自动定稿；解释器退出阶段的异常静默
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------ 渲染
    def _stream(self):
        return self.file if self.file is not None else sys.stdout

    def _use_color(self, stream) -> bool:
        """主题色开关：显式指定优先，否则仅终端 TTY 启用（重定向为纯文本）。"""
        if self.colour is not None:
            return self.colour
        return _is_tty(stream)

    def render(self, force: bool = False) -> None:
        """把当前状态渲染为一行并以 \\r 结尾输出（终端原地重绘）。

        终端下带 Melog 主题色；pad 计算按可见宽度（ANSI 码零显示宽度）。
        内容与上次渲染相同且非强制时跳过：屏幕已是最新，日志文件侧也不
        重复落盘；force 用于屏幕行被消息擦除、子条覆盖后的修复性重绘。
        """
        stream = self._stream()
        line, plain = self._format(self._use_color(stream))
        if not force and plain == self._last_plain:
            return
        # 行变短时用空格覆盖残留（文件侧由 Mirror 截断重写并剥离颜色码）
        pad = " " * max(0, self._rendered_len - len(plain))
        stream.write(line + pad + "\r")
        stream.flush()
        self._rendered_len = len(plain) + len(pad)
        self._last_render = time.monotonic()
        self._last_plain = plain

    def _format(self, use_color: bool) -> Tuple[str, str]:
        """渲染当前状态，返回 (终端行[含颜色码], 纯文本行)。

        布局：[n/total] 最前，指标随后，条形图/百分比/[耗时<剩余 速率]
        殿后；desc 提供时置于行首。各段定宽：n 右对齐到 total 宽度、
        百分比固定宽、指标值右对齐到各自历史最宽宽度，因此数值位数
        变化不引起行宽摆动，尾部逐帧位置稳定（仅出现更宽数值时整体
        右移一次）。进度条的填充/剩余两段直接拼接为一段（中间无空格），
        仅颜色不同。
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
            parts.append(seg(str(self.desc), _BOLD_WHITE))

        rate = self.n / elapsed if elapsed > 0 and self.n > 0 else 0.0
        postfix = ("", "")
        if self.postfix:
            plain_cells = []
            color_cells = []
            for k, v in self.postfix.items():
                val = _fmt_value(v)
                w = self._value_w.get(k, 0)
                if len(val) > w:
                    w = len(val)
                    self._value_w[k] = w
                val = val.rjust(w)
                plain_cells.append(f"{k}={val}")
                # 指标名灰、数值白，视觉上把名字与读数分开
                color_cells.append(f"{_DIM}{k}={_RESET}{_WHITE}{val}{_RESET}")
            if not use_color:
                color_cells = plain_cells
            postfix = (" ".join(color_cells), " ".join(plain_cells))

        time_part = _fmt_clock(elapsed)
        if self.total and rate > 0:
            time_part += f"<{_fmt_clock((self.total - self.n) / rate)}"
        if rate >= 1:
            rate_part = f"{rate:.1f}{self.unit}/s"
        elif rate > 0:
            rate_part = f"{1 / rate:.1f}s/{self.unit}"
        else:
            rate_part = ""

        # 尾段 [耗时<剩余 + 速率]：时间灰、速率黄，拆两段拼回同一对中括号
        if rate_part:
            tail = [seg(f"[{time_part}", _DIM), seg(f"{rate_part}]", _YELLOW)]
        else:
            tail = [seg(f"[{time_part}]", _DIM)]

        if self.total:
            frac = min(max(self.n / self.total, 0.0), 1.0)
            filled = int(frac * BAR_WIDTH)
            bar = self._render_bar(filled, use_color)
            n_str = str(self.n).rjust(len(str(self.total)))
            parts.append(postfix)
            parts.append((bar, _FILL * filled + _EMPTY * (BAR_WIDTH - filled)))
            parts.append(seg(f"{100 * frac:5.1f}%", _ACCENT_BOLD))
            parts.append(seg(f"[{n_str}/{self.total}]", _CYAN))
        else:
            parts.append(postfix)
            parts.append(seg(f"[{self.n}{self.unit}]", _CYAN))
        parts.extend(tail)

        drop_empty = lambda p: bool(p[0])  # noqa: E731  # 空片段不占位，避免多余空格
        kept = [p for p in parts if drop_empty(p)]
        return " ".join(p[0] for p in kept), " ".join(p[1] for p in kept)

    @staticmethod
    def _render_bar(filled: int, use_color: bool) -> str:
        """进度条本体：填充段逐格做紫→粉线性插值（无色块台阶），剩余轨道深灰。"""
        if not use_color:
            return _FILL * filled + _EMPTY * (BAR_WIDTH - filled)
        out = []
        for i in range(filled):
            t = i / (filled - 1) if filled > 1 else 0.0
            r, g, b = (round(s + (e - s) * t) for s, e in zip(_GRAD_FROM, _GRAD_TO))
            out.append(f"\x1b[38;2;{r};{g};{b}m{_FILL}")
        if filled:
            out.append(_RESET)
        if filled < BAR_WIDTH:
            out.append(f"{_BAR_EMPTY}{_EMPTY * (BAR_WIDTH - filled)}{_RESET}")
        return "".join(out)
