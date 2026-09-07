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

终端下自适应列宽渲染（重定向 / 管道 / 日志文件按固定宽度）：内容不超行，
绝不自动换行——
- 进度条吃掉固定段（desc / 指标 / 百分比 / 计数 / 耗时）之外的**全部剩余
  列**，恰好占满整行（宽终端下条形随之加长）；
- 剩余列不足时进度条收缩到最小宽度（指标区优先保全）；
- 条已最小仍放不下指标时，指标区截断显示、以省略号收尾（终端下避免换行
  破坏原地重绘；宽字符按 2 列计，截断不超界）。

每帧以 ``\\r`` 结尾：终端原地重绘；若标准输入输出已被 Mirror 接管，
日志文件里也保持同一行进度条（就地刷新、节流落盘，见 melog.storage.mirror）。
颜色只在输出流为终端（TTY）时启用，重定向 / 管道 / 日志文件始终纯文本。
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__all__ = ["tqdm"]

BAR_WIDTH = 24  # 进度条基准字符宽度（终端下为弹性宽度，不小于 _BAR_MIN）
_BAR_MIN = 10  # 进度条最小字符宽度（终端列不足时收缩到此，指标区再让位）
_ELLIPSIS = "…"  # 指标区截断收尾的省略号
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


def _term_width(stream) -> Optional[int]:
    """终端可用列数（仅 TTY；重定向 / 管道返回 None，按固定宽度渲染）。

    进度条行按列数自适应（占满整行、收缩、截断），管道与日志文件没有
    列宽概念，保持固定宽度布局。
    """
    if not _is_tty(stream):
        return None
    try:
        cols = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, ValueError, OSError):
        try:  # 自定义流无 fileno：退回 COLUMNS 环境变量 / 默认列数
            cols = shutil.get_terminal_size().columns
        except (AttributeError, ValueError, OSError):
            return None
    return int(cols) or None


def _char_w(ch: str) -> int:
    """单字符显示列数：East Asian Wide / Fullwidth（CJK、全角标点等）按 2。"""
    o = ord(ch)
    if 0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 \
            or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F \
            or 0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6 \
            or 0x20000 <= o <= 0x3FFFD:
        return 2
    return 1


def _display_width(s: str) -> int:
    """字符串显示列数（ANSI 零宽由调用方保证不在 s 中；宽字符按 2 计）。"""
    return sum(_char_w(ch) for ch in s)


def _fit_text(s: str, width: int) -> str:
    """按显示列数截断 s（宽字符不超界）；恰好放下则原样返回。

    保留左侧内容（数值右对齐时左侧空白先让位）；width <= 0 返回空串。
    """
    if _display_width(s) <= width:
        return s
    out = []
    w = 0
    for ch in s:
        cw = _char_w(ch)
        if w + cw > width:
            break
        out.append(ch)
        w += cw
    return "".join(out)


def _fit_cells(cells: List[Tuple[str, str, int, str, str]],
               budget: int) -> List[Tuple[str, str, int, str, str]]:
    """把指标格截进 budget 显示列（终端下进度条已最小时让指标区收尾）。

    装不下的整格丢弃；剩余空间够 "k=…" 时最后一格部分保留、以省略号
    收尾（省略号灰显，绝不超界）。cells 为 (纯文本, 颜色文本, 显示宽,
    键名, 值文本)，格间空格计入预算。
    """
    if budget <= 0:
        return []
    out: List[Tuple[str, str, int, str, str]] = []
    used = 0
    for plain, color, w, k, val in cells:
        gap = 1 if out else 0
        if used + gap + w <= budget:
            out.append((plain, color, w))
            used += gap + w
            continue
        rest = budget - used - gap  # 本格可用列
        key_w = _display_width(k)
        if rest >= key_w + 2:  # "k=" 与省略号至少可容（值可空 → "k=…"）
            val_cut = _fit_text(val, rest - key_w - 2).lstrip()  # 右对齐空白先让位
            plain_cut = f"{k}={val_cut}{_ELLIPSIS}"
            color_cut = (f"{_DIM}{k}={_RESET}{_WHITE}{val_cut}{_RESET}"
                         f"{_DIM}{_ELLIPSIS}{_RESET}")
            out.append((plain_cut, color_cut, _display_width(plain_cut)))
        break  # 再后的整格一律丢弃
    return out


def _fmt_clock(seconds: float) -> str:
    """秒 -> m:ss / h:mm:ss。"""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_value(value: Any) -> str:
    """postfix 指标值：浮点用科学计数法保留 3 位小数（宽度恒定、小数点对齐），
    其余原样。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3e}"
    return str(value)


class tqdm:
    """进度条：迭代器 / 手动 update 两种用法，接口对齐 tqdm.tqdm。

    终端（TTY）下按终端列数自适应渲染：进度条为弹性段，吃掉固定段
    之外的全部剩余列恰好占满整行；剩余不足时收缩到最小宽度（指标区
    优先保全），仍放不下的指标以省略号收尾——内容不超行、绝不换行，
    宽字符按 2 列计。重定向 / 管道 / 日志文件无列宽概念，按固定宽度
    （BAR_WIDTH）渲染、指标不截断。

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
            # 先换行再渲染首帧：bar 从新行开始，不与既有输出同行
            stream.write("\n")
            stream.flush()
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

        终端下带 Melog 主题色，并按终端列数自适应（占满整行 / 收缩 /
        截断，见 _format）；pad 计算按可见宽度（ANSI 码零显示宽度）。
        内容与上次渲染相同且非强制时跳过：屏幕已是最新，日志文件侧也不
        重复落盘；force 用于屏幕行被消息擦除、子条覆盖后的修复性重绘。
        """
        stream = self._stream()
        line, plain = self._format(self._use_color(stream), _term_width(stream))
        plain_w = _display_width(plain)
        if not force and plain == self._last_plain:
            return
        # 行变短时用空格覆盖残留（文件侧由 Mirror 截断重写并剥离颜色码）
        pad = " " * max(0, self._rendered_len - plain_w)
        stream.write(line + pad + "\r")
        stream.flush()
        self._rendered_len = plain_w + len(pad)
        self._last_render = time.monotonic()
        self._last_plain = plain

    def _format(self, use_color: bool, width: Optional[int] = None) -> Tuple[str, str]:
        """渲染当前状态，返回 (终端行[含颜色码], 纯文本行)。

        布局：[n/total] 最前，指标随后，条形图/百分比/[耗时<剩余 速率]
        殿后；desc 提供时置于行首。各段定宽：n 右对齐到 total 宽度、
        百分比固定宽、指标值右对齐到各自历史最宽宽度，因此数值位数
        变化不引起行宽摆动，尾部逐帧位置稳定（仅出现更宽数值时整体
        右移一次）。进度条的填充/剩余两段直接拼接为一段（中间无空格），
        仅颜色不同。

        width 为终端列数时按列自适应（内容不超行、不换行）：
        - 进度条为弹性段，吃掉固定段（desc / 指标 / 百分比 / 计数 / 耗时）
          之外的**全部剩余列**，恰好占满整行（宽终端下条形随之加长）；
        - 剩余列不足时进度条收缩到 _BAR_MIN（指标区优先保全）；
        - 条已最小仍放不下指标时，指标区截断、以省略号收尾。
        width 为 None（重定向 / 管道 / 日志文件侧没有列宽概念）时按固定
        宽度渲染，进度条恒为 BAR_WIDTH、指标不截断（与旧行为一致）。
        """
        def seg(text: str, code: str = "") -> Tuple[str, str]:
            if not text:
                return ("", "")
            if use_color and code:
                return (f"{code}{text}{_RESET}", text)
            return (text, text)

        elapsed = time.monotonic() - self._t0
        desc_seg = seg(str(self.desc), _BOLD_WHITE) if self.desc else ("", "")

        rate = self.n / elapsed if elapsed > 0 and self.n > 0 else 0.0
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
            n_str = str(self.n).rjust(len(str(self.total)))
            pct_seg = seg(f"{100 * frac:5.1f}%", _ACCENT_BOLD)
            count_seg = seg(f"[{n_str}/{self.total}]", _CYAN)
        else:
            frac = None
            pct_seg = ("", "")
            count_seg = seg(f"[{self.n}{self.unit}]", _CYAN)

        # 指标格 (纯文本, 颜色文本, 显示宽, 键名, 值文本)：值定宽右对齐，
        # 供弹性布局整格丢弃 / 部分截断（颜色版本由截断后重建）
        cells: List[Tuple[str, str, int, str, str]] = []
        postfix_w = 0
        for k, v in self.postfix.items():
            val = _fmt_value(v)
            vw = _display_width(val)
            w = self._value_w.get(k, 0)
            if vw > w:
                w = vw
                self._value_w[k] = w
            val = " " * (w - vw) + val  # 显示宽右对齐（宽字符按 2 计）
            plain = f"{k}={val}"
            # 指标名灰、数值白，视觉上把名字与读数分开
            color = f"{_DIM}{k}={_RESET}{_WHITE}{val}{_RESET}"
            cells.append((plain, color, _display_width(plain), k, val))
        postfix_w = sum(c[2] for c in cells) + max(0, len(cells) - 1)

        # ---- 弹性布局（仅终端有列数时）：占满整行 → 收缩进度条 → 截断指标
        bar_w: Optional[int] = BAR_WIDTH if self.total else None
        if width is not None:
            fixed_segs = [s for s in (desc_seg, pct_seg, count_seg, *tail) if s[0]]
            fixed_w = sum(_display_width(p[1]) for p in fixed_segs)

            def _budget(cells_present: bool) -> int:
                # 列预算：扣掉固定段与各段间空格后的弹性空间
                n = len(fixed_segs) + (1 if cells_present else 0) + (1 if self.total else 0)
                return width - fixed_w - max(0, n - 1)

            if cells:
                budget = _budget(True)
                if self.total:
                    bar_w = budget - postfix_w  # 指标 Desired 之外全给进度条（占满整行）
                    if bar_w < _BAR_MIN:
                        # 剩余不足：进度条先收缩到最小宽度，指标区再让位
                        fit = _fit_cells(cells, budget - _BAR_MIN)
                        if fit:
                            dropped = len(fit) < len(cells)  # 有整格装不下
                            used = sum(c[2] for c in fit) + max(0, len(fit) - 1)
                            # 末格已被部分截断（自带省略号）时不再补标记
                            if dropped and budget - used >= 2 \
                                    and not fit[-1][0].endswith(_ELLIPSIS):
                                # 补一个省略号标记（需求优先级高于条宽下限，下限至多让 2 列）
                                marker = (_ELLIPSIS, f"{_DIM}{_ELLIPSIS}{_RESET}", 1)
                                fit = fit + [marker]
                                used += 2  # 格间空格 + 省略号
                            cells = fit
                            bar_w = budget - used  # 指标截断后的剩余列归还进度条（恰好占满）
                        else:  # 指标一格都放不下：整段让位，进度条吃满剩余列
                            cells = []
                            bar_w = max(1, _budget(False))
                else:  # 无进度条段（未绑 total）：指标独占剩余列，超宽省略号截断
                    cells = _fit_cells(cells, budget)
            elif self.total:
                bar_w = max(1, _budget(False))

        postfix: Tuple[str, str] = ("", "")
        if cells:
            plain_cells = [c[0] for c in cells]
            color_cells = [c[1] for c in cells] if use_color else plain_cells
            postfix = (" ".join(color_cells), " ".join(plain_cells))

        parts: List[Tuple[str, str]] = [desc_seg]
        if self.total:
            filled = int(frac * bar_w)
            parts.append(postfix)
            parts.append((self._render_bar(filled, bar_w, use_color),
                          _FILL * filled + _EMPTY * (bar_w - filled)))
            parts.append(pct_seg)
            parts.append(count_seg)
        else:
            parts.append(postfix)
            parts.append(count_seg)
        parts.extend(tail)

        drop_empty = lambda p: bool(p[0])  # noqa: E731  # 空片段不占位，避免多余空格
        kept = [p for p in parts if drop_empty(p)]
        return " ".join(p[0] for p in kept), " ".join(p[1] for p in kept)

    @staticmethod
    def _render_bar(filled: int, width: int, use_color: bool) -> str:
        """进度条本体：填充段逐格做紫→粉线性插值（无色块台阶），剩余轨道深灰。"""
        if not use_color:
            return _FILL * filled + _EMPTY * (width - filled)
        out = []
        for i in range(filled):
            t = i / (filled - 1) if filled > 1 else 0.0
            r, g, b = (round(s + (e - s) * t) for s, e in zip(_GRAD_FROM, _GRAD_TO))
            out.append(f"\x1b[38;2;{r};{g};{b}m{_FILL}")
        if filled:
            out.append(_RESET)
        if filled < width:
            out.append(f"{_BAR_EMPTY}{_EMPTY * (width - filled)}{_RESET}")
        return "".join(out)
