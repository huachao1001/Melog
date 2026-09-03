"""Melog：轻量级训练监控库。

核心能力：
- 训练指标记录与 JSONL 持久化
- 多 GPU 指标合并（torch.distributed all_reduce）
- 控制台实时进度条（tqdm 兼容）与控制台日志镜像
- Web 可视化（FastAPI + WebSocket + ECharts）
"""

from __future__ import annotations

import atexit
import builtins
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from .distributed import get_rank, reduce_metrics
from .media import sanitize_name, save_audio, save_image
from .metrics import MetricGroup
from .mirror import Mirror
from .tqdm import _is_tty, tqdm
from .web.media_store import MediaStore
from .web.server import WebServer
from .web.store import MetricStore

__all__ = ["Melog", "tqdm"]

# 控制台消息色（SGR 标准色，随终端主题）；log 走终端默认色（黑字）
_GREEN, _RED, _YELLOW = "\x1b[32m", "\x1b[31m", "\x1b[33m"
_RESET = "\x1b[0m"

# 被 print 拦截顶掉的原生 print，以及当前已拦截 print 的实例栈（close 时还原）
_ORIG_PRINT = None
_PRINT_PATCHED: list = []


def _free_port() -> int:
    """向系统要一个当前空闲的 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Melog:
    """训练监控主入口（内部实现；公开入口为 melog.init()）。

    用法：
        >>> import melog
        >>> logger = melog.init(log_dir="runs/demo")
        >>> for step in logger.progress(range(100)):
        ...     logger.scalar({"loss": loss})
    """

    def __init__(
        self,
        project: str = "melog",
        output_dir: Optional[str] = None,
        enable_web: bool = True,
        web_host: str = "127.0.0.1",
        web_port: Optional[int] = None,
        enable_progress: bool = True,
        reduce_op: str = "mean",
        flush_every: int = 1,
        max_plot_points: int = 2000,
    ):
        """
        Args:
            project: 项目名，作为输出子目录。
            output_dir: 指标持久化根目录，默认 ./melog_runs。
            enable_web: 是否启动 Web 可视化服务（仅 rank0 生效）。
            web_host: Web 服务监听地址；web_port 为 None 时自动选空闲端口。
            enable_progress: 是否启用控制台进度条（仅 rank0 生效）。
            reduce_op: 多 GPU 合并方式，"mean" 或 "sum"。
            flush_every: 每 N 次 log 落盘一次 JSONL。
            max_plot_points: Web 历史曲线单指标最大点数，超出自动降采样；
                JSONL 落盘始终保留全量数据。
        """
        self.project = project
        self.reduce_op = reduce_op
        self.web_port = web_port if web_port is not None else _free_port()
        self._flush_every = max(1, flush_every)
        self._enable_progress = enable_progress
        self._rank = get_rank()
        self._is_primary = self._rank == 0
        self._step = 0
        self._epoch: Optional[int] = None  # 当前 epoch（用户传入后粘滞生效）
        self._epoch_step = 0  # 当前 epoch 内步数（未显式传入时内部统计）
        self._epoch_base = 0  # 当前 epoch 起始处的全局 x（跨 epoch 连续）
        self._epoch_bases: Dict[int, int] = {}  # 各 epoch 的全局 x 基准（媒体跨 epoch 定位）
        self._last_x = 0  # 最近一次记录的全局 x（媒体默认附着于此）
        self._last_epoch: Optional[int] = None  # 最近一次记录的 epoch
        self._pending = 0
        self._closed = False
        self._lock = threading.Lock()
        self._colors: Dict[str, str] = {}  # 用户指定的指标颜色（名称 -> CSS 颜色）

        self._run_dir = self._prepare_run_dir(output_dir)
        self._log_file = self._run_dir / "metrics.melog"

        self.store = MetricStore()
        self.media = MediaStore()
        self._web: Optional[WebServer] = None
        if enable_web and self._is_primary:
            self._web = WebServer(
                self.store,
                media_store=self.media,
                host=web_host,
                port=self.web_port,
                max_points=max_plot_points,
                log_file=str(self._log_file),
            )
            self._web.start()

        self._progress: Optional[tqdm] = None
        # 控制台日志镜像（仅 rank0）：进度条与 print 同步写入 console.log
        self.mirror: Optional[Mirror] = None
        if self._is_primary:
            self.mirror = Mirror(self._run_dir / "console.log")
            self.mirror.hook_stdio()
            # 拦截官方 print：用户代码里的 print(...) 内部改走 self.log()
            self._patch_print()
        if self._web is not None:
            # 端口可能随机分配，启动时打印面板地址（同步进 console.log）
            self.log(f"Web 可视化: {self._web.url}")
        # 注册为全局活动实例（见 melog.current / 模块级 melog.log 等便捷接口）
        _set_active(self)
        # 进程退出时自动收尾（落盘剩余指标 / 定稿进度条 / 停 Web / 还原 print）
        atexit.register(self.close)

    # ------------------------------------------------------------------ 运行目录
    def _prepare_run_dir(self, output_dir: Optional[str]) -> Path:
        root = Path(output_dir) if output_dir else Path.cwd() / "melog_runs"
        run_dir = root / self.project / time.strftime("%Y%m%d_%H%M%S")
        # 多进程时仅 rank0 创建目录，其余进程等待
        if self._is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "rank.txt").write_text(str(self._rank), encoding="utf-8")
        return run_dir

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def web_url(self) -> Optional[str]:
        """Web 面板地址（未启用 Web 时为 None）。"""
        return self._web.url if self._web is not None else None

    # ------------------------------------------------------------------ 训练上下文
    def progress(
        self,
        iterable: Iterable,
        description: str = "",
        total: Optional[float] = None,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> tqdm:
        """tqdm 风格进度条：直接包裹可迭代对象，迭代时自动推进，无需手动 update。

        用法与 tqdm.tqdm 一致::

            for batch in logger.progress(loader):
                logger.scalar({"loss": loss})   # 指标实时显示在进度条上

        传入 epoch 时，进入进度条即绑定该 epoch（epoch 内步数清零、全局
        x 从上一位置接续），bar 内的 scalar() / log_group() 无需再传
        epoch，bar 结束后沿用，直至下一个 epoch::

            for epoch in range(epochs):
                for _ in logger.progress(loader, epoch=epoch):
                    logger.scalar({"loss": loss})

        进度条布局：[n/total] 最前，指标其后，条形图/百分比/耗时殿后；
        description 提供时显示在行首（默认不显示）。total 缺省时自动取
        len(iterable)。进度条实时渲染到控制台，并经 Mirror 同步进
        console.log；非 rank0 或设置 MELOG_DISABLE_PROGRESS=1 时静默。
        迭代自然结束后自动解除登记，可再次调用 progress()（如每个
        epoch 一条进度条）。
        """
        if self._progress is not None:
            raise RuntimeError("progress() 上下文不可嵌套")
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                self._epoch = epoch
                self._epoch_step = 0
                self._epoch_base = self._step
                self._epoch_bases[epoch] = self._epoch_base
        disable = (not self._is_primary) or not self._enable_progress or _progress_disabled()
        bar = tqdm(iterable=iterable, total=total, desc=description, disable=disable, **kwargs)
        return self._register_progress(bar)

    def _register_progress(self, bar: tqdm) -> tqdm:
        """登记当前进度条（scalar() 的 postfix/advance 作用其上），close 后自动解除。"""
        self._progress = bar
        original_close = bar.close

        def _close() -> None:
            original_close()
            if self._progress is bar:
                self._progress = None

        bar.close = _close
        return bar

    # ------------------------------------------------------------------ 记录指标
    def scalar(
        self,
        metrics: Dict[str, Union[float, int, Any]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        advance: int = 0,
        commit: bool = True,
    ) -> Dict[str, float]:
        """记录一批指标。

        多 GPU 场景下先做 all_reduce 合并（默认取均值），再由 rank0
        持久化、推送到 Web、刷新进度条。

        Args:
            metrics: 指标名 -> 数值（float / int / 0 维 tensor）。
            step: 当前 epoch 内的步数；未启用 epoch 时为全局步数，
                缺省时内部自增（epoch 模式下每个 epoch 从 0 重新计步）。
            epoch: 当前 epoch 序号，曲线图据此标注 epoch 分界；
                缺省沿用当前绑定的 epoch（progress(epoch=...) 绑定或
                上次显式传入），从未设置则不记录 epoch。
            advance: 额外推进进度条的步数（progress() 迭代每次已自动
                推进 1，缺省 0；仅一个迭代内多次 scalar() 等场景需要传入）。
            commit: 是否推进内部 step 计数。
        Returns:
            合并后的指标（rank>0 也返回，便于本地打印）。
        """
        merged = reduce_metrics(metrics, op=self.reduce_op)

        if not self._is_primary:
            return merged

        with self._lock:
            if epoch is not None and epoch != self._epoch:
                # 切换 epoch：epoch 内步数清零，全局 x 从上一位置接续
                self._epoch = epoch
                self._epoch_step = 0
                self._epoch_base = self._step
                self._epoch_bases[epoch] = self._epoch_base
            if self._epoch is None:
                x = step if step is not None else self._step
                out_epoch = None
            else:
                s = step if step is not None else self._epoch_step
                x = self._epoch_base + s
                out_epoch = self._epoch
            self.store.add(x, merged, out_epoch)
            self._push_web(x, merged, out_epoch)
            self._update_progress(merged)
            if advance and self._progress is not None:
                self._progress.update(advance)
            self._maybe_flush()
            self._last_x, self._last_epoch = x, out_epoch
            if commit:
                self._epoch_step = (step + 1) if step is not None else (self._epoch_step + 1)
                self._step = x + 1
        return merged

    # 兼容 wandb 风格别名
    # 兼容 wandb 风格别名
    log_metrics = scalar

    # ------------------------------------------------------------------ 记录媒体
    def image(
        self,
        name: str,
        data: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """记录一帧图像（曲线图之外的"图像"页签展示）。

        数据落盘到 run_dir/media/image/<name>/，元数据写入日志文件并
        实时推送到 Web；多 GPU 下仅 rank0 落盘，其余 rank 直接返回。

        Args:
            name: 图像名，支持 "train/sample" 层级命名（页面按名建卡片）。
            data: 文件路径 / PIL.Image / numpy / torch 张量
                （(H,W) 灰度或 (H,W,C)，C=1/3/4；浮点自动映射 0-255）。
            step / epoch: 显式指定展示位置，语义与 scalar() 一致（epoch
                模式下 step 为 epoch 内步数）；缺省附着到最近一次
                scalar() 的位置。均不推进 step 计数。
            caption: 配文，随图显示在卡片上（如样本说明、预测对比）。
        """
        self._log_media("image", name, data, step=step, epoch=epoch, caption=caption,
                        save=lambda out_dir, stem: save_image(data, out_dir, stem))

    def audio(
        self,
        name: str,
        data: Any,
        sr: int = 22050,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """记录一段音频（"音频"页签展示，浏览器内直接播放）。

        Args:
            name: 音频名，支持层级命名。
            data: 文件路径（wav/mp3/flac 等按原格式复制）/ numpy / torch
                波形（(N,) 单声道或 (N, 声道数)；浮点按 [-1,1] 裁剪）。
            sr: 采样率（data 为路径时忽略，沿用文件本身格式）。
            step / epoch: 显式指定展示位置，语义与 scalar() 一致（epoch
                模式下 step 为 epoch 内步数）；缺省附着到最近一次
                scalar() 的位置。均不推进 step 计数。
            caption: 配文，随音频显示在卡片上（如转写文本、听感说明）。
        """
        self._log_media("audio", name, data, step=step, epoch=epoch, sr=sr, caption=caption,
                        save=lambda out_dir, stem: save_audio(data, out_dir, stem, sr))

    def _log_media(self, kind: str, name: str, data: Any, save, step, epoch,
                   sr=None, caption=None) -> None:
        """媒体记录公共流程：定位 -> 落盘 -> 索引 -> 日志 -> 推送。"""
        if not self._is_primary:
            return
        safe = sanitize_name(name)
        with self._lock:
            x, e = self._media_position(step, epoch)
            rel = f"media/{kind}/{safe}/{save(self._run_dir / 'media' / kind / safe, f'{int(x):09d}')}"
            record: Dict[str, Any] = {"type": kind, "metric": name, "step": int(x), "file": rel}
            if e is not None:
                record["epoch"] = e
            if sr is not None:
                record["sr"] = sr
            if caption:
                record["caption"] = caption
            self.media.add(kind, name, x, rel, e, sr=sr, caption=caption)
            self._append_journal(record)
            self._push_web_media(kind, name, x, e, rel, sr=sr, caption=caption)

    def _media_position(self, step: Optional[int], epoch: Optional[int]):
        """媒体条目的展示位置：坐标规则与 scalar() 一致，但不推进任何计数器。

        缺省附着最近一次 scalar() 的位置；epoch 模式下 step 为 epoch 内
        步数，x = 该 epoch 的全局基准 + step。
        """
        if step is None and epoch is None:
            return self._last_x, self._last_epoch
        e = epoch if epoch is not None else self._epoch
        if e is None:
            return (step if step is not None else self._last_x), None
        base = self._epoch_bases.get(e, self._epoch_base)
        s = step if step is not None else self._epoch_step
        return base + s, e

    def _append_journal(self, record: Dict[str, Any]) -> None:
        """把一条记录追加到日志文件（调用方需持有 _lock）。"""
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _push_web_media(self, kind: str, name: str, step: int, epoch: Optional[int],
                        relpath: str, sr: Optional[int] = None,
                        caption: Optional[str] = None) -> None:
        if self._web is not None:
            self._web.publish_media(kind, name, step, epoch, relpath, sr=sr, caption=caption)

    def log_group(
        self,
        group: MetricGroup,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        advance: int = 0,
        reset: bool = False,
    ) -> Dict[str, float]:
        """记录一组 Metric 指标（跨 GPU 同步由 group.compute() 内部完成）。

        所有 rank 都应调用本方法；仅 rank0 持久化与展示。

        Args:
            group: MetricGroup 实例。
            step: 当前 epoch 内的步数，缺省时内部自增。
            epoch: 当前 epoch 序号，缺省沿用当前绑定的 epoch
                （progress(epoch=...) 绑定或上次显式传入）。
            advance: 额外推进进度条的步数，epoch 级记录默认不推进。
            reset: 记录后是否重置组内指标（开启新一轮 epoch 统计）。
        """
        values = group.compute()
        result = self.scalar(values, step=step, epoch=epoch, advance=advance)
        if reset:
            group.reset()
        return result

    # ------------------------------------------------------------------ 面板配色
    def set_colors(self, colors: Dict[str, str]) -> None:
        """指定指标在 Web 面板中的颜色（指标名 -> CSS 颜色字符串）。

        未指定的指标仍按名称 hash 自动配色；同名指标重复设置会覆盖。
        颜色随 colors.json 落盘（历史日志重新加载时一并恢复），并实时
        推送到已连接的面板；多 GPU 下所有 rank 都可调用（仅 rank0 落盘）。

        Args:
            colors: 如 {"recall/class_0": "#ef4444", "loss": "steelblue"}，
                支持 #RGB/#RRGGBB 等 CSS 颜色写法。
        """
        with self._lock:
            self._colors.update(colors)
        if not self._is_primary:
            return
        (self._run_dir / "colors.json").write_text(
            json.dumps(self._colors, ensure_ascii=False), encoding="utf-8"
        )
        if self._web is not None:
            self._web.set_colors(dict(self._colors))

    # ------------------------------------------------------------------ 控制台消息
    def log(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """普通控制台输出，签名对齐 print；终端默认色（黑字），无图标前缀。

        实例存活期间官方 print 被拦截到本方法；多个参数自动转 str()
        后以 sep 拼接。
        """
        self._emit("", "", values, sep, end, flush)

    def success(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """绿色文字 + ✔ 前缀。"""
        self._emit("✔", _GREEN, values, sep, end, flush)

    def error(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """红色文字 + ✘ 前缀。"""
        self._emit("✘", _RED, values, sep, end, flush)

    def warn(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """黄色文字 + ⚠ 前缀。"""
        self._emit("⚠", _YELLOW, values, sep, end, flush)

    def _emit(self, icon: str, color: str, values: tuple, sep: str,
              end: str, flush: bool) -> None:
        """控制台消息统一出口：转 str、拼图标、按 TTY 着色，写入当前 stdout。"""
        text = sep.join(str(v) for v in values)
        line = f"{icon} {text}" if icon else text
        stream = sys.stdout
        if color and _is_tty(stream):
            line = f"{color}{line}{_RESET}"
        stream.write(line + end)
        if flush:
            stream.flush()

    def _patch_print(self) -> None:
        """拦截官方 print：用户代码的 print(...) 内部改走 self.log()。"""
        global _ORIG_PRINT
        if not _PRINT_PATCHED:
            _ORIG_PRINT = builtins.print
        _PRINT_PATCHED.append(self)
        builtins.print = self._print_proxy

    def _unpatch_print(self) -> None:
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

    def _push_web(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        if self._web is not None:
            self._web.publish(step, metrics, epoch=epoch)

    def _update_progress(self, metrics: Dict[str, float]) -> None:
        if self._progress is not None:
            self._progress.set_postfix(metrics)

    def _maybe_flush(self) -> None:
        self._pending += 1
        if self._pending >= self._flush_every:
            self._flush()
            self._pending = 0

    def _flush(self) -> None:
        records = self.store.drain()
        if not records:
            return
        with open(self._log_file, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ 收尾
    def close(self) -> None:
        """落盘剩余指标，定稿进度条与日志镜像，停止 Web 服务。

        进程退出时经 atexit 自动调用，无需手动收尾；若本实例是全局
        活动实例（见 melog.current），收尾后一并清空。
        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if self._is_primary:
            self._flush()
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self.mirror is not None:
            self._unpatch_print()
            self.mirror.unhook_stdio()
            self.mirror.close()
            self.mirror = None
        if self._web is not None:
            self._web.stop()
            self._web = None
        if _get_active() is self:
            _set_active(None)

    def __enter__(self) -> "Melog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------- 全局共享
# 最近一次创建的 Melog 实例作为全局活动实例，供模块级便捷接口使用；
# 实例本身线程安全（内部有锁），跨模块共享无需额外处理。
_active: Optional["Melog"] = None


def _get_active() -> Optional["Melog"]:
    return _active


def _set_active(inst: Optional["Melog"]) -> None:
    global _active
    _active = inst


def current() -> "Melog":
    """返回全局共享的 Melog 实例；尚未创建时抛出 RuntimeError。"""
    if _active is None:
        raise RuntimeError("尚未创建 Melog 实例：请先调用 melog.init(...)（或 Melog(...)）")
    return _active


def init(log_dir: str = "./melog_runs", web_port: Optional[int] = None, **kwargs: Any) -> Melog:
    """创建并激活全局共享的 Melog 实例（melog 的唯一公开入口）。

    Args:
        log_dir: 日志保存路径；本次运行的指标 / 媒体 / console.log 落在
            其下的时间戳子目录中，路径末级目录名作为项目名展示。
        web_port: Web 监听端口；缺省自动选择一个空闲端口。
        **kwargs: 其余高级参数（enable_web / enable_progress / reduce_op /
            flush_every / max_plot_points，以及 project 覆盖项目名等）。

    入口处调用一次，之后项目任意位置可直接使用模块级 melog.scalar() 等
    接口。再次调用会用新实例替换当前活动实例。
    """
    log_dir = Path(log_dir)
    kwargs.setdefault("project", log_dir.name)
    return Melog(output_dir=str(log_dir.parent), web_port=web_port, **kwargs)


def scalar(
    metrics: Dict[str, Union[float, int, Any]],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    advance: int = 0,
    commit: bool = True,
) -> Dict[str, float]:
    """模块级便捷接口：等价于 ``current().scalar(...)``。"""
    return current().scalar(metrics, step=step, epoch=epoch, advance=advance, commit=commit)


def log(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().log(...)``。"""
    current().log(*values, sep=sep, end=end, flush=flush)


def success(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().success(...)``。"""
    current().success(*values, sep=sep, end=end, flush=flush)


def error(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().error(...)``。"""
    current().error(*values, sep=sep, end=end, flush=flush)


def warn(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().warn(...)``。"""
    current().warn(*values, sep=sep, end=end, flush=flush)


def log_group(
    group: MetricGroup,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    advance: int = 0,
    reset: bool = False,
) -> Dict[str, float]:
    """模块级便捷接口：等价于 ``current().log_group(...)``。"""
    return current().log_group(group, step=step, epoch=epoch, advance=advance, reset=reset)


def image(
    name: str,
    data: Any,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().image(...)``。"""
    current().image(name, data, step=step, epoch=epoch, caption=caption)


def audio(
    name: str,
    data: Any,
    sr: int = 22050,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().audio(...)``。"""
    current().audio(name, data, sr=sr, step=step, epoch=epoch, caption=caption)


def set_colors(colors: Dict[str, str]) -> None:
    """模块级便捷接口：等价于 ``current().set_colors(...)``。"""
    current().set_colors(colors)


def _progress_disabled() -> bool:
    # CI 日志里进度条噪音大，保留开关
    return os.environ.get("MELOG_DISABLE_PROGRESS", "0") == "1"
