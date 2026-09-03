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
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

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


class _EpochEndIterable:
    """包装可迭代对象：仅在自然耗尽时触发一次回调（提前 break / 异常不触发）。

    供 stepsbar(..., metrics=...) 实现 epoch 末自动记录。回调发生在最后
    一个元素之后、StopIteration 传给进度条之前，此时进度条尚未关闭，
    记录的指标值能渲染进 postfix。所有 rank 都会执行回调（compute() 是
    集合操作，各 rank 必须在同一位置调用；落盘与展示由 log_group 内部
    仅 rank0 处理）。
    """

    def __init__(self, iterable: Iterable, on_end: Callable[[], None]):
        self._src = iterable
        self._it = iter(iterable)
        self._on_end = on_end
        self._fired = False

    def __iter__(self) -> "_EpochEndIterable":
        return self

    def __next__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            if not self._fired:
                self._fired = True
                self._on_end()
            raise

    def __len__(self) -> int:
        # 透传 len()，tqdm 才能自动取 total（无 len 时 TypeError 由 tqdm 捕获）
        return len(self._src)  # type: ignore[arg-type]


class _BarFrame:
    """stepsbar 栈帧：一条打开的进度条及其挂载的指标组。"""

    __slots__ = ("bar", "metrics")

    def __init__(self, bar: tqdm, metrics: Optional[MetricGroup]):
        self.bar = bar
        self.metrics = metrics


class _Axis:
    """训练坐标轴：全局 x 与 epoch 内步数的唯一裁决者。

    坐标不接受手动指定，完全由 stepsbar 驱动（scalar 写入与媒体定位
    共用同一实现，避免规则漂移）：
    - 未绑定 epoch（没用 stepsbar(epoch=...)）：x 即全局提交计数
    - epoch 模式：x = 该 epoch 的全局基准 + epoch 内已提交步数；
      步数每次提交自增，全局 x 跨 epoch 连续接续

    写入型记录（scalar）走 resolve_commit + commit，会推进计数器；
    附着型记录（媒体）走 resolve_attach，只读、绝不推进任何计数。
    """

    def __init__(self) -> None:
        self.step = 0  # 下一个全局 x（= 已提交记录数）
        self.epoch: Optional[int] = None  # 当前绑定 epoch（粘滞，幂等切换）
        self.epoch_step = 0  # 当前 epoch 内已提交步数
        self.bases: Dict[int, int] = {}  # 各 epoch 的全局 x 基准
        self.last_x = 0  # 最近一次记录的全局 x
        self.last_epoch: Optional[int] = None  # 最近一次记录的 epoch

    @property
    def base(self) -> int:
        """当前 epoch 的全局 x 基准（未绑定 epoch 时为 0）。"""
        return self.bases.get(self.epoch, 0) if self.epoch is not None else 0

    def bind_epoch(self, epoch: int) -> None:
        """进入 epoch（幂等）：epoch 内步数清零，全局 x 从上一位置接续。"""
        if epoch != self.epoch:
            self.epoch = epoch
            self.epoch_step = 0
            self.bases[epoch] = self.step

    def resolve_commit(self) -> "tuple[int, Optional[int]]":
        """scalar 写入位置：下一个空槽（epoch 模式为 epoch 内步数）。"""
        if self.epoch is None:
            return self.step, None
        return self.base + self.epoch_step, self.epoch

    def resolve_attach(self) -> "tuple[int, Optional[int]]":
        """媒体附着位置：最近一次 scalar() 的提交位置（只读，不推进计数）。"""
        return self.last_x, self.last_epoch

    def commit(self, x: int, epoch: Optional[int]) -> None:
        """scalar 记录后推进计数器（附着型记录不调用）。"""
        self.last_x, self.last_epoch = x, epoch
        if epoch is not None:
            self.epoch_step += 1
        self.step = x + 1


class Melog:
    """训练监控主入口（内部实现；公开入口为 melog.init()）。

    用法：
        >>> import melog
        >>> logger = melog.init(log_dir="runs/demo")
        >>> for step in logger.stepsbar(range(100)):
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
        self._flush_every = max(1, flush_every)
        self._enable_progress = enable_progress
        self._rank = get_rank()
        self._is_primary = self._rank == 0
        self._axis = _Axis()  # 全局 x / epoch 计数与定位的唯一所有者
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
                port=web_port,
                max_points=max_plot_points,
                log_file=str(self._log_file),
            )
            self._web.start()

        self._bars: List[_BarFrame] = []  # 打开中的进度条栈（栈顶 = 当前环境）
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
    def stepsbar(
        self,
        iterable: Iterable,
        total: Optional[float] = None,
        epoch: Optional[int] = None,
        metrics: Optional[MetricGroup] = None,
        reset: bool = False,
        **kwargs: Any,
    ) -> tqdm:
        """tqdm 风格进度条：直接包裹可迭代对象，迭代时自动推进，无需手动 update。

        本库按 epoch 组织训练记录：**每个 epoch 的循环必须用 stepsbar
        包裹**并传入 epoch，坐标（epoch/step）由它统一管理——scalar() /
        log_group() / image() / audio() 都没有坐标参数，记录自动依附
        当前 epoch 与下一个空槽；不用 stepsbar 包裹的记录退化为全局
        自增 x、无 epoch 分界。

        用法与 tqdm.tqdm 一致::

            for batch in logger.stepsbar(loader):
                logger.scalar({"loss": loss})   # 指标实时显示在进度条上

        传入 epoch 时，进入进度条即绑定该 epoch（epoch 内步数清零、全局
        x 从上一位置接续），bar 结束后沿用，直至下一个 epoch；行首描述
        自动标为 "epoch N"（需自定义时透传 tqdm 的 desc=...）::

            for epoch in range(epochs):
                for _ in logger.stepsbar(loader, epoch=epoch):
                    logger.scalar({"loss": loss})   # 坐标自动依附 epoch

        传入 metrics（MetricGroup）时，进度条实时显示本卡本地值——每次
        feed() 后零通信刷新 postfix（实际渲染的只有 rank0，即主卡本地
        值；无观测的指标与非数值结果自动跳过）。迭代自然结束即 gather
        所有 rank 的状态、log_group 全局值一次（reset=True 则记录后重
        置组内指标），曲线上得到跨 GPU 精确合并的结果::

            for _ in logger.stepsbar(loader, epoch=e, metrics=metrics, reset=True):
                metrics.feed(...)

        自动记录仅在循环自然跑完时触发：提前 break / 抛异常不会记录
        （此时各 rank 的进度可能不一致，自动 compute() 的 all_gather 会
        互相等待甚至挂死；需要中途落盘请显式调用 scalar() / log_group()）。
        所有 rank 都会触发回调，compute() 在各 rank 同一位置执行，落盘
        仅 rank0。

        total 缺省时自动取 len(iterable)。进度条实时渲染到控制台，并经
        Mirror 同步进 console.log；非 rank0 或设置 MELOG_DISABLE_PROGRESS=1
        时静默。迭代自然结束后自动出栈，可再次调用 stepsbar()（如每个
        epoch 一条进度条）。

        允许嵌套（如训练 bar 内嵌验证 bar）：内部以栈管理，current_bar()
        返回栈顶即当前环境；scalar() / log_group() 的 postfix 与 advance
        自动作用于栈顶，下层 bar 暂停渲染（计数与 postfix 照常更新），
        栈顶关闭后自动恢复下层渲染。提前 break 的 bar 请 close()（或用
        with 包裹），否则会一直留在栈中占位。
        """
        if metrics is not None and not isinstance(metrics, MetricGroup):
            raise TypeError(f"metrics 须为 MetricGroup，收到 {type(metrics).__name__}")
        if epoch is not None:
            with self._lock:
                self._axis.bind_epoch(epoch)
        if metrics is not None:
            iterable = _EpochEndIterable(
                iterable, lambda: self.log_group(metrics, reset=reset)
            )
        disable = (not self._is_primary) or not self._enable_progress or _progress_disabled()
        if epoch is not None and "desc" not in kwargs:
            kwargs["desc"] = f"epoch {epoch}"
        bar = tqdm(iterable=iterable, total=total, disable=disable, **kwargs)
        return self._register_progress(bar, metrics=metrics)

    def current_bar(self) -> Optional[tqdm]:
        """当前栈顶进度条（无打开的 bar 时为 None）。

        stepsbar() 允许嵌套、以栈管理：scalar() / log_group() 的 postfix
        与 advance 自动作用于栈顶；深层函数需要手动推进、读数或写
        postfix 时用它获取当前 bar，免层层传参。
        """
        with self._lock:
            return self._bars[-1].bar if self._bars else None

    def _register_progress(self, bar: tqdm, metrics: Optional[MetricGroup] = None) -> tqdm:
        """压栈登记进度条（栈顶 = 当前环境），关闭后自动出栈。

        挂载 metrics 时，feed() 的实时刷新只作用于自己的 bar（即使被
        上层覆盖，postfix 数据照常更新，恢复渲染时可见）。
        """
        with self._lock:
            for frame in self._bars:  # 覆盖下层：只有栈顶渲染
                frame.bar.covered = True
            self._bars.append(_BarFrame(bar, metrics))
            if metrics is not None:
                def _display_local() -> None:
                    # 实时显示：本卡本地值，零通信；NaN 与非数值（如混淆矩阵）不上 postfix
                    snap = {
                        k: v
                        for k, v in metrics.local().items()
                        if isinstance(v, (int, float)) and v == v
                    }
                    bar.set_postfix(snap)

                metrics._on_feed = _display_local
        bar.on_close = lambda: self._forget_progress(bar)
        return bar

    def _forget_progress(self, bar: tqdm) -> None:
        """bar 关闭：出栈并解除其指标组钩子；恢复新栈顶的渲染。"""
        with self._lock:
            for i, frame in enumerate(self._bars):
                if frame.bar is bar:
                    del self._bars[i]
                    if frame.metrics is not None:
                        frame.metrics._on_feed = None  # bar 已关闭，停止实时刷新
                    break
            if self._bars:
                top = self._bars[-1].bar
                top.covered = False
                top.refresh()

    # ------------------------------------------------------------------ 记录指标
    def scalar(
        self,
        metrics: Dict[str, Union[float, int, Any]],
        advance: int = 0,
    ) -> Dict[str, float]:
        """记录一批指标；坐标（epoch/step）由 stepsbar 自动管理。

        多 GPU 场景下先做 all_reduce 合并（默认取均值），再由 rank0
        持久化、推送到 Web、刷新进度条。

        坐标规则：epoch 由 stepsbar(epoch=...) 绑定，step 取 epoch 内
        下一个空槽（内部自增，全局 x 跨 epoch 连续接续）；未用 stepsbar
        时退化为全局自增。要控制记录粒度（每步 / 每 N 步窗口），调整
        调用 scalar() 的频率即可，无需也无法手动指定坐标。

        Args:
            metrics: 指标名 -> 数值（float / int / 0 维 tensor）。
            advance: 额外推进进度条的步数（stepsbar() 迭代每次已自动
                推进 1，缺省 0；仅一个迭代内多次 scalar() 等场景需要传入）。
        Returns:
            合并后的指标（rank>0 也返回，便于本地打印）。
        """
        merged = reduce_metrics(metrics, op=self.reduce_op)

        if not self._is_primary:
            return merged

        with self._lock:
            x, out_epoch = self._axis.resolve_commit()
            self.store.add(x, merged, out_epoch)
            self._push_web(x, merged, out_epoch)
            self._update_progress(merged)
            if advance and self._bars:
                self._bars[-1].bar.update(advance)
            self._maybe_flush()
            self._axis.commit(x, out_epoch)
        return merged

    # 兼容 wandb 风格别名
    log_metrics = scalar

    # ------------------------------------------------------------------ 记录媒体
    def image(
        self,
        name: str,
        data: Any,
        caption: Optional[str] = None,
    ) -> None:
        """记录一帧图像（曲线图之外的"图像"页签展示）。

        数据落盘到 run_dir/media/image/<name>/，元数据写入日志文件并
        实时推送到 Web；多 GPU 下仅 rank0 落盘，其余 rank 直接返回。

        Args:
            name: 图像名，支持 "train/sample" 层级命名（页面按名建卡片）。
            data: 文件路径 / PIL.Image / numpy / torch 张量
                （(H,W) 灰度或 (H,W,C)，C=1/3/4；浮点自动映射 0-255）。
            caption: 配文，随图显示在卡片上（如样本说明、预测对比）。

        位置自动附着到最近一次 scalar() / log_group() 的记录处，不推进计数。
        """
        self._log_media("image", name, data, caption=caption,
                        save=lambda out_dir, stem: save_image(data, out_dir, stem))

    def audio(
        self,
        name: str,
        data: Any,
        sr: int = 22050,
        caption: Optional[str] = None,
    ) -> None:
        """记录一段音频（"音频"页签展示，浏览器内直接播放）。

        Args:
            name: 音频名，支持层级命名。
            data: 文件路径（wav/mp3/flac 等按原格式复制）/ numpy / torch
                波形（(N,) 单声道或 (N, 声道数)；浮点按 [-1,1] 裁剪）。
            sr: 采样率（data 为路径时忽略，沿用文件本身格式）。
            caption: 配文，随音频显示在卡片上（如转写文本、听感说明）。

        位置自动附着到最近一次 scalar() / log_group() 的记录处，不推进计数。
        """
        self._log_media("audio", name, data, sr=sr, caption=caption,
                        save=lambda out_dir, stem: save_audio(data, out_dir, stem, sr))

    def _log_media(self, kind: str, name: str, data: Any, save,
                   sr=None, caption=None) -> None:
        """媒体记录公共流程：定位 -> 落盘 -> 索引 -> 日志 -> 推送。"""
        if not self._is_primary:
            return
        safe = sanitize_name(name)
        with self._lock:
            x, e = self._axis.resolve_attach()
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
        advance: int = 0,
        reset: bool = False,
    ) -> Dict[str, float]:
        """记录一组 Metric 指标（跨 GPU 同步由 group.compute() 内部完成）。

        所有 rank 都应调用本方法；仅 rank0 持久化与展示。坐标自动依附
        当前绑定的 epoch（stepsbar）与下一个空槽。

        Args:
            group: MetricGroup 实例。
            advance: 额外推进进度条的步数，epoch 级记录默认不推进。
            reset: 记录后是否重置组内指标（开启新一轮 epoch 统计）。
        """
        values = group.compute()
        result = self.scalar(values, advance=advance)
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
        if self._bars:
            self._bars[-1].bar.set_postfix(metrics)

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
        while self._bars:  # 定稿所有打开中的进度条（栈顶先关，逐层恢复渲染）
            self._bars[-1].bar.close()
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
    advance: int = 0,
) -> Dict[str, float]:
    """模块级便捷接口：等价于 ``current().scalar(...)``。"""
    return current().scalar(metrics, advance=advance)


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
    advance: int = 0,
    reset: bool = False,
) -> Dict[str, float]:
    """模块级便捷接口：等价于 ``current().log_group(...)``。"""
    return current().log_group(group, advance=advance, reset=reset)


def current_bar() -> Optional[tqdm]:
    """模块级便捷接口：等价于 ``current().current_bar()``。"""
    return current().current_bar()


def image(
    name: str,
    data: Any,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().image(...)``。"""
    current().image(name, data, caption=caption)


def audio(
    name: str,
    data: Any,
    sr: int = 22050,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().audio(...)``。"""
    current().audio(name, data, sr=sr, caption=caption)


def set_colors(colors: Dict[str, str]) -> None:
    """模块级便捷接口：等价于 ``current().set_colors(...)``。"""
    current().set_colors(colors)


def _progress_disabled() -> bool:
    # CI 日志里进度条噪音大，保留开关
    return os.environ.get("MELOG_DISABLE_PROGRESS", "0") == "1"
