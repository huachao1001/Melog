"""Melog：轻量级训练监控库。

核心能力：
- 训练指标记录与 JSONL 持久化
- 多 GPU 指标合并（torch.distributed all_reduce）
- 控制台实时进度条显示指标
- Web 可视化（FastAPI + WebSocket + ECharts）
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .distributed import get_rank, reduce_metrics
from .media import sanitize_name, save_audio, save_image
from .metrics import MetricGroup
from .progress import TrainProgress
from .web.media_store import MediaStore
from .web.server import WebServer
from .web.store import MetricStore

__all__ = ["Melog"]


class Melog:
    """训练监控主入口。

    用法：
        >>> from melog import Melog
        >>> mlog = Melog(project="demo", enable_web=True)
        >>> with mlog.train(total=100) as bar:
        ...     for step in range(100):
        ...         loss = 1.0 / (step + 1)
        ...         mlog.log({"loss": loss, "lr": 1e-3})
        ...         bar.advance(1)
    """

    def __init__(
        self,
        project: str = "melog",
        output_dir: Optional[str] = None,
        enable_web: bool = True,
        web_host: str = "127.0.0.1",
        web_port: int = 8666,
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
            web_host / web_port: Web 服务监听地址。
            enable_progress: 是否启用控制台进度条（仅 rank0 生效）。
            reduce_op: 多 GPU 合并方式，"mean" 或 "sum"。
            flush_every: 每 N 次 log 落盘一次 JSONL。
            max_plot_points: Web 历史曲线单指标最大点数，超出自动降采样；
                JSONL 落盘始终保留全量数据。
        """
        self.project = project
        self.reduce_op = reduce_op
        self._flush_every = max(1, flush_every)
        self._rank = get_rank()
        self._is_primary = self._rank == 0
        self._step = 0
        self._epoch: Optional[int] = None  # 当前 epoch（用户传入后粘滞生效）
        self._epoch_step = 0  # 当前 epoch 内步数（未显式传入时内部统计）
        self._epoch_base = 0  # 当前 epoch 起始处的全局 x（跨 epoch 连续）
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
                port=web_port,
                max_points=max_plot_points,
                log_file=str(self._log_file),
            )
            self._web.start()

        self._progress: Optional[TrainProgress] = None

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

    # ------------------------------------------------------------------ 训练上下文
    def train(self, total: int, description: str = "train") -> "TrainProgress":
        """返回训练进度条上下文管理器（非分布式或 rank0 才真正渲染）。"""
        if self._progress is not None:
            raise RuntimeError("train() 上下文不可嵌套")
        if not self._is_primary or not _progress_enabled():
            return _NullProgress()
        self._progress = TrainProgress(total=total, description=description)
        return self._progress

    # ------------------------------------------------------------------ 记录指标
    def log(
        self,
        metrics: Dict[str, Union[float, int, Any]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        advance: int = 1,
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
                缺省沿用上一次传入的值，从未传入则不记录 epoch。
            advance: 进度条前进步数。
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
            self._maybe_flush()
            self._last_x, self._last_epoch = x, out_epoch
            if commit:
                self._epoch_step = (step + 1) if step is not None else (self._epoch_step + 1)
                self._step = x + 1
        return merged

    # 兼容 wandb 风格别名
    log_metrics = log

    # ------------------------------------------------------------------ 记录媒体
    def log_image(
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
            step / epoch: 缺省附着到最近一次 log() 的位置，不推进 step 计数。
            caption: 配文，随图显示在卡片上（如样本说明、预测对比）。
        """
        self._log_media("image", name, data, step=step, epoch=epoch, caption=caption,
                        save=lambda out_dir, stem: save_image(data, out_dir, stem))

    def log_audio(
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
            step / epoch: 缺省附着到最近一次 log() 的位置，不推进 step 计数。
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
        """媒体条目的展示位置：缺省附着最近一次 log()，不推进任何计数器。"""
        if step is None and epoch is None:
            return self._last_x, self._last_epoch
        e = epoch if epoch is not None else self._last_epoch
        x = step if step is not None else self._last_x
        return x, e

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
            epoch: 当前 epoch 序号，缺省沿用上一次传入的值。
            advance: 进度条前进步数，epoch 级记录默认不推进。
            reset: 记录后是否重置组内指标（开启新一轮 epoch 统计）。
        """
        values = group.compute()
        result = self.log(values, step=step, epoch=epoch, advance=advance)
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

    def _push_web(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        if self._web is not None:
            self._web.publish(step, metrics, epoch=epoch)

    def _update_progress(self, metrics: Dict[str, float]) -> None:
        if self._progress is not None:
            self._progress.show_metrics(metrics)

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
    def finish(self) -> None:
        """落盘剩余指标并停止 Web 服务。"""
        if self._closed:
            return
        self._closed = True
        if self._is_primary:
            self._flush()
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._web is not None:
            self._web.stop()
            self._web = None

    def close(self) -> None:
        self.finish()

    def __enter__(self) -> "Melog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()


class _NullProgress:
    """非主进程下的空进度条，保持接口一致。"""

    def advance(self, n: int = 1) -> None:
        pass

    def show_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


def _progress_enabled() -> bool:
    # 非 TTY 环境下 rich 也能渲染，但 CI 日志里进度条噪音大，保留开关
    return os.environ.get("MELOG_DISABLE_PROGRESS", "0") != "1"
