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
from .progress import TrainProgress
from .web.server import MetricStore, WebServer

__all__ = ["Melog"]


class Melog:
    """训练监控主入口。

    用法：
        >>> from melog import Melog
        >>> logger = Melog(project="demo", enable_web=True)
        >>> with logger.train(total=100) as bar:
        ...     for step in range(100):
        ...         loss = 1.0 / (step + 1)
        ...         logger.log({"loss": loss, "lr": 1e-3})
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
        self._pending = 0
        self._closed = False
        self._lock = threading.Lock()

        self._run_dir = self._prepare_run_dir(output_dir)
        self._log_file = self._run_dir / "metrics.melog"

        self.store = MetricStore()
        self._web: Optional[WebServer] = None
        if enable_web and self._is_primary:
            self._web = WebServer(self.store, host=web_host, port=web_port, max_points=max_plot_points)
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
        advance: int = 1,
        commit: bool = True,
    ) -> Dict[str, float]:
        """记录一批指标。

        多 GPU 场景下先做 all_reduce 合并（默认取均值），再由 rank0
        持久化、推送到 Web、刷新进度条。

        Args:
            metrics: 指标名 -> 数值（float / int / 0 维 tensor）。
            step: 全局步数，缺省时内部自增。
            advance: 进度条前进步数。
            commit: 是否推进内部 step 计数。
        Returns:
            合并后的指标（rank>0 也返回，便于本地打印）。
        """
        merged = reduce_metrics(metrics, op=self.reduce_op)

        if not self._is_primary:
            return merged

        if step is None:
            step = self._step
        with self._lock:
            self.store.add(step, merged)
            self._push_web(step, merged)
            self._update_progress(merged)
            self._maybe_flush()
            if commit:
                self._step = step + 1
        return merged

    # 兼容 wandb 风格别名
    log_metrics = log

    def _push_web(self, step: int, metrics: Dict[str, float]) -> None:
        if self._web is not None:
            self._web.publish(step, metrics)

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
