"""Melog：轻量级训练监控库。

核心能力：
- 训练指标记录与二进制持久化（melog 容器，会话文件带时间戳）
- 断点续训：同一目录重跑自动回灌历史、按 epoch 截断重叠区
- 多 GPU 指标合并（torch.distributed all_reduce）
- 控制台实时进度条（tqdm 兼容）与控制台日志镜像
- Web 可视化（FastAPI + WebSocket + ECharts）

目录划分：
- core.py（本模块）：Melog 主类——组合各组件、调度记录、管理生命周期
- api/：全局入口 melog.init() 与模块级便捷接口（melog.scalar() 等）
- tracking/：记录上下文——坐标（axis）、进度条（steps_bar）、控制台消息（console）
- storage/：持久化与产物——二进制日志（journal / melog_file）、媒体
  （media / media_log）、控制台镜像（mirror）
- metrics/：指标计算与跨 GPU 同步；web/：可视化面板；utils/：通用工具
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .metrics import MetricGroup
from .storage.journal import Journal
from .storage.media_log import MediaLog
from .storage.melog_file import MelogFile, MelogFileReader
from .storage.mirror import Mirror
from .tracking.axis import Axis
from .tracking.console import Console
from .utils.bar_stack import BarStack
from .utils.distributed import get_rank, reduce_metrics
from .utils.tqdm import tqdm
from .web.loader import LogLoader
from .web.media_store import MediaStore
from .web.server import WebServer
from .web.store import MetricStore

__all__ = ["Melog", "current"]


class Melog:
    """训练监控主入口（内部实现；公开入口为 melog.init()）。

    组合的组件（各司其职，详见各模块文档）：
    - Axis      全局 x / epoch 坐标裁决
    - BarStack  进度条栈（嵌套、恢复渲染）
    - Console   控制台消息 + 官方 print 拦截
    - Journal   metrics-<时间戳>.melog 二进制日志落盘
    - MediaLog  图像 / 音频记录流程
    - Mirror    console.log 镜像 + stdio 接管
    - WebServer 实时面板

    用法：
        >>> import melog
        >>> melog.init(log_dir="runs/demo")
        >>> from melog import StepsBar
        >>> for step in StepsBar(range(100)):
        ...     melog.scalar({"loss": loss})
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
            flush_every: 每 N 次 log 落盘一次（二进制日志按批成块）。
            max_plot_points: Web 历史曲线单指标最大点数，超出自动降采样；
                日志文件始终保留全量数据。
        """
        self.project = project
        self.reduce_op = reduce_op
        self._enable_progress = enable_progress
        self._rank = get_rank()
        self._is_primary = self._rank == 0
        self._axis = Axis()  # 全局 x / epoch 计数与定位的唯一所有者
        self._closed = False
        self._lock = threading.Lock()
        self._colors: Dict[str, str] = {}  # 用户指定的指标颜色（名称 -> CSS 颜色）

        self._run_dir = self._prepare_run_dir(output_dir)
        self._log_file = self._next_log_file()

        self.store = MetricStore()
        self.media = MediaStore()
        self._journal = Journal(self._log_file, self.store, flush_every=flush_every)
        self._media_log = MediaLog(self)
        if self._is_primary:
            self._restore_history()  # 断点续训：回灌同目录历史日志
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
            if self._colors:  # 历史日志自带配色时一并恢复
                self._web.set_colors(dict(self._colors))

        self._bars = BarStack()  # 打开中的进度条栈（栈顶 = 当前环境）
        self._console = Console()  # 控制台消息 + print 拦截
        # 控制台日志镜像（仅 rank0）：进度条与 print 同步写入 console.log
        self.mirror: Optional[Mirror] = None
        if self._is_primary:
            self.mirror = Mirror(self._run_dir / "console.log")
            self.mirror.hook_stdio()
            # 拦截官方 print：用户代码里的 print(...) 内部改走 console.log()
            self._console.patch_print()
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
        # log_dir 即 run 目录（不再自动套时间戳子目录）：重跑同一目录
        # 即断点续训，历史日志被回灌、重叠区被截断覆盖
        run_dir = root / self.project
        # 多进程时仅 rank0 创建目录，其余进程等待
        if self._is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "rank.txt").write_text(str(self._rank), encoding="utf-8")
        return run_dir

    def _next_log_file(self) -> Path:
        """本次会话的日志文件：带启动时间戳，多次训练互不覆盖。"""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._run_dir / f"metrics-{ts}.melog"
        i = 1
        while path.exists():  # 同秒内多次初始化（如测试）时顺延
            path = self._run_dir / f"metrics-{ts}-{i}.melog"
            i += 1
        return path

    def _restore_history(self) -> None:
        """断点续训：回灌同 run 目录的历史日志（仅 rank0）。

        指标灌入内存 store（Web 面板立即显示完整曲线）、坐标轴状态
        （step / 各 epoch 基准）从日志重建，媒体索引与用户配色一并
        恢复。此后重新进入某个历史 epoch 时按其基准截断重叠区。
        """
        for f in sorted(self._run_dir.glob("metrics*.melog")):
            if f == self._log_file:
                continue  # 跳过本次会话自己的（空）文件
            reader = MelogFileReader(f)
            for step, epoch, values in reader.records():
                self.store.add(step, values, epoch, persist=False)  # 已落盘，仅回灌展示
                self._axis.absorb(step, epoch)
            for rec in reader.media():
                kind, name, step = rec.get("type"), rec.get("metric"), rec.get("step")
                if isinstance(kind, str) and isinstance(name, str) and isinstance(step, int):
                    self.media.add(kind, name, step, rec.get("file", ""),
                                   rec.get("epoch") if isinstance(rec.get("epoch"), int) else None,
                                   sr=rec.get("sr") if isinstance(rec.get("sr"), int) else None,
                                   caption=rec.get("caption") if isinstance(rec.get("caption"), str) else None)
        try:  # 恢复上次运行的用户配色（如有）
            data = json.loads((self._run_dir / "colors.json").read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._colors.update(data)
        except (OSError, ValueError):
            pass

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def web_url(self) -> Optional[str]:
        """Web 面板地址（未启用 Web 时为 None）。"""
        return self._web.url if self._web is not None else None

    # ------------------------------------------------------------------ 进度条栈
    def current_bar(self) -> Optional[tqdm]:
        """当前栈顶进度条（无打开的 bar 时为 None）。

        StepsBar 允许嵌套、以栈管理：scalar() 的 postfix 与 advance
        自动作用于栈顶；深层函数需要手动推进、读数或写
        postfix 时用它获取当前 bar，免层层传参。
        """
        return self._bars.top()

    # ------------------------------------------------------------------ 记录指标
    def scalar(
        self,
        metrics: Union[Dict[str, Union[float, int, Any]], MetricGroup],
        advance: int = 0,
    ) -> Dict[str, float]:
        """记录一批指标；坐标（epoch/step）由 StepsBar 自动管理。

        多 GPU 场景下先做 all_reduce 合并（默认取均值），再由 rank0
        持久化、推送到 Web、刷新进度条。

        坐标规则：epoch 由 StepsBar(epoch=...) 绑定，step 取 epoch 内
        下一个空槽（内部自增，全局 x 跨 epoch 连续接续）；未用 StepsBar
        时退化为全局自增。要控制记录粒度（每步 / 每 N 步窗口），调整
        调用 scalar() 的频率即可，无需也无法手动指定坐标。

        Args:
            metrics: 指标名 -> 数值（float / int / 0 维 tensor）；
                或直接传 MetricGroup——跨 GPU 同步合并由内部完成，
                返回值在各 rank 上一致（开启新一轮统计需再调 metrics.reset()）。
            advance: 额外推进进度条的步数（StepsBar 迭代每次已自动
                推进 1，缺省 0；仅一个迭代内多次 scalar() 等场景需要传入）。
        Returns:
            合并后的指标（rank>0 也返回，便于本地打印）。
        """
        if isinstance(metrics, MetricGroup):
            metrics = metrics._compute()
        merged = reduce_metrics(metrics, op=self.reduce_op)

        if not self._is_primary:
            return merged

        with self._lock:
            x, out_epoch = self._axis.resolve_commit()
            self._journal.add(x, merged, out_epoch)
            self._push_web(x, merged, out_epoch)
            self._bars.update_top(merged)
            if advance:
                self._bars.advance_top(advance)
            self._axis.commit(x, out_epoch)
        return merged

    # 兼容 wandb 风格别名
    log_metrics = scalar

    def _record_local(self, values: Dict[str, Any]) -> None:
        """记录本卡本地指标值（零通信；StepsBar 挂接的指标组 feed 时实时调用）。

        与 scalar() 的区别：不做跨 rank 合并，坐标规则相同（依附当前
        绑定的 epoch 与下一个空槽）。仅 rank0 落盘，其余 rank 直接返回。
        """
        if not self._is_primary:
            return
        with self._lock:
            x, out_epoch = self._axis.resolve_commit()
            self._journal.add(x, values, out_epoch)
            self._push_web(x, values, out_epoch)
            self._axis.commit(x, out_epoch)

    # ------------------------------------------------------------------ 断点续训
    def _bind_epoch(self, epoch: int) -> None:
        """StepsBar 绑定 epoch 的统一入口（各 rank 都调用）。

        rank0 上若该 epoch 在历史日志中已有记录（中断残留的重叠区），
        先物理截断 x >= 该 epoch 基准的旧记录并回滚坐标轴，再正常绑定；
        其余 rank 坐标不参与落盘，直接绑定。
        """
        with self._lock:
            if self._is_primary:
                cut = self._axis.cut_on_rebind(epoch)
                if cut is not None:
                    self._apply_truncation(cut, epoch)
            self._axis.bind_epoch(epoch)

    def _apply_truncation(self, cut: int, epoch: int) -> None:
        """截断 x >= cut 的历史记录（调用方需持有锁且已判定需要截断）。

        中断残留总在最近一个历史会话文件的尾部，物理截断之（本会话
        文件若已产生越界记录也一并处理）；内存历史、坐标轴同步回滚，
        并向已连接的面板广播全量历史（前端整体替换，不会出现 x 轴
        回退的折线）。
        """
        self._journal.flush()
        last = (None, None)
        prev_file = self._last_history_file()
        if prev_file is not None:
            last = MelogFile.truncate(prev_file, cut)
        if last == (None, None):
            last = self._journal.truncate_from(cut)
        self.store.truncate(cut)
        self._axis.rollback(cut, last)
        self.warn(f"epoch {epoch} 存在历史记录，截断 x >= {cut} 的重叠数据后继续")
        if self._web is not None:
            self._web.broadcast_history()

    def _last_history_file(self) -> Optional[Path]:
        """最近一个历史会话文件（不含本会话正在写的）；无则 None。"""
        files = [f for f in LogLoader.session_files(self._run_dir)
                 if f != self._log_file]
        return files[-1] if files else None

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

        位置自动附着到最近一次 scalar() 的记录处，不推进计数。
        """
        self._media_log.image(name, data, caption=caption)

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

        位置自动附着到最近一次 scalar() 的记录处，不推进计数。
        """
        self._media_log.audio(name, data, sr=sr, caption=caption)

    def _log_group(
        self,
        group: MetricGroup,
        advance: int = 0,
        reset: bool = False,
    ) -> Dict[str, float]:
        """合并记录一组 Metric 指标（内部方法，供 StepsBar epoch 末自动调用）。

        用户侧等价写法::

            melog.scalar(group)   # 跨 GPU 同步合并由 scalar 内部完成
            group.reset()         # 需要开启新一轮统计时

        所有 rank 都应调用；仅 rank0 持久化与展示。坐标自动依附
        当前绑定的 epoch（StepsBar）与下一个空槽。

        Args:
            group: MetricGroup 实例。
            advance: 额外推进进度条的步数，epoch 级记录默认不推进。
            reset: 记录后是否重置组内指标（开启新一轮 epoch 统计）。
        """
        result = self.scalar(group, advance=advance)
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
        self._console.log(*values, sep=sep, end=end, flush=flush)

    def success(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """绿色文字 + ✔ 前缀。"""
        self._console.success(*values, sep=sep, end=end, flush=flush)

    def error(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """红色文字 + ✘ 前缀。"""
        self._console.error(*values, sep=sep, end=end, flush=flush)

    def warn(self, *values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """黄色文字 + ⚠ 前缀。"""
        self._console.warn(*values, sep=sep, end=end, flush=flush)

    def _push_web(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        if self._web is not None:
            self._web.publish(step, metrics, epoch=epoch)

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
            self._journal.flush()
            self._journal.close()
        self._bars.close_all()  # 定稿所有打开中的进度条（栈顶先关，逐层恢复渲染）
        if self.mirror is not None:
            self._console.unpatch_print()
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
