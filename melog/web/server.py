"""Web 服务：uvicorn 线程生命周期管理。"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .app import STATIC_DIR, ApiRoutes
from .fs import FileBrowser
from .loader import LogLoader, MediaLoader
from .media_store import MediaStore
from .media_view import MediaView
from .store import MetricStore
from .view import MetricView
from .ws import WsHub

_log = logging.getLogger("melog.web")


class WebServer:
    """组装各功能组件并在后台线程运行 uvicorn。

    组件划分：
    - MetricStore: 实时指标历史（与 core 共享）
    - MetricView:  展示视图（实时/历史切换）
    - MediaStore:  实时媒体索引（与 core 共享）
    - MediaView:   媒体展示视图（实时/历史切换 + 文件白名单解析）
    - WsHub:       WebSocket 客户端与广播
    - FileBrowser: 文件系统浏览
    - LogLoader / MediaLoader: 二进制日志解析（指标 / 媒体，会话合并）
    - ApiRoutes:   路由注册
    """

    def __init__(
        self,
        store: MetricStore,
        media_store: Optional[object] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        max_points: int = 2000,
        log_file: str = "",
    ):
        self.store = store
        self.host = host
        self.port = port  # None 时 start() 自动选择空闲端口
        self.view = MetricView(store, max_points)
        self.media_store = media_store if isinstance(media_store, MediaStore) else MediaStore()
        self.media_view = MediaView(self.media_store)
        # 媒体相对路径以 run_dir 为根（media/<kind>/<name>/…）；历史加载时随日志切换
        self.media_view.set_base(Path(log_file).parent if log_file else None)
        self.hub = WsHub()
        # 文件浏览器默认打开当前展示日志所在目录；无日志时退回进程 cwd
        default_dir = str(Path(log_file).parent) if log_file else str(Path.cwd())
        self.routes = ApiRoutes(
            self.view, self.hub, FileBrowser, LogLoader, MediaLoader,
            media_view=self.media_view, default_dir=default_dir,
        )
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[object] = None
        self._started = threading.Event()
        self._build_app()

    # ------------------------------------------------------------------ App
    def _build_app(self) -> None:
        app = FastAPI(title="Melog", docs_url=None, redoc_url=None)
        self.routes.register(app)

        # 静态资源与首页禁用强缓存：JS/CSS/HTML 更新后浏览器立即生效（仍走 ETag 304 协商缓存）
        @app.middleware("http")
        async def no_cache_static(request: Request, call_next):
            response = await call_next(request)
            path = request.url.path
            if path.startswith("/static") or path == "/":
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        self.app = app

    # ------------------------------------------------------------------ 推送
    def publish(self, step: int, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        """线程安全：从训练主线程向所有 WebSocket 客户端广播增量指标。"""
        payload: Dict[str, object] = {"type": "update", "step": step, "metrics": metrics}
        if epoch is not None:
            payload["epoch"] = epoch
        self.hub.publish(payload)

    def publish_categories(self, categories) -> None:
        """线程安全：广播新增的大类别（前端据此划分卡片分区）。"""
        self.view.add_categories(categories)
        self.hub.publish({"type": "categories", "categories": sorted(self.view.categories)})

    def set_categories(self, categories) -> None:
        """线程安全：设置大类别集合（历史恢复时整体替换）并广播。"""
        self.view.set_categories(categories)
        self.hub.publish({"type": "categories", "categories": sorted(self.view.categories)})

    def set_colors(self, colors: Dict[str, str]) -> None:
        """线程安全：更新用户指定颜色（指标名 -> CSS 颜色）并广播。"""
        self.view.set_colors(colors)
        self.hub.publish({"type": "colors", "colors": dict(colors)})

    def broadcast_history(self) -> None:
        """线程安全：向所有面板广播全量历史（续训截断重叠区后整体替换）。"""
        self.hub.publish({"type": "history", "metrics": self.view.snapshot(),
                          "categories": sorted(self.view.categories)})

    def publish_media(self, kind: str, name: str, step: int, epoch: Optional[int],
                      relpath: str, sr: Optional[int] = None,
                      caption: Optional[str] = None) -> None:
        """线程安全：从训练主线程向所有 WebSocket 客户端广播新媒体条目。"""
        payload: Dict[str, object] = {"type": kind, "name": name, "step": step,
                                      "url": self.media_url(relpath)}
        if epoch is not None:
            payload["epoch"] = epoch
        if sr is not None:
            payload["sr"] = sr
        if caption:
            payload["caption"] = caption
        self.hub.publish(payload)

    def media_url(self, relpath: str) -> str:
        from ..storage.media import media_url

        return media_url(relpath)

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        # 端口被占用时自动向后顺延，避免 bind 失败拖垮训练
        self.port = self._find_free_port(self.port if self.port is not None else 8666)
        self._thread = threading.Thread(target=self._serve, name="melog-web", daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)
        # 等到端口真正可连接再返回，避免并发实例的端口探测竞态
        self._wait_listening()
        _log.info("Melog Web 已启动: http://%s:%d", self.host, self.port)

    def _wait_listening(self, timeout: float = 10) -> bool:
        import socket
        import time

        probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex((probe_host, self.port)) == 0:
                    return True
            time.sleep(0.1)
        return False

    @staticmethod
    def _find_free_port(start: int, host: str = "127.0.0.1", tries: int = 20) -> int:
        """从 start 起向后找到第一个空闲端口。"""
        import socket

        for port in range(start, start + tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    return port
                except OSError:
                    continue
        return start

    def _serve(self) -> None:
        import uvicorn

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.hub.set_loop(loop)
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error", loop="asyncio")
        self._server = uvicorn.Server(config)
        self._started.set()
        try:
            loop.run_until_complete(self._server.serve())
        except Exception as e:  # 端口占用等场景不拖垮训练
            _log.warning("Melog Web 服务异常退出: %s", e)

    def stop(self) -> None:
        if self._server is not None:
            server = self._server
            self.hub.call_soon_threadsafe(lambda: setattr(server, "should_exit", True))
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
