"""Web 服务：uvicorn 线程生命周期管理。"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .app import STATIC_DIR, ApiRoutes
from .fs import FileBrowser
from .loader import LogLoader
from .store import MetricStore
from .view import MetricView
from .ws import WsHub

logger = logging.getLogger("melog.web")


class WebServer:
    """组装各功能组件并在后台线程运行 uvicorn。

    组件划分：
    - MetricStore: 实时指标历史（与 core 共享）
    - MetricView:  展示视图（实时/历史切换）
    - WsHub:       WebSocket 客户端与广播
    - FileBrowser: 文件系统浏览
    - LogLoader:   JSONL 解析
    - ApiRoutes:   路由注册
    """

    def __init__(self, store: MetricStore, host: str = "127.0.0.1", port: int = 8666, max_points: int = 2000):
        self.store = store
        self.host = host
        self.port = port
        self.view = MetricView(store, max_points)
        self.hub = WsHub()
        self.routes = ApiRoutes(self.view, self.hub, FileBrowser, LogLoader)
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[object] = None
        self._started = threading.Event()
        self._build_app()

    # ------------------------------------------------------------------ App
    def _build_app(self) -> None:
        app = FastAPI(title="Melog", docs_url=None, redoc_url=None)
        self.routes.register(app)

        # 静态资源禁用强缓存：JS/CSS 更新后浏览器立即生效（仍走 ETag 304 协商缓存）
        @app.middleware("http")
        async def no_cache_static(request: Request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        self.app = app

    # ------------------------------------------------------------------ 推送
    def publish(self, step: int, metrics: Dict[str, float]) -> None:
        """线程安全：从训练主线程向所有 WebSocket 客户端广播增量指标。"""
        self.hub.publish({"type": "update", "step": step, "metrics": metrics})

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, name="melog-web", daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)
        logger.info("Melog Web 已启动: http://%s:%d", self.host, self.port)

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
            logger.warning("Melog Web 服务异常退出: %s", e)

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
