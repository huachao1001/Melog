"""API 路由：静态页面 + 指标/文件浏览/加载接口 + WebSocket。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from .fs import FileBrowser
    from .loader import LogLoader
    from .view import MetricView
    from .ws import WsHub

STATIC_DIR = Path(__file__).parent / "static"


class ApiRoutes:
    """把各功能组件封装成 FastAPI 路由。"""

    def __init__(self, view: "MetricView", hub: "WsHub", browser: "FileBrowser", loader: "LogLoader", default_dir: str = ""):
        self.view = view
        self.hub = hub
        self.browser = browser
        self.loader = loader
        self.initial_default = default_dir  # 初始展示日志所在目录
        self.default_dir = default_dir      # 文件浏览器默认打开目录（随当前展示日志联动）

    # ------------------------------------------------------------------ 注册
    def register(self, app: FastAPI) -> None:
        app.get("/", response_class=HTMLResponse)(self.index)
        app.get("/api/metrics")(self.api_metrics)
        app.get("/api/fs")(self.api_fs)
        app.post("/api/load")(self.api_load)
        app.post("/api/unload")(self.api_unload)
        app.websocket("/ws")(self.ws_endpoint)

    # ------------------------------------------------------------------ 页面
    async def index(self):
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ 指标
    async def api_metrics(self):
        return {"project": "run", "metrics": self.view.snapshot()}

    # ------------------------------------------------------------------ 文件浏览
    async def api_fs(self, path: str = ""):
        if not path:
            return {"roots": self.browser.list_roots(), "default": self.default_dir}
        try:
            return self.browser.list_dir(path)
        except FileNotFoundError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except (NotADirectoryError, PermissionError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    # ------------------------------------------------------------------ 加载/卸载
    async def api_load(self, request: Request):
        body = await request.json()
        path = Path(body.get("path", ""))
        if not path.exists():
            return JSONResponse(status_code=400, content={"error": f"路径不存在: {path}"})
        try:
            log_file = self.browser.find_latest_log(path) if path.is_dir() else path
        except FileNotFoundError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        series = self.loader.parse(log_file)
        if not series:
            return JSONResponse(status_code=400, content={"error": "文件中没有可解析的指标"})
        self.view.set_loaded(series)
        self.default_dir = str(log_file.parent)  # 默认浏览目录跟随当前展示日志
        await self.hub.broadcast({"type": "history", "metrics": self.view.snapshot()})
        return {"ok": True, "count": len(series), "path": str(log_file)}

    async def api_unload(self):
        """切回当前实时运行视图。"""
        self.view.clear_loaded()
        self.default_dir = self.initial_default
        await self.hub.broadcast({"type": "history", "metrics": self.view.snapshot()})
        return {"ok": True}

    # ------------------------------------------------------------------ WebSocket
    async def ws_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        self.hub.attach(websocket)
        try:
            # 建连即补发全量历史（降采样后），避免前端漏数据
            await websocket.send_json({"type": "history", "metrics": self.view.snapshot()})
            while True:
                await websocket.receive_text()  # 保持连接，忽略客户端消息
        except WebSocketDisconnect:
            pass
        finally:
            self.hub.detach(websocket)
