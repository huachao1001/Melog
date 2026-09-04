"""API 路由：静态页面 + 指标/媒体/文件浏览/加载接口 + WebSocket。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from .fs import FileBrowser
    from .loader import LogLoader, MediaLoader
    from .media_view import MediaView
    from .view import MetricView
    from .ws import WsHub

STATIC_DIR = Path(__file__).parent / "static"


class ApiRoutes:
    """把各功能组件封装成 FastAPI 路由。"""

    def __init__(
        self,
        view: "MetricView",
        hub: "WsHub",
        browser: "FileBrowser",
        loader: "LogLoader",
        media_loader: Optional["MediaLoader"] = None,
        media_view: Optional["MediaView"] = None,
        default_dir: str = "",
    ):
        self.view = view
        self.hub = hub
        self.browser = browser
        self.loader = loader
        self.media_loader = media_loader
        self.media_view = media_view
        self.initial_default = default_dir  # 初始展示日志所在目录
        self.default_dir = default_dir      # 文件浏览器默认打开目录（随当前展示日志联动）

    # ------------------------------------------------------------------ 注册
    def register(self, app: FastAPI) -> None:
        app.get("/", response_class=HTMLResponse)(self.index)
        app.get("/api/metrics")(self.api_metrics)
        app.get("/api/media")(self.api_media)
        app.get("/api/media/file")(self.api_media_file)
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

    # ------------------------------------------------------------------ 媒体
    async def api_media(self):
        return {"media": self.media_view.snapshot() if self.media_view else {}}

    async def api_media_file(self, path: str = ""):
        """回源媒体文件：仅允许当前视图媒体目录内的白名单路径。"""
        if self.media_view is None:
            return JSONResponse(status_code=404, content={"error": "媒体服务不可用"})
        file = self.media_view.resolve(path)
        if file is None:
            return JSONResponse(status_code=404, content={"error": f"媒体不存在: {path}"})
        return FileResponse(file)

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
        files = self.loader.session_files(path)  # 会话文件按时间戳合并成完整曲线
        if not files:
            return JSONResponse(status_code=400, content={"error": f"目录下未找到 .melog 日志: {path}"})
        series = self.loader.parse(files)
        if not series:
            return JSONResponse(status_code=400, content={"error": "文件中没有可解析的指标"})
        run_dir = files[0].parent
        colors = self._read_colors(run_dir)  # 该日志运行时的用户指定颜色（如有）
        categories = self.loader.categories(files)  # 日志中声明的大类别
        media = self._read_media(files)
        self.view.set_loaded(series, colors, categories)
        if self.media_view is not None:
            self.media_view.set_loaded(media, run_dir)
        self.default_dir = str(run_dir)  # 默认浏览目录跟随当前展示日志
        await self.hub.broadcast({"type": "history", "metrics": self.view.snapshot(),
                                  "categories": sorted(self.view.categories)})
        await self.hub.broadcast({"type": "colors", "colors": colors})
        await self._broadcast_media_history()
        return {"ok": True, "count": len(series), "path": str(files[0])}

    async def api_unload(self):
        """切回当前实时运行视图。"""
        self.view.clear_loaded()
        if self.media_view is not None:
            self.media_view.clear_loaded()
        self.default_dir = self.initial_default
        await self.hub.broadcast({"type": "history", "metrics": self.view.snapshot(),
                                  "categories": sorted(self.view.categories)})
        await self.hub.broadcast({"type": "colors", "colors": self.view.colors})
        await self._broadcast_media_history()
        return {"ok": True}

    def _read_media(self, files: list) -> dict:
        """解析日志中的媒体记录；无 MediaLoader 或解析失败时返回空。"""
        if self.media_loader is None or not files:
            return {}
        try:
            return self.media_loader.parse(files)
        except OSError:
            return {}

    async def _broadcast_media_history(self):
        if self.media_view is None:
            return
        await self.hub.broadcast({"type": "media_history", "media": self.media_view.snapshot()})

    @staticmethod
    def _read_colors(run_dir: Path) -> dict:
        """读取 run 目录的 colors.json（用户指定颜色），缺失或损坏时返回空。"""
        path = run_dir / "colors.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    # ------------------------------------------------------------------ WebSocket
    async def ws_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        self.hub.attach(websocket)
        try:
            # 建连即补发全量历史（降采样后），避免前端漏数据；随后下发类别、颜色与媒体索引
            await websocket.send_json({"type": "history", "metrics": self.view.snapshot(),
                                       "categories": sorted(self.view.categories)})
            await websocket.send_json({"type": "colors", "colors": self.view.colors})
            if self.media_view is not None:
                await websocket.send_json({"type": "media_history", "media": self.media_view.snapshot()})
            while True:
                await websocket.receive_text()  # 保持连接，忽略客户端消息
        except WebSocketDisconnect:
            pass
        finally:
            self.hub.detach(websocket)
