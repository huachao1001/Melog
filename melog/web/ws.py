"""WebSocket 客户端管理与消息广播。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Set


class WsHub:
    """维护 WebSocket 客户端集合；支持事件循环内广播与跨线程发布。"""

    def __init__(self):
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """由服务线程在启动时调用，供跨线程发布使用。"""
        self._loop = loop

    def attach(self, websocket) -> None:
        self._clients.add(websocket)

    def detach(self, websocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        """向所有客户端广播，自动剔除失效连接。"""
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def publish(self, payload: Dict[str, Any]) -> None:
        """线程安全：从训练主线程向所有客户端广播。"""
        if not self._clients or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self._loop)

    def call_soon_threadsafe(self, callback) -> None:
        """线程安全：在事件循环线程中调度回调（用于停止服务等控制操作）。"""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(callback)
