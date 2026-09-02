"""Web 服务接口测试（进程内，不占真实端口）。"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from melog.web.server import WebServer
from melog.web.store import MetricStore


@pytest.fixture
def app_client():
    server = WebServer(store=MetricStore(), host="127.0.0.1", port=8999)
    with TestClient(server.app) as client:
        yield client, server


def test_index_html_offline(app_client):
    client, _ = app_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Melog" in resp.text
    # 不允许任何在线 CDN 引用
    assert "cdn.jsdelivr.net" not in resp.text
    assert "http://" not in resp.text.replace('location.protocol', '').replace('location.host', '')


def test_echarts_served_locally(app_client):
    client, _ = app_client
    resp = client.get("/static/echarts.min.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert len(resp.content) > 500_000  # 完整的 echarts.min.js
    assert b"echarts" in resp.content


def test_api_metrics_empty(app_client):
    client, _ = app_client
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    assert resp.json() == {"project": "run", "metrics": {}}


def test_api_metrics_snapshot(app_client):
    client, server = app_client
    server.store.add(0, {"loss": 1.0})
    resp = client.get("/api/metrics")
    assert resp.json()["metrics"]["loss"] == [{"step": 0, "value": 1.0}]


def test_websocket_history(app_client):
    client, server = app_client
    server.store.add(3, {"loss": 0.5})
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "history"
        assert msg["metrics"]["loss"] == [{"step": 3, "value": 0.5}]


def test_websocket_sends_colors(app_client):
    """建连先发 history 再发 colors（可为空 dict）。"""
    client, server = app_client
    server.set_colors({"loss": "#ef4444"})
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "history"
        msg = ws.receive_json()
        assert msg["type"] == "colors"
        assert msg["colors"]["loss"] == "#ef4444"


def test_set_colors_updates_view(app_client):
    """WebServer.set_colors 为整体替换；增量合并在 Melog.set_colors 层（见 test_core）。"""
    client, server = app_client
    server.set_colors({"recall/class_0": "#f00"})
    assert server.view.colors == {"recall/class_0": "#f00"}
    server.set_colors({"recall/class_0": "#f00", "loss": "steelblue"})
    assert server.view.colors == {"recall/class_0": "#f00", "loss": "steelblue"}


def test_load_restores_colors_sidecar(tmp_path):
    """加载历史日志时恢复其 colors.json，卸载后回到实时颜色。"""
    import json

    log = tmp_path / "metrics.melog"
    log.write_text('{"metric": "loss", "step": 0, "value": 1.0}\n', encoding="utf-8")
    (tmp_path / "colors.json").write_text(
        json.dumps({"loss": "#123456"}), encoding="utf-8"
    )
    server = WebServer(store=MetricStore(), port=8990, log_file=str(tmp_path / "live" / "metrics.melog"))
    client = TestClient(server.app)
    server.set_colors({"loss": "#abcdef"})  # 实时颜色

    assert client.post("/api/load", json={"path": str(log)}).json()["ok"] is True
    assert server.view.colors == {"loss": "#123456"}

    client.post("/api/unload")
    assert server.view.colors == {"loss": "#abcdef"}


def test_publish_to_websocket(app_client):
    client, server = app_client
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # history
        ws.receive_json()  # colors
        # 模拟后台线程推送
        time.sleep(0.1)
        import asyncio

        payload = {"type": "update", "step": 1, "metrics": {"loss": 0.9}}
        asyncio.run(server.hub.broadcast(payload))
        msg = ws.receive_json()
        assert msg["type"] == "update"
        assert msg["metrics"]["loss"] == 0.9


# ---------------------------------------------------------------- 文件浏览
def test_fs_roots():
    import sys

    server = WebServer(store=MetricStore(), port=8998)
    client = TestClient(server.app)
    info = client.get("/api/fs").json()
    # 任何平台都应有可浏览的根：Windows 盘符（X:/），POSIX 为 "/"
    assert info["roots"]
    if sys.platform != "win32":
        assert info["roots"] == ["/"]


def test_fs_root_dir_posix():
    """POSIX 根目录 "/" 可列举，且没有上一级（此前面包屑在此崩溃）。"""
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX 专属")
    server = WebServer(store=MetricStore(), port=8992)
    client = TestClient(server.app)
    info = client.get("/api/fs", params={"path": "/"}).json()
    assert info["path"] == "/"
    assert info["parent"] is None
    assert isinstance(info["dirs"], list)


def test_fs_listing(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "metrics.melog").write_text('{"metric": "loss", "step": 0, "value": 1.0}\n', encoding="utf-8")
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")  # 非 .melog 不列出
    server = WebServer(store=MetricStore(), port=8997)
    client = TestClient(server.app)
    info = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert info["dirs"] == ["sub"]
    assert info["files"] == ["metrics.melog"]
    assert info["parent"] is not None


def test_fs_missing_path():
    server = WebServer(store=MetricStore(), port=8996)
    client = TestClient(server.app)
    assert client.get("/api/fs", params={"path": "Z:/no/such"}).status_code == 400


def test_load_missing_file():
    server = WebServer(store=MetricStore(), port=8995)
    client = TestClient(server.app)
    resp = client.post("/api/load", json={"path": "Z:/not/exist.jsonl"})
    assert resp.status_code == 400


def test_load_and_unload_jsonl(tmp_path):
    jl = tmp_path / "metrics.melog"
    jl.write_text(
        '{"metric": "loss", "step": 0, "value": 1.0}\n'
        '{"metric": "loss", "step": 1, "value": 0.5}\n'
        "坏行不应崩溃\n",
        encoding="utf-8",
    )
    server = WebServer(store=MetricStore(), port=8991)
    client = TestClient(server.app)

    resp = client.post("/api/load", json={"path": str(jl)})
    assert resp.json()["ok"] is True
    assert client.get("/api/metrics").json()["metrics"]["loss"] == [
        {"step": 0, "value": 1.0},
        {"step": 1, "value": 0.5},
    ]

    client.post("/api/unload")
    assert client.get("/api/metrics").json()["metrics"] == {}


def test_load_dir_picks_latest_jsonl(tmp_path):
    old = tmp_path / "a" / "metrics.melog"
    old.parent.mkdir(parents=True)
    old.write_text('{"metric": "acc", "step": 0, "value": 0.5}\n', encoding="utf-8")
    new = tmp_path / "b" / "metrics.melog"
    new.parent.mkdir(parents=True)
    new.write_text('{"metric": "acc", "step": 0, "value": 0.9}\n', encoding="utf-8")
    import os

    os.utime(old, (1, 1))  # old 更旧，应选中 new

    server = WebServer(store=MetricStore(), port=8994)
    client = TestClient(server.app)
    body = client.post("/api/load", json={"path": str(tmp_path)}).json()
    assert body["ok"] is True
    assert body["path"] == str(new)
    assert client.get("/api/metrics").json()["metrics"]["acc"][0]["value"] == 0.9


def test_load_dir_without_jsonl(tmp_path):
    server = WebServer(store=MetricStore(), port=8993)
    client = TestClient(server.app)
    assert client.post("/api/load", json={"path": str(tmp_path)}).status_code == 400
