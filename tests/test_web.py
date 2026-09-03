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


def test_store_snapshot_carries_epoch():
    store = MetricStore()
    store.add(0, {"loss": 1.0}, epoch=2)
    store.add(1, {"loss": 0.5})  # 未传 epoch 的点不带该字段
    snap = store.snapshot()
    assert snap["loss"] == [
        {"step": 0, "value": 1.0, "epoch": 2},
        {"step": 1, "value": 0.5},
    ]


def test_loader_parses_epoch(tmp_path):
    from melog.web.loader import LogLoader

    log = tmp_path / "metrics.melog"
    log.write_text(
        '{"metric": "loss", "step": 0, "value": 1.0, "epoch": 0}\n'
        '{"metric": "loss", "step": 5, "value": 0.5, "epoch": 1}\n'
        '{"metric": "loss", "step": 6, "value": 0.4}\n'  # 旧格式无 epoch
        "坏行跳过\n",
        encoding="utf-8",
    )
    series = LogLoader.parse(log)
    assert series["loss"] == [(0, 1.0, 0), (5, 0.5, 1), (6, 0.4, None)]


def test_publish_carries_epoch(app_client):
    client, server = app_client
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # history
        ws.receive_json()  # colors
        ws.receive_json()  # media_history
        time.sleep(0.1)
        import asyncio

        asyncio.run(server.hub.broadcast({"type": "update", "step": 3, "metrics": {"loss": 0.9}, "epoch": 1}))
        msg = ws.receive_json()
        assert msg["type"] == "update"
        assert msg["epoch"] == 1


def test_websocket_history(app_client):
    client, server = app_client
    server.store.add(3, {"loss": 0.5})
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "history"
        assert msg["metrics"]["loss"] == [{"step": 3, "value": 0.5}]
        assert ws.receive_json()["type"] == "colors"
        assert ws.receive_json()["type"] == "media_history"


def test_websocket_sends_colors(app_client):
    """建连先发 history 再发 colors（可为空 dict），随后 media_history。"""
    client, server = app_client
    server.set_colors({"loss": "#ef4444"})
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "history"
        msg = ws.receive_json()
        assert msg["type"] == "colors"
        assert msg["colors"]["loss"] == "#ef4444"
        assert ws.receive_json()["type"] == "media_history"


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
        ws.receive_json()  # media_history
        # 模拟后台线程推送
        time.sleep(0.1)
        import asyncio

        payload = {"type": "update", "step": 1, "metrics": {"loss": 0.9}}
        asyncio.run(server.hub.broadcast(payload))
        msg = ws.receive_json()
        assert msg["type"] == "update"
        assert msg["metrics"]["loss"] == 0.9


# ---------------------------------------------------------------- 媒体接口
def test_ws_sends_media_history(app_client):
    client, server = app_client
    server.media_store.add("image", "pred", 0, "media/image/pred/000000000.png", epoch=1)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "history"
        assert ws.receive_json()["type"] == "colors"
        msg = ws.receive_json()
        assert msg["type"] == "media_history"
        entry = msg["media"]["image"]["pred"][0]
        assert entry["step"] == 0 and entry["epoch"] == 1
        assert entry["url"].startswith("/api/media/file?path=")


def test_api_media_snapshot(app_client):
    client, server = app_client
    assert client.get("/api/media").json() == {"media": {"image": {}, "audio": {}}}
    server.media_store.add("audio", "tone", 2, "media/audio/tone/000000002.wav", epoch=0)
    media = client.get("/api/media").json()["media"]
    assert media["audio"]["tone"][0]["step"] == 2
    assert media["audio"]["tone"][0]["url"].startswith("/api/media/file?path=")


def test_publish_media_payload(app_client):
    client, server = app_client
    sent = []
    server.hub.publish = sent.append
    server.publish_media("image", "pred", 3, 1, "media/image/pred/000000003.png",
                         caption="epoch 1 的样本")
    server.publish_media("audio", "tone", 4, None, "media/audio/tone/000000004.wav", sr=8000)
    assert sent[0] == {
        "type": "image", "name": "pred", "step": 3, "epoch": 1,
        "caption": "epoch 1 的样本",
        "url": "/api/media/file?path=media/image/pred/000000003.png",
    }
    assert sent[1]["type"] == "audio" and sent[1]["sr"] == 8000
    assert "epoch" not in sent[1] and "caption" not in sent[1]


def test_api_media_file_whitelist(tmp_path):
    pytest.importorskip("PIL")
    run = tmp_path / "run"
    run.mkdir()
    img_dir = run / "media/image/pred"
    img_dir.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (4, 4), (255, 0, 0)).save(img_dir / "000000000.png")

    server = WebServer(store=MetricStore(), port=8989, log_file=str(run / "metrics.melog"))
    client = TestClient(server.app)

    resp = client.get("/api/media/file", params={"path": "media/image/pred/000000000.png"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("image/")

    # 穿越与未知路径一律 404
    assert client.get("/api/media/file", params={"path": "../../etc/passwd"}).status_code == 404
    assert client.get("/api/media/file", params={"path": "media/nope.png"}).status_code == 404
    assert client.get("/api/media/file", params={"path": ""}).status_code == 404


def test_load_broadcasts_media_history(tmp_path):
    """加载历史日志后，WS 收到的 media_history 指向该日志的媒体。"""
    pytest.importorskip("PIL")
    import asyncio

    run = tmp_path / "old_run"
    (run / "media/image/hist").mkdir(parents=True)
    from PIL import Image

    Image.new("L", (4, 4)).save(run / "media/image/hist/000000003.png")
    log = run / "metrics.melog"
    log.write_text(
        '{"metric": "loss", "step": 0, "value": 1.0}\n'
        '{"type": "image", "metric": "hist", "step": 3, "file": "media/image/hist/000000003.png"}\n',
        encoding="utf-8",
    )
    server = WebServer(store=MetricStore(), port=8988, log_file=str(tmp_path / "live" / "metrics.melog"))
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json(); ws.receive_json(); ws.receive_json()  # history/colors/media_history
        assert client.post("/api/load", json={"path": str(log)}).json()["ok"] is True
        time.sleep(0.1)
        msgs = [ws.receive_json() for _ in range(3)]
        media_msg = next(m for m in msgs if m["type"] == "media_history")
        entry = media_msg["media"]["image"]["hist"][0]
        assert entry["step"] == 3

        # 回源到历史日志的媒体文件
        from urllib.parse import unquote, urlparse, parse_qs

        rel = parse_qs(urlparse(entry["url"]).query)["path"][0]
        file_resp = client.get("/api/media/file", params={"path": rel})
        assert file_resp.status_code == 200

        client.post("/api/unload")
        time.sleep(0.1)
        msgs = [ws.receive_json() for _ in range(3)]
        media_msg = next(m for m in msgs if m["type"] == "media_history")
        assert media_msg["media"]["image"] == {}  # 实时视图无媒体


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
