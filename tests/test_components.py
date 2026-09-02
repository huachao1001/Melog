"""Web 组件级测试：MetricStore / FileBrowser / LogLoader / MetricView / WsHub。"""

import pytest

from melog.web.fs import FileBrowser
from melog.web.loader import LogLoader
from melog.web.store import MetricStore
from melog.web.view import MetricView
from melog.web.ws import WsHub


# ---------------------------------------------------------------- MetricStore
def test_store_snapshot_with_downsample():
    store = MetricStore()
    for i in range(100):
        store.add(i, {"loss": i * 0.1})
    snap = store.snapshot(max_points=10)
    assert len(snap["loss"]) == 10
    assert len(store.snapshot()["loss"]) == 100


def test_store_drain_empties_pending():
    store = MetricStore()
    store.add(0, {"loss": 1.0})
    store.add(1, {"loss": 2.0})
    assert len(store.drain()) == 2
    assert store.drain() == []


# ---------------------------------------------------------------- FileBrowser
def test_list_dir_basic(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "run.melog").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    info = FileBrowser.list_dir(str(tmp_path))
    assert info["dirs"] == ["sub"]
    assert info["files"] == ["run.melog"]  # 仅列出 .melog 文件


def test_list_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileBrowser.list_dir(str(tmp_path / "nope"))


def test_find_latest_log(tmp_path):
    old = tmp_path / "a" / "metrics.melog"
    old.parent.mkdir(parents=True)
    old.write_text("x", encoding="utf-8")
    new = tmp_path / "b" / "metrics.melog"
    new.parent.mkdir(parents=True)
    new.write_text("x", encoding="utf-8")
    import os

    os.utime(old, (1, 1))
    assert FileBrowser.find_latest_log(tmp_path) == new


def test_find_latest_log_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileBrowser.find_latest_log(tmp_path)


# ---------------------------------------------------------------- LogLoader
def test_loader_parse(tmp_path):
    jl = tmp_path / "metrics.melog"
    jl.write_text(
        '{"metric": "loss", "step": 1, "value": 0.5}\n'
        '{"metric": "loss", "step": 0, "value": 1.0}\n'  # 乱序
        "坏行\n",
        encoding="utf-8",
    )
    series = LogLoader.parse(jl)
    assert series["loss"] == [(0, 1.0), (1, 0.5)]  # 按 step 排序


# ---------------------------------------------------------------- MetricView
def test_view_switches_between_realtime_and_loaded():
    store = MetricStore()
    view = MetricView(store, max_points=100)
    store.add(0, {"live": 1.0})
    assert view.snapshot()["live"] == [{"step": 0, "value": 1.0}]

    view.set_loaded({"hist": [(5, 0.5)]})
    assert view.snapshot() == {"hist": [{"step": 5, "value": 0.5}]}

    view.clear_loaded()
    assert "live" in view.snapshot()


# ---------------------------------------------------------------- WsHub
def test_hub_publish_without_clients_or_loop():
    hub = WsHub()
    hub.publish({"x": 1})  # 无客户端/无 loop 时不抛异常
