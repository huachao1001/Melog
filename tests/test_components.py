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
    """二进制日志解析：乱序 step 排序；无 epoch 记 None；坏文件返回空。"""
    from melog.storage.melog_file import MelogFile

    f = MelogFile(tmp_path / "metrics-1.melog")
    f.add_batch([{"metric": "loss", "step": 1, "value": 0.5}])
    f.add_batch([{"metric": "loss", "step": 0, "value": 1.0}])
    f.close()
    (tmp_path / "broken.melog").write_bytes(b"\xc0\x01garbage")  # 非 melog 格式

    series = LogLoader.parse(tmp_path)  # 目录：合并全部会话文件
    assert series["loss"] == [(0, 1.0, None), (1, 0.5, None)]  # 按 step 排序
    assert LogLoader.parse(tmp_path / "broken.melog") == {}


def test_loader_merges_session_files(tmp_path):
    """同目录多个时间戳会话文件按文件名序合并为完整时间线。"""
    from melog.storage.melog_file import MelogFile

    f1 = MelogFile(tmp_path / "metrics-20260903_101010.melog")
    f1.add_batch([{"metric": "loss", "step": 0, "value": 1.0, "epoch": 0}])
    f1.close()
    f2 = MelogFile(tmp_path / "metrics-20260903_102020.melog")
    f2.add_batch([{"metric": "loss", "step": 1, "value": 0.5, "epoch": 1}])
    f2.close()

    series = LogLoader.parse(tmp_path)
    assert series["loss"] == [(0, 1.0, 0), (1, 0.5, 1)]
    # 指定任一会话文件也合并同目录全部会话
    assert LogLoader.parse(f2._path)["loss"] == [(0, 1.0, 0), (1, 0.5, 1)]


# ---------------------------------------------------------------- MetricView
def test_view_switches_between_realtime_and_loaded():
    store = MetricStore()
    view = MetricView(store, max_points=100)
    store.add(0, {"live": 1.0})
    assert view.snapshot()["live"] == [{"step": 0, "value": 1.0}]

    view.set_loaded({"hist": [(5, 0.5, None)]})
    assert view.snapshot() == {"hist": [{"step": 5, "value": 0.5}]}

    view.clear_loaded()
    assert "live" in view.snapshot()


def test_view_colors_realtime_vs_loaded():
    """颜色随视图切换：历史日志自带颜色优先，卸载后回到实时颜色。"""
    store = MetricStore()
    view = MetricView(store, max_points=100)
    assert view.colors == {}

    view.set_colors({"loss": "#ef4444"})
    assert view.colors == {"loss": "#ef4444"}

    view.set_loaded({"hist": [(0, 1.0, None)]}, colors={"acc": "#3b82f6"})
    assert view.colors == {"acc": "#3b82f6"}

    view.set_loaded({"hist2": [(0, 2.0, None)]})  # 无 colors.json 的历史日志
    assert view.colors == {}

    view.clear_loaded()
    assert view.colors == {"loss": "#ef4444"}


# ---------------------------------------------------------------- WsHub
def test_hub_publish_without_clients_or_loop():
    hub = WsHub()
    hub.publish({"x": 1})  # 无客户端/无 loop 时不抛异常
