"""Melog 核心单元测试。"""

import json
import threading

import pytest

from melog.core import Melog
from melog.distributed import reduce_metrics
from melog.web.server import MetricStore


@pytest.fixture
def logger(tmp_path):
    lg = Melog(project="test", output_dir=str(tmp_path), enable_web=False)
    yield lg
    lg.finish()


# ---------------------------------------------------------------- MetricStore
def test_store_add_and_snapshot():
    store = MetricStore()
    store.add(0, {"loss": 1.0, "acc": 0.5})
    store.add(1, {"loss": 0.5, "acc": 0.7})
    snap = store.snapshot()
    assert snap["loss"] == [{"step": 0, "value": 1.0}, {"step": 1, "value": 0.5}]
    assert snap["acc"][-1] == {"step": 1, "value": 0.7}


def test_store_thread_safety():
    store = MetricStore()

    def worker(tid):
        for i in range(100):
            store.add(i, {f"m{tid}": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    snap = store.snapshot()
    assert len(snap) == 4
    assert all(len(v) == 100 for v in snap.values())


# ---------------------------------------------------------------- reduce_metrics
def test_reduce_single_process():
    out = reduce_metrics({"loss": 1.5, "acc": 0.5})
    assert out == {"loss": 1.5, "acc": 0.5}
    assert all(isinstance(v, float) for v in out.values())


def test_reduce_empty():
    assert reduce_metrics({}) == {}


def test_reduce_invalid_type():
    with pytest.raises(TypeError):
        reduce_metrics({"bad": "text"})


# ---------------------------------------------------------------- Melog
def test_log_records_and_persists(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False, flush_every=2)
    for step in range(5):
        lg.log({"loss": 1.0 / (step + 1)}, advance=1)
    lg.finish()

    lines = (tmp_path / "t").glob("**/metrics.melog")
    path = next(lines)
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert records[0] == {"metric": "loss", "step": 0, "value": 1.0}


def test_log_returns_merged(logger):
    out = logger.log({"loss": 0.25})
    assert out == {"loss": 0.25}


def test_set_colors_merges_and_persists(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    lg.set_colors({"recall/class_0": "#ef4444"})
    lg.set_colors({"loss": "steelblue"})  # 增量合并，不覆盖前一次
    lg.finish()

    path = next((tmp_path / "t").glob("**/colors.json"))
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "recall/class_0": "#ef4444",
        "loss": "steelblue",
    }


def test_step_auto_increment(logger):
    logger.log({"a": 1})
    logger.log({"a": 2})
    assert logger.store.snapshot()["a"] == [
        {"step": 0, "value": 1.0},
        {"step": 1, "value": 2.0},
    ]


def test_explicit_step(logger):
    logger.log({"a": 1}, step=100)
    assert logger.store.snapshot()["a"][0]["step"] == 100


# ---------------------------------------------------------------- epoch 支持
def test_log_with_epoch_and_step(logger):
    """epoch + 当前 epoch 内 step：全局 x 跨 epoch 连续，记录携带 epoch。"""
    for step in range(3):
        logger.log({"a": step}, epoch=0, step=step)
    for step in range(3):
        logger.log({"a": 10 + step}, epoch=1, step=step)
    snap = logger.store.snapshot()["a"]
    assert [p["step"] for p in snap] == [0, 1, 2, 3, 4, 5]
    assert [p["epoch"] for p in snap] == [0, 0, 0, 1, 1, 1]


def test_epoch_only_internal_step_count(logger):
    """只传 epoch 不传 step：每个 epoch 内部从 0 重新计步。"""
    for _ in range(3):
        logger.log({"a": 1}, epoch=0)
    for _ in range(2):
        logger.log({"a": 2}, epoch=1)
    snap = logger.store.snapshot()["a"]
    assert [(p["step"], p["epoch"]) for p in snap] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1),
    ]


def test_epoch_sticky_after_first_call(logger):
    """epoch 粘滞：只传一次后，后续不传也记录同一 epoch。"""
    logger.log({"a": 1}, epoch=7, step=0)
    logger.log({"a": 2})
    snap = logger.store.snapshot()["a"]
    assert [(p["step"], p["epoch"]) for p in snap] == [(0, 7), (1, 7)]


def test_explicit_epoch_step_syncs_internal_counter(logger):
    """显式传 step 后再省略，内部计数从显式值接续。"""
    logger.log({"a": 1}, epoch=0, step=10)
    logger.log({"a": 2}, epoch=0)
    snap = logger.store.snapshot()["a"]
    assert [p["step"] for p in snap] == [10, 11]


def test_epoch_records_persist_with_epoch_key(logger):
    """启用 epoch 时 JSONL 记录带 epoch 字段。"""
    logger.log({"a": 1.5}, epoch=2, step=5)
    logger.finish()
    path = next(logger.run_dir.parent.glob("**/metrics.melog"))
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"metric": "a", "step": 5, "value": 1.5, "epoch": 2}


def test_no_epoch_records_omit_epoch_key(logger):
    """未启用 epoch 时 JSONL 记录不带 epoch 字段（兼容旧格式）。"""
    logger.log({"a": 1.0})
    logger.finish()
    path = next(logger.run_dir.parent.glob("**/metrics.melog"))
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"metric": "a", "step": 0, "value": 1.0}
    assert "epoch" not in rec


def test_log_group_with_epoch(logger):
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    group.update(m=3.0)
    logger.log_group(group, epoch=1, step=4)
    snap = logger.store.snapshot()["m"]
    assert snap == [{"step": 4, "value": 3.0, "epoch": 1}]


def test_train_context_returns_progress(logger):
    with logger.train(total=10) as bar:
        for _ in range(3):
            logger.log({"loss": 0.1})
            bar.advance(1)
        assert bar.completed == 3


def test_finish_idempotent(logger):
    logger.log({"a": 1})
    logger.finish()
    logger.finish()  # 不应抛异常


def test_run_dir_created(tmp_path):
    lg = Melog(project="run1", output_dir=str(tmp_path), enable_web=False)
    assert (lg.run_dir).exists()
    lg.finish()
