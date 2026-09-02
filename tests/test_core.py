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
