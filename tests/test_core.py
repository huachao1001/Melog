"""Melog 核心单元测试。"""

import builtins
import json
import sys
import threading

import pytest

from melog.core import Melog
from melog.distributed import reduce_metrics
from melog.web.server import MetricStore


@pytest.fixture
def logger(tmp_path):
    lg = Melog(project="test", output_dir=str(tmp_path), enable_web=False)
    yield lg
    lg.close()


def read_log(tmp_path) -> str:
    return (tmp_path / "test").glob("**/console.log").__next__().read_text(encoding="utf-8")


# ---------------------------------------------------------------- 控制台消息
def test_atexit_auto_close_registered(tmp_path, monkeypatch):
    """实例创建即注册 atexit 自动收尾，close 后注销且重复调用安全。"""
    import atexit

    registered = []
    monkeypatch.setattr(atexit, "register", lambda f: registered.append(f))
    lg = Melog(project="test", output_dir=str(tmp_path), enable_web=False)
    assert registered == [lg.close]
    lg.close()
    lg.close()  # 幂等


def test_console_messages(tmp_path):
    """log/success/error/warn：多参数转 str()、图标前缀、非 TTY 不着色。"""
    saved = sys.stdout
    lg = Melog(project="test", output_dir=str(tmp_path), enable_web=False)
    try:
        lg.log("hello", {"k": 2}, 3)
        lg.success("saved")
        lg.error("boom")
        lg.warn("careful")
    finally:
        lg.close()
    assert sys.stdout is saved
    text = read_log(tmp_path)
    assert "hello {'k': 2} 3\n" in text      # 多参数 + 对象 str()
    assert "✔ saved\n" in text
    assert "✘ boom\n" in text
    assert "⚠ careful\n" in text
    assert "\x1b[" not in text               # 非 TTY 无 ANSI


def test_print_intercepted_to_log(tmp_path):
    """官方 print 被拦截改走 log；close 后还原；file 指定时走原生 print。"""
    saved, orig_print = sys.stdout, builtins.print
    lg = Melog(project="test", output_dir=str(tmp_path), enable_web=False)
    try:
        assert builtins.print is not orig_print      # 已拦截
        print("via print", 123)
        print(end="")                                # end 透传
        builtins.print("direct", file=sys.stderr)    # file 指定 → 原生 print
        assert builtins.print is not orig_print
    finally:
        lg.close()
    assert builtins.print is orig_print              # 已还原
    assert sys.stdout is saved
    text = read_log(tmp_path)
    assert "via print 123\n" in text


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
def test_scalar_records_and_persists(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False, flush_every=2)
    for step in range(5):
        lg.scalar({"loss": 1.0 / (step + 1)}, advance=1)
    lg.close()

    lines = (tmp_path / "t").glob("**/metrics.melog")
    path = next(lines)
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert records[0] == {"metric": "loss", "step": 0, "value": 1.0}


def test_scalar_returns_merged(logger):
    out = logger.scalar({"loss": 0.25})
    assert out == {"loss": 0.25}


def test_set_colors_merges_and_persists(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    lg.set_colors({"recall/class_0": "#ef4444"})
    lg.set_colors({"loss": "steelblue"})  # 增量合并，不覆盖前一次
    lg.close()

    path = next((tmp_path / "t").glob("**/colors.json"))
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "recall/class_0": "#ef4444",
        "loss": "steelblue",
    }


def test_step_auto_increment(logger):
    logger.scalar({"a": 1})
    logger.scalar({"a": 2})
    assert logger.store.snapshot()["a"] == [
        {"step": 0, "value": 1.0},
        {"step": 1, "value": 2.0},
    ]


# ---------------------------------------------------------------- epoch 支持
def test_scalar_coords_managed_by_stepsbar(logger):
    """坐标由 stepsbar 统一管理：epoch 内自动计步、全局 x 跨 epoch 连续、记录携带 epoch。"""
    for e in range(2):
        for _ in logger.stepsbar(range(3), epoch=e):
            logger.scalar({"a": 1.0})
    snap = logger.store.snapshot()["a"]
    assert [(p["step"], p["epoch"]) for p in snap] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1),
    ]


def test_epoch_sticky_after_bar(logger):
    """epoch 粘滞：bar 结束后沿用绑定值，直至下一个 epoch。"""
    for _ in logger.stepsbar(range(1), epoch=7):
        logger.scalar({"a": 1})
    logger.scalar({"a": 2})  # bar 已结束：沿用绑定的 epoch
    snap = logger.store.snapshot()["a"]
    assert [(p["step"], p["epoch"]) for p in snap] == [(0, 7), (1, 7)]


def test_epoch_records_persist_with_epoch_key(logger):
    """绑定 epoch 后 JSONL 记录带 epoch 字段。"""
    for _ in logger.stepsbar(range(1), epoch=2):
        logger.scalar({"a": 1.5})
    logger.close()
    path = next(logger.run_dir.parent.glob("**/metrics.melog"))
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"metric": "a", "step": 0, "value": 1.5, "epoch": 2}


def test_no_epoch_records_omit_epoch_key(logger):
    """未启用 epoch 时 JSONL 记录不带 epoch 字段（兼容旧格式）。"""
    logger.scalar({"a": 1.0})
    logger.close()
    path = next(logger.run_dir.parent.glob("**/metrics.melog"))
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"metric": "a", "step": 0, "value": 1.0}
    assert "epoch" not in rec


def test_log_group_uses_bound_epoch(logger):
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    for _ in logger.stepsbar(range(2), epoch=1):
        group.feed(m=3.0)
    logger.log_group(group)  # 自动依附绑定的 epoch 与下一个空槽
    snap = logger.store.snapshot()["m"]
    assert snap == [{"step": 0, "value": 3.0, "epoch": 1}]


def test_stepsbar_returns_bar(logger):
    bar = logger.stepsbar(range(10))
    for _ in bar:
        logger.scalar({"loss": 0.1})
    assert bar.n == 10  # 迭代自动推进


def test_stepsbar_binds_epoch(logger):
    """stepsbar(epoch=...) 绑定当前 epoch：bar 内 scalar 免传；步数每轮清零、全局 x 接续。"""
    for e in range(2):
        for _ in logger.stepsbar(range(3), epoch=e):
            logger.scalar({"loss": 0.5})
    logger.scalar({"loss": 0.25})  # bar 结束后沿用绑定的 epoch
    recs = logger.store.snapshot()["loss"]
    assert [r["epoch"] for r in recs] == [0, 0, 0, 1, 1, 1, 1]
    assert [r["step"] for r in recs] == [0, 1, 2, 3, 4, 5, 6]


# -------------------------------------------------------------- stepsbar 自动记录
def test_stepsbar_auto_log_on_complete(logger):
    """stepsbar(metrics=...)：自然迭代结束自动 log_group；reset=True 记录后清零。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    bar = logger.stepsbar(range(3), epoch=0, metrics=group, reset=True)
    assert bar.total == 3  # len() 透传，total 自动检测不受包装影响
    for _ in bar:
        group.feed(m=2.0)
    assert logger.store.snapshot()["m"] == [{"step": 0, "value": 2.0, "epoch": 0}]
    assert group.compute() != group.compute()  # 已重置 -> NaN


def test_stepsbar_auto_log_each_epoch(logger):
    """每个 epoch 一条 bar：各自在末尾自动记录一次。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    for e in range(2):
        for i in logger.stepsbar(range(3), epoch=e, metrics=group, reset=True):
            group.feed(m=float(i + 1))
    snap = logger.store.snapshot()["m"]
    assert [(r["epoch"], r["value"]) for r in snap] == [(0, 2.0), (1, 2.0)]


def test_stepsbar_no_auto_log_on_break(logger):
    """提前 break：不触发自动记录（各 rank 迭代进度可能不一致）。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    bar = logger.stepsbar(range(10), epoch=0, metrics=group, reset=True)
    for _ in bar:
        group.feed(m=1.0)
        break
    bar.close()  # 手动收尾进度条；自动记录不触发
    assert logger.store.snapshot() == {}


def test_stepsbar_no_auto_log_on_exception(logger):
    """循环内抛异常：不触发自动记录。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    bar = logger.stepsbar(range(5), epoch=0, metrics=group)
    with pytest.raises(RuntimeError):
        for _ in bar:
            group.feed(m=1.0)
            raise RuntimeError("boom")
    bar.close()
    assert logger.store.snapshot() == {}


def test_stepsbar_metrics_type_check(logger):
    with pytest.raises(TypeError):
        logger.stepsbar(range(3), metrics={"m": 1.0})


def test_stepsbar_realtime_postfix(logger):
    """stepsbar(metrics=...)：每次 feed 后 postfix 实时刷新为本卡本地值（零通信）。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    bar = logger.stepsbar(range(4), epoch=0, metrics=group, reset=True)
    group.feed(m=1.0)
    assert bar.postfix["m"] == pytest.approx(1.0)
    group.feed(m=3.0)
    assert bar.postfix["m"] == pytest.approx(2.0)  # 运行中均值 (1+3)/2
    for _ in bar:
        pass
    # 自然结束：gather 全局值落盘并重置；曲线得到精确结果
    assert logger.store.snapshot()["m"] == [{"step": 0, "value": 2.0, "epoch": 0}]
    assert group.compute() != group.compute()  # 已重置 -> NaN


def test_stepsbar_postfix_skips_nan(logger):
    """无观测（NaN）与非数值指标不上 postfix，其余正常显示。"""
    from melog.metrics import Count, Mean, MetricGroup

    group = MetricGroup({"a": Mean(), "n": Count()})
    bar = logger.stepsbar(range(2), epoch=0, metrics=group)
    group.feed(n=5)  # a 尚无观测 -> NaN
    assert "a" not in bar.postfix
    assert bar.postfix["n"] == 5
    bar.close()


def test_stepsbar_hook_cleared_on_close(logger):
    """bar 关闭后解除实时刷新钩子：继续 feed 不报错、不再影响已关闭的条。"""
    from melog.metrics import Mean, MetricGroup

    group = MetricGroup({"m": Mean()})
    bar = logger.stepsbar(range(2), epoch=0, metrics=group)
    bar.close()
    group.feed(m=1.0)  # 钩子已解除：不报错
    assert logger.store.snapshot() == {}


def test_stepsbar_feed_routes_to_own_bar(logger):
    """嵌套时各 bar 的 metrics 组只刷新自己的条，不窜到栈顶。"""
    from melog.metrics import Mean, MetricGroup

    train = MetricGroup({"m": Mean()})
    val = MetricGroup({"m": Mean()})
    outer = logger.stepsbar(range(3), epoch=0, metrics=train)
    inner = logger.stepsbar(range(2), epoch=0, metrics=val)
    train.feed(m=1.0)
    val.feed(m=9.0)
    assert outer.postfix["m"] == pytest.approx(1.0)
    assert inner.postfix["m"] == pytest.approx(9.0)
    inner.close()  # 栈顶关闭：outer 恢复渲染，数据不受影响
    assert outer.postfix["m"] == pytest.approx(1.0)
    outer.close()


def test_close_idempotent(logger):
    logger.scalar({"a": 1})
    logger.close()
    logger.close()  # 不应抛异常


def test_run_dir_created(tmp_path):
    lg = Melog(project="run1", output_dir=str(tmp_path), enable_web=False)
    assert (lg.run_dir).exists()
    lg.close()


# ---------------------------------------------------------------- 全局共享
def test_global_shared_instance(tmp_path):
    """入口 init 一次，模块级接口在任何位置可用；close 后清空活动实例。"""
    import melog as pkg
    from melog.core import _get_active

    lg = pkg.init(tmp_path / "g", enable_web=False)
    try:
        assert pkg.current() is lg
        pkg.scalar({"a": 1.5})
        pkg.set_colors({"loss": "#123456"})
        assert lg.store.snapshot()["a"] == [{"step": 0, "value": 1.5}]
        assert lg._colors == {"loss": "#123456"}
    finally:
        lg.close()

    assert _get_active() is None  # close 清空活动实例
    with pytest.raises(RuntimeError):
        pkg.current()
    with pytest.raises(RuntimeError):
        pkg.scalar({"a": 1.0})


def test_global_reinit_switches_active_instance(tmp_path):
    """再次 init 用新实例替换活动实例；旧实例仍可显式使用。"""
    import melog as pkg

    first = pkg.init(tmp_path / "g1", enable_web=False)
    second = pkg.init(tmp_path / "g2", enable_web=False)
    try:
        assert pkg.current() is second
        pkg.scalar({"a": 1.0})
        assert second.store.snapshot()["a"] and not first.store.snapshot()
        first.scalar({"b": 2.0})  # 旧实例显式调用不受影响
        assert first.store.snapshot()["b"][0]["value"] == 2.0
    finally:
        second.close()  # 收尾活动实例
        first.close()
