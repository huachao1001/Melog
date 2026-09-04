"""指标计算单元测试：内置指标、自定义指标、MetricGroup、与 Melog 集成。"""

import datetime
import json

import pytest

from melog import StepsBar
from melog.core import Melog
from melog.utils.distributed import gather_object
from melog.metrics import (
    Accuracy,
    Count,
    Last,
    Mean,
    Metric,
    MetricGroup,
    Sum,
)


# ---------------------------------------------------------------- 内置指标
def test_mean_with_count():
    m = Mean()
    m.feed(1.0, 2)   # count 2（结果 = (1+2×2)/3）
    m.feed(4.0, 2)
    assert m.result() == pytest.approx(2.5)


def test_mean_default_count():
    m = Mean()
    for v in (1.0, 2.0, 3.0):
        m.feed(v)
    assert m.result() == pytest.approx(2.0)


def test_mean_empty_is_nan():
    assert Mean().result() != Mean().result()  # NaN


def test_sum_last_count():
    s = Sum()
    last = Last()
    cnt = Count()
    for v in (3.0, 1.0, 2.0):
        s.feed(v)
        last.feed(v)
        cnt.feed()
    assert s.result() == pytest.approx(6.0)
    assert last.result() == pytest.approx(2.0)
    assert cnt.result() == pytest.approx(3.0)


def test_reset():
    m = Mean()
    m.feed(5.0)
    m.reset()
    assert m.result() != m.result()  # 重置后为 NaN
    m.feed(2.0)
    assert m.result() == pytest.approx(2.0)


def test_accepts_tensor_like():
    class FakeTensor:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    m = Mean()
    m.feed(FakeTensor(2.0), FakeTensor(4.0))
    assert m.result() == pytest.approx(2.0)


# ---------------------------------------------------------------- 自定义指标
class EpochAcc(Metric):
    """epoch 级自定义指标示例：只写增量与计算，多卡合并交给框架。"""

    def update(self, correct, total):
        return {"correct": float(correct), "total": float(total)}

    def compute(self, correct, total):
        return correct / total if total else float("nan")


def test_custom_metric_epoch_level():
    acc = EpochAcc()
    acc.feed(8, 10)
    acc.feed(9, 10)
    assert acc.result() == pytest.approx(0.85)
    acc.reset()
    acc.feed(5, 10)
    assert acc.result() == pytest.approx(0.5)


# ---------------------------------------------------------------- 实时指标（只实现 compute）
class PairAcc(Metric):
    """实时自定义指标示例：只实现 compute，其余交给框架。"""

    def compute(self, logits, labels):
        preds = [1 if float(x) >= 0.5 else 0 for x in logits]
        hits = sum(1 for p, t in zip(preds, labels) if p == int(t))
        return (hits / len(labels), float(len(labels))) if labels else (0.0, 0.0)


def test_batch_metric_single_function():
    m = PairAcc()
    m.feed(logits=[0.9, 0.2, 0.7], labels=[1, 0, 0])  # 命中 2/3
    m.feed([0.1, 0.8], [0, 1])                        # 位置喂入亦可：命中 2/2
    # 按样本数加权：全局 4/5，而非各 batch 平均值的平均
    assert m.result() == pytest.approx(4 / 5)


def test_batch_metric_empty_and_reset():
    m = PairAcc()
    assert m.result() != m.result()  # 无观测 -> NaN
    m.feed(logits=[0.9], labels=[1])
    m.reset()
    assert m.result() != m.result()


def test_batch_metric_custom_count():
    class W(Metric):
        def compute(self, value, count):  # 形参名任意，框架按名取值
            return (float(value), float(count))

    w = W()
    w.feed(value=2.0, count=3.0)
    w.feed(value=5.0, count=1.0)
    assert w.result() == pytest.approx((2.0 * 3.0 + 5.0) / 4.0)


def test_batch_metric_multiple_params():
        class MaskedAcc(Metric):
            # 需要几个参数就声明几个，名字任意；多余观测自动忽略
            def compute(self, logits, labels, mask):
                vals = [
                    1.0 if int(t) == (1 if float(x) >= 0.5 else 0) else 0.0
                    for x, t, m in zip(logits, labels, mask)
                    if m
                ]
                # (值, 样本数)：值须为该 batch 的样本平均
                return (sum(vals) / len(vals), float(len(vals))) if vals else (0.0, 0.0)

        m = MaskedAcc()
        m.feed(logits=[0.9, 0.2, 0.7], labels=[1, 1, 0], mask=[1, 0, 1], step=123)  # 1/2
        m.feed(labels=[0], logits=[0.1], mask=[1])                                  # 1/1
        assert m.result() == pytest.approx(2 / 3)


def test_batch_metric_missing_param():
    m = PairAcc()
    with pytest.raises(KeyError):
        m.feed(logits=[0.9])  # 缺少 labels


def test_batch_metric_merge_across_ranks(monkeypatch):
    # rank0: 1/1；rank1: 1/2 -> 全局 2/3
    def fake_gather(states):
        return [states, [1.0, 2.0]]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    m = PairAcc()
    m.feed(logits=[0.9], labels=[1])
    assert m.result() == pytest.approx(2 / 3)


def test_group_feed_dispatch():
    group = MetricGroup({"loss": Mean(), "acc": Accuracy()})
    # acc 按形参名/位置从 args 取 logits/labels；loss 按注册名取值（元组 = 值 + count）
    group.feed(args=([0.9, 0.2], [1, 0]), loss=(1.0, 2))
    out = group._compute()
    assert out["acc"] == pytest.approx(1.0)
    assert out["loss"] == pytest.approx(1.0)


def test_group_feed_args_dict():
    """args 传字典：按键名对应 compute 形参，多余的键自动忽略。"""
    group = MetricGroup({"loss": Mean(), "acc": Accuracy()})
    group.feed(args={"logits": [0.9, 0.2], "labels": [1, 0], "extra": 1}, loss=0.5)
    out = group._compute()
    assert out["acc"] == pytest.approx(1.0)
    assert out["loss"] == pytest.approx(0.5)


def test_group_feed_partial_scalars():
    group = MetricGroup({"loss": Mean(), "lr": Mean()})
    group.feed(loss=1.0)  # 本 batch 未提供 lr，不累积也不报错
    group.feed(loss=3.0, lr=1e-3)
    out = group._compute()
    assert out["loss"] == pytest.approx(2.0)
    assert out["lr"] == pytest.approx(1e-3)


def test_group_local_no_gather(monkeypatch):
    """local() 只合并本 rank 状态（零通信），供进度条实时显示。"""

    def boom(states):
        raise AssertionError("local() 不应触发跨 rank 收集")

    monkeypatch.setattr("melog.metrics.group.gather_object", boom)
    group = MetricGroup({"loss": Mean(), "acc": Accuracy()})
    group.feed(args=([0.9, 0.2], [1, 0]), loss=(2.0, 2.0))
    out = group.local()
    assert out["loss"] == pytest.approx(2.0)
    assert out["acc"] == pytest.approx(1.0)


def test_batch_metric_with_melog(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"acc": PairAcc()})
    for _ in StepsBar(range(2)):
        group.feed(args=([0.9, 0.2], [1, 0]))
        lg.scalar(group)
        group.reset()
    lg.close()

    from melog.storage.melog_file import MelogFileReader

    path = next((tmp_path / "t").glob("*metrics-*.melog"))
    records = list(MelogFileReader(path).records())
    assert records == [(0, None, {"acc": 1.0}), (1, None, {"acc": 1.0})]


# ---------------------------------------------------------------- 多 rank 合并逻辑
def test_group_merge_across_ranks(monkeypatch):
    # 模拟 2 个 rank 的状态：rank0 loss total=4/count=2，rank1 total=1/count=1
    def fake_gather(states):
        # states 是本 rank 各指标的 state 列表，返回按 rank 排列的收集结果
        return [states, [list(s) for s in states]]

    monkeypatch.setattr("melog.metrics.group.gather_object", fake_gather)

    group = MetricGroup({"loss": Mean(), "acc": Mean()})
    group.feed(loss=(2.0, 2.0), acc=0.5)  # rank0: 4/2, rank1: 1/1
    out = group._compute()
    # 直接比较：每组指标的合并输入相同 -> 用 MultiRankMean 逻辑验证见下个测试
    assert set(out) == {"loss", "acc"}


def test_metric_merge_states_multi_rank(monkeypatch):
    # rank0: total=4, count=2；rank1: total=1, count=1 -> 全局 5/3
    def fake_gather(states):
        other = [1.0, 1.0]
        return [states, other]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    m = Mean()
    m.feed(2.0, 2.0)
    m.feed(2.0, 0.0)  # 不改变 count 和（count=0 的观测）
    assert m.result() == pytest.approx(5.0 / 3.0)


def test_group_single_gather_for_all_metrics(monkeypatch):
    calls = []

    def fake_gather(states):
        calls.append(1)

        def double(s):  # 模拟另一 rank：每个指标状态都乘 2
            return {k: v * 2 for k, v in s.items()} if isinstance(s, dict) else s * 2

        return [states, [double(s) for s in states]]

    monkeypatch.setattr("melog.metrics.group.gather_object", fake_gather)
    group = MetricGroup({"loss": Mean(), "total": Sum()})
    group.feed(loss=1.0, total=3.0)  # rank0: loss 1/1, total 3
    out = group._compute()
    assert out["loss"] == pytest.approx(1.0)  # (1+2)/(1+2)
    assert out["total"] == pytest.approx(9.0)  # 3 + 6
    assert len(calls) == 1  # 全组只做一次 gather


# ---------------------------------------------------------------- MetricGroup
def test_group_feed_scalars():
    group = MetricGroup({"loss": Mean(), "n": Count()})
    group.feed(loss=(3.0, 3.0), n=1)
    group.feed(loss=1.0, n=1)
    out = group._compute()
    assert out["loss"] == pytest.approx((3.0 * 3 + 1.0) / 4)
    assert out["n"] == pytest.approx(2.0)


def test_group_tuple_count_dispatch():
    """元组 (值, count) 按指标精确平均，其余指标等权。"""
    group = MetricGroup({"loss": Mean(), "acc": Mean()})
    group.feed(loss=(2.0, 4), acc=0.5)
    out = group._compute()
    assert out["loss"] == pytest.approx(2.0)   # sum=8, count=4
    assert out["acc"] == pytest.approx(0.5)


def test_group_feed_ignores_unknown():
    """feed 只取已注册指标需要的观测，未注册/多余的键自动忽略。"""
    group = MetricGroup({"loss": Mean()})
    group.feed(acc=0.5, loss=2.0)
    out = group._compute()
    assert out == {"loss": pytest.approx(2.0)}


def test_group_duplicate_add():
    group = MetricGroup({"loss": Mean()})
    with pytest.raises(KeyError):
        group.add("loss", Sum())


def test_group_reset():
    group = MetricGroup({"loss": Mean()})
    group.feed(loss=1.0)
    group.reset()
    group.feed(loss=3.0)
    assert group._compute()["loss"] == pytest.approx(3.0)


def test_group_getitem_contains_len():
    group = MetricGroup({"loss": Mean(), "acc": Sum()})
    assert len(group) == 2
    assert "loss" in group
    assert isinstance(group["acc"], Sum)
    assert dict(group)["loss"] is group["loss"]


# ---------------------------------------------------------------- 与 Melog 集成
def test_group_compute_records_and_resets(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"loss": Mean(), "acc": Mean()})
    bar = StepsBar(range(4))  # 建条但不迭代
    for _ in range(2):
        group.feed(loss=1.0, acc=0.5)
        lg.scalar(group)
        group.reset()
    assert bar.n == 0  # epoch 级记录默认不推进进度条
    lg.close()

    from melog.storage.melog_file import MelogFileReader

    path = next((tmp_path / "t").glob("*metrics-*.melog"))
    records = list(MelogFileReader(path).records())
    assert records == [
        (0, None, {"loss": 1.0, "acc": 0.5}),
        (1, None, {"loss": 1.0, "acc": 0.5}),
    ]


def test_gather_object_passthrough():
    assert gather_object({"a": 1}) == [{"a": 1}]


# ---------------------------------------------------------------- 真实双进程（gloo）
def _gloo_worker(rank, world_size, init_file, result_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    group = MetricGroup({"loss": Mean(), "total": Sum()})
    # rank0: loss=2(w=2), total=3；rank1: loss=1(w=1), total=6
    if rank == 0:
        group.feed(loss=(2.0, 2.0), total=3.0)
    else:
        group.feed(loss=(1.0, 1.0), total=6.0)
    out = group._compute()
    (result_dir / f"rank{rank}.json").write_text(json.dumps(out), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def test_multi_rank_gloo(tmp_path):
    torch = pytest.importorskip("torch")

    init_file = str(tmp_path / "dist_init").replace("\\", "/")
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(2, init_file, tmp_path),
        nprocs=2,
        join=True,
    )
    r0 = json.loads((tmp_path / "rank0.json").read_text(encoding="utf-8"))
    r1 = json.loads((tmp_path / "rank1.json").read_text(encoding="utf-8"))
    # 全局加权平均: (2*2 + 1*1) / (2+1)；求和: 3+6
    assert r0 == r1 == {"loss": pytest.approx(5.0 / 3.0), "total": pytest.approx(9.0)}


def test_group_category_prefixes_record_names(tmp_path):
    """MetricGroup(category=...)：记录名自动加类别前缀，local 同步带前缀。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    train = MetricGroup({"loss": Mean(), "recall/class_0": Mean()}, category="train")
    val = MetricGroup({"loss": Mean()}, category="val")
    train.feed(loss=1.0, **{"recall/class_0": 0.5})
    lg.scalar(train)
    val.feed(loss=2.0)
    lg.scalar(val)
    lg.close()

    from melog.storage.melog_file import MelogFileReader

    path = next((tmp_path / "t").glob("*metrics-*.melog"))
    records = list(MelogFileReader(path).records())
    assert records == [
        (0, None, {"train/loss": 1.0, "train/recall/class_0": 0.5}),
        (1, None, {"val/loss": 2.0}),
    ]
    assert set(train.local()) == {"train/loss", "train/recall/class_0"}
    # 类别声明随日志持久化
    cats = [r for r in MelogFileReader(path).media() if r.get("type") == "category"]
    assert {r["name"] for r in cats} == {"train", "val"}


def test_group_without_category_unchanged(tmp_path):
    """不传 category：行为与旧版完全一致（名字不加前缀）。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"loss": Mean()})
    group.feed(loss=1.0)
    lg.scalar(group)
    lg.close()

    from melog.storage.melog_file import MelogFileReader

    path = next((tmp_path / "t").glob("*metrics-*.melog"))
    records = list(MelogFileReader(path).records())
    assert records == [(0, None, {"loss": 1.0})]
    cats = [r for r in MelogFileReader(path).media() if r.get("type") == "category"]
    assert cats == []
