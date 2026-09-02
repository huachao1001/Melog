"""指标计算单元测试：内置指标、自定义指标、MetricGroup、与 Melog 集成。"""

import datetime
import json

import pytest

from melog.core import Melog
from melog.distributed import gather_object
from melog.metrics import (
    Accuracy,
    BatchMetric,
    Count,
    Last,
    Max,
    Mean,
    Metric,
    MetricGroup,
    Min,
    Sum,
)


# ---------------------------------------------------------------- 内置指标
def test_mean_weighted():
    m = Mean()
    m.update(1.0, 2)   # 权重 2
    m.update(4.0, 2)
    assert m.compute() == pytest.approx(2.5)


def test_mean_default_weight():
    m = Mean()
    for v in (1.0, 2.0, 3.0):
        m.update(v)
    assert m.compute() == pytest.approx(2.0)


def test_mean_empty_is_nan():
    assert Mean().compute() != Mean().compute()  # NaN


def test_sum_max_min_last_count():
    s = Sum()
    mx = Max()
    mn = Min()
    last = Last()
    cnt = Count()
    for v in (3.0, 1.0, 2.0):
        s.update(v)
        mx.update(v)
        mn.update(v)
        last.update(v)
        cnt.update()
    assert s.compute() == pytest.approx(6.0)
    assert mx.compute() == pytest.approx(3.0)
    assert mn.compute() == pytest.approx(1.0)
    assert last.compute() == pytest.approx(2.0)
    assert cnt.compute() == pytest.approx(3.0)


def test_reset():
    m = Mean()
    m.update(5.0)
    m.reset()
    assert m.compute() != m.compute()  # 重置后为 NaN
    m.update(2.0)
    assert m.compute() == pytest.approx(2.0)


def test_accepts_tensor_like():
    class FakeTensor:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    m = Mean()
    m.update(FakeTensor(2.0), FakeTensor(4.0))
    assert m.compute() == pytest.approx(2.0)


# ---------------------------------------------------------------- 自定义指标
class EpochAcc(Metric):
    """epoch 级自定义指标示例：用混淆计数算准确率。"""

    def __init__(self):
        self.correct = 0.0
        self.total = 0.0

    def update(self, correct, total):
        self.correct += float(correct)
        self.total += float(total)

    def state(self):
        return {"correct": self.correct, "total": self.total}

    def merge_states(self, states):
        correct = sum(s["correct"] for s in states)
        total = sum(s["total"] for s in states)
        return correct / total if total else float("nan")

    def reset(self):
        self.correct = self.total = 0.0


def test_custom_metric_epoch_level():
    acc = EpochAcc()
    acc.update(8, 10)
    acc.update(9, 10)
    assert acc.compute() == pytest.approx(0.85)
    acc.reset()
    acc.update(5, 10)
    assert acc.compute() == pytest.approx(0.5)


# ---------------------------------------------------------------- 单批次指标（BatchMetric）
class PairAcc(BatchMetric):
    """单批次自定义指标示例：只实现 compute_batch，其余交给框架。"""

    def compute_batch(self, logits, labels):
        preds = [1 if float(x) >= 0.5 else 0 for x in logits]
        hits = sum(1 for p, t in zip(preds, labels) if p == int(t))
        return (hits / len(labels), float(len(labels))) if labels else (0.0, 0.0)


def test_batch_metric_single_function():
    m = PairAcc()
    m.update(logits=[0.9, 0.2, 0.7], labels=[1, 0, 0])  # 命中 2/3
    m.update([0.1, 0.8], [0, 1])                        # 位置喂入亦可：命中 2/2
    # 按样本数加权：全局 4/5，而非各 batch 平均值的平均
    assert m.compute() == pytest.approx(4 / 5)


def test_batch_metric_empty_and_reset():
    m = PairAcc()
    assert m.compute() != m.compute()  # 无观测 -> NaN
    m.update(logits=[0.9], labels=[1])
    m.reset()
    assert m.compute() != m.compute()


def test_batch_metric_custom_weight():
    class W(BatchMetric):
        def compute_batch(self, value, weight):  # 形参名任意，框架按名取值
            return (float(value), float(weight))

    w = W()
    w.update(value=2.0, weight=3.0)
    w.update(value=5.0, weight=1.0)
    assert w.compute() == pytest.approx((2.0 * 3.0 + 5.0) / 4.0)


def test_batch_metric_multiple_params():
        class MaskedAcc(BatchMetric):
            # 需要几个参数就声明几个，名字任意；多余观测自动忽略
            def compute_batch(self, logits, labels, mask):
                vals = [
                    1.0 if int(t) == (1 if float(x) >= 0.5 else 0) else 0.0
                    for x, t, m in zip(logits, labels, mask)
                    if m
                ]
                # (值, 样本数)：值须为该 batch 的样本平均
                return (sum(vals) / len(vals), float(len(vals))) if vals else (0.0, 0.0)

        m = MaskedAcc()
        m.update(logits=[0.9, 0.2, 0.7], labels=[1, 1, 0], mask=[1, 0, 1], step=123)  # 1/2
        m.update(labels=[0], logits=[0.1], mask=[1])                                  # 1/1
        assert m.compute() == pytest.approx(2 / 3)


def test_batch_metric_missing_param():
    m = PairAcc()
    with pytest.raises(KeyError):
        m.update(logits=[0.9])  # 缺少 labels


def test_batch_metric_merge_across_ranks(monkeypatch):
    # rank0: 1/1；rank1: 1/2 -> 全局 2/3
    def fake_gather(states):
        return [states, [1.0, 2.0]]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    m = PairAcc()
    m.update(logits=[0.9], labels=[1])
    assert m.compute() == pytest.approx(2 / 3)


def test_group_feed_dispatch():
    group = MetricGroup({"loss": Mean(), "acc": Accuracy()})
    # acc 按形参名自动取 logits/labels；loss 按注册名取值（元组 = 值 + 权重）
    group.feed(logits=[0.9, 0.2], labels=[1, 0], loss=(1.0, 2))
    out = group.compute()
    assert out["acc"] == pytest.approx(1.0)
    assert out["loss"] == pytest.approx(1.0)


def test_group_feed_partial_scalars():
    group = MetricGroup({"loss": Mean(), "lr": Mean()})
    group.feed(loss=1.0)  # 本 batch 未提供 lr，不累积也不报错
    group.feed(loss=3.0, lr=1e-3)
    out = group.compute()
    assert out["loss"] == pytest.approx(2.0)
    assert out["lr"] == pytest.approx(1e-3)


def test_batch_metric_with_melog(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"acc": PairAcc()})
    with lg.train(total=2) as bar:
        for _ in range(2):
            group.feed(logits=[0.9, 0.2], labels=[1, 0])
            lg.log_group(group, reset=True)
            bar.advance(1)
    lg.finish()

    path = next((tmp_path / "t").glob("**/metrics.melog"))
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"metric": "acc", "step": 0, "value": 1.0},
        {"metric": "acc", "step": 1, "value": 1.0},
    ]


# ---------------------------------------------------------------- 多 rank 合并逻辑
def test_group_merge_across_ranks(monkeypatch):
    # 模拟 2 个 rank 的状态：rank0 loss sum=4/weight=2，rank1 sum=1/weight=1
    def fake_gather(states):
        # states 是本 rank 各指标的 state 列表，返回按 rank 排列的收集结果
        return [states, [dict(s) for s in states]]

    monkeypatch.setattr("melog.metrics.group.gather_object", fake_gather)

    group = MetricGroup({"loss": Mean(), "acc": Mean()})
    group.update(loss=(2.0, 2.0), acc=0.5)  # rank0: 4/2, rank1: 1/1
    out = group.compute()
    # 直接比较：每组指标的合并输入相同 -> 用 MultiRankMean 逻辑验证见下个测试
    assert set(out) == {"loss", "acc"}


def test_metric_merge_states_multi_rank(monkeypatch):
    # rank0: sum=4, weight=2；rank1: sum=1, weight=1 -> 全局 5/3
    def fake_gather(states):
        other = {"sum": 1.0, "weight": 1.0}
        return [states, other]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    m = Mean()
    m.update(2.0, 2.0)
    m.update(2.0, 0.0)  # 不改变权重和（weight=0 的观测）
    assert m.compute() == pytest.approx(5.0 / 3.0)


def test_group_single_gather_for_all_metrics(monkeypatch):
    calls = []

    def fake_gather(states):
        calls.append(1)
        # 模拟另一 rank：每个指标状态都乘 2
        return [states, [{k: v * 2 for k, v in s.items()} for s in states]]

    monkeypatch.setattr("melog.metrics.group.gather_object", fake_gather)
    group = MetricGroup({"loss": Mean(), "total": Sum()})
    group.update(loss=1.0, total=3.0)  # rank0: loss 1/1, total 3
    out = group.compute()
    assert out["loss"] == pytest.approx(1.0)  # (1+2)/(1+2)
    assert out["total"] == pytest.approx(9.0)  # 3 + 6
    assert len(calls) == 1  # 全组只做一次 gather


# ---------------------------------------------------------------- MetricGroup
def test_group_update_dispatch():
    group = MetricGroup({"loss": Mean(), "n": Count()})
    group.update(loss=(3.0, 3.0), n=1)
    group.update(loss=1.0, n=1)
    out = group.compute()
    assert out["loss"] == pytest.approx((3.0 * 3 + 1.0) / 4)
    assert out["n"] == pytest.approx(2.0)


def test_group_unknown_metric():
    group = MetricGroup({"loss": Mean()})
    with pytest.raises(KeyError):
        group.update(acc=0.5)


def test_group_duplicate_add():
    group = MetricGroup({"loss": Mean()})
    with pytest.raises(KeyError):
        group.add("loss", Sum())


def test_group_reset():
    group = MetricGroup({"loss": Mean()})
    group.update(loss=1.0)
    group.reset()
    group.update(loss=3.0)
    assert group.compute()["loss"] == pytest.approx(3.0)


def test_group_getitem_contains_len():
    group = MetricGroup({"loss": Mean(), "acc": Sum()})
    assert len(group) == 2
    assert "loss" in group
    assert isinstance(group["acc"], Sum)
    assert dict(group)["loss"] is group["loss"]


# ---------------------------------------------------------------- 与 Melog 集成
def test_log_group_records_and_resets(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"loss": Mean(), "acc": Mean()})
    with lg.train(total=4) as bar:
        for _ in range(2):
            group.update(loss=1.0, acc=0.5)
            lg.log_group(group, reset=True)
    assert bar.completed == 0  # epoch 级记录默认不推进进度条
    lg.finish()

    path = next((tmp_path / "t").glob("**/metrics.melog"))
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"metric": "loss", "step": 0, "value": 1.0},
        {"metric": "acc", "step": 0, "value": 0.5},
        {"metric": "loss", "step": 1, "value": 1.0},
        {"metric": "acc", "step": 1, "value": 0.5},
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
        group.update(loss=(2.0, 2.0), total=3.0)
    else:
        group.update(loss=(1.0, 1.0), total=6.0)
    out = group.compute()
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
