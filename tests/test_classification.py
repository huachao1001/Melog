"""分类指标单元测试：预测函数、Accuracy / P / R / F1 / 混淆矩阵、多 rank 合并、集成。"""

import json

import pytest

from melog.core import Melog
from melog.metrics import Accuracy, ConfusionMatrix, F1, MetricGroup, Precision, Recall, preds_from_logits


# ---------------------------------------------------------------- 预测函数
def test_preds_binary_threshold():
    assert preds_from_logits([0.9, 0.2, 0.5]) == [1, 0, 1]


def test_preds_binary_rejects_1d_with_num_classes():
    with pytest.raises(ValueError):
        preds_from_logits([0.9, 0.2], num_classes=2)


def test_preds_2d_is_argmax_even_without_num_classes():
    # 两列 logits 的二分类直接 argmax，与 sigmoid 阈值等价
    assert preds_from_logits([[0.1, 0.9], [0.8, 0.2]]) == [1, 0]


def test_preds_multiclass_argmax():
    logits = [[0.1, 0.9, 0.0], [0.6, 0.3, 0.1]]
    assert preds_from_logits(logits, num_classes=3) == [1, 0]


def test_preds_multiclass_topk():
    logits = [[0.1, 0.5, 0.4]]
    assert preds_from_logits(logits, num_classes=3, topk=2) == [[1, 2]]


def test_preds_accepts_tensor():
    torch = pytest.importorskip("torch")
    out = preds_from_logits(torch.tensor([[0.1, 0.9], [0.7, 0.3]]), num_classes=2)
    assert out == [1, 0]


def test_preds_length_mismatch_raises():
    acc = Accuracy(num_classes=2)
    with pytest.raises(ValueError):
        acc.update([[0.9, 0.1], [0.2, 0.8]], [1])


# ---------------------------------------------------------------- Accuracy
def test_accuracy_binary():
    acc = Accuracy()  # 二分类
    acc.update([0.9, 0.8, 0.6, 0.3, 0.2], [1, 0, 1, 1, 0])
    assert acc.compute() == pytest.approx(3 / 5)


def test_accuracy_multiclass_tensor():
    torch = pytest.importorskip("torch")
    acc = Accuracy(num_classes=3)
    logits = torch.tensor(
        [
            [0.9, 0.05, 0.05],  # 0, label 0 ✓
            [0.4, 0.5, 0.1],    # 1, label 1 ✓
            [0.2, 0.3, 0.5],    # 2, label 2 ✓
            [0.8, 0.1, 0.1],    # 0, label 2 ✗
            [0.1, 0.2, 0.7],    # 2, label 1 ✗
        ]
    )
    labels = torch.tensor([0, 1, 2, 2, 1])
    acc.update(logits, labels)
    assert acc.compute() == pytest.approx(3 / 5)


def test_accuracy_topk():
    torch = pytest.importorskip("torch")
    acc1 = Accuracy(num_classes=3)
    acc2 = Accuracy(num_classes=3, topk=2)
    logits = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.4, 0.5, 0.1],
            [0.2, 0.3, 0.5],
            [0.8, 0.1, 0.1],  # label 2：top1=0 ✗，top2 含 1 仍 ✗
            [0.1, 0.2, 0.7],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2, 1])
    for acc in (acc1, acc2):
        acc.update(logits, labels)
    assert acc1.compute() == pytest.approx(3 / 5)
    assert acc2.compute() == pytest.approx(4 / 5)


def test_accuracy_reset_and_empty():
    acc = Accuracy()
    assert acc.compute() != acc.compute()  # 无观测 -> NaN
    acc.update([0.9, 0.1], [1, 0])
    acc.reset()
    assert acc.compute() != acc.compute()


# ---------------------------------------------------------------- Precision / Recall / F1
def test_prf_binary():
    scores = [0.9, 0.8, 0.6, 0.3, 0.2]
    labels = [1, 0, 1, 1, 0]
    p, r, f1 = Precision(), Recall(), F1()
    for m in (p, r, f1):
        m.update(scores, labels)
    # 预测 [1,1,1,0,0]：tp=2 fp=1 fn=1 tn=1
    assert p.compute() == pytest.approx(2 / 3)
    assert r.compute() == pytest.approx(2 / 3)
    assert f1.compute() == pytest.approx(2 / 3)


def test_prf_multiclass_averages():
    logits = [
        [0.9, 0.05, 0.05],
        [0.4, 0.5, 0.1],
        [0.2, 0.3, 0.5],
        [0.8, 0.1, 0.1],  # pred 0, label 2
        [0.1, 0.2, 0.7],  # pred 2, label 1
    ]
    labels = [0, 1, 2, 2, 1]
    p_macro = Precision(num_classes=3)
    r_macro = Recall(num_classes=3)
    f1_macro = F1(num_classes=3)
    f1_micro = F1(num_classes=3, average="micro")
    f1_weighted = F1(num_classes=3, average="weighted")
    for m in (p_macro, r_macro, f1_macro, f1_micro, f1_weighted):
        m.update(logits, labels)
    assert p_macro.compute() == pytest.approx(2 / 3)
    assert r_macro.compute() == pytest.approx(2 / 3)
    assert f1_macro.compute() == pytest.approx((2 / 3 + 2 / 3 + 0.5) / 3)
    assert f1_micro.compute() == pytest.approx(0.6)
    assert f1_weighted.compute() == pytest.approx(0.6)


def test_prf_unknown_average():
    with pytest.raises(ValueError):
        Precision(num_classes=3, average="bad")._from_counts({(0, 0): 1.0})


# ---------------------------------------------------------------- ConfusionMatrix
def test_confusion_binary():
    cm = ConfusionMatrix()
    cm.update([0.9, 0.8, 0.6, 0.3, 0.2], [1, 0, 1, 1, 0])
    assert cm.compute() == [[1.0, 1.0], [1.0, 2.0]]  # [[tn, fp], [fn, tp]]


def test_confusion_multiclass_infer_classes():
    cm = ConfusionMatrix()
    cm.update([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]], [0, 2])
    assert cm.compute() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_confusion_multiclass_explicit_k():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(
        [[0.9, 0.05, 0.05], [0.4, 0.5, 0.1], [0.2, 0.3, 0.5], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
        [0, 1, 2, 2, 1],
    )
    assert cm.compute() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------- 多 rank 合并逻辑
def test_accuracy_merge_across_ranks(monkeypatch):
    # rank0: 3/5；rank1: 1/1 -> 全局 4/6
    def fake_gather(states):
        return [states, [3.0, 5.0]]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    acc = Accuracy()
    acc.update([0.9], [1])
    assert acc.compute() == pytest.approx(4 / 6)


def test_prf_merge_across_ranks(monkeypatch):
    # rank0: tp=2 fp=1；rank1: tp=3 -> precision (2+3)/(2+3+1) = 5/6
    def fake_gather(states):
        return [states, {(1, 1): 3.0}]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    p = Precision()
    p.update([0.9, 0.8, 0.6], [1, 0, 1])
    assert p.compute() == pytest.approx(5 / 6)


# ---------------------------------------------------------------- MetricGroup / Melog 集成
def test_group_dispatch_classification():
    group = MetricGroup({"acc": Accuracy(num_classes=2), "binary_acc": Accuracy()})
    group.update(acc=([[0.1, 0.9], [0.8, 0.2]], [1, 0]), binary_acc=([0.7, 0.3], [1, 0]))
    out = group.compute()
    assert out["acc"] == pytest.approx(1.0)
    assert out["binary_acc"] == pytest.approx(1.0)


def test_log_group_persists_accuracy(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"acc": Accuracy(num_classes=2)})
    logits = [[0.1, 0.9], [0.7, 0.3]]
    labels = [1, 0]
    group.update(acc=(logits, labels))
    lg.log_group(group, reset=True)
    lg.finish()

    path = next((tmp_path / "t").glob("**/metrics.melog"))
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"metric": "acc", "step": 0, "value": 1.0}]
