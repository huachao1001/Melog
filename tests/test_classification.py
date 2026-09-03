"""分类指标单元测试：预测函数、Accuracy / P / R / F1 / AUC / 混淆矩阵、多 rank 合并、集成。"""

import json

import pytest

from melog.core import Melog
from melog.metrics import Accuracy, AUC, ConfusionMatrix, F1, MetricGroup, Precision, Recall, preds_from_logits


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
        acc.feed([[0.9, 0.1], [0.2, 0.8]], [1])


# ---------------------------------------------------------------- Accuracy
def test_accuracy_binary():
    acc = Accuracy()  # 二分类
    acc.feed([0.9, 0.8, 0.6, 0.3, 0.2], [1, 0, 1, 1, 0])
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
    acc.feed(logits, labels)
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
        acc.feed(logits, labels)
    assert acc1.compute() == pytest.approx(3 / 5)
    assert acc2.compute() == pytest.approx(4 / 5)


def test_accuracy_reset_and_empty():
    acc = Accuracy()
    assert acc.compute() != acc.compute()  # 无观测 -> NaN
    acc.feed([0.9, 0.1], [1, 0])
    acc.reset()
    assert acc.compute() != acc.compute()


# ---------------------------------------------------------------- Precision / Recall / F1
def test_prf_binary():
    scores = [0.9, 0.8, 0.6, 0.3, 0.2]
    labels = [1, 0, 1, 1, 0]
    p, r, f1 = Precision(), Recall(), F1()
    for m in (p, r, f1):
        m.feed(scores, labels)
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
        m.feed(logits, labels)
    assert p_macro.compute() == pytest.approx(2 / 3)
    assert r_macro.compute() == pytest.approx(2 / 3)
    assert f1_macro.compute() == pytest.approx((2 / 3 + 2 / 3 + 0.5) / 3)
    assert f1_micro.compute() == pytest.approx(0.6)
    assert f1_weighted.compute() == pytest.approx(0.6)


def test_prf_unknown_average():
    with pytest.raises(ValueError):
        Precision(num_classes=3, average="bad")._from_counts({(0, 0): 1.0})


def test_prf_class_index_single_class():
    # 逐类指标：只看指定类别（one-vs-rest）
    rec2 = Recall(num_classes=3, class_index=2)
    rec2.feed([[0.1, 0.2, 0.9], [0.9, 0.1, 0.2], [0.2, 0.1, 0.8]], [2, 0, 2])
    assert rec2.compute() == pytest.approx(1.0)  # 类 2：tp=2, fn=0

    rec2.reset()
    rec2.feed([[0.1, 0.2, 0.9], [0.9, 0.1, 0.2], [0.2, 0.9, 0.3]], [2, 0, 2])
    assert rec2.compute() == pytest.approx(0.5)  # tp=1, fn=1
    prec2 = Precision(num_classes=3, class_index=2)
    prec2.feed([[0.1, 0.2, 0.9], [0.9, 0.1, 0.2], [0.2, 0.9, 0.3]], [2, 0, 2])
    assert prec2.compute() == pytest.approx(1.0)  # tp=1, fp=0

    # 与 macro 对比：类 1 从未作为真实标签出现（recall 为 NaN 被跳过），
    # macro = (1.0 + 0.5) / 2
    r_macro = Recall(num_classes=3)
    r_macro.feed([[0.1, 0.2, 0.9], [0.9, 0.1, 0.2], [0.2, 0.9, 0.3]], [2, 0, 2])
    assert r_macro.compute() == pytest.approx(0.75)


def test_prf_class_index_absent_class_is_nan():
    rec = Recall(num_classes=3, class_index=1)
    rec.feed([[0.9, 0.1, 0.2]], [0])
    assert rec.compute() != rec.compute()  # 该类无样本 -> NaN


# ---------------------------------------------------------------- ConfusionMatrix
def test_confusion_binary():
    cm = ConfusionMatrix()
    cm.feed([0.9, 0.8, 0.6, 0.3, 0.2], [1, 0, 1, 1, 0])
    assert cm.compute() == [[1.0, 1.0], [1.0, 2.0]]  # [[tn, fp], [fn, tp]]


def test_confusion_multiclass_infer_classes():
    cm = ConfusionMatrix()
    cm.feed([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]], [0, 2])
    assert cm.compute() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_confusion_multiclass_explicit_k():
    cm = ConfusionMatrix(num_classes=3)
    cm.feed(
        [[0.9, 0.05, 0.05], [0.4, 0.5, 0.1], [0.2, 0.3, 0.5], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
        [0, 1, 2, 2, 1],
    )
    assert cm.compute() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------- AUC
def test_auc_binary_perfect_and_inverted():
    auc = AUC()
    auc.feed([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0])
    assert auc.compute() == pytest.approx(1.0)
    auc.reset()
    auc.feed([0.1, 0.3, 0.8, 0.9], [1, 1, 0, 0])
    assert auc.compute() == pytest.approx(0.0)


def test_auc_known_value_and_ties():
    auc = AUC()
    # 正类得分 {0.9, 0.2}，负类 {0.1, 0.8}：赢 3 对输 1 对
    auc.feed([0.9, 0.1, 0.8, 0.2], [1, 0, 0, 1])
    assert auc.compute() == pytest.approx(0.75)
    # 全部并列 -> 0.5
    auc.reset()
    auc.feed([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
    assert auc.compute() == pytest.approx(0.5)


def test_auc_2d_binary_uses_positive_column():
    auc = AUC()
    auc.feed([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], [1, 0, 1, 0])
    # 正类列得分 [0.9, 0.6]，负类列 [0.2, 0.3] -> 全胜
    assert auc.compute() == pytest.approx(1.0)


def test_auc_multiclass_requires_class_index():
    auc = AUC()
    with pytest.raises(ValueError):
        auc.feed([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]], [0, 1])


def test_auc_multiclass_one_vs_rest():
    logits = [
        [0.3, 0.5, 0.9],
        [0.6, 0.1, 0.4],
        [0.2, 0.8, 0.3],
        [0.9, 0.2, 0.5],
        [0.7, 0.3, 0.2],
    ]
    labels = [2, 0, 1, 0, 2]
    a0 = AUC(class_index=0)
    a0.feed(logits, labels)
    # 类 0 列得分 [0.3, 0.6, 0.2, 0.9, 0.7]；正类 {0.9, 0.6}，负类 {0.3, 0.2, 0.7}
    # 0.9 胜 3 对，0.6 胜 2 对（输给 0.7）-> 5/6
    assert a0.compute() == pytest.approx(5 / 6)
    # 与"取出该列直接算二分类 AUC"等价
    direct = AUC()
    direct.feed([0.3, 0.6, 0.2, 0.9, 0.7], [0, 1, 0, 1, 0])
    assert direct.compute() == pytest.approx(a0.compute())


def test_auc_empty_is_nan():
    auc = AUC()
    assert auc.compute() != auc.compute()
    auc.feed([0.9, 0.1], [1, 1])  # 只有正类 -> 无定义
    assert auc.compute() != auc.compute()


# ---------------------------------------------------------------- 多 rank 合并逻辑
def test_accuracy_merge_across_ranks(monkeypatch):
    # rank0: 3/5；rank1: 1/1 -> 全局 4/6
    def fake_gather(states):
        return [states, [3.0, 5.0]]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    acc = Accuracy()
    acc.feed([0.9], [1])
    assert acc.compute() == pytest.approx(4 / 6)


def test_prf_merge_across_ranks(monkeypatch):
    # rank0: tp=2 fp=1；rank1: tp=3 -> precision (2+3)/(2+3+1) = 5/6
    def fake_gather(states):
        return [states, {(1, 1): 3.0}]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    p = Precision()
    p.feed([0.9, 0.8, 0.6], [1, 0, 1])
    assert p.compute() == pytest.approx(5 / 6)


def test_auc_merge_across_ranks(monkeypatch):
    # rank0 前半 + rank1 后半，合并后与全量一致（AUC = 8/9）
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    labels = [1, 0, 1, 0, 0, 1]

    def fake_gather(states):
        return [states, [(0.2, False), (0.7, False), (0.3, True)]]

    monkeypatch.setattr("melog.metrics.base.gather_object", fake_gather)
    auc = AUC()
    auc.feed(scores[:3], labels[:3])
    assert auc.compute() == pytest.approx(8 / 9)


# ---------------------------------------------------------------- MetricGroup / Melog 集成
def test_group_dispatch_classification():
    group = MetricGroup({"acc": Accuracy(num_classes=2), "binary_acc": Accuracy()})
    group["acc"].feed([[0.1, 0.9], [0.8, 0.2]], [1, 0])
    group["binary_acc"].feed([0.7, 0.3], [1, 0])
    out = group.compute()
    assert out["acc"] == pytest.approx(1.0)
    assert out["binary_acc"] == pytest.approx(1.0)


def test_log_group_persists_accuracy(tmp_path):
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    group = MetricGroup({"acc": Accuracy(num_classes=2)})
    logits = [[0.1, 0.9], [0.7, 0.3]]
    labels = [1, 0]
    group["acc"].feed(logits, labels)
    lg.log_group(group, reset=True)
    lg.close()

    path = next((tmp_path / "t").glob("**/metrics.melog"))
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"metric": "acc", "step": 0, "value": 1.0}]
