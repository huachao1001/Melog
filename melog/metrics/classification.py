"""内置分类指标：Accuracy / Precision / Recall / F1 / ConfusionMatrix。

均继承 BatchMetric：框架把每个 batch 的 logits 与 labels 回调给
compute_batch，累积、跨 GPU 合并、reset 由框架完成。
logits -> 预测类别的转换由预测函数完成（默认 preds_from_logits，
可通过 predictor 参数替换为自定义函数）。

默认预测规则（preds_from_logits，可用 predictor 参数替换）：
- 一维得分：按 threshold 判二分类
- 二维 (N, K)：按行 argmax，K=2 的二分类同样适用
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base import BatchMetric

__all__ = [
    "preds_from_logits",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",
    "ConfusionMatrix",
]

# 预测函数签名：(logits, num_classes, threshold, topk) -> 每个样本的预测
Predictor = Callable[[Any, Optional[int], float, Optional[int]], Sequence[Any]]


def _plain(values: Any) -> Any:
    """torch tensor / numpy 数组 -> 纯 Python 列表，其余原样返回。"""
    return values.tolist() if hasattr(values, "tolist") else values


def _check_pair(preds: Sequence[Any], targets: Sequence[Any]) -> None:
    if len(preds) != len(targets):
        raise ValueError(f"logits 与 labels 数量不一致: {len(preds)} != {len(targets)}")


def preds_from_logits(
    logits: Any,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    topk: Optional[int] = None,
) -> List[Any]:
    """默认预测函数：把 logits 转成类别预测。

    Args:
        logits: 一维得分序列按二分类阈值判定；二维 (N, K) 结构按行
            argmax（两列 logits 的二分类同样适用）。支持 torch tensor /
            numpy / 嵌套列表。
        num_classes: 类别数，None 表示二分类（一维输入时须为 None）。
        threshold: 一维得分的二分类判定阈值。
        topk: 返回每行得分最高的前 topk 个类别索引（Accuracy(topk) 用）。
    """
    data = _plain(logits)
    if data and isinstance(data[0], (list, tuple)):
        # 二维：多分类 argmax；topk 时返回每行前 topk 个类别
        if topk is None:
            return [max(range(len(row)), key=lambda i: float(row[i])) for row in data]
        return [
            sorted(range(len(row)), key=lambda i: float(row[i]), reverse=True)[:topk]
            for row in data
        ]
    if num_classes is not None:
        raise ValueError("一维输入按二分类阈值判定，请勿设置 num_classes")
    return [1 if float(x) >= threshold else 0 for x in data]


class Accuracy(BatchMetric):
    """准确率。

    继承 BatchMetric：只实现单 batch 计算，累积与跨 GPU 合并由框架完成。

    Args:
        num_classes / threshold / predictor: 同 preds_from_logits。
        topk: 大于 1 时，真实类别出现在前 topk 个预测类别中即算正确。
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        predictor: Optional[Predictor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.topk = topk
        self._predictor = predictor or preds_from_logits

    def compute_batch(self, logits: Any, labels: Any) -> Tuple[float, float]:
        preds = self._predictor(logits, self.num_classes, self.threshold, self.topk)
        targets = _plain(labels)
        _check_pair(preds, targets)
        hits = sum(
            1.0 if (int(t) in p if isinstance(p, (list, tuple)) else int(p) == int(t)) else 0.0
            for p, t in zip(preds, targets)
        )
        # (值, 样本数)：各 batch 样本数不同时，框架按样本数加权出全局准确率
        return (hits / len(targets), float(len(targets))) if targets else (0.0, 0.0)


class _CountMetric(BatchMetric):
    """基于 (pred, target) 计数的分类指标公共基类，跨 GPU 自动合并计数。

    计数型指标的全局结果（如 macro F1）无法由各 batch 值加权平均还原，
    因此重写 state / merge_states 以计数状态参与框架的同步合并，并重写
    _consume 把 compute_batch 的逐样本预测对累积为计数（feed 仍由
    基类负责：接收观测并按形参名转发给 compute_batch）。
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        predictor: Optional[Predictor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self._predictor = predictor or preds_from_logits
        self._counts: Dict[Tuple[int, int], float] = defaultdict(float)

    def compute_batch(self, logits: Any, labels: Any) -> List[Tuple[int, int]]:
        preds = self._predictor(logits, self.num_classes, self.threshold, None)
        targets = _plain(labels)
        _check_pair(preds, targets)
        return [(int(p), int(t)) for p, t in zip(preds, targets)]

    def _consume(self, out: List[Tuple[int, int]]) -> None:
        for key in out:
            self._counts[key] += 1.0

    def state(self) -> Dict[Tuple[int, int], float]:
        return dict(self._counts)

    def merge_states(self, states: List[Dict[Tuple[int, int], float]]) -> Any:
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        for s in states:
            for key, n in s.items():
                counts[key] += n
        return self._from_counts(counts)

    def reset(self) -> None:
        self._counts = defaultdict(float)

    def _from_counts(self, counts: Dict[Tuple[int, int], float]) -> Any:
        raise NotImplementedError


def _prf(mode: str, tp: float, fp: float, fn: float) -> float:
    if mode == "precision":
        return tp / (tp + fp) if tp + fp else float("nan")
    if mode == "recall":
        return tp / (tp + fn) if tp + fn else float("nan")
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def _per_class(counts: Dict[Tuple[int, int], float]) -> Dict[int, Tuple[float, float, float]]:
    """每个出现过的类别 -> (tp, fp, fn)。"""
    classes = {p for p, _ in counts} | {t for _, t in counts}
    out: Dict[int, Tuple[float, float, float]] = {}
    for c in classes:
        tp = counts.get((c, c), 0.0)
        fp = sum(n for (p, t), n in counts.items() if p == c and t != c)
        fn = sum(n for (p, t), n in counts.items() if t == c and p != c)
        out[c] = (tp, fp, fn)
    return out


class _PRF(_CountMetric):
    """Precision / Recall / F1 公共实现。"""

    _mode = "precision"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: Optional[str] = None,
        threshold: float = 0.5,
        predictor: Optional[Predictor] = None,
        class_index: Optional[int] = None,
    ):
        super().__init__(num_classes=num_classes, threshold=threshold, predictor=predictor)
        self.average = average
        self.class_index = class_index

    def _from_counts(self, counts: Dict[Tuple[int, int], float]) -> float:
        if not counts:
            return float("nan")
        if self.class_index is not None:  # 单类别（one-vs-rest），用于逐类曲线
            tp, fp, fn = _per_class(counts).get(self.class_index, (0.0, 0.0, 0.0))
            return _prf(self._mode, tp, fp, fn)
        if self.num_classes is None:  # 二分类：报告正类指标
            return _prf(
                self._mode,
                counts.get((1, 1), 0.0),
                counts.get((1, 0), 0.0),
                counts.get((0, 1), 0.0),
            )
        average = self.average or "macro"
        per = _per_class(counts)
        if average == "micro":
            tp = sum(v[0] for v in per.values())
            fp = sum(v[1] for v in per.values())
            fn = sum(v[2] for v in per.values())
            return _prf(self._mode, tp, fp, fn)
        if average == "macro":
            vals = [r for r in (_prf(self._mode, *v) for v in per.values()) if r == r]
            return sum(vals) / len(vals) if vals else float("nan")
        if average == "weighted":
            support: Dict[int, float] = defaultdict(float)
            for (_, t), n in counts.items():
                support[t] += n
            total = sum(support.values())
            acc = sum(
                _prf(self._mode, *v) * support[c]
                for c, v in per.items()
                if support[c] and _prf(self._mode, *v) == _prf(self._mode, *v)
            )
            return acc / total if total else float("nan")
        raise ValueError(f"未知的 average: {average}")


class Precision(_PRF):
    """精确率。average: None（二分类=正类，多分类=macro）/ "macro" / "micro" / "weighted"。

    class_index: 指定单类别（one-vs-rest）只算该类，配合 Web 端
    "precision/class_2" 式命名可逐类分图绘制曲线。
    """

    _mode = "precision"


class Recall(_PRF):
    """召回率。average 同 Precision，class_index 同 Precision。"""

    _mode = "recall"


class F1(_PRF):
    """F1 分数。average 同 Precision，class_index 同 Precision。"""

    _mode = "f1"


class ConfusionMatrix(_CountMetric):
    """混淆矩阵；compute() 返回嵌套列表，行=真实类别、列=预测类别。

    二分类为 [[tn, fp], [fn, tp]]；多分类为 K×K（K 取 num_classes，
    未指定时按出现过的类别推断）。值为矩阵而非标量，适合直接读取，
    不用于曲线记录。
    """

    def _from_counts(self, counts: Dict[Tuple[int, int], float]) -> List[List[float]]:
        if self.num_classes is not None:
            k = self.num_classes
        else:
            seen = {p for p, _ in counts} | {t for _, t in counts}
            k = max(2, (max(seen) + 1) if seen else 2)
        matrix = [[0.0] * k for _ in range(k)]
        for (p, t), n in counts.items():
            if 0 <= p < k and 0 <= t < k:
                matrix[t][p] += n
        return matrix


def _rank_auc(scores: List[float], labels: List[int]) -> float:
    """Mann-Whitney U 形式的 AUC：正类得分名次和法，并列得分取平均名次。"""
    n_pos = sum(1 for t in labels if t == 1)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:  # 只有一类时 AUC 无定义
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum_pos = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 名次从 1 起，并列取平均
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class AUC(BatchMetric):
    """ROC AUC（阈值无关的排序指标）。

    二分类：logits 为一维正类得分（(N, 2) 亦可，默认取第 pos_index 列），
    compute() 返回 float —— 二分类无需逐类拆分，单曲线即可。
    多分类：按 one-vs-rest 逐类计算，为每个类别各建一个实例并指定
    class_index，配合 "auc/class_0" 式命名由 Web 端按前缀分组绘图；
    未指定 class_index 的多分类输入会报错提示。

    跨 GPU：状态为逐样本 (得分, 是否正类) 对，合并即拼接后整体计算，
    与单进程全量结果一致。
    """

    def __init__(self, pos_index: int = 1, class_index: Optional[int] = None):
        super().__init__()
        self.pos_index = pos_index
        self.class_index = class_index
        self._pairs: List[Tuple[float, int]] = []

    def compute_batch(self, logits: Any, labels: Any) -> List[Tuple[float, int]]:
        rows = _plain(logits)
        targets = _plain(labels)
        _check_pair(rows, targets)
        if rows and isinstance(rows[0], (list, tuple)):
            if self.class_index is None and len(rows[0]) > 2:
                raise ValueError(
                    "多分类 AUC 请为每个类别各建一个实例并指定 class_index，"
                    "如 AUC(class_index=i)，命名 auc/class_i 供 Web 分组绘图"
                )
            col = self.class_index if self.class_index is not None else self.pos_index
            return [(float(row[col]), int(t) == col) for row, t in zip(rows, targets)]
        if self.class_index is not None:
            raise ValueError("一维输入为二分类正类得分，请勿设置 class_index")
        return [(float(s), bool(t)) for s, t in zip(rows, targets)]

    def _consume(self, out: List[Tuple[float, int]]) -> None:
        self._pairs.extend(out)

    def state(self) -> List[Tuple[float, int]]:
        return list(self._pairs)

    def merge_states(self, states: List[List[Tuple[float, int]]]) -> float:
        pairs = [p for s in states for p in s]
        if not pairs:
            return float("nan")
        return _rank_auc([s for s, _ in pairs], [1 if t else 0 for _, t in pairs])

    def reset(self) -> None:
        self._pairs = []
