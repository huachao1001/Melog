"""指标计算与跨 GPU 同步。

- 基础指标：Mean / Sum / Last / Count
- 分类指标：Accuracy / Precision / Recall / F1 / ConfusionMatrix / AUC
- 自定义指标：继承 Metric，只实现 compute()（实时指标）；需要 epoch 末
  计算的再加 prepare()（备料增量），跨 GPU 合并由框架自动完成
- MetricGroup：具名指标集合，一次同步合并全部；feed() 自动分发
  logits/labels 与标量观测
"""

from .base import Metric
from .basic import Count, Last, Mean, Sum
from .classification import (
    Accuracy,
    AUC,
    ConfusionMatrix,
    F1,
    Precision,
    Recall,
    preds_from_logits,
)
from .group import MetricGroup

__all__ = [
    "Metric",
    "Mean",
    "Sum",
    "Last",
    "Count",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",
    "ConfusionMatrix",
    "preds_from_logits",
    "MetricGroup",
]
