"""指标计算与跨 GPU 同步。

- 基础指标：Mean / Sum / Last / Count
- 分类指标：Accuracy / Precision / Recall / F1 / ConfusionMatrix，
  继承 BatchMetric，框架自动传入 logits 与 labels 并完成累积合并
- 自定义单批次指标：继承 BatchMetric 只实现 compute_batch() 一个函数，
  累积、合并、reset、分布式同步全部由框架完成
- 自定义 epoch 级指标：继承 Metric 实现 update / state / merge_states /
  reset，跨 GPU 的状态收集与单进程直通由基类 compute() 完成
- MetricGroup：具名指标集合，一次同步合并全部；feed() 自动分发
  logits/labels 与标量观测
"""

from .base import BatchMetric, Metric
from .basic import Count, Last, Mean, ScalarMetric, Sum
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
    "BatchMetric",
    "ScalarMetric",
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
