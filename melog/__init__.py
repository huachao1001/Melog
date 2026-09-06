"""Melog 包入口。"""

from .api import (
    StepsBar,
    audio,
    current,
    current_bar,
    error,
    image,
    init,
    log,
    scalar,
    set_colors,
    stepsbar,
    success,
    tqdm,
    warn,
)
from .metrics import (
    Accuracy,
    AUC,
    ConfusionMatrix,
    Count,
    F1,
    Last,
    Mean,
    Metric,
    MetricGroup,
    Precision,
    Recall,
    Sum,
)

try:  # 安装后的版本随包元数据（与 pyproject 一致）；源码直接导入无元数据时回退
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("melog")
except Exception:  # pragma: no cover
    __version__ = "1.4.0"  # 发布脚本（scripts/release.py）升版本时同步更新此回退值
__all__ = [
    "init",
    "current",
    "current_bar",
    "tqdm",
    "StepsBar",
    "stepsbar",
    "scalar",
    "log",
    "image",
    "audio",
    "success",
    "error",
    "warn",
    "set_colors",
    "Metric",
    "MetricGroup",
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
    "__version__",
]
