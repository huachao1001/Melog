"""全局入口与模块级便捷接口。

melog.init() 创建并激活全局共享实例（不返回）；之后项目任意位置
直接使用本模块定义的模块级接口（melog.scalar / melog.log 等），
无需持有实例。需要实例本身时用 melog.current() 取回。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from ..core import Melog, current
from ..metrics import MetricGroup
from ..tracking.steps_bar import StepsBar
from ..utils.tqdm import tqdm

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
]


def init(log_dir: str = "./melog_runs", web_port: Optional[int] = None, **kwargs: Any) -> None:
    """创建并激活全局共享的 Melog 实例（melog 的唯一公开入口）。

    不返回实例：入口处调用一次后，项目任意位置直接使用模块级
    melog.scalar() 等接口（需要实例本身时用 melog.current() 取回）。

    Args:
        log_dir: 日志保存路径；本次运行的指标 / 媒体 / console.log 落在
            其下的时间戳子目录中，路径末级目录名作为项目名展示。
        web_port: Web 监听端口；缺省自动选择一个空闲端口。
        **kwargs: 其余高级参数（enable_web / enable_progress / reduce_op /
            flush_every / max_plot_points，以及 project 覆盖项目名等）。

    再次调用会用新实例替换当前活动实例。
    """
    log_dir = Path(log_dir)
    kwargs.setdefault("project", log_dir.name)
    Melog(output_dir=str(log_dir.parent), web_port=web_port, **kwargs)


def stepsbar(
    iterable: Iterable,
    total: Optional[float] = None,
    epoch: Optional[int] = None,
    metrics: Optional[MetricGroup] = None,
    **kwargs: Any,
) -> StepsBar:
    """模块级便捷接口：等价于 ``StepsBar(iterable, ...)``。"""
    return StepsBar(iterable, total=total, epoch=epoch, metrics=metrics, **kwargs)


def scalar(
    metrics: Union[Dict[str, Any], MetricGroup],
    advance: int = 0,
) -> Dict[str, float]:
    """模块级便捷接口：等价于 ``current().scalar(...)``。"""
    return current().scalar(metrics, advance=advance)


def log(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().log(...)``。"""
    current().log(*values, sep=sep, end=end, flush=flush)


def success(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().success(...)``。"""
    current().success(*values, sep=sep, end=end, flush=flush)


def error(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().error(...)``。"""
    current().error(*values, sep=sep, end=end, flush=flush)


def warn(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """模块级便捷接口：等价于 ``current().warn(...)``。"""
    current().warn(*values, sep=sep, end=end, flush=flush)


def current_bar() -> Optional[tqdm]:
    """模块级便捷接口：等价于 ``current().current_bar()``。"""
    return current().current_bar()


def image(
    name: str,
    data: Any,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().image(...)``。"""
    current().image(name, data, caption=caption)


def audio(
    name: str,
    data: Any,
    sr: int = 22050,
    caption: Optional[str] = None,
) -> None:
    """模块级便捷接口：等价于 ``current().audio(...)``。"""
    current().audio(name, data, sr=sr, caption=caption)


def set_colors(colors: Dict[str, str]) -> None:
    """模块级便捷接口：等价于 ``current().set_colors(...)``。"""
    current().set_colors(colors)
