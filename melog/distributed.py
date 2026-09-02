"""多 GPU 指标合并。

基于 torch.distributed 的 all_reduce 实现跨进程聚合；
torch 未安装或未初始化分布式时，自动退化为单进程直通。
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "reduce_metrics",
]

try:
    import torch
    import torch.distributed as dist

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def is_distributed() -> bool:
    """当前是否处于已初始化的分布式训练环境。"""
    if not _TORCH_OK:
        return False
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_tensor(value: Any):
    # 已是张量则直接用，标量放到当前默认设备上参与 all_reduce
    if _TORCH_OK and torch.is_tensor(value):
        if value.ndim == 0:
            return value
        raise ValueError(f"仅支持标量或 0 维 tensor，收到 shape={tuple(value.shape)}")
    if not _is_number(value):
        raise TypeError(f"指标值须为数值类型，收到 {type(value).__name__}")
    if not _TORCH_OK:
        return value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(float(value), device=device, dtype=torch.float64)


def reduce_metrics(metrics: Dict[str, Any], op: str = "mean") -> Dict[str, float]:
    """合并所有进程的指标。

    Args:
        metrics: 指标字典，值为数值或 0 维 tensor。
        op: "mean"（默认，按 world_size 取平均）或 "sum"。
    """
    if not metrics:
        return {}
    if not is_distributed():
        # 单进程直通，避免不必要的拷贝
        out = {}
        for k, v in metrics.items():
            if _TORCH_OK and torch.is_tensor(v):
                out[k] = float(v.item())
            elif _is_number(v):
                out[k] = float(v)
            else:
                raise TypeError(f"指标值须为数值类型，收到 {type(v).__name__}")
        return out

    tensors = {k: _to_tensor(v) for k, v in metrics.items()}
    world_size = dist.get_world_size()
    for t in tensors.values():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    divisor = world_size if op == "mean" else 1
    return {k: float(t.item()) / divisor for k, t in tensors.items()}
