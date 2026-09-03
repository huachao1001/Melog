"""多 GPU 训练示例：MetricGroup 指标 + 跨进程自动同步。

运行：
    torchrun --nproc_per_node=2 examples/multi_gpu_train.py
未安装 torch 时自动退化为单进程，行为与单 GPU 示例一致。

指标分三类：
- loss / acc：各 batch 等权平均（跨 GPU 自动合并）
- seen：样本数求和
- lr：最近一次喂入值
"""

import math
import os
import time

import melog
from melog import Last, Mean, MetricGroup, StepsBar, Sum
from melog.utils.distributed import get_rank, get_world_size, is_distributed

STEPS = 200
EPOCHS = 3


def main():
    rank = get_rank()
    melog.init("melog_runs/demo-multi")

    if rank == 0:
        url = melog.current().web_url
        ws = f", Web: {url}" if url else ""
        print(f"world_size={get_world_size()} 分布式={is_distributed()}{ws}")

    metrics = MetricGroup(
        {
            "loss": Mean(),   # 各 batch 等权平均
            "acc": Mean(),
            "seen": Sum(),    # 求和
            "lr": Last(),     # 最近一次喂入值
        }
    )

    for epoch in range(EPOCHS):
        # metrics=...：bar 实时显示本卡本地值（零通信）；迭代自然结束时
        # gather 合并全局值落盘并 reset（提前 break / 异常不触发）
        for step in StepsBar(range(STEPS), epoch=epoch, metrics=metrics, reset=True):
            # 模拟各 GPU 有差异的本地观测
            base = 2.0 * math.exp(-step / 60) + 0.05
            local_loss = base + 0.01 * (rank + 1) * math.sin(step / 9 + rank)
            local_acc = 1 - local_loss / 2.1

            # 只需累积本地值；跨 GPU 合并在 compute() 内自动完成
            metrics.feed(
                loss=local_loss,
                acc=local_acc,
                seen=16 + step % 8,
                lr=1e-3 * (0.98 ** (epoch * STEPS + step)),
            )
            time.sleep(0.02)

    if rank == 0:
        print(f"指标已落盘: {melog.current().run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
