"""多 GPU 训练示例：MetricGroup 指标 + 跨进程自动同步。

运行：
    torchrun --nproc_per_node=2 examples/multi_gpu_train.py
未安装 torch 时自动退化为单进程，行为与单 GPU 示例一致。

指标分三类：
- loss / acc：加权平均（按 batch_size 加权，跨 GPU 自动合并）
- seen：样本数求和
- best_acc：历史最大值
"""

import math
import os
import time

from melog import Melog, Mean, Max, MetricGroup, Sum
from melog.distributed import get_rank, get_world_size, is_distributed

STEPS = 200
EPOCHS = 3


def main():
    rank = get_rank()
    logger = Melog(project="demo-multi", web_port=8666)

    if rank == 0:
        ws = f", Web: {logger._web.url}" if logger._web else ""
        print(f"world_size={get_world_size()} 分布式={is_distributed()}{ws}")

    metrics = MetricGroup(
        {
            "loss": Mean(),      # 加权平均
            "acc": Mean(),       # 加权平均
            "seen": Sum(),       # 求和
            "best_acc": Max(),   # 历史最大
            "lr": Mean(),        # Last 亦可：多卡时取 rank0 的值
        }
    )

    with logger.train(total=STEPS, description=f"rank{rank}") as bar:
        for epoch in range(EPOCHS):
            for step in range(STEPS):
                # 模拟各 GPU 有差异的本地观测
                base = 2.0 * math.exp(-step / 60) + 0.05
                local_loss = base + 0.01 * (rank + 1) * math.sin(step / 9 + rank)
                local_acc = 1 - local_loss / 2.1
                batch_size = 16 + step % 8

                # 只需累积本地值；跨 GPU 合并在 compute() 内自动完成
                metrics.update(
                    loss=(local_loss, batch_size),
                    acc=(local_acc, batch_size),
                    seen=batch_size,
                    best_acc=local_acc,
                    lr=1e-3 * (0.98 ** (epoch * STEPS + step)),
                )
                bar.advance(1)
                time.sleep(0.02)

            # epoch 末：所有 rank 统一调用，得到全局一致结果；记录后重置
            logger.log_group(metrics, reset=True)

    logger.finish()
    if rank == 0:
        print(f"指标已落盘: {logger.run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
