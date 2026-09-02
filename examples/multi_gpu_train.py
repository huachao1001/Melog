"""多 GPU 训练示例：指标跨进程合并（取均值）后仅 rank0 记录与展示。

运行：
    torchrun --nproc_per_node=2 examples/multi_gpu_train.py
未安装 torch 时自动退化为单进程，行为与单 GPU 示例一致。
"""

import math
import os
import time

from melog import Melog
from melog.distributed import get_rank, get_world_size, is_distributed

STEPS = 200


def main():
    rank = get_rank()
    logger = Melog(project="demo-multi", web_port=8666)

    if rank == 0:
        ws = f", Web: {logger._web.url}" if logger._web else ""
        print(f"world_size={get_world_size()} 分布式={is_distributed()}{ws}")

    with logger.train(total=STEPS, description=f"rank{rank}") as bar:
        for step in range(STEPS):
            # 模拟各 GPU 有差异的 loss，all_reduce 后取均值
            base = 2.0 * math.exp(-step / 60) + 0.05
            local_loss = base + 0.01 * (rank + 1) * math.sin(step / 9 + rank)
            local_acc = 1 - local_loss / 2.1
            logger.log({"loss": local_loss, "acc": local_acc})
            bar.advance(1)
            time.sleep(0.03)

    logger.finish()
    if rank == 0:
        print(f"指标已落盘: {logger.run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
