"""单进程训练示例：控制台进度条 + Web 可视化。

运行：python examples/single_gpu_train.py
浏览器打开 http://127.0.0.1:8666 查看实时曲线。
"""

import math
import time

from melog import Melog

EPOCHS = 4
STEPS = 50


def main():
    logger = Melog(project="demo-single", web_port=8666)
    print(f"Web 可视化: {logger._web.url if logger._web else '未启用'}")

    for epoch in range(EPOCHS):
        # 每个 epoch 一条进度条；每步恰好 log 一次，step 自动计数无需传
        for step in logger.progress(range(STEPS)):
            g = epoch * STEPS + step
            # 模拟一段收敛曲线
            loss = 2.0 * math.exp(-g / 60) + 0.05 + 0.02 * math.sin(g / 7)
            acc = 1 - loss / 2.1
            # 传入 epoch，曲线上标注 epoch 分界
            logger.log({"loss": loss, "acc": acc, "lr": 1e-3 * (0.98 ** g)}, epoch=epoch)
            time.sleep(0.03)

    logger.finish()
    print(f"指标已落盘: {logger.run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
