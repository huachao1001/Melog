"""单进程训练示例：控制台进度条 + Web 可视化。

运行：python examples/single_gpu_train.py
浏览器打开启动时打印的 Web 地址查看实时曲线。
"""

import math
import time

import melog
from melog import StepsBar

EPOCHS = 4
STEPS = 50


def main():
    melog.init("melog_runs/demo-single")

    for epoch in range(EPOCHS):
        # 每个 epoch 一条进度条并绑定 epoch；坐标（epoch/step）自动管理
        for step in StepsBar(range(STEPS), epoch=epoch):
            g = epoch * STEPS + step
            # 模拟一段收敛曲线
            loss = 2.0 * math.exp(-g / 60) + 0.05 + 0.02 * math.sin(g / 7)
            acc = 1 - loss / 2.1
            # 记录自动依附当前 epoch，曲线上标注 epoch 分界
            melog.scalar({"loss": loss, "acc": acc, "lr": 1e-3 * (0.98 ** g)})
            time.sleep(0.03)

    print(f"指标已落盘: {melog.current().run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
