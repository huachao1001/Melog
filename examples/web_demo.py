"""Web 展示测试脚本：持续推送多组指标，供浏览器实时查看曲线。

运行：python examples/web_demo.py
浏览器打开启动时打印的 Web 地址，应看到 loss/acc/lr/grad_norm 四条实时曲线。
按 Ctrl+C 停止。
"""

import math
import random
import time

import melog

STEPS = 3000       # 足够长，方便观察
INTERVAL = 0.5     # 每 0.5s 推送一次


def main():
    logger = melog.init("melog_runs/web-demo")
    print("浏览器打开查看实时曲线，Ctrl+C 退出")

    try:
        for step in logger.progress(range(STEPS)):
            loss = 2.0 * math.exp(-step / 150) + 0.05 + 0.02 * math.sin(step / 11) + random.uniform(-0.01, 0.01)
            acc = min(0.99, 1 - loss / 2.1 + random.uniform(-0.005, 0.005))
            grad_norm = 1.0 * math.exp(-step / 300) + random.uniform(0, 0.2)
            logger.scalar({
                "loss": loss,
                "acc": acc,
                "lr": 1e-3 * (0.995 ** step),
                "grad_norm": grad_norm,
            })
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
