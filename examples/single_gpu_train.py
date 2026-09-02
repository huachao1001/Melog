"""单进程训练示例：控制台进度条 + Web 可视化。

运行：python examples/single_gpu_train.py
浏览器打开 http://127.0.0.1:8666 查看实时曲线。
"""

import math
import time

from melog import Melog

STEPS = 200


def main():
    logger = Melog(project="demo-single", web_port=8666)
    print(f"Web 可视化: {logger._web.url if logger._web else '未启用'}")

    with logger.train(total=STEPS, description="demo") as bar:
        for step in range(STEPS):
            # 模拟一段收敛曲线
            loss = 2.0 * math.exp(-step / 60) + 0.05 + 0.02 * math.sin(step / 7)
            acc = 1 - loss / 2.1
            logger.log({"loss": loss, "acc": acc, "lr": 1e-3 * (0.98 ** step)})
            bar.advance(1)
            time.sleep(0.03)

    logger.finish()
    print(f"指标已落盘: {logger.run_dir / 'metrics.melog'}")


if __name__ == "__main__":
    main()
