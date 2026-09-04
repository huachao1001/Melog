"""长时间进度条模拟：观察终端与控制台日志的进度条效果。

运行（另开一个终端实时查看日志文件）：

    python examples/progress_live_demo.py                # 默认 120 秒
    python examples/progress_live_demo.py 300            # 指定时长（秒）
    tail -f melog_runs/progress-demo/console-<时间戳>.log

终端里进度条原地实时刷新；日志文件里进度条行同样就地刷新（始终一行
在动，与终端内容一致，进度条字符替换为加高的块状样式更醒目，行首不
带时间戳）。消息插入时进度条行定稿、消息独占一行（带 [HH:MM:SS]
时间戳），进度条在消息下方重新开始。每次运行生成独立文件
console-<启动时间戳>.log，不跨会话追加。注意：就地刷新对 tail -f
不可见（无新增字节），查看进度条请用编辑器实时刷新。
"""

import math
import sys
import time

import melog
from melog import StepsBar

EPOCHS = 3
STEPS = 200


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    # 按目标时长反推每步间隔，保证跑满指定时间
    interval = max(0.01, duration / (EPOCHS * STEPS))

    melog.init("melog_runs/progress-demo")
    melog.log(f"开始模拟：{EPOCHS} epochs x {STEPS} steps，约 {EPOCHS * STEPS * interval:.0f} 秒")
    melog.log(f"控制台日志: {melog.current().console_log}")

    t0 = time.monotonic()
    for epoch in range(EPOCHS):
        melog.log(f"epoch {epoch} 开始")
        for step in StepsBar(range(STEPS), epoch=epoch, desc=f"epoch {epoch}"):
            g = epoch * STEPS + step
            loss = 2.0 * math.exp(-g / 120) + 0.05 + 0.02 * math.sin(g / 7)
            acc = 1 - loss / 2.1
            melog.scalar({"loss": loss, "acc": acc, "lr": 1e-3 * (0.98**g)})
            time.sleep(interval)

    melog.success(f"模拟完成，总耗时 {time.monotonic() - t0:.0f}s，日志: {melog.current().run_dir}")


if __name__ == "__main__":
    main()
