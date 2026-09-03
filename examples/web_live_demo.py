"""实时训练输出指标 + Web 面板渲染验证（约 2 分钟）。

运行：python examples/web_live_demo.py
启动后浏览器打开终端打印的 Web 地址，曲线实时生长；
跑完后可用  melog /tmp/opencode/melog-live  离线回看。
"""
import time

import melog
from melog import Mean, MetricGroup, StepsBar, Sum

melog.init("/tmp/opencode/melog-live", web_port=8765)
melog.log("Web 地址:", melog.current().web_url)

metrics = MetricGroup({
    "loss": Mean(),
    "acc": Mean(),
    "seen": Sum(),
})

EPOCHS, STEPS = 5, 60
for epoch in range(EPOCHS):
    for step in StepsBar(range(STEPS), epoch=epoch, metrics=metrics):
        g = epoch * STEPS + step
        loss = 2.0 * pow(0.97, g) + 0.3
        acc = 1 - loss / 2.5
        metrics.feed(loss=loss, acc=acc, seen=8)
        time.sleep(0.1)  # 放慢以便观察实时推送
print("done")
