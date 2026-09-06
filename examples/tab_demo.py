"""面板分区（tab）演示：左侧切换栏按 tab 独立展示（约 1.5 分钟）。

运行：python examples/tab_demo.py
启动后浏览器打开终端打印的 Web 地址，确认三点：
1. 左侧切换栏出现 train / val / test（声明顺序），默认打开 train；
2. 点击切换后只显示该分区的卡片；分区内按指标注册名分卡
   （train 块的 loss 卡、lr 卡互不合并）；
3. 分区内逐类命名（recall/class_0..2）照常合并为一张多系列卡片。
"""
import time

import melog
from melog import Last, Mean, MetricGroup, StepsBar, Sum

melog.init("outputs/melog-tab", web_port=8766)
melog.log("Web 地址:", melog.current().web_url)

CLASSES = 3


def make_metrics():
    """用户只需注册要监控的指标；分区归属由 StepsBar 的 tab=... 指定。"""
    return MetricGroup({
        "loss": Mean(),
        "acc": Mean(),
        "seen": Sum(),
        "lr": Last(),
        **{
            f"recall/class_{c}": Mean()
            for c in range(CLASSES)
        },
    })


train = make_metrics()
val = make_metrics()
test = make_metrics()

EPOCHS, STEPS = 4, 40


def observe(group, g, n=8):
    """模拟一批观测：各指标随全局步数 g 收敛。"""
    loss = 2.0 * pow(0.97, g) + 0.3
    group.feed(
        loss=loss,
        acc=1 - loss / 2.5,
        seen=n,
        lr=1e-3 * pow(0.98, g),
        **{
            f"recall/class_{c}": 1 - loss / 2.5 - 0.03 * c
            for c in range(CLASSES)
        },
    )


for epoch in range(EPOCHS):
    # 训练：feed 即实时记录本 tab 的曲线（面板 train 分块实时生长）；
    # 进度条 postfix 始终显示注册名（loss=...），分区前缀只用于落盘
    for step in StepsBar(range(STEPS), epoch=epoch, tab="train",
                         metrics=train):
        observe(train, epoch * STEPS + step)
        time.sleep(0.05)
    # 验证：不逐 batch 写曲线，epoch 末手动落盘一次（同样指定分区）
    for _ in StepsBar(range(10)):
        observe(val, epoch * STEPS + 10)
    melog.scalar(val, tab="val")
    val.reset()

# 测试：只记录一次
for _ in StepsBar(range(10)):
    observe(test, 10_000)
melog.scalar(test, tab="test")
test.reset()

melog.log("跑完了，左侧切换栏应有 train / val / test 三个 tab", "确认后 Ctrl+C 退出")
while True:
    time.sleep(1)  # 保持 Web 服务存活以便查看面板
