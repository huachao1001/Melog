"""category 大类别分区演示：train / val / test 垂直分块绘制（约 1.5 分钟）。

运行：python examples/category_demo.py
启动后浏览器打开终端打印的 Web 地址，确认三点：
1. 面板按大类别分成三个垂直分块（train / val / test 各占整行）；
2. 分块内按指标名分卡（train 块的 loss 卡、lr 卡互不合并）；
3. 分块内逐类命名（recall/class_0..2）照常合并为一张多系列卡片。
"""
import time

import melog
from melog import Last, Mean, MetricGroup, StepsBar, Sum

melog.init("/tmp/opencode/melog-category", web_port=8766)
melog.log("Web 地址:", melog.current().web_url)

CLASSES = 3


def make_metrics(category):
    """同一套指标定义，按大类别实例化：记录名自动加 category/ 前缀。"""
    return MetricGroup(
        {
            "loss": Mean(),
            "acc": Mean(),
            "seen": Sum(),
            "lr": Last(),
            **{f"recall/class_{c}": Mean() for c in range(CLASSES)},
        },
        category=category,
    )


train = make_metrics("train")
val = make_metrics("val")
test = make_metrics("test")

EPOCHS, STEPS = 4, 40


def observe(group, g, n=8):
    """模拟一批观测：各指标随全局步数 g 收敛。"""
    loss = 2.0 * pow(0.97, g) + 0.3
    group.feed(
        loss=loss,
        acc=1 - loss / 2.5,
        seen=n,
        lr=1e-3 * pow(0.98, g),
        **{f"recall/class_{c}": 1 - loss / 2.5 - 0.03 * c for c in range(CLASSES)},
    )


for epoch in range(EPOCHS):
    # 训练：feed 即实时记录本 category 的曲线（面板 train 分块实时生长）
    for step in StepsBar(range(STEPS), epoch=epoch, metrics=train):
        observe(train, epoch * STEPS + step)
        time.sleep(0.05)
    # 验证：不逐 batch 写曲线，epoch 末手动落盘一次
    for _ in StepsBar(range(10), metrics=val):
        observe(val, epoch * STEPS + 10)
    melog.scalar(val)
    val.reset()

# 测试：只记录一次
for _ in StepsBar(range(10), metrics=test):
    observe(test, 10_000)
melog.scalar(test)
test.reset()

melog.log("跑完了，面板应看到 train / val / test 三个垂直分块", "确认后 Ctrl+C 退出")
while True:
    time.sleep(1)  # 保持 Web 服务存活以便查看面板
