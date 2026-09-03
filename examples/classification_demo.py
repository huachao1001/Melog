"""多分类指标可视化演示：逐类 recall / F1 / AUC 曲线（同图多色对比）。

运行：python examples/classification_demo.py
浏览器打开终端打印的地址，应看到：
- acc / loss：普通单曲线卡片
- recall、f1、auc：每类一条曲线的分组卡片（legend 可逐类开关）
- 每个 epoch 起点有一条灰色虚线分界（标注 e0 / e1 / …），悬浮显示 epoch · step

模拟一个 4 类分类器，各类难度与样本量不同：
- 类别 0：简单、样本多，快速收敛
- 类别 1：中等
- 类别 2：困难，收敛慢，早期常与类别 0 混淆
- 类别 3：稀有（10% 样本），曲线噪声大，早期常与类别 1 混淆

指标用 "指标/类别" 层级命名（如 auc/class_2），Web 端自动按前缀分组。
二分类无需逐类拆分：Precision() / Recall() / F1() 默认报告正类，
AUC() 直接吃一维正类得分，名字不带 "/" 即可，渲染为普通单曲线卡片。

Ctrl+C 停止。
"""

import math
import random
import time

from melog import Accuracy, AUC, F1, Mean, Melog, MetricGroup, Recall

CLASSES = 4
EPOCHS = 5        # epoch 数：log 时传入 epoch，曲线按 epoch 画分界线
STEPS = 120       # 每个 epoch 的步数
BATCH = 64        # 每步模拟的 batch 大小
LOG_EVERY = 10    # 每 10 步汇总记录一次（相当于一个验证窗口）
INTERVAL = 0.05   # 每步间隔（秒），放慢以便观察曲线生长

SUPPORT = [0.40, 0.30, 0.20, 0.10]   # 各类采样比例（类别 3 稀有）
GAIN = [2.2, 1.8, 1.2, 0.9]          # 训练对各类得分的抬升幅度（越小越难学）
CONFUSE = {2: (0, 0.30), 3: (1, 0.40)}  # 类 -> (易混类, 早期混淆概率)


def simulate_batch(step, rng):
    """模拟第 step 步：模型能力随 step 对数增长，返回 (logits, labels)。"""
    progress = 1 - math.exp(-step / 150)
    labels, logits = [], []
    for _ in range(BATCH):
        y = rng.choices(range(CLASSES), weights=SUPPORT)[0]
        row = [rng.gauss(0, 1.0) for _ in range(CLASSES)]
        row[y] += GAIN[y] * progress + 0.15
        confuse_cls, p = CONFUSE.get(y, (None, 0.0))
        if confuse_cls is not None and rng.random() < p * (1 - progress):
            row[confuse_cls] += 1.0
        logits.append(row)
        labels.append(y)
    return logits, labels


def main():
    rng = random.Random(7)
    logger = Melog(project="demo-multiclass", web_port=8666)
    # 可选：为个别指标固定颜色（覆盖自动配色，其余仍按名称 hash 分配）
    logger.set_colors({"recall/class_2": "#ef4444", "recall/class_3": "#94a3b8"})
    url = logger._web.url if logger._web else "未启用"
    print(f"Web 可视化: {url} （重点看 recall / f1 / auc 三张多系列卡片，Ctrl+C 退出）")

    metrics = MetricGroup(
        {
            "loss": Mean(),
            "acc": Accuracy(num_classes=CLASSES),
            **{f"recall/class_{c}": Recall(num_classes=CLASSES, class_index=c) for c in range(CLASSES)},
            **{f"f1/class_{c}": F1(num_classes=CLASSES, class_index=c) for c in range(CLASSES)},
            **{f"auc/class_{c}": AUC(class_index=c) for c in range(CLASSES)},
        }
    )

    try:
        for epoch in range(EPOCHS):
            # 每个 epoch 一条进度条；每 LOG_EVERY 步才 log 一次，step 需显式传
            for step in logger.progress(range(STEPS)):
                g = epoch * STEPS + step  # 全局步数：模型能力按它增长
                logits, labels = simulate_batch(g, rng)
                loss = 1.8 * math.exp(-g / 180) + 0.4 + rng.gauss(0, 0.02)
                metrics.feed(logits=logits, labels=labels, loss=(loss, len(labels)))
                if step % LOG_EVERY == LOG_EVERY - 1:
                    out = metrics.compute()
                    # 窗口内个别类可能无样本（NaN），跳过不记录，曲线稍后补上
                    # 传入 epoch + 当前 epoch 的 step：曲线上标注 epoch 分界
                    logger.log({k: v for k, v in out.items() if v == v},
                               epoch=epoch, step=step)
                    metrics.reset()
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
