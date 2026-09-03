"""多模态演示：曲线 + 图像 + 音频三个页签。

运行：python examples/media_demo.py
浏览器打开终端打印的地址，用 header 页签切换：
- 曲线页签：loss 收敛曲线（含 epoch 分界线）
- 图像页签：sample/grid 卡片——滑杆回放各 step 的图像，点击看原图
- 音频页签：sample/tone 卡片——播放随训练由沙哑变纯净的音色

媒体接口：
    logger.image("名字", 图片路径/PIL/numpy/torch, step=?, epoch=?)
    logger.audio("名字", 音频路径/numpy/torch, sr=采样率, step=?, epoch=?)
step/epoch 缺省时自动附着到最近一次 scalar() 的位置（本示例即这种用法）。

Ctrl+C 停止。
"""

import math
import time

import numpy as np

import melog

EPOCHS = 3
STEPS = 40        # 每个 epoch 步数
IMG_EVERY = 10    # 每 10 步记录一帧图像
AUD_EVERY = 15    # 每 15 步记录一段音频
INTERVAL = 0.05


def make_image(step: int, total: int) -> np.ndarray:
    """合成一帧 (H,W,3) 渐变图：亮斑位置随训练逐渐收敛到中心。"""
    h = w = 96
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [xx / w, yy / h, 0.5 + 0.5 * np.sin(xx / 7.0 + step / 3.0)], axis=-1
    )
    t = step / total
    cy, cx = h / 2 + (0.5 - t) * 30, w / 2 - (0.5 - t) * 30
    glow = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 60.0)
    img[..., 0] += glow * 0.8
    return np.clip(img, 0.0, 1.0)


def make_audio(step: int, total: int, sr: int = 22050) -> np.ndarray:
    """合成 1.2s 波形：噪声随训练衰减、主音逐渐变纯（沙哑 -> 纯净）。"""
    t = np.arange(int(sr * 1.2)) / sr
    purity = step / total
    tone = 0.5 * np.sin(2 * np.pi * (330 + 220 * purity) * t)
    tone *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)  # 缓慢颤音
    noise = np.random.default_rng(step).normal(0, 0.25 * (1 - purity), len(t))
    return np.clip(tone * 0.7 + noise, -1.0, 1.0)


def main():
    logger = melog.init("melog_runs/demo-media")   # 日志保存路径；Web 地址启动时自动打印
    print("页签切换 曲线/图像/音频，Ctrl+C 退出")

    total = EPOCHS * STEPS
    try:
        for epoch in range(EPOCHS):
            # tqdm 风格：包裹可迭代对象即自动推进，scalar() 指标实时显示在条上
            for step in logger.progress(range(STEPS)):
                g = epoch * STEPS + step
                loss = 1.6 * math.exp(-g / 80) + 0.2
                logger.scalar({"loss": loss}, epoch=epoch)
                if g % IMG_EVERY == 0:
                    # caption：随图显示的配文（textContent 渲染，支持换行）
                    logger.image("sample/grid", make_image(g, total),
                                   caption=f"亮斑位置 t={g / total:.2f}（应逐渐移向中心）")
                if g % AUD_EVERY == 0:
                    logger.audio("sample/tone", make_audio(g, total), sr=22050,
                                   caption=f"纯净度 {g / total:.0%}，噪声已衰减")
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n手动停止")
        return

    # 保持 Web 服务运行，方便在浏览器里拖滑杆回放图像/音频；历史查看也可用: melog <run_dir>
    print("\n训练结束，浏览器中可回放图像/音频（滑杆拖动）。按回车退出。")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass  # 进程退出时自动收尾（atexit）


if __name__ == "__main__":
    main()
