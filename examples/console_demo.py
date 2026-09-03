"""控制台消息演示：log / success / error / warn + print 拦截。

运行：python examples/console_demo.py
（在真实终端里运行才能看到颜色；重定向时自动退化为纯文本）
"""

import time

from melog import Melog


def main():
    logger = Melog(project="console-demo", enable_web=False)

    logger.log("普通消息：多参数", {"k": 1}, 2)   # 终端默认色（黑字），无前缀
    logger.success("模型保存成功 -> checkpoint.pt")  # 绿色 ✔
    logger.error("加载失败：文件不存在")             # 红色 ✘
    logger.warn("学习率过大，可能不收敛")            # 黄色 ⚠

    # 官方 print 被拦截，内部改走 log()（自动同步 console.log）
    print("这行是用官方 print 打的，内部已改走 log()")

    for step in logger.progress(range(30)):
        logger.scalar({"loss": 1.0 / (step + 1)})
        time.sleep(0.05)

    logger.finish()
    print("finish 后 print 已还原为原生行为（本行不再进 console.log）")


if __name__ == "__main__":
    main()
