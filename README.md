# Melog

轻量级训练监控库：**多 GPU 指标合并 + 控制台实时进度条 + Web 可视化**。

```text
epoch 3 loss=0.2153 acc=0.8974 lr=8.2e-04 ━━━━━━━━━━━──────────  45.0% [90/200] [0:03<0:04 30.0it/s]
```

## 特性

- **控制台实时进度条**：自研 tqdm（用法与 tqdm.tqdm 一致），`[n/total]` 领先、指标紧随其后实时刷新；进度条与 print 同步镜像到 console.log（进度条行 2 秒节流刷新）
- **多 GPU 指标合并**：基于 `torch.distributed` all_reduce 跨进程聚合（默认取均值），仅 rank0 记录与展示；未装 torch 自动退化单进程
- **Web 可视化**：FastAPI + WebSocket + ECharts，后台线程运行，实时推送曲线，断线自动重连
- **持久化**：指标自动写入 JSONL，供离线分析

## 安装

```bash
pip install -e .            # 基础安装
```

多 GPU 合并基于 `torch.distributed`，假定环境中已装好 PyTorch；未装 torch 时自动退化单进程。

## 快速开始

```python
import melog
from melog import StepsBar

melog.init("runs/my-exp")            # 日志保存路径；端口缺省自动选空闲端口
                                     # Web 地址启动时自动打印，也可读 melog.current().web_url

for step in StepsBar(range(1000)):   # tqdm 风格：自动推进，无需手动 update
    loss = train_one_step()
    melog.scalar({"loss": loss, "lr": 1e-3})   # 记录 + 刷新进度条指标 + 推送 Web
```

训练期间浏览器打开启动时打印的 Web 地址（即 `melog.current().web_url`）查看实时曲线。

## 全局共享

`melog.init` 是唯一入口，创建的实例自动成为全局活动实例。入口处 `init` 一次，
项目任何地方直接用模块级接口，无需层层传递实例：

```python
import melog

melog.init(log_dir="runs/my-exp")   # 日志保存路径；端口缺省自动选空闲端口

# 任意其他模块中：
import melog
melog.scalar({"loss": 0.5})
melog.image("sample", img)
# 收尾：进程退出时自动完成，无需调用
```

- `log_dir` 末级目录名即项目名（`runs/my-exp` → 项目 `my-exp`），本次运行落在
  `runs/my-exp/<时间戳>/` 下；`project=` 可覆盖项目名
- 最近一次创建的实例即全局活动实例（`melog.current()` 取回），收尾后清空
- 进程退出时经 atexit 自动收尾：落盘剩余指标、定稿进度条、停 Web、还原 print，
  无需任何手动调用
- 模块级 `scalar / image / audio / log / success / error / warn / set_colors / current_bar` 与实例方法等价
- 实例内部有锁，多线程 / 多模块共享安全；多 GPU 约定不变

## 曲线上体现 epoch

本库**按 epoch 组织训练记录**：每个 epoch 的循环必须用 `StepsBar` 包裹并传入
`epoch`，坐标（epoch / step）由它统一管理——`scalar()` / `image()` /
`audio()` 都**没有坐标参数**，记录自动依附当前 epoch 与下一个空槽：

```python
from melog import StepsBar

for epoch in range(epochs):
    for _ in StepsBar(loader, epoch=epoch):        # 行首自动标注 "epoch N"
        loss = train_one_step()
        melog.scalar({"loss": loss, "lr": lr})     # 坐标自动依附当前 epoch
```

- `StepsBar(epoch=...)` 进入进度条即绑定 epoch：epoch 内步数清零、全局 x 从上一位置
  接续；bar 结束后沿用绑定值，直至下一个 epoch
- `step` 为**当前 epoch 内**的记录序号，内部自增（每个 epoch 从 0 重新计步）；
  完全没用 `StepsBar(epoch=...)` 时退化为全局自增 x、不标注 epoch 分界
- 要控制记录粒度（每步 / 每 N 步窗口），调整调用 `scalar()` 的频率即可，无需手动指定坐标
- Web 曲线在每个 epoch 起点画分界虚线（标注 `e0` / `e1` / …），悬浮提示显示 `epoch N · step X`
- `MetricGroup` 末尾收尾交给 `StepsBar` 的自动记录（见下文），epoch 沿用绑定值

## 控制台消息

print 风格的控制台输出接口：多参数自动转 `str()`、以 `sep` 拼接，签名对齐 `print`
（支持 `sep` / `end` / `flush`）：

```python
melog.log("普通消息", {"k": 1})   # 终端默认色（黑字），无前缀
melog.success("保存完成")         # 绿色 ✔
melog.error("加载失败")           # 红色 ✘
melog.warn("学习率过大")          # 黄色 ⚠
```

实例存活期间（仅 rank0），官方 `print(...)` 会被拦截内部改走 `log()`——普通打印
自动带上图标/配色并同步进 console.log，进程退出收尾后还原原生 print。颜色仅在真实
终端（TTY）启用，重定向 / console.log 始终纯文本。

## 记录图像与音频

除指标曲线外，Web 端 header 可在 **曲线 / 图像 / 音频** 三个页签间切换。图像与音频用
`image` / `audio` 记录，Web 端按名字建卡片、滑杆按 step 回放（图像点击看原图，
音频在线播放）；文件自动落盘到 `run_dir/media/`，元数据随日志持久化，历史日志加载时
媒体一并恢复：

```python
melog.scalar({"loss": loss})            # 坐标自动依附当前 epoch（StepsBar 绑定）
melog.image("train/sample", img)        # 路径 / PIL / numpy / torch
melog.image("val/sample", img)          # 自动附着最近一次 scalar() 的位置
melog.audio("val/audio", wav, sr=16000) # 路径(wav/mp3/…) / numpy / torch 波形
```

- 图像 / 音频自动附着到**最近一次 `scalar()` 的位置**，不推进计数；
  坐标（epoch / step）由 `StepsBar` 统一管理，接口无坐标参数
- `caption="..."` 可为每条图像 / 音频配一段文字（如样本说明、转写文本），
  显示在卡片上、随滑杆切换；换行会被保留
- 图像：`(H,W)` 灰度或 `(H,W,C)`（C=1/3/4），浮点自动映射 0-255，统一存为 PNG
- 音频：`(N,)` 单声道或 `(N, 声道数)`，浮点按 [-1,1] 裁剪存为 16bit WAV；
  传文件路径则按原格式复制
- 数组编码需要 `pillow`（仅图像）：`pip install pillow`

## 指标计算（多 GPU 自动同步）

内置 `Mean` / `Sum` / `Last` / `Count`，按 epoch 组织在 `MetricGroup` 中使用：

```python
import melog
from melog import Last, Mean, MetricGroup, StepsBar, Sum

melog.init("runs/my-exp")
metrics = MetricGroup({
    "loss": Mean(),      # 各 batch 等权平均
    "acc": Mean(),
    "seen": Sum(),       # 求和
    "lr": Last(),        # 最近一次喂入值
})

for epoch in range(epochs):
    # metrics=... 传入后：每次 feed 自动把本卡本地值写入日志/面板
    # （实时曲线，零通信），epoch 末自动跨 GPU 合并出全局值再记录
    # 一次并 reset 清零
    for _ in StepsBar(range(steps), epoch=epoch, metrics=metrics):
        metrics.feed(loss=loss, acc=acc, seen=batch_size, lr=lr)
    # feed(..., write=False) 只累积内存（如验证集不想逐 batch 写曲线）；
    # epoch 末仍自动跨 GPU 合并记录 + reset，无需手动 scalar
```
- `Mean` 默认按 StepsBar 自动识别的**批次样本数**精确平均（feed 无需传
  batch_size；各 batch 等样本数时即等权）；多 GPU 下合并为全局样本平均，
  而非"各卡平均值的平均"
- 批次样本数自动识别（tensor / numpy 的 shape[0]、字典、列表 / 元组递归）；
  识别失败（如迭代 range）回退等权平均并警告一次。需手动指定时传**元组**
  `(值, 观测数)`：`metrics.feed(loss=(loss, token_num))`，显式值优先
- `melog.scalar(metrics)` 随时可落盘当前累计值；**必须算完一个 epoch 才有意义的指标**，
  在 epoch 末统一记录一次即可（交给 StepsBar 自动执行，或手动调用）
- 跨 GPU 合并是集合操作：**所有 rank 必须以相同顺序执行**，返回值各 rank 一致；单进程自动直通
- 实时 + 精确一步到位：`StepsBar(loader, epoch=e, metrics=metrics)`——
  每次 feed 自动把本卡本地值写入日志/面板（零通信，仅 rank0 落盘），bar 同步实时显示；
  迭代自然结束时自动 gather 所有 rank 合并出**全局值**再记录一次（提前 break / 抛异常不触发，
  以免各 rank 在 all_gather 处互相等待；所有 rank 都会执行，落盘仅 rank0）。
  `feed(..., write=False)` 关闭逐 batch 实时写入（如验证集场景），epoch 末
  仍自动合并记录，无需手动 scalar

### feed 如何分发观测

`metrics.feed(args=..., **scalars)` 把一个 batch 的观测一次喂入，两类指标
**分开传、各取所需**。以

```python
metrics = MetricGroup({"loss": Mean(), "macc": MaskedAcc()})
metrics.feed(args={"logits": logits, "labels": labels, "mask": mask},
             loss=(loss, batch_size))
```

为例，一次 feed 内部的流转：

- **`args=`：观测型指标**（如 `"macc"` 与所有内置分类指标）的观测，
  单独成组——**字典**按键名对应各指标 `compute` / `prepare` 的形参
  （推荐，形参多时更可读），自动分发给形参名匹配的指标，多余的键忽略；
  **元组**按位置喂给未被注册名喂入的指标。
  缺少必需形参才抛 `KeyError`。
- **`**scalars`：按注册名喂入的指标**（`Mean` / `Sum` / `Last` / `Count`，如 `"loss"`）：
  按**注册名**找同名键——取出 `loss=(loss, batch_size)`；是元组就展开为
  `feed(loss, batch_size)` 加权累积，普通数值则等权。本 batch 没有同名键就跳过
  （不累积也不报错）。

一句话：**观测型指标的观测放 `args`，标量指标按注册名"点名取值"**。两类规则
互不干扰，所以同一个 feed 调用可以同时喂两类指标；无主的多余观测两边都不收。

单独使用某个指标时规则一致：标量指标位置喂入 `Mean().feed(value, count)`；
观测型指标具名或位置均可 `MaskedAcc().feed(logits=..., labels=..., mask=...)`，
框架同样按 `compute` / `prepare` 形参名组装。

### 分类指标

内置 `Accuracy` / `Precision` / `Recall` / `F1` / `ConfusionMatrix`，接口与基础指标一致，
`feed(logits, labels)` 直接接收模型输出与标签：

```python
from melog import Accuracy, F1, MetricGroup, Mean, Precision

metrics = MetricGroup({
    "loss": Mean(),
    "acc": Accuracy(),                 # 二分类：一维得分按阈值 0.5 判定
    "acc5": Accuracy(topk=5),          # top-5 准确率（多分类）
    "f1": F1(num_classes=10),          # 多分类：二维 (N, K) logits 按行 argmax
})

# 验证集：write=False 不逐 batch 写曲线，epoch 末自动跨 GPU 合并记录
for logits, labels in StepsBar(val_loader, epoch=epoch, metrics=val_metrics):
    # feed：观测型指标的观测放 args（元组按位置 / 字典按键名），
    # 标量指标按注册名喂入（loss 自动按批次样本数平均）
    val_metrics.feed(args=(logits, labels), loss=loss, write=False)
# 无需手动 scalar：StepsBar 结束时自动合并记录并 reset
```

- `Accuracy(topk=k)`：真实类别在前 k 个预测中即算正确
- `Precision / Recall / F1` 的 `average`：`None`（二分类=正类，多分类=macro）/ `"macro"` / `"micro"` / `"weighted"`
- `ConfusionMatrix` 的 `compute()` 返回矩阵（行=真实、列=预测），适合直接读取而非画曲线
- 预测规则由 `preds_from_logits` 实现，可传 `predictor=` 替换（如多标签、分割等自定义转换）

### 自定义指标

统一继承 `Metric`，按指标何时出值选择实现方式：

**实时指标——只实现 `compute()`**：每次喂入立即用本批观测算出指标值，
形参名和个数完全由你定义，框架按形参名自动从 `feed()` 的观测中取值回调；
各 batch 结果按观测数加权平均、跨 GPU 合并，全部由框架完成：

```python
from melog import Metric

class MaskedAcc(Metric):
    """需要几个参数就声明几个，logits/labels 仅为示例。"""
    def compute(self, logits, labels, mask):
        hits = ((logits.argmax(-1) == labels) & mask).sum()
        n = mask.sum()
        return (hits / n, n)          # 返回 (值, 观测数)：按样本数平均出全局结果

# 训练循环里：位置或具名喂入均可，多余观测自动忽略
metric.feed(logits, labels, mask)
metric.feed(logits=logits, labels=labels, mask=mask)
```

- `compute` 返回 `(值, 观测数)` 元组：各 batch 按观测数（如样本数）平均（样本数不同时务必带上）；
  只返回 float 时各 batch 等权平均
- 组合使用时交给 `MetricGroup.feed(...)` 统一分发：

```python
metrics = MetricGroup({"loss": Mean(), "macc": MaskedAcc()})

# 每个 batch：feed 把观测累积进各指标的内存状态并自动记录本卡实时值
# （观测型指标观测放 args，标量指标按注册名，Mean 自动按批次样本数平均）
for logits, labels, mask in StepsBar(loader, epoch=epoch, metrics=metrics):
    metrics.feed(args={"logits": logits, "labels": labels, "mask": mask},
                 loss=loss)

# epoch 末无需任何手动调用：StepsBar 结束时自动跨 GPU 合并记录 + reset
```

不用 StepsBar 包裹时（如独立验证脚本）才需要手动落盘：
`melog.scalar(metrics)` 跨 GPU 合并记录，`metrics.reset()` 清零开启下一轮。

**epoch 级指标——加实现 `prepare()`**：全局结果无法由各 batch 值加权平均还原时
（如 macro F1、AUC），每次喂入先用 `prepare()`"备料"——接收同样的观测，
返回本批次贡献的增量（数值 / 字典 / 列表），框架自动累积并跨 GPU 合并
（数值求和、字典按键合并、列表拼接）；epoch 末把合并后的总量交给
`compute()` 算出全局结果：

```python
from melog import Metric

class F1(Metric):
    """epoch 末才能计算的指标：累积混淆计数，末尾统一算。"""
    def prepare(self, tp, fp, fn):      # 每个 batch：本批次贡献的计数
        return {"tp": tp, "fp": fp, "fn": fn}

    def compute(self, tp, fp, fn):      # epoch 末：由总量算出全局值
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")

# 使用（通常放进 MetricGroup / StepsBar 自动记录）：
f1 = F1()
f1.feed(tp=2, fp=1, fn=0)   # 位置或具名喂入均可
f1.result()                  # 跨 GPU 合并并计算（单进程直通）
```

## 多 GPU

代码无需修改，用 `torchrun` 启动即可：

```bash
torchrun --nproc_per_node=4 examples/multi_gpu_train.py
```

代码无需修改：每个 rank 进程各自跑同一份 melog 代码，`melog.init` 后
库自动感知分布式环境（`torch.distributed` 已初始化即生效；未装 torch
或单进程运行则所有集合操作退化为本地直通，行为完全一致）。

### 卡间如何传递

通信只发生在"记录"时，`feed()` 永远零通信。两种原语按数据形态分工：

- **`all_reduce`（数值求和）**：`melog.scalar({...})` 的普通数值指标
  （float / int / 0 维 tensor，内部统一转 float64 张量）。每个 rank 的
  同名指标相加后除以卡数（`reduce_op="mean"`，可选 `"sum"` 不除），
  各 rank 得到一致的全局值。
- **`all_gather_object`（pickle 收集）**：`MetricGroup` 的合并。每个
  rank 把本地状态整体 pickle 后互相收集（结果按 rank 顺序排列），
  再用各指标的合并规则还原全局值——实时指标（Mean / Accuracy 等）
  状态是 `[加权总和, 观测数]` 两行数值，按观测数加权平均；epoch 级
  指标（F1 / AUC 等）状态是 prepare 备料的增量，数值求和、字典按键
  合并、列表拼接。一组指标无论几个，**每个 epoch 只做一次收集**。

各 rank 状态是纯 Python 数据（数值 / 嵌套字典 / 列表），可 pickle 即可，
不要求张量、不要求同构——各 rank 观测到的类别可以不同（如 F1 的计数
字典按键求和自动对齐）。

### rank 分工与同步纪律

- **rank0** 负责所有落盘与展示：JSONL 日志、Web 服务、进度条渲染、
  feed 实时值的写入；其余 rank 静默但仍参与每次集合操作（集合操作
  是全员的，谁都不能缺席）。
- **同步纪律**：所有 rank 必须以相同顺序、相同次数执行集合操作，
  否则互相等待挂死。库的 API 按此设计——StepsBar 只在循环自然结束时
  触发自动合并（提前 break / 异常不触发，因为各 rank 迭代进度可能
  不一致）；手动 `melog.scalar(...)` / `melog.scalar(metrics)` 时
  确保所有 rank 执行同一位置即可（正常训练代码天然满足）。
- `reduce_op` 只影响 `melog.scalar({...})` 的数值合并；MetricGroup
  的合并规则由指标语义决定（加权平均 / 求和 / 拼接），不受它影响。

### 时序一览（2 卡示例）

```text
feed()   ── rank0: 本地累积 + 实时值落盘(write=True)   零通信
         ── rank1: 本地累积
epoch 末 ── all_gather_object(MetricGroup 全组状态, 1 次)
         ── 各 rank 各自合并出一致的全局值
         ── rank0: 全局值写日志/面板 + reset 清零
```

## API

### `melog.init(...)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `log_dir` | `./melog_runs` | 日志保存路径；本次运行落在其下时间戳子目录，末级目录名即项目名 |
| `web_port` | 随机空闲端口 | Web 监听端口（`web_host` 默认 `127.0.0.1`，地址启动时自动打印，也可读 `melog.current().web_url`） |
| `enable_web` | `True` | 启动 Web 服务（仅 rank0） |
| `enable_progress` | `True` | 启用控制台进度条 |
| `reduce_op` | `"mean"` | 多 GPU 合并方式 |
| `flush_every` | `1` | 每 N 次 scalar 落盘一次 |
| `project` | `log_dir` 末级目录名 | 覆盖项目名 |

### 主要方法

- `scalar(metrics, advance=0)` — 记录一批指标（dict 或 MetricGroup，后者跨 GPU 合并由内部完成）；坐标由 `StepsBar` 自动管理（epoch 绑定 + 内部计步），调用频率即记录粒度（见上文）
- `image(name, data, caption=None)` / `audio(name, data, sr=22050, ...)` — 记录图像 / 音频，自动附着最近一次记录位置，Web 端页签展示（见上文）
- `StepsBar(iterable, epoch=None, metrics=None)` — tqdm 风格训练进度条（`from melog import StepsBar`，模块级 `melog.stepsbar(...)` 等价），**epoch 循环必须用它包裹**：包裹可迭代对象即自动推进，`scalar()` 指标实时显示在条上；`epoch=...` 绑定当前 epoch 并统一管理坐标；`metrics=...` 传入 MetricGroup 时 bar 实时显示本卡本地值（feed 零通信刷新），迭代自然结束自动 gather 全局值合并记录并重置组内指标（提前 break / 异常不触发）（见上文）
- 允许嵌套（如训练 bar 内嵌验证 bar）：内部以栈管理，`current_bar()` 返回栈顶即当前环境；`scalar()` 的 postfix 与 `advance` 自动作用于栈顶，下层 bar 暂停渲染（数据照常累计），栈顶关闭后自动恢复下层渲染；提前 break / 抛异常时 bar 自动出栈（如需立即定稿可 `close()` 或用 with），否则等引用释放时兜底
- `current_bar()` — 当前栈顶进度条（无打开的 bar 时 `None`）；深层函数需要手动推进 / 读数 / 写 postfix 时取它，免层层传参
- `log / success / error / warn` — print 风格控制台消息（图标 + 彩色文字），`print` 被拦截改走 `log()`（见上文）
- 收尾无需手动调用：进程退出时经 atexit 自动落盘剩余指标、定稿进度条、停 Web、还原 print
- 全局共享：`melog.init(...)` 创建活动实例后，模块级 `melog.scalar(...)` 等可在任意位置直接调用（见上文）

进度条显示的指标可通过环境变量 `MELOG_DISABLE_PROGRESS=1` 全局关闭。

## CLI 快速查看

安装后可直接用命令行查看历史日志，自动打开浏览器：

```bash
melog F:/runs/exp1/metrics.melog   # 指定日志文件
melog F:/runs/exp1                 # 指定目录（自动取最新 metrics.melog）
melog                              # 缺省在 ./melog_runs 中查找
melog F:/runs/exp1 --port 9000 --no-browser  # 自定义端口 / 不开浏览器
```

## 项目结构

```text
melog/
├── __init__.py      # 包入口：公开 API 导出
├── core.py          # Melog 主类：组合组件、调度记录、生命周期
├── api/             # 全局入口 melog.init() + 模块级便捷接口（melog.scalar 等）
├── cli/             # 命令行入口：melog <path>
├── tracking/        # 训练记录上下文
│   ├── axis.py      # Axis：全局 x / epoch 坐标的唯一裁决者
│   ├── steps_bar.py # StepsBar：tqdm 风格训练进度条（epoch 绑定 + 自动记录）
│   └── console.py   # Console：控制台消息（log/success/error/warn）+ print 拦截
├── storage/         # 持久化与产物
│   ├── journal.py   # Journal：JSONL 日志落盘（批量写 + 即时追加）
│   ├── media.py     # 图像/音频落盘编码（路径复制或数组编码）
│   ├── media_log.py # MediaLog：媒体记录流程（定位->落盘->索引->日志->推送）
│   └── mirror.py    # Mirror：控制台日志镜像（进度条就地刷新 + stdio 接管）
├── metrics/         # 指标计算与跨 GPU 同步
│   ├── base.py      # Metric 基类：compute 实时出值 / prepare+compute epoch 出值
│   ├── basic.py     # Mean / Sum / Last / Count
│   ├── classification.py  # Accuracy / Precision / Recall / F1 / ConfusionMatrix
│   └── group.py     # MetricGroup：具名指标集合
├── web/             # Web 可视化面板
│   ├── server.py    # WebServer：uvicorn 线程生命周期
│   ├── app.py       # ApiRoutes：路由注册（指标/媒体/文件浏览/加载/WS）
│   ├── store.py     # MetricStore：内存指标历史
│   ├── view.py      # MetricView：实时/历史视图切换
│   ├── media_store.py  # MediaStore：实时媒体索引
│   ├── media_view.py   # MediaView：媒体视图切换 + 文件白名单解析
│   ├── fs.py        # FileBrowser：文件浏览
│   ├── loader.py    # LogLoader / MediaLoader：JSONL 解析
│   ├── ws.py        # WsHub：WebSocket 广播
│   └── static/      # 前端（js 按类分模块）
└── utils/           # 通用工具类
    ├── tqdm.py               # 自研进度条（tqdm 兼容，样式重设计）
    ├── downsample.py         # 曲线降采样
    ├── distributed.py        # 多 GPU all_reduce / all_gather 原语
    ├── bar_stack.py          # BarStack / BarFrame：进度条栈帧管理（嵌套、恢复渲染）
    └── epoch_end_iterable.py # EpochEndIterable：自然耗尽触发回调的迭代包装
```

## 开发

```bash
pip install -e ".[dev]"
pytest tests -q
```
