# Melog

轻量级训练监控库：**多 GPU 指标合并 + 控制台实时进度条 + Web 可视化**。

```text
demo ██████████░░░░░░  45% 90/200 0:00:03 loss=0.2153  acc=0.8974  lr=8.2e-04
```

## 特性

- **控制台实时进度条**：训练期间在 rich 进度条尾部实时刷新最新指标值
- **多 GPU 指标合并**：基于 `torch.distributed` all_reduce 跨进程聚合（默认取均值），仅 rank0 记录与展示；未装 torch 自动退化单进程
- **Web 可视化**：FastAPI + WebSocket + ECharts，后台线程运行，实时推送曲线，断线自动重连
- **持久化**：指标自动写入 JSONL，供离线分析

## 安装

```bash
pip install -e .            # 基础安装
pip install -e ".[torch]"   # 多 GPU 合并需要 torch
```

## 快速开始

```python
from melog import Melog

mlog = Melog(project="my-exp", web_port=8666)

with mlog.train(total=1000) as bar:
    for step in range(1000):
        loss = train_one_step()
        mlog.log({"loss": loss, "lr": 1e-3})   # 记录 + 更新进度条 + 推送 Web
        bar.advance(1)

mlog.finish()   # 落盘剩余指标并停止 Web 服务
```

训练期间浏览器打开 `http://127.0.0.1:8666` 查看实时曲线。

## 曲线上体现 epoch

`log()` 可传入 `epoch` 与当前 epoch 内的 `step`，Web 曲线会在每个 epoch 起点画分界虚线
（标注 `e0` / `e1` / …），悬浮提示显示 `epoch N · step X`；两者都不传时内部自动统计 step，
行为与旧版一致：

```python
for epoch in range(epochs):
    for step in range(steps):
        mlog.log({"loss": loss, "lr": lr}, epoch=epoch, step=step)
```

- `epoch` 缺省时**粘滞沿用**上一次传入的值（整个 epoch 内只需传一次），从未传入则不记录 epoch
- `step` 为**当前 epoch 内**的步数，缺省内部自增（每个 epoch 从 0 重新计步）；
  未启用 epoch 时 `step` 含义不变（全局步数，缺省自增）

## 记录图像与音频

除指标曲线外，Web 端 header 可在 **曲线 / 图像 / 音频** 三个页签间切换。图像与音频用
`log_image` / `log_audio` 记录，Web 端按名字建卡片、滑杆按 step 回放（图像点击看原图，
音频在线播放）；文件自动落盘到 `run_dir/media/`，元数据随日志持久化，历史日志加载时
媒体一并恢复：

```python
mlog.log({"loss": loss}, epoch=epoch, step=step)
mlog.log_image("train/sample", img)        # 路径 / PIL / numpy / torch，附着当前 step
mlog.log_audio("val/audio", wav, sr=16000) # 路径(wav/mp3/…) / numpy / torch 波形
```

- `step` / `epoch` 缺省时自动附着到**最近一次 `log()` 的位置**，不推进 step 计数
- `caption="..."` 可为每条图像 / 音频配一段文字（如样本说明、转写文本），
  显示在卡片上、随滑杆切换；换行会被保留
- 图像：`(H,W)` 灰度或 `(H,W,C)`（C=1/3/4），浮点自动映射 0-255，统一存为 PNG
- 音频：`(N,)` 单声道或 `(N, 声道数)`，浮点按 [-1,1] 裁剪存为 16bit WAV；
  传文件路径则按原格式复制
- 数组编码需要 `pillow`（仅图像）：`pip install pillow`

## 指标计算（多 GPU 自动同步）

内置 `Mean` / `Sum` / `Max` / `Min` / `Last` / `Count`，按 epoch 组织在 `MetricGroup` 中使用：

```python
from melog import Melog, Mean, Max, MetricGroup, Sum

mlog = Melog(project="my-exp")
metrics = MetricGroup({
    "loss": Mean(),      # 加权平均：update(loss, batch_size)
    "acc": Mean(),
    "seen": Sum(),       # 求和
    "best_acc": Max(),   # 历史最大值
    "lr": Mean(),        # 取最近值
})

with mlog.train(total=steps * epochs) as bar:
    for epoch in range(epochs):
        for _ in range(steps):
            metrics.update(loss=(loss, batch_size), acc=(acc, batch_size),
                           seen=batch_size, best_acc=acc, lr=lr)
            bar.advance(1)
        mlog.log_group(metrics, reset=True)   # epoch 末：同步 + 记录 + 重置
```

- `Mean` 是**全局加权平均**：各 rank 的 `value × weight` 求和后除以总权重，不是"各卡平均值的平均"
- `Mean` / `Sum` / `Max` / `Min` 可随时 `compute()`；**必须算完一个 epoch 才有意义的指标**，在 epoch 末统一调用 `compute()`（或 `log_group(..., reset=True)`）即可
- `compute()` 是集合操作：**所有 rank 必须以相同顺序调用**，返回值各 rank 一致；单进程自动直通
- `mlog.log_group(group, reset=True)` 等价于 `mlog.log(group.compute()); group.reset()`

### 分类指标

内置 `Accuracy` / `Precision` / `Recall` / `F1` / `ConfusionMatrix`，接口与基础指标一致，
`update(logits, labels)` 直接接收模型输出与标签：

```python
from melog import Accuracy, F1, MetricGroup, Mean, Precision

metrics = MetricGroup({
    "loss": Mean(),
    "acc": Accuracy(),                 # 二分类：一维得分按阈值 0.5 判定
    "acc5": Accuracy(topk=5),          # top-5 准确率（多分类）
    "f1": F1(num_classes=10),          # 多分类：二维 (N, K) logits 按行 argmax
})

for logits, labels in val_loader:
    # feed：框架按各指标的形参名/注册名自动分发，无需逐个传 (logits, labels)
    metrics.feed(logits=logits, labels=labels, loss=(loss, batch_size))
mlog.log_group(metrics, reset=True)  # epoch 末：跨 GPU 同步 + 记录 + 重置
```

- `Accuracy(topk=k)`：真实类别在前 k 个预测中即算正确
- `Precision / Recall / F1` 的 `average`：`None`（二分类=正类，多分类=macro）/ `"macro"` / `"micro"` / `"weighted"`
- `ConfusionMatrix` 的 `compute()` 返回矩阵（行=真实、列=预测），适合直接读取而非画曲线
- 预测规则由 `preds_from_logits` 实现，可传 `predictor=` 替换（如多标签、分割等自定义转换）

### 自定义指标

**单批次指标（推荐）**：继承 `BatchMetric`，只实现 `compute_batch()` 一个函数。
形参名和个数完全由你定义，框架按形参名自动从 `update()` / `feed()` 的观测中取值回调；
累积、跨 GPU 合并、reset 全部由框架完成：

```python
from melog import BatchMetric

class MaskedAcc(BatchMetric):
    """需要几个参数就声明几个，logits/labels 仅为示例。"""
    def compute_batch(self, logits, labels, mask):
        hits = ((logits.argmax(-1) == labels) & mask).sum()
        n = mask.sum()
        return (hits / n, n)          # 返回 (值, 权重)：按样本数加权出全局结果

# 训练循环里：位置或具名喂入均可，多余观测自动忽略
metric.update(logits, labels, mask)
metric.update(logits=logits, labels=labels, mask=mask)
```

- `compute_batch` 返回 `(值, 权重)` 元组：各 batch 按权重加权平均（样本数不同时务必带上权重）；
  只返回 float 时各 batch 等权平均
- 组合使用时交给 `MetricGroup.feed(...)` 统一分发：

```python
metrics = MetricGroup({"loss": Mean(), "macc": MaskedAcc()})
metrics.feed(logits=logits, labels=labels, mask=mask, loss=(loss, batch_size))
mlog.log_group(metrics, reset=True)
```

**epoch 级指标**：全局结果无法由各 batch 值加权平均还原时（如 macro F1、AUC），
继承 `Metric` 实现完整契约，跨 GPU 状态收集仍由基类完成：

```python
from melog import Metric

class F1(Metric):
    """epoch 末才能计算的指标：累积混淆计数，末尾统一算。"""
    def __init__(self):
        self.tp = self.fp = self.fn = 0.0

    def update(self, tp, fp, fn):          # 每个 batch 累积本地计数
        self.tp += tp; self.fp += fp; self.fn += fn

    def state(self):                        # 导出可 pickle 的本地状态
        return [self.tp, self.fp, self.fn]

    def merge_states(self, states):         # states: 所有 rank 的状态（按 rank 顺序）
        tp = sum(s[0] for s in states); fp = sum(s[1] for s in states)
        fn = sum(s[2] for s in states)
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")

    def reset(self):
        self.tp = self.fp = self.fn = 0.0
```

## 多 GPU

代码无需修改，用 `torchrun` 启动即可：

```bash
torchrun --nproc_per_node=4 examples/multi_gpu_train.py
```

- 各 rank 的指标自动 all_reduce 合并（`reduce_op="mean"`，可选 `"sum"`）
- 仅 rank0 持久化、启动 Web、渲染进度条，其余 rank 静默
- 指标值支持 float / int / 0 维 tensor

## API

### `Melog(...)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `project` | `"melog"` | 项目名（输出子目录） |
| `output_dir` | `./melog_runs` | 持久化根目录 |
| `enable_web` | `True` | 启动 Web 服务（仅 rank0） |
| `web_host` / `web_port` | `127.0.0.1:8666` | Web 监听地址 |
| `enable_progress` | `True` | 启用控制台进度条 |
| `reduce_op` | `"mean"` | 多 GPU 合并方式 |
| `flush_every` | `1` | 每 N 次 log 落盘一次 |

### 主要方法

- `log(metrics, step=None, epoch=None, advance=1)` — 记录一批指标；`epoch` / `step` 缺省内部自增，传入后曲线标注 epoch 分界（见上文）
- `log_image(name, data, step=None, epoch=None)` / `log_audio(name, data, sr=22050, ...)` — 记录图像 / 音频，Web 端页签展示（见上文）
- `train(total)` — 返回进度条上下文管理器（`bar.advance(n)` 推进）
- `finish()` — 落盘并停止 Web 服务

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
├── core.py          # Melog 主类：记录、调度、JSONL 落盘、媒体记录
├── cli.py           # 命令行入口：melog <path>
├── distributed.py   # 多 GPU all_reduce / all_gather 原语
├── media.py         # 图像/音频落盘（路径复制或数组编码）
├── metrics/         # 指标计算与跨 GPU 同步
│   ├── base.py      # Metric / BatchMetric 基类（自定义指标继承其一）
│   ├── basic.py     # Mean / Sum / Max / Min / Last / Count
│   ├── classification.py  # Accuracy / Precision / Recall / F1 / ConfusionMatrix
│   └── group.py     # MetricGroup：具名指标集合
├── downsample.py    # 曲线降采样
├── progress.py      # rich 进度条 + 指标列
└── web/
    ├── server.py    # WebServer：uvicorn 线程生命周期
    ├── app.py       # ApiRoutes：路由注册（指标/媒体/文件浏览/加载/WS）
    ├── store.py     # MetricStore：内存指标历史
    ├── view.py      # MetricView：实时/历史视图切换
    ├── media_store.py  # MediaStore：实时媒体索引
    ├── media_view.py   # MediaView：媒体视图切换 + 文件白名单解析
    ├── fs.py        # FileBrowser：文件浏览
    ├── loader.py    # LogLoader / MediaLoader：JSONL 解析
    ├── ws.py        # WsHub：WebSocket 广播
    └── static/      # 前端（js 按类分模块）
```

## 开发

```bash
pip install -e ".[dev]"
pytest tests -q
```
