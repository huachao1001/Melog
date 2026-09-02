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

logger = Melog(project="my-exp", web_port=8666)

with logger.train(total=1000) as bar:
    for step in range(1000):
        loss = train_one_step()
        logger.log({"loss": loss, "lr": 1e-3})   # 记录 + 更新进度条 + 推送 Web
        bar.advance(1)

logger.finish()   # 落盘剩余指标并停止 Web 服务
```

训练期间浏览器打开 `http://127.0.0.1:8666` 查看实时曲线。

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

- `log(metrics, step=None, advance=1)` — 记录一批指标；`step` 缺省自增
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
├── core.py          # Melog 主类：记录、调度、JSONL 落盘
├── cli.py           # 命令行入口：melog <path>
├── distributed.py   # 多 GPU all_reduce 合并
├── downsample.py    # 曲线降采样
├── progress.py      # rich 进度条 + 指标列
└── web/
    ├── server.py    # WebServer：uvicorn 线程生命周期
    ├── app.py       # ApiRoutes：路由注册
    ├── store.py     # MetricStore：内存指标历史
    ├── view.py      # MetricView：实时/历史视图切换
    ├── fs.py        # FileBrowser：文件浏览
    ├── loader.py    # LogLoader：JSONL 解析
    ├── ws.py        # WsHub：WebSocket 广播
    └── static/      # 前端（js 按类分模块）
```

## 开发

```bash
pip install -e ".[dev]"
pytest tests -q
```
