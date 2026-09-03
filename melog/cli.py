"""命令行入口：melog <path> 快速查看训练日志。

用法：
    melog F:/runs/exp1/metrics.jsonl   # 指定日志文件
    melog F:/runs/exp1                 # 指定目录（自动取最新 metrics.jsonl）
    melog                              # 缺省在 ./melog_runs 中查找
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

from .web.fs import FileBrowser
from .web.loader import LogLoader
from .web.server import WebServer
from .web.store import MetricStore


def _resolve_log_file(path: Path) -> Path:
    """文件直接用；目录取其中最新的 metrics.jsonl。"""
    if path.is_file():
        return path
    return FileBrowser.find_latest_log(path)


def _load_into_store(store: MetricStore, log_file: Path) -> int:
    """把日志灌入内存 store，作为静态视图数据源。"""
    series = LogLoader.parse(log_file)
    count = 0
    for name, pts in series.items():
        for step, value, epoch in pts:
            store.add(step, {name: value}, epoch)
            count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="melog",
        description="Melog 训练日志可视化：melog <.melog 日志文件或目录>",
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help=".melog 日志文件或包含它的目录；缺省时在 ./melog_runs 中查找",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8666, help="监听端口（默认 8666）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    if args.path:
        target = Path(args.path)
        if not target.exists():
            print(f"路径不存在: {target}", file=sys.stderr)
            return 1
    else:
        target = Path("melog_runs")
        if not target.exists():
            print("未指定路径，且当前目录下没有 melog_runs/", file=sys.stderr)
            return 1

    try:
        log_file = _resolve_log_file(target)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    store = MetricStore()
    if _load_into_store(store, log_file) == 0:
        print("文件中没有可解析的指标", file=sys.stderr)
        return 1

    server = WebServer(store, host=args.host, port=args.port, log_file=str(log_file))
    server.start()

    print(f"Melog 可视化: {server.url}")
    print(f"日志文件: {log_file}")
    print("Ctrl+C 退出")
    if not args.no_browser:
        webbrowser.open(server.url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
