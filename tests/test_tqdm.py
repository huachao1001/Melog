"""自研 tqdm 与控制台日志镜像测试。"""

import io
import sys

import pytest

from melog.core import Melog
from melog.mirror import Mirror
from melog.tqdm import tqdm


def read(path) -> str:
    return path.read_bytes().decode("utf-8")


# ---------------------------------------------------------------- Mirror
def test_mirror_bar_refresh_in_place(tmp_path):
    """进度条行（\\r 结尾）在文件中就地刷新，只保留最后一行。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("train ██░░ 10%\r")
    m.write("train ████ 30%\r")
    m.close()
    assert read(tmp_path / "c.log") == "train ████ 30%\n"  # close 定稿为普通行


def test_mirror_bar_throttled(tmp_path):
    """进度条行按节流间隔落盘：间隔内跳过，定稿时补最新内容。"""
    t = [0.0]
    m = Mirror(tmp_path / "c.log", throttle=2.0, clock=lambda: t[0])
    m.write("bar 10%\r")       # 首帧立即写
    t[0] = 1.0
    m.write("bar 20%\r")       # 间隔内 → 跳过
    assert read(tmp_path / "c.log") == "bar 10%\r"
    t[0] = 3.0
    m.write("bar 30%\r")       # 超过节流间隔 → 刷新
    assert read(tmp_path / "c.log") == "bar 30%\r"
    m.write("hello\n")         # 普通行：先定稿最新进度条（40%）再追加
    m.write("bar 40%\r")
    m.write("done\n")
    assert read(tmp_path / "c.log") == "bar 30%\nhello\nbar 40%\ndone\n"
    m.close()


def test_mirror_finalize_uses_latest_content(tmp_path):
    """节流期间文件落后时，定稿补写最新进度条内容而非文件里的旧内容。"""
    t = [0.0]
    m = Mirror(tmp_path / "c.log", throttle=2.0, clock=lambda: t[0])
    m.write("bar 10%\r")
    t[0] = 1.0
    m.write("bar 99%\r")       # 被节流
    m.write("ok\n")            # 定稿应写 99%
    assert read(tmp_path / "c.log") == "bar 99%\nok\n"
    m.close()


def test_mirror_resume_trailing_cr(tmp_path):
    """打开已有文件：最后一行以 \\r 结尾（不可见空白字符）→ 视为进度条行，就地刷新。"""
    p = tmp_path / "c.log"
    p.write_bytes("旧进度 50%\r".encode("utf-8"))
    m = Mirror(p, throttle=0)
    m.write("新进度 80%\r")
    m.close()
    assert read(p) == "新进度 80%\n"


def test_mirror_partial_line_buffered_then_joined(tmp_path):
    """无终止符的输出先缓冲，与后续内容拼接成完整行（print(end='') 场景）。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("abc")             # 未成行，缓冲
    m.write("def\n")           # 成行
    m.close()
    assert read(tmp_path / "c.log") == "abcdef\n"


def test_mirror_hook_stdio(tmp_path, capsys):
    """hook 后 print 同时进控制台与文件；unhook 还原。"""
    console = io.StringIO()
    saved_out = sys.stdout
    m = Mirror(tmp_path / "c.log", throttle=0)
    m._saved = (console, sys.stderr)  # 直接预置，绕过对真实 stdout 的替换
    sys.stdout = _TeeOf(console, m)
    try:
        print("你好镜像")
    finally:
        sys.stdout = saved_out
    m.close()
    assert console.getvalue() == "你好镜像\n"
    assert read(tmp_path / "c.log") == "你好镜像\n"


def _TeeOf(console, mirror):
    from melog.mirror import _Tee

    return _Tee(console, mirror)


# ---------------------------------------------------------------- tqdm
def test_tqdm_iterable_mode():
    out = io.StringIO()
    items = list(tqdm(range(3), desc="处理", file=out, mininterval=0))
    assert items == [0, 1, 2]
    text = out.getvalue()
    assert text.startswith("处理 ")
    assert "3/3" in text and "100.0%" in text
    assert text.endswith("\n")  # close 定稿换行
    assert "█" in text and "░" in text


def test_tqdm_manual_update_and_postfix():
    out = io.StringIO()
    with tqdm(total=10, desc="train", file=out, mininterval=0) as bar:
        bar.update(4)
        bar.set_postfix(loss=0.215300001)
        line = out.getvalue().splitlines()[-1] if "\n" in out.getvalue() else out.getvalue()
    assert "4/10" in out.getvalue()
    assert "loss=0.2153" in out.getvalue()
    assert bar.n == 4


def test_tqdm_advance_alias():
    out = io.StringIO()
    bar = tqdm(total=5, file=out, mininterval=0, disable=False)
    bar.advance(2)
    assert bar.n == 2
    bar.close()


def test_tqdm_disable_is_silent():
    out = io.StringIO()
    items = list(tqdm(range(3), file=out, disable=True))
    assert items == [0, 1, 2]
    assert out.getvalue() == ""


def test_tqdm_write_goes_through_stream():
    out = io.StringIO()
    bar = tqdm(total=5, file=out, mininterval=0)
    tqdm.write("hello", file=out)
    bar.close()
    assert "hello\n" in out.getvalue()


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_tqdm_color_on_tty_plain_on_pipe():
    """终端 TTY 输出 Melog 主题色；管道/重定向为纯文本。"""
    out = _TTY()
    bar = tqdm(total=4, desc="train", file=out, mininterval=0)
    bar.update(2)
    bar.close()
    assert "\x1b[38;2;168;85;247m" in out.getvalue()  # 主题紫

    out2 = io.StringIO()
    tqdm(total=4, file=out2, mininterval=0).close()
    assert "\x1b[" not in out2.getvalue()

    out3 = io.StringIO()
    tqdm(total=4, file=out3, mininterval=0, colour=False).close()
    assert "\x1b[" not in out3.getvalue()


def test_mirror_strips_ansi(tmp_path):
    """日志文件剥离 ANSI 颜色码，保持纯文本。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("\x1b[38;2;168;85;247mtrain\x1b[0m \x1b[1;38;2;168;85;247m50.0%\x1b[0m\r")
    m.close()
    assert read(tmp_path / "c.log") == "train 50.0%\n"


# ---------------------------------------------------------------- Melog 集成
def test_melog_mirrors_console_log(tmp_path, capsys):
    """训练期间 print 与进度条同步进 console.log；finish 后还原 stdout。"""
    saved_stdout = sys.stdout
    mlog = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        assert sys.stdout is not saved_stdout  # 已接管
        with mlog.train(total=2, description="train") as bar:
            mlog.log({"loss": 0.5})
            bar.update(1)
            print("step done")
            bar.update(1)
    finally:
        mlog.finish()
    assert sys.stdout is saved_stdout  # 已还原

    log_path = next((tmp_path / "t").glob("**/console.log"))
    text = read(log_path)
    assert "step done\n" in text
    assert "loss=0.5" in text          # 进度条 postfix
    assert "2/2" in text               # 定稿的最终进度


def test_melog_console_log_progress_line_uses_cr(tmp_path):
    """日志文件里的进度条行以 \\r 结尾（不可见空白字符标记），定稿后变 \\n。"""
    mlog = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        with mlog.train(total=1, description="train") as bar:
            bar.update(1)
            # 训练中途：最后一行应为 \r 结尾的进度条行
            path = next((tmp_path / "t").glob("**/console.log"))
            assert path.read_bytes().endswith(b"\r")
    finally:
        mlog.finish()
    path = next((tmp_path / "t").glob("**/console.log"))
    assert path.read_bytes().endswith(b"\n")  # finish 定稿
