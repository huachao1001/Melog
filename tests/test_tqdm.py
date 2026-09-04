"""自研 tqdm 与控制台日志镜像测试。"""

import io
import re
import sys

import pytest

from melog import StepsBar
from melog.core import Melog
from melog.storage.mirror import Mirror
from melog.tracking.console import Console
from melog.utils.bar_stack import BarStack
from melog.utils.tqdm import tqdm


def read(path) -> str:
    return path.read_bytes().decode("utf-8")


# ---------------------------------------------------------------- Mirror
def test_mirror_bar_rewrote_in_place(tmp_path):
    """进度条行（\\r 结尾）就地刷新：文件里始终一行、实时重写（时间戳由渲染自带）。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("[10:00:00] train ██░░ 10%\r")
    m.write("[10:00:00] train ████ 30%\r")
    assert read(tmp_path / "c.log") == "[10:00:00] train ████ 30%\r"
    m.close()  # 定稿：进度条行升级为完整行
    assert read(tmp_path / "c.log") == "[10:00:00] train ████ 30%\n"


def test_mirror_bar_throttled(tmp_path):
    """就地刷新按节流间隔写文件：间隔内跳过（内容暂存），定稿时写最新内容。"""
    t = [0.0]
    m = Mirror(tmp_path / "c.log", throttle=2.0, clock=lambda: t[0])
    m.write("bar 10%\r")       # 首帧立即写
    t[0] = 1.0
    m.write("bar 20%\r")       # 间隔内 → 文件仍是 10%
    assert read(tmp_path / "c.log") == "bar 10%\r"
    t[0] = 3.0
    m.write("bar 30%\r")       # 超过节流间隔 → 就地覆盖为 30%
    assert read(tmp_path / "c.log") == "bar 30%\r"
    m.write("hello\n")         # 普通行：先定稿进度条再追加
    m.write("bar 40%\r")
    m.write("done\n")
    assert read(tmp_path / "c.log") == "bar 30%\nhello\nbar 40%\ndone\n"
    m.close()


def test_mirror_finalize_uses_latest_content(tmp_path):
    """节流期间文件落后时，定稿写最新进度条内容而非文件里的旧内容。"""
    t = [0.0]
    m = Mirror(tmp_path / "c.log", throttle=2.0, clock=lambda: t[0])
    m.write("bar 10%\r")
    t[0] = 1.0
    m.write("bar 99%\r")       # 被节流
    m.write("ok\n")            # 定稿应写 99%
    assert read(tmp_path / "c.log") == "bar 99%\nok\n"
    m.close()


def test_mirror_message_restarts_bar_line(tmp_path):
    """消息插入：进度条行定稿、消息独占一行；后续帧在消息下方重新开一条就地刷新行。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("[10:00:00] bar 50%\r")
    m.write("[10:00:00] hello\n")    # 定稿 bar 行 + 消息行
    m.write("[10:00:01] bar 60%\r")  # 消息下方重新开始一条就地刷新行
    assert read(tmp_path / "c.log") == "[10:00:00] bar 50%\n[10:00:00] hello\n[10:00:01] bar 60%\r"
    m.close()
    assert read(tmp_path / "c.log") == "[10:00:00] bar 50%\n[10:00:00] hello\n[10:00:01] bar 60%\n"


def test_mirror_resume_trailing_cr(tmp_path):
    """打开未完的进度条行（\\r 结尾）：后续刷新继续就地覆盖该行。"""
    p = tmp_path / "c.log"
    p.write_bytes("旧进度 50%\r".encode("utf-8"))
    m = Mirror(p, throttle=0)
    m.write("新进度 80%\r")
    assert read(p) == "新进度 80%\r"  # 继续就地覆盖未完行
    m.close()
    assert read(p) == "新进度 80%\n"


def test_mirror_bar_tall_characters(tmp_path):
    """文件侧进度条加高：细线字符替换为块状字符（█ 填充 / ░ 剩余）。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("train ━━━━━───────  35.0%\r")
    assert read(tmp_path / "c.log") == "train █████░░░░░░░  35.0%\r"
    m.close()
    assert read(tmp_path / "c.log") == "train █████░░░░░░░  35.0%\n"


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
    from melog.storage.mirror import _Tee

    return _Tee(console, mirror)


# ---------------------------------------------------------------- tqdm
def test_tqdm_iterable_mode():
    out = io.StringIO()
    items = list(tqdm(range(3), desc="处理", file=out, mininterval=0))
    assert items == [0, 1, 2]
    text = out.getvalue()
    assert text.startswith("\n处理 ")  # 首帧前先换行（bar 从新行开始），行不带时间戳（时间戳只用于消息行）
    assert "3/3" in text and "100.0%" in text
    assert text.endswith("\n")  # close 定稿换行
    assert "━" in text


def test_tqdm_layout_order():
    """布局：指标最前，条形图其后，[n/total] 在百分比后、耗时尾段前。"""
    out = io.StringIO()
    with tqdm(total=10, file=out, mininterval=0) as bar:
        bar.update(1)
        bar.set_postfix(loss=0.5)
    line = [s for s in out.getvalue().rstrip("\n").split("\r") if s.strip()][-1].rstrip()
    # 段序：指标 < 条形 < 百分比 < [n/total] < 耗时尾段（n 右对齐到 total 宽度）
    assert line.index("loss=0.5") < line.index("━") < line.index("10.0%")
    assert line.index("10.0%") < line.index("[ 1/10]") < line.index("[0:00<")
    assert not line.startswith("train")  # 无 desc 时不显示前缀


def test_tqdm_postfix_stable_width():
    """指标值定宽右对齐：常规数值行宽恒定；更宽数值只右移一次不回摆。"""
    out = io.StringIO()
    bar = tqdm(total=10, file=out, mininterval=0)
    for loss in (1.183, 0.796, 0.5615):
        bar.set_postfix(loss=loss)
    bar.close()
    lines = [s for s in out.getvalue().rstrip("\n").split("\r") if "loss=" in s]
    assert len({len(s) for s in lines}) == 1  # 常规范围：各帧行宽一致
    assert "loss=0.5615" in lines[-1]

    out2 = io.StringIO()
    bar2 = tqdm(total=10, file=out2, mininterval=0)
    bar2.set_postfix(loss=0.5615)
    bar2.set_postfix(loss=12.34567)  # 更宽 → 定宽字段扩一次
    bar2.set_postfix(loss=0.796)  # 变窄 → 字段保持宽度，不回摆
    bar2.close()
    lines2 = [s for s in out2.getvalue().rstrip("\n").split("\r") if "loss=" in s]
    assert len(lines2[0]) < len(lines2[1]) == len(lines2[-1])  # 行宽只增不减


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
    bar.update(2)
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
    bar.set_postfix(loss=0.5)
    bar.close()
    text = out.getvalue()
    assert "\x1b[38;2;168;85;247m" in text   # 主题紫（渐变起点 / 百分比）
    grad_colors = set(re.findall(r"\x1b\[38;2;(\d+;\d+;\d+)m", text))
    assert len(grad_colors) >= 10            # 逐格插值：填充段颜色连续变化
    assert "\x1b[38;2;236;72;153m" in text   # 渐变末端粉
    assert "\x1b[38;2;97;214;214m" in text   # [n/total] 青
    assert "\x1b[38;2;250;204;21m" in text   # 速率黄
    assert "\x1b[1;38;2;230;233;238m" in text  # desc 加粗白
    assert "\x1b[38;2;230;233;238m" in text    # [n/total] 白 / 数值白
    assert "\x1b[38;2;128;134;145m" in text    # 次要信息灰（指标名 / 耗时）
    assert "\x1b[38;2;70;74;80m" in text       # 空心条深灰
    assert "\x1b[?25l" in text                 # 渲染期间隐藏光标（块状光标压行首）
    assert text.endswith("\x1b[?25h")          # close 后恢复光标

    out2 = io.StringIO()
    tqdm(total=4, file=out2, mininterval=0).close()
    assert "\x1b[" not in out2.getvalue()

    out3 = io.StringIO()
    tqdm(total=4, file=out3, mininterval=0, colour=False).close()
    assert "\x1b[" not in out3.getvalue()


def test_melog_progress_iterates_and_autoupdates(tmp_path):
    """StepsBar 包裹可迭代对象：迭代自动推进，scalar() 指标显示在条尾。"""
    saved_stdout = sys.stdout
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        items = list(StepsBar(range(3), epoch=0))
        assert items == [0, 1, 2]
    finally:
        lg.close()
    assert sys.stdout is saved_stdout

    log_path = next((tmp_path / "t").glob("**/*console-*.log"))
    text = read(log_path)
    assert "3/3" in text               # 迭代自动推进到终点
    assert "epoch=0" in text           # epoch 传入时行首自动标注


def test_melog_progress_shows_log_postfix(tmp_path):
    """StepsBar 期间 scalar() 的指标经 postfix 实时写进 console.log。"""
    saved_stdout = sys.stdout
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        for _ in StepsBar(range(2)):
            lg.scalar({"loss": 0.5})
    finally:
        lg.close()
    log_path = next((tmp_path / "t").glob("**/*console-*.log"))
    assert "loss=0.5" in read(log_path)


def test_melog_progress_reusable_across_epochs(tmp_path):
    """进度条自然结束后自动解除登记，同一 Melog 可再次开条。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        first = list(StepsBar(range(2), epoch=0))
        second = list(StepsBar(range(2), epoch=1))
        assert first == [0, 1] and second == [0, 1]
    finally:
        lg.close()


def test_melog_progress_nesting_stack(tmp_path):
    """StepsBar 允许嵌套（栈管理）：栈顶为当前环境，关闭后自动恢复下层。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        assert lg.current_bar() is None
        outer = StepsBar(range(3), epoch=0)
        inner = StepsBar(range(2), epoch=0)
        assert lg.current_bar() is inner
        assert outer.covered and not inner.covered  # 只有栈顶渲染

        lg.scalar({"loss": 1.0})  # postfix 作用于栈顶
        assert inner.postfix.get("loss") == pytest.approx(1.0)
        assert "loss" not in outer.postfix

        inner.close()  # 栈顶关闭：出栈并恢复下层渲染与登记
        assert lg.current_bar() is outer
        assert not outer.covered
        lg.scalar({"loss": 2.0})
        assert outer.postfix.get("loss") == pytest.approx(2.0)

        outer.close()
        assert lg.current_bar() is None
    finally:
        lg.close()


def test_melog_current_bar_module_level(tmp_path):
    """模块级 current_bar()：任意位置取当前栈顶 bar，免层层传参。"""
    import melog as pkg

    pkg.init(tmp_path / "cb", enable_web=False)
    try:
        assert pkg.current_bar() is None
        bar = pkg.stepsbar(range(2))
        assert pkg.current_bar() is bar
        bar.close()
        assert pkg.current_bar() is None
    finally:
        pkg.current().close()


def test_melog_log_advance_opt_in(tmp_path):
    """scalar(advance=N) 可选推进进度条；缺省 0（StepsBar 迭代已自动推进）。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        bar = StepsBar(range(5))
        lg.scalar({"loss": 1.0})
        assert bar.n == 0
        lg.scalar({"loss": 1.0}, advance=2)
        assert bar.n == 2
    finally:
        lg.close()


def test_mirror_strips_ansi(tmp_path):
    """日志文件剥离 ANSI 颜色码，保持纯文本。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("\x1b[38;2;168;85;247mtrain\x1b[0m \x1b[1;38;2;168;85;247m50.0%\x1b[0m\r")
    m.close()
    assert read(tmp_path / "c.log") == "train 50.0%\n"


def test_mirror_drops_clear_line_artifacts(tmp_path):
    """行清除序列（\\r + 空格覆盖 + \\r，leave=False 关闭时产生）不清掉已刷新的进度条行。"""
    m = Mirror(tmp_path / "c.log", throttle=0)
    m.write("bar 50%\r")
    m.write("\r" + " " * 20 + "\r")  # leave=False 关闭的清除序列
    m.close()
    assert read(tmp_path / "c.log") == "bar 50%\n"


# ---------------------------------------------------------------- 控制台消息
class _SpyBar:
    """记录 clear_line / refresh 调用的假进度条。"""

    def __init__(self):
        self.cleared = 0
        self.refreshes = 0

    def clear_line(self):
        self.cleared += 1

    def refresh(self, force: bool = False):
        self.refreshes += 1


def test_console_message_clears_bar_without_redraw():
    """TTY 上消息打印前擦除进度条行；之后不主动重绘，进度条由下一次渲染重新开始。"""
    out = _TTY()
    saved = sys.stdout
    sys.stdout = out
    try:
        spy = _SpyBar()
        console = Console(top_bar=lambda: spy)
        console.log("hello")
    finally:
        sys.stdout = saved
    assert spy.cleared == 1 and spy.refreshes == 0
    assert "hello" in out.getvalue()


def test_console_message_plain_stream_untouched():
    """非 TTY（重定向）时不插擦除码、不重绘，行为与从前一致。"""
    out = io.StringIO()
    saved = sys.stdout
    sys.stdout = out
    try:
        bar = tqdm(total=5, file=out, mininterval=0)
        console = Console(top_bar=lambda: bar)
        console.log("hello")
        bar.close()
    finally:
        sys.stdout = saved
    assert "\x1b[" not in out.getvalue()


def test_bar_stack_cover_top_erases_displayed_line():
    """嵌套开条前 cover_top 擦除栈顶已渲染的行，子条首帧不压出残迹。"""
    out = _TTY()
    stack = BarStack()
    bar = tqdm(total=5, file=out, mininterval=0)
    stack.push(bar)
    stack.cover_top()
    assert "\x1b[2K" in out.getvalue()       # 栈顶行已被擦除
    bar.close()


# ---------------------------------------------------------------- Melog 集成
def test_melog_mirrors_console_log(tmp_path, capsys):
    """训练期间 print 与进度条同步进会话控制台日志；close 后还原 stdout。"""
    saved_stdout = sys.stdout
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        assert sys.stdout is not saved_stdout  # 已接管
        for _ in StepsBar(range(2)):
            lg.scalar({"loss": 0.5})
            print("step done")
    finally:
        lg.close()
    assert sys.stdout is saved_stdout  # 已还原

    log_path = next((tmp_path / "t").glob("**/*console-*.log"))
    assert log_path.name.startswith("console-") and log_path.name.endswith(".log")
    text = read(log_path)
    assert "step done\n" in text
    assert "loss=0.5" in text          # 进度条 postfix
    assert "2/2" in text               # 定稿的最终进度


def test_melog_file_lines_match_terminal(tmp_path, capsys):
    """控制台日志的每一行都与终端显示一致（进度条字符替换回细线后逐行可对上）。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        for step in StepsBar(range(3), epoch=0):
            lg.scalar({"loss": 1.0 / (step + 1)})
            if step == 1:
                lg.log("一条消息")
    finally:
        lg.close()

    file_lines = [ln.strip() for ln in read(lg.console_log).splitlines() if ln.strip()]
    term = capsys.readouterr().out
    term_lines = [ln.strip() for ln in re.split(r"[\r\n]", term) if ln.strip()]
    assert file_lines, "日志文件不为空"
    back = str.maketrans({"█": "━", "░": "─"})  # 文件侧加高字符还原为细线比对
    for fl in file_lines:
        assert fl.translate(back) in term_lines, f"文件行未在终端出现: {fl}"
    # 文件侧进度条为加高块状字符，消息行保留 [MM-DD HH:MM:SS][文件名] 前缀
    bar_lines = [ln for ln in file_lines if "█" in ln or "░" in ln]
    assert bar_lines, "文件中应有加高块状进度条"
    msg_lines = [ln for ln in file_lines if "一条消息" in ln]
    assert msg_lines and all(
        re.match(r"^\[\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[test_tqdm\] ", ln)
        for ln in msg_lines
    )


def test_melog_console_log_per_session(tmp_path):
    """每次 init 生成独立的 console-<时间戳>.log，不跨会话追加。"""
    lg1 = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        lg1.log("第一次会话")
        first = lg1.console_log
    finally:
        lg1.close()
    lg2 = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        lg2.log("第二次会话")
        second = lg2.console_log
        assert read(second) == read(second)  # 可重复读
        assert "第一次会话" not in read(second)
    finally:
        lg2.close()
    assert first != second
    assert "第二次会话" in read(second)
    assert "第一次会话" in read(first)


def test_melog_console_log_progress_line_uses_cr(tmp_path):
    """训练中途控制台日志已有就地刷新的进度条行，close 后定稿为完整行。"""
    lg = Melog(project="t", output_dir=str(tmp_path), enable_web=False)
    try:
        for _ in StepsBar(range(1)):
            path = lg.console_log
            assert "[0/1]" in path.read_bytes().decode("utf-8")  # 首帧已落盘
    finally:
        lg.close()
    assert lg.console_log.read_bytes().endswith(b"\n")  # 定稿为完整行
