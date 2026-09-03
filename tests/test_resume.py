"""断点续训测试：会话文件时间戳、历史回灌、重叠区截断。"""

import pytest

from melog import StepsBar
from melog.core import Melog
from melog.storage.melog_file import MelogFileReader
from melog.web.loader import LogLoader


def make(tmp_path, project="exp"):
    return Melog(project=project, output_dir=str(tmp_path), enable_web=False)


def read_records(run_dir) -> list:
    """按启动时间序合并 run 目录全部会话日志为 (step, epoch, values) 列表。"""
    out = []
    for f in LogLoader.session_files(run_dir):
        out.extend(MelogFileReader(f).records())
    return out


def train_epochs(m, lengths, start=0):
    """模拟训练：每个 epoch 一个 StepsBar，每步记一次 loss=epoch+步号。"""
    for e in range(start, start + len(lengths)):
        for i in StepsBar(range(lengths[e - start]), epoch=e):
            m.scalar({"loss": float(e + i / 100)})


# ---------------------------------------------------------------- 会话文件
def test_session_files_timestamped_and_distinct(tmp_path):
    """多次训练同一目录：各写一个带时间戳的会话文件，互不覆盖。"""
    m1 = make(tmp_path)
    m1.scalar({"loss": 1.0})
    m1.close()
    m2 = make(tmp_path)
    m2.scalar({"loss": 2.0})
    m2.close()
    files = sorted((tmp_path / "exp").glob("metrics-*.melog"))
    assert len(files) == 2
    assert [len(r) for r in (list(MelogFileReader(f).records()) for f in files)] == [1, 1]


# ---------------------------------------------------------------- 历史回灌
def test_resume_restores_history(tmp_path):
    """重跑同一目录：历史曲线回灌面板，坐标轴从中断处接续。"""
    m1 = make(tmp_path)
    train_epochs(m1, [3, 3])  # epoch 0/1 各 3 步 → x = 0..5
    m1.close()
    assert [(s, e) for s, e, _ in read_records(tmp_path / "exp")] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1),
    ]

    m2 = make(tmp_path)  # "重启"：不开新目录
    assert m2.run_dir == m1.run_dir
    snap = m2.store.snapshot()["loss"]  # 历史回灌到面板
    assert [p["step"] for p in snap] == [0, 1, 2, 3, 4, 5]
    assert m2._axis.step == 6 and m2._axis.bases == {0: 0, 1: 3}
    train_epochs(m2, [2], start=2)  # 从 epoch 2 继续训练
    m2.close()
    assert [(s, e) for s, e, _ in read_records(tmp_path / "exp")] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2),
    ]


def test_resume_truncates_overlapping_epoch(tmp_path):
    """中断残留（2 epoch + 100 step）→ 从 epoch 2 重训：重叠区被清除覆盖。

    折线在 x 轴上不回退：旧会话文件物理截断到 epoch 2 基准，新会话
    文件从基准处接续。
    """
    m1 = make(tmp_path)
    train_epochs(m1, [3, 3])  # 完整的 epoch 0/1
    for i in StepsBar(range(100), epoch=2):  # epoch 2 训了 100 步后"中断"
        m1.scalar({"loss": 9.0})
    m1.close()

    m2 = make(tmp_path)  # 从 2 epoch 的 checkpoint 继续训练
    sessions = LogLoader.session_files(tmp_path / "exp")
    old_session = sessions[0]
    old_before = len(list(MelogFileReader(old_session).records()))
    assert old_before == 106  # 6 + 100 条旧记录

    train_epochs(m2, [4], start=2)  # 绑定 epoch 2 → 自动截断，重训 4 步
    m2.close()

    # 旧会话文件只保留 epoch 0/1 的 6 条，epoch 2 的 100 条残留被清除
    assert len(list(MelogFileReader(old_session).records())) == 6
    records = read_records(tmp_path / "exp")
    assert [(s, e) for s, e, _ in records] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1),
        (6, 2), (7, 2), (8, 2), (9, 2),
    ]
    # 新会话文件只含 epoch 2 重训的 4 条
    new_session = LogLoader.session_files(tmp_path / "exp")[1]
    assert [s for s, _, _ in MelogFileReader(new_session).records()] == [6, 7, 8, 9]


def test_resume_new_epoch_no_truncation(tmp_path):
    """checkpoint 在 epoch 末、重启后进入全新 epoch：不截断，直接接续。"""
    m1 = make(tmp_path)
    train_epochs(m1, [3])
    m1.close()

    m2 = make(tmp_path)
    train_epochs(m2, [2], start=1)  # epoch 1 从未训过
    m2.close()
    assert [(s, e) for s, e, _ in read_records(tmp_path / "exp")] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1),
    ]


def test_resume_rebind_old_epoch_after_truncation(tmp_path):
    """截断续训后继续跑后续 epoch：旧的后续 epoch 基准已失效，不再误截断。"""
    m1 = make(tmp_path)
    train_epochs(m1, [3, 3])  # epoch 0/1 各 3 步
    for _ in StepsBar(range(5), epoch=2):  # epoch 2 中断残留
        m1.scalar({"loss": 9.0})
    m1.close()

    m2 = make(tmp_path)
    train_epochs(m2, [2, 2], start=2)  # 重训 epoch 2，再跑 epoch 3
    m2.close()
    assert [(s, e) for s, e, _ in read_records(tmp_path / "exp")] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1),
        (6, 2), (7, 2), (8, 3), (9, 3),
    ]


def test_resume_truncation_updates_store(tmp_path):
    """截断后面板内存历史同步回滚（已连接的客户端经 history 广播整体替换）。"""
    m1 = make(tmp_path)
    train_epochs(m1, [3])
    for _ in StepsBar(range(5), epoch=1):
        m1.scalar({"loss": 9.0})
    m1.close()

    m2 = make(tmp_path)
    train_epochs(m2, [2], start=1)
    snap = m2.store.snapshot()["loss"]
    # 旧 epoch1 的 5 步残留被清除，重训 2 步从 x=3 接续，无 x 轴回退的点
    assert [p["step"] for p in snap] == [0, 1, 2, 3, 4]
    m2.close()


def test_resume_restores_media_and_colors(tmp_path):
    """续训回灌不止指标：媒体索引与用户配色一并恢复。"""
    pytest.importorskip("PIL")
    import numpy as np

    m1 = make(tmp_path)
    m1.scalar({"loss": 1.0})
    m1.image("pred", np.zeros((4, 4), dtype=np.uint8), caption="样本")
    m1.set_colors({"loss": "#123456"})
    m1.close()

    m2 = make(tmp_path)
    assert m2.media.snapshot()["image"]["pred"][0]["caption"] == "样本"
    assert m2._colors == {"loss": "#123456"}
    m2.close()


def test_resume_nested_same_epoch_bar_no_truncation(tmp_path):
    """嵌套 bar 重入当前 epoch（如 train 内嵌 val）：幂等，不触发截断。"""
    m = make(tmp_path)
    outer = StepsBar(range(3), epoch=0)
    for _ in outer:
        m.scalar({"loss": 1.0})
        inner = StepsBar(range(2), epoch=0)
        for _ in inner:
            m.scalar({"loss": 1.0})
    m.close()
    assert [s for s, _, _ in read_records(tmp_path / "exp")] == list(range(9))


def test_resume_truncated_torn_tail(tmp_path):
    """上次进程被杀留下残缺尾部：重启后自动修复，续训不受影响。"""
    m1 = make(tmp_path)
    train_epochs(m1, [3])
    m1.close()
    run_dir = tmp_path / "exp"
    log = next(run_dir.glob("metrics-*.melog"))
    with open(log, "r+b") as f:  # 模拟写 block 中途被杀
        f.seek(0, 2)
        f.write(b"\x01\x00\x00\x99")

    m2 = make(tmp_path)
    train_epochs(m2, [2], start=1)
    m2.close()
    assert [(s, e) for s, e, _ in read_records(run_dir)] == [
        (0, 0), (1, 0), (2, 0), (3, 1), (4, 1),
    ]
