"""降采样逻辑测试。"""

import pytest

from melog.downsample import downsample


def test_no_downsample_under_threshold():
    pts = [(i, float(i)) for i in range(100)]
    out = downsample(pts, 200)
    assert out == pts


def test_none_means_unlimited():
    pts = [(i, float(i)) for i in range(5000)]
    assert len(downsample(pts, None)) == 5000


def test_length_capped():
    pts = [(i, float(i)) for i in range(10_000)]
    out = downsample(pts, 2000)
    assert len(out) == 2000


def test_endpoints_preserved():
    pts = [(i, float(i * i)) for i in range(1000)]
    out = downsample(pts, 100)
    assert out[0] == (0, 0.0)
    assert out[-1] == (999, 999.0 * 999.0)


def test_mean_converges_for_linear():
    # 线性序列降采样后均值仍应落在原直线上
    pts = [(i, 2.0 * i) for i in range(1000)]
    out = downsample(pts, 50)
    for step, value in out[1:-1]:
        assert value == pytest.approx(2.0 * step, abs=1e-6)


def test_invalid_threshold_passthrough():
    pts = [(i, float(i)) for i in range(10)]
    assert len(downsample(pts, 1)) == 10  # max_points < 2 不处理


def test_snapshot_downsampled(tmp_path):
    from melog.web.server import MetricStore

    store = MetricStore()
    for i in range(5000):
        store.add(i, {"loss": i * 0.001})
    snap = store.snapshot(max_points=1000)
    assert len(snap["loss"]) == 1000
    # 全量仍保留在内存历史中
    assert len(store.snapshot()["loss"]) == 5000


def test_downsample_keeps_epoch():
    # 分桶均值取 step/value，epoch 取桶内最后一个非空值
    pts = [(0, 1.0, 0), (1, 2.0, 0), (2, 3.0, 1), (3, 4.0, 1)]
    out = downsample(pts, 2)
    assert out[0] == (0, 1.0, 0)
    assert out[1][2] == 1


def test_downsample_2tuple_passthrough():
    pts = [(i, float(i)) for i in range(100)]
    out = downsample(pts, 10)
    assert all(len(p) == 2 for p in out)
