"""媒体记录测试：图像/音频落盘、媒体索引、视图解析与 Melog 接口。"""

import json
import wave

import pytest

from melog.core import Melog
from melog.media import read_wav_info, sanitize_name, save_audio, save_image
from melog.web.loader import LogLoader, MediaLoader
from melog.web.media_store import MediaStore
from melog.web.media_view import MediaView


np = pytest.importorskip("numpy")
pytest.importorskip("PIL")


# ---------------------------------------------------------------- 名称消毒
def test_sanitize_name_keeps_hierarchy():
    assert sanitize_name("train/sample_0") == "train/sample_0"
    assert sanitize_name("a//b") == "a/b"


def test_sanitize_name_blocks_traversal_and_illegal():
    assert sanitize_name("../etc/passwd") != "../etc/passwd"
    assert ".." not in sanitize_name("a/../b").split("/")
    assert sanitize_name("a b/c:d") == "a_b/c_d"
    with pytest.raises(ValueError):
        sanitize_name("")
    with pytest.raises(ValueError):
        sanitize_name("//")


# ---------------------------------------------------------------- 图像落盘
def test_save_image_numpy_gray_rgb_and_float(tmp_path):
    gray = (np.random.rand(8, 10) * 255).astype(np.uint8)
    name = save_image(gray, tmp_path, "000000001")
    assert name == "000000001.png"

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    assert save_image(rgb, tmp_path, "000000002") == "000000002.png"

    floats = np.linspace(0.0, 1.0, 8 * 10).reshape(8, 10)  # 浮点自动映射 0-255
    save_image(floats, tmp_path, "000000003")

    from PIL import Image

    assert Image.open(tmp_path / "000000001.png").size == (10, 8)
    assert Image.open(tmp_path / "000000002.png").mode == "RGB"
    assert Image.open(tmp_path / "000000003.png").getpixel((0, 0)) == 0


def test_save_image_rgba_and_channel1(tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert save_image(rgba, tmp_path, "a") == "a.png"
    assert save_image(np.zeros((4, 4, 1), dtype=np.uint8), tmp_path, "b") == "b.png"

    from PIL import Image

    assert Image.open(tmp_path / "a.png").mode == "RGBA"


def test_save_image_from_file_copies_original_format(tmp_path):
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGB", (6, 6), (10, 20, 30)).save(src)
    out = tmp_path / "out"
    name = save_image(src, out, "000000005")
    assert name == "000000005.png"
    assert Image.open(out / name).getpixel((0, 0)) == (10, 20, 30)


def test_save_image_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 3, 4, 5)), tmp_path, "x")
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 3, 2)), tmp_path, "x")  # C=2 不支持
    with pytest.raises(TypeError):
        save_image(12345, tmp_path, "x")
    with pytest.raises(FileNotFoundError):
        save_image(tmp_path / "nope.png", tmp_path, "x")


# ---------------------------------------------------------------- 音频落盘
def test_save_audio_numpy_mono_and_stereo(tmp_path):
    sr = 8000
    mono = np.sin(np.linspace(0, 100, 800)).astype(np.float64)
    name = save_audio(mono, tmp_path, "000000001", sr)
    assert name == "000000001.wav"
    info = read_wav_info(tmp_path / name)
    assert info == {"channels": 1, "sr": sr, "seconds": pytest.approx(0.1)}

    stereo = np.zeros((100, 2))
    stereo[:, 1] = 0.5
    save_audio(stereo, tmp_path, "000000002", sr)
    info = read_wav_info(tmp_path / "000000002.wav")
    assert info["channels"] == 2

    # int16 波形原样写入
    ints = (np.sin(np.linspace(0, 10, 500)) * 10000).astype(np.int16)
    save_audio(ints, tmp_path, "000000003", sr)
    with wave.open(str(tmp_path / "000000003.wav"), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnframes() == 500


def test_save_audio_clips_and_accepts_path(tmp_path):
    # 浮点越界部分裁剪到 [-1, 1]，不溢出
    save_audio(np.array([2.0, -2.0, 0.5]), tmp_path, "clip", 8000)
    with wave.open(str(tmp_path / "clip.wav"), "rb") as w:
        import struct

        samples = struct.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes()))
    assert samples[0] == 32767 and samples[1] == -32767

    src = tmp_path / "src.wav"
    save_audio(np.zeros(100), tmp_path, "src", 8000)
    out = tmp_path / "out"
    assert save_audio(src, out, "000000009", 8000) == "000000009.wav"  # 路径按原格式复制

    with pytest.raises(ValueError):
        save_audio(np.zeros((2, 3, 4)), tmp_path, "bad", 8000)


def test_save_audio_transposes_channel_first(tmp_path):
    # (C, N) 常见于 torch 风格，自动转置为 (N, C)
    data = np.zeros((2, 100))
    data[0] = 1.0
    save_audio(data, tmp_path, "cf", 8000)
    info = read_wav_info(tmp_path / "cf.wav")
    assert info["channels"] == 2


# ---------------------------------------------------------------- MediaStore
def test_media_store_add_sort_dedupe_and_cap():
    store = MediaStore(max_per_name=3)
    for step in (5, 1, 3):
        store.add("image", "pred", step, f"f{step}.png", epoch=step // 3)
    store.add("image", "pred", 3, "f3-override.png")  # 同 step 覆盖
    snap = store.snapshot()
    assert [e["step"] for e in snap["image"]["pred"]] == [1, 3, 5]
    assert snap["image"]["pred"][1]["file"] == "f3-override.png"
    assert snap["image"]["pred"][0]["epoch"] == 0

    store.add("image", "pred", 7, "f7.png")
    store.add("image", "pred", 9, "f9.png")  # 超上限丢最旧
    snap = store.snapshot()
    assert [e["step"] for e in snap["image"]["pred"]] == [3, 5, 7, 9][1:]

    store.add("audio", "tone", 0, "a.wav", sr=8000)
    assert list(store.snapshot()["audio"]) == ["tone"]


# ---------------------------------------------------------------- 日志解析
def test_loaders_split_metrics_and_media(tmp_path):
    log = tmp_path / "metrics.melog"
    log.write_text(
        '{"metric": "loss", "step": 0, "value": 1.0, "epoch": 0}\n'
        '{"type": "image", "metric": "pred", "step": 0, "epoch": 0, "file": "media/image/pred/000000000.png"}\n'
        '{"type": "audio", "metric": "tone", "step": 5, "file": "media/audio/tone/000000005.wav", "sr": 16000}\n'
        "坏行\n",
        encoding="utf-8",
    )
    metrics = LogLoader.parse(log)
    assert metrics["loss"] == [(0, 1.0, 0)]  # 媒体记录不混入指标

    media = MediaLoader.parse(log)
    assert media["image"]["pred"] == [
        {"step": 0, "epoch": 0, "file": "media/image/pred/000000000.png"}
    ]
    assert media["audio"]["tone"] == [{"step": 5, "file": "media/audio/tone/000000005.wav", "sr": 16000}]


# ---------------------------------------------------------------- MediaView
def test_media_view_urls_and_resolve(tmp_path):
    store = MediaStore()
    store.add("image", "pred", 0, "media/image/pred/000000000.png", epoch=1)
    view = MediaView(store)
    view.set_base(tmp_path)

    snap = view.snapshot()
    entry = snap["image"]["pred"][0]
    assert entry["url"].startswith("/api/media/file?path=")
    assert "epoch" in entry

    target = tmp_path / "media/image/pred/000000000.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fake")
    resolved = view.resolve("media/image/pred/000000000.png")
    assert resolved == target
    assert view.resolve("../../etc/passwd") is None
    assert view.resolve("media/missing.png") is None
    assert view.resolve("") is None


def test_media_view_loaded_switch(tmp_path):
    store = MediaStore()
    store.add("image", "live", 0, "media/image/live/x.png")
    view = MediaView(store)
    view.set_base(tmp_path / "live_run")
    assert list(view.snapshot()["image"]) == ["live"]

    loaded_root = tmp_path / "old_run"
    view.set_loaded({"image": {"hist": [{"step": 3, "file": "y.png"}]}}, loaded_root)
    assert list(view.snapshot()["image"]) == ["hist"]
    assert view._loaded_base == loaded_root

    view.clear_loaded()
    assert list(view.snapshot()["image"]) == ["live"]


# ---------------------------------------------------------------- Melog 接口
@pytest.fixture
def logger(tmp_path):
    lg = Melog(project="m", output_dir=str(tmp_path), enable_web=False)
    yield lg
    lg.finish()


def test_image_attaches_to_last_scalar_position(logger):
    logger.scalar({"loss": 1.0}, epoch=2, step=7)
    logger.image("pred", np.zeros((4, 4), dtype=np.uint8))
    entries = logger.media.snapshot()["image"]["pred"]
    assert entries[0]["step"] == 7 and entries[0]["epoch"] == 2
    # 不推进 step 计数
    logger.scalar({"loss": 2.0})
    assert logger.store.snapshot()["loss"][-1]["step"] == 8


def test_media_explicit_position_and_files(logger):
    logger.scalar({"loss": 1.0})
    img = np.full((3, 3, 3), 200, dtype=np.uint8)
    audio = np.zeros(100, dtype=np.float64)
    logger.image("sample/a", img, step=42, epoch=1)
    logger.audio("sample/a", audio, sr=8000, step=42, epoch=1)

    media_root = logger.run_dir / "media"
    assert (media_root / "image/sample/a/000000042.png").is_file()
    assert (media_root / "audio/sample/a/000000042.wav").is_file()

    recs = [json.loads(l) for l in logger._log_file.read_text(encoding="utf-8").splitlines()]
    img_rec, aud_rec = recs[-2], recs[-1]
    assert img_rec == {"type": "image", "metric": "sample/a", "step": 42, "epoch": 1,
                       "file": "media/image/sample/a/000000042.png"}
    assert aud_rec["type"] == "audio" and aud_rec["sr"] == 8000

    snap = logger.media.snapshot()
    assert snap["image"]["sample/a"][0]["file"] == "media/image/sample/a/000000042.png"
    assert snap["audio"]["sample/a"][0]["sr"] == 8000


def test_media_before_any_scalar_defaults_to_zero(logger):
    logger.image("early", np.zeros((2, 2), dtype=np.uint8))
    entries = logger.media.snapshot()["image"]["early"]
    assert entries[0]["step"] == 0 and "epoch" not in entries[0]


def test_media_caption_roundtrip(logger):
    """caption 随记录贯通：journal / 内存索引 / 历史解析；无配文则不写该字段。"""
    logger.image("cap/img", np.zeros((2, 2), dtype=np.uint8),
                     step=1, epoch=0, caption="第一张\n说明文字")
    logger.image("cap/img", np.ones((2, 2), dtype=np.uint8), step=2)
    logger.audio("cap/tone", np.zeros(10), sr=8000, step=1, caption="转写文本")

    recs = [json.loads(l) for l in logger._log_file.read_text(encoding="utf-8").splitlines()]
    assert recs[-3]["caption"] == "第一张\n说明文字"
    assert "caption" not in recs[-2]  # 无配文的条目不带该字段
    assert recs[-1]["caption"] == "转写文本"

    snap = logger.media.snapshot()
    assert snap["image"]["cap/img"][0]["caption"] == "第一张\n说明文字"
    assert "caption" not in snap["image"]["cap/img"][1]

    parsed = MediaLoader.parse(logger._log_file)
    assert parsed["image"]["cap/img"][0]["caption"] == "第一张\n说明文字"
    assert "caption" not in parsed["image"]["cap/img"][1]
    assert parsed["audio"]["cap/tone"][0]["caption"] == "转写文本"


def test_media_path_copy(logger, tmp_path):
    from PIL import Image

    src = tmp_path / "in.png"
    Image.new("L", (4, 4)).save(src)
    logger.image("copied", src)
    assert (logger.run_dir / "media/image/copied/000000000.png").is_file()


def test_media_bad_name_rejected(logger):
    from melog.media import sanitize_name

    with pytest.raises(ValueError):
        sanitize_name("")
