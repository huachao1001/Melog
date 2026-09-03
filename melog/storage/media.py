"""媒体记录：图像 / 音频落盘与 URL 构造。

- save_image：文件路径直接复制；numpy / torch / PIL 数组编码为 PNG
- save_audio：文件路径直接复制；numpy / torch 一维波形写为 16bit WAV

只依赖标准库 + 调用方自带的 numpy/Pillow/torch：路径复制无需任何第三方
编码库，数组编码仅在真正传入数组时才 import PIL。
"""

from __future__ import annotations

import re
import shutil
import wave
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

__all__ = ["save_image", "save_audio", "media_url", "sanitize_name"]


def sanitize_name(name: str) -> str:
    """指标名 -> 相对路径：'/' 保留为子目录层级，其余字符消毒。

    名字里常见的 "train/sample_0" 会存为 media/image/train/sample_0/…；
    每段仅保留字母数字与 -_.，杜绝路径穿越。
    """
    if not name or not name.strip():
        raise ValueError("媒体名称不能为空")
    parts = [p for p in name.split("/") if p]
    if not parts:
        raise ValueError(f"非法的媒体名称: {name!r}")
    safe = []
    for p in parts:
        cleaned = re.sub(r"[^\w\-.]", "_", p).strip("._") or "_"
        if cleaned in (".", ".."):
            cleaned = "_"
        safe.append(cleaned)
    return "/".join(safe)


def media_url(relpath: str) -> str:
    """媒体文件相对路径 -> 服务端 URL（file 路由带白名单校验）。"""
    return "/api/media/file?path=" + quote(relpath)


def _as_numpy(data):
    """torch tensor -> numpy（其余原样返回）。"""
    if hasattr(data, "detach"):  # torch.Tensor
        data = data.detach().cpu().numpy()
    return data


def _to_uint8(arr):
    import numpy as np

    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype("float64")
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 1.0:  # [0,1] / [-1,1] 浮点图按满量程拉伸
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_image(data: Union[str, Path, object], out_dir: Path, stem: str) -> str:
    """保存一帧图像到 out_dir，返回落盘文件名。

    data 支持：文件路径（按原格式复制）/ PIL.Image / numpy / torch
    （2D 灰度，3D (H,W,C)，C 取 1/3/4）。浮点数组自动映射到 0-255。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (str, Path)):
        src = Path(data)
        if not src.is_file():
            raise FileNotFoundError(f"图像文件不存在: {src}")
        dst = out_dir / f"{stem}{src.suffix.lower() or '.png'}"
        shutil.copyfile(src, dst)
        return dst.name

    if hasattr(data, "save") and hasattr(data, "mode"):  # PIL.Image
        from PIL import Image  # noqa: F401  # 确认 PIL 可用

        dst = out_dir / f"{stem}.png"
        data.save(dst)
        return dst.name

    import numpy as np

    arr = _as_numpy(data)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"不支持的图像类型: {type(data).__name__}")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    else:
        raise ValueError(f"不支持的图像形状: {arr.shape}（期望 (H,W) 或 (H,W,C)，C=1/3/4）")

    from PIL import Image

    dst = out_dir / f"{stem}.png"
    Image.fromarray(_to_uint8(np.ascontiguousarray(arr)), mode=mode).save(dst)
    return dst.name


def save_audio(data: Union[str, Path, object], out_dir: Path, stem: str, sr: int) -> str:
    """保存一段音频到 out_dir，返回落盘文件名。

    data 支持：文件路径（wav/mp3/flac 等按原格式复制）/ numpy / torch
    一维波形 (N,) 或二维 (N, 声道数)。浮点波形按 [-1,1] 裁剪写为
    16bit WAV（标准库 wave，无需第三方依赖）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (str, Path)):
        src = Path(data)
        if not src.is_file():
            raise FileNotFoundError(f"音频文件不存在: {src}")
        dst = out_dir / f"{stem}{src.suffix.lower() or '.wav'}"
        shutil.copyfile(src, dst)
        return dst.name

    import numpy as np

    arr = _as_numpy(data)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"不支持的音频类型: {type(data).__name__}")
    if arr.ndim == 2:  # (N, C)：声道放最后
        if arr.shape[0] < arr.shape[1]:  # (C, N) 常见于 torch，自动转置
            arr = arr.T
    elif arr.ndim != 1:
        raise ValueError(f"不支持的音频形状: {arr.shape}（期望 (N,) 或 (N, 声道数)）")

    if arr.dtype == np.int16:
        pcm = np.ascontiguousarray(arr)
    else:
        pcm = np.clip(np.asarray(arr, dtype="float64"), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    frames = pcm.tobytes()

    dst = out_dir / f"{stem}.wav"
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(frames)
    return dst.name


def read_wav_info(path: Union[str, Path]) -> Optional[dict]:
    """读取 WAV 元信息（声道/采样率/时长），非 WAV 返回 None。"""
    try:
        with wave.open(str(path), "rb") as w:
            return {
                "channels": w.getnchannels(),
                "sr": w.getframerate(),
                "seconds": w.getnframes() / w.getframerate(),
            }
    except (wave.Error, IsADirectoryError):
        return None
