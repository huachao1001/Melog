"""metrics.melog 二进制容器：紧凑编码的写入 / 读取 / 重叠截断。

文件布局（小端）::

    magic  8 字节  b"MELOG" + 版本号 1 + 2 字节保留
    block* 直到文件末尾，每个 block：
        header 1 字节  高 4 位 flags（bit0 = payload 经 zlib 压缩），
                       低 4 位 type
        length 4 字节  uint32 payload 长度
        payload

block 类型（type）::

    1 METRIC  指标批量记录（一次提交的一批指标合并为一条记录）
    2 MEDIA   单条媒体 / JSON 记录（utf-8 JSON 文本，默认压缩存储）
    3 NAME    指标名符号表项：varint id + utf-8 名称（名字首次使用时写入）

METRIC payload（记录按提交顺序排列，step 相对上一条记录做增量编码）::

    varint n                      记录条数
    每条记录：
        zigzag varint             step 增量（相对文件内上一条记录，首条相对 0）
        varint                    0 = 无 epoch，否则 epoch + 1
        varint k                  该条记录的指标个数
        k × (varint id, f64)      指标名 id（查符号表）+ 8 字节 float64 值

空间对比：一条 3 指标记录约 5(block 头) + 1 + 1 + 1 + 3×9 = 35 字节，
同等内容的 JSONL 文本约 150+ 字节。

容错：进程中途被杀可能留下残缺的尾部 block（header 或 payload 不完整），
读取时遇到即停止（之前的 block 不受影响）；写入器打开已存在的文件时
会先扫描并截掉残缺尾部，保证追加不会污染后续数据。
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

__all__ = ["MelogFile", "MelogFileReader"]

_PATH = Union[str, Path]
# 一条指标记录：(step, epoch, {指标名: 值})
Row = Tuple[int, Optional[int], Dict[str, float]]


# ------------------------------------------------------------------ varint
def _put_varint(buf: bytearray, v: int) -> None:
    """无符号 LEB128 编码（v 须 >= 0）。"""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _get_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """解码无符号 LEB128，返回 (值, 新位置)；越界抛 IndexError。"""
    result = shift = 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos + 1
        shift += 7
        pos += 1


def _zigzag_encode(v: int) -> int:
    """有符号整数 -> 无符号（负增量也能紧凑存）。"""
    return (v << 1) if v >= 0 else ((-v << 1) - 1)


def _zigzag_decode(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


# ------------------------------------------------------------------ block 编解码
def _block_bytes(btype: int, flags: int, payload: bytes) -> bytes:
    return struct.pack("<BI", flags | btype, len(payload)) + payload


def _iter_blocks(f):
    """依次产出 (type, flags, payload)；遇到残缺尾部即停止。"""
    while True:
        head = f.read(5)
        if len(head) < 5:
            return
        header, length = head[0], struct.unpack("<I", head[1:5])[0]
        payload = f.read(length)
        if len(payload) < length:
            return  # 尾部被截断（进程中途被杀）
        yield header & 0x0F, header & 0xF0, payload


def _decode_metric(payload: bytes, id2name: Dict[int, str],
                   prev_step: int) -> Tuple[List[Row], int]:
    """解码 METRIC payload，返回 (记录列表, 新的 step 基准)。

    step 为相对上一条记录的增量，基准跨 block 连续，由调用方传入并
    接续；id2name 随 NAME block 逐步建立，缺失视为文件损坏抛 KeyError。
    """
    n, pos = _get_varint(payload, 0)
    rows: List[Row] = []
    for _ in range(n):
        delta, pos = _get_varint(payload, pos)
        step = prev_step + _zigzag_decode(delta)
        prev_step = step
        code, pos = _get_varint(payload, pos)
        epoch = None if code == 0 else code - 1
        k, pos = _get_varint(payload, pos)
        values: Dict[str, float] = {}
        for _ in range(k):
            nid, pos = _get_varint(payload, pos)
            (value,) = struct.unpack_from("<d", payload, pos)
            pos += 8
            name = id2name.get(nid)
            if name is None:
                raise KeyError(nid)  # 符号表缺失 = 文件损坏，到此为止
            values[name] = value
        rows.append((step, epoch, values))
    return rows, prev_step


def _decode_media(_btype: int, flags: int, payload: bytes) -> Optional[Dict]:
    if flags & MelogFile._FLAG_ZLIB:
        payload = zlib.decompress(payload)
    rec = json.loads(payload.decode("utf-8"))
    return rec if isinstance(rec, dict) else None


def _encode_rows(rows: List[Row], id_of: Dict[str, int], prev_step: int) -> bytes:
    """把记录行编码为 METRIC payload（id_of 提供名字 -> id，prev 为增量基准）。"""
    buf = bytearray()
    _put_varint(buf, len(rows))
    for step, epoch, values in rows:
        _put_varint(buf, _zigzag_encode(step - prev_step))
        prev_step = step
        _put_varint(buf, 0 if epoch is None else epoch + 1)
        _put_varint(buf, len(values))
        for name, value in values.items():
            _put_varint(buf, id_of[name])
            buf.extend(struct.pack("<d", value))
    return bytes(buf)


# ------------------------------------------------------------------ 写入器
class MelogFile:
    """metrics.melog 写入器：会话文件从头创建，追加式写 block。

    由 Journal 持有：指标记录经 add_batch 成块写入（落盘节奏由调用方
    控制），媒体记录经 append_media 即时写入。打开已存在且非空的文件时
    先修复残缺尾部并重扫符号表，保证续写不污染数据。
    """

    MAGIC = b"MELOG\x01\x00\x00"
    TYPE_METRIC, TYPE_MEDIA, TYPE_NAME = 1, 2, 3
    _FLAG_ZLIB = 0x10  # header 高 4 位：payload 已 zlib 压缩

    def __init__(self, path: _PATH):
        self._path = Path(path)
        self._names: Dict[str, int] = {}  # 指标名 -> id（文件内符号表）
        self._prev_step = 0  # 上一条记录的 step（增量编码基准）
        self._file = self._open()

    # ------------------------------------------------------------ 写入接口
    def add_batch(self, records: List[Dict]) -> None:
        """写入一批指标记录（Journal.flush 的暂存记录，按 step 归并成行）。"""
        if not records:
            return
        rows: List[Row] = []
        for rec in records:
            step, epoch = rec["step"], rec.get("epoch")
            if rows and rows[-1][0] == step and rows[-1][1] == epoch:
                rows[-1][2][rec["metric"]] = float(rec["value"])
            else:
                rows.append((step, epoch, {rec["metric"]: float(rec["value"])}))
        for row in rows:  # 新指标名先写符号表项（读取按顺序建表）
            for name in row[2]:
                if name not in self._names:
                    self._define(name)
        self._write_block(self.TYPE_METRIC, _encode_rows(rows, self._names, self._prev_step))
        self._prev_step = rows[-1][0]

    def append_media(self, record: Dict) -> None:
        """即时写入一条媒体 / JSON 记录（zlib 压缩）。"""
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        self._write_block(self.TYPE_MEDIA, zlib.compress(payload), compress=True)

    def flush(self) -> None:
        """兼容 Journal 旧接口：写入器无内部缓冲，无需落盘动作。"""

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def truncate_from(self, cut_step: int) -> Tuple[Optional[int], Optional[int]]:
        """物理截断本文件尾部 step >= cut_step 的记录（续训清除重叠区）。

        返回截断后最后一条保留记录的 (step, epoch)；无保留记录时返回
        (None, None)。调用方需保证此前已 flush（无未落盘记录）。
        """
        self._file.close()
        result = MelogFile.truncate(self._path, cut_step)
        self._file = self._open()  # 重开并重扫符号表 / 增量基准
        return result

    # ------------------------------------------------------------ 内部
    def _open(self):
        p = self._path
        if p.exists() and p.stat().st_size:
            f = open(p, "r+b")
            end = self._repair_tail(f)  # 截掉被杀进程留下的残缺尾部
            f.truncate(end)
            self._load_state(f)
            f.seek(0, 2)
            return f
        p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, "w+b")
        f.write(self.MAGIC)
        f.flush()
        return f

    def _load_state(self, f) -> None:
        """从文件头重扫符号表与 step 增量基准（打开已有文件时续写用）。"""
        f.seek(0)
        f.read(len(self.MAGIC))
        id2name: Dict[int, str] = {}
        prev = 0
        try:
            for btype, _flags, payload in _iter_blocks(f):
                if btype == self.TYPE_NAME:
                    nid, pos = _get_varint(payload, 0)
                    name = payload[pos:].decode("utf-8")
                    id2name[nid] = name
                    self._names.setdefault(name, len(self._names))
                elif btype == self.TYPE_METRIC:
                    _rows, prev = _decode_metric(payload, id2name, prev)
        except (IndexError, struct.error, KeyError):
            pass  # 残缺尾部：修复时已截掉，这里读到即停
        self._prev_step = prev

    def _define(self, name: str) -> None:
        nid = len(self._names)
        self._names[name] = nid
        buf = bytearray()
        _put_varint(buf, nid)
        buf.extend(name.encode("utf-8"))
        self._write_block(self.TYPE_NAME, bytes(buf))

    def _write_block(self, btype: int, payload: bytes, compress: bool = False) -> None:
        flags = self._FLAG_ZLIB if compress else 0
        self._file.write(struct.pack("<BI", flags | btype, len(payload)))
        self._file.write(payload)
        self._file.flush()

    @staticmethod
    def _repair_tail(f) -> int:
        """返回最后一个完整 block 的结束偏移（残缺尾部由此截掉）。"""
        f.seek(0)
        f.read(len(MelogFile.MAGIC))
        end = f.tell()
        for _btype, _flags, _payload in _iter_blocks(f):
            end = f.tell()
        return end

    # ------------------------------------------------------------ 截断
    @staticmethod
    def truncate(path: _PATH, cut_step: int) -> Tuple[Optional[int], Optional[int]]:
        """截断指定日志文件中 step >= cut_step 的所有记录（含媒体记录）。

        截断边界可能落在 block 中间：受影响 block 里 step < cut 的记录
        重编码后原位保留，其后的内容全部丢弃。返回截断后最后一条保留
        记录的 (step, epoch)；无保留记录时返回 (None, None)。
        """
        p = Path(path)
        if not p.exists() or p.stat().st_size <= len(MelogFile.MAGIC):
            return (None, None)
        with open(p, "r+b") as f:
            f.seek(0)
            if f.read(len(MelogFile.MAGIC)) != MelogFile.MAGIC:
                return (None, None)
            blocks: List[Tuple[int, int, int, bytes]] = []  # (offset, type, flags, payload)
            base = f.tell()
            for btype, flags, payload in _iter_blocks(f):
                blocks.append((base, btype, flags, payload))
                base = f.tell()

            # 找到第一个含 step >= cut 记录的 block；同时维护符号表
            id2name: Dict[int, str] = {}
            cut_at = None
            last_kept: Optional[Row] = None
            prev = 0  # step 增量基准（跨 block 连续）
            for i, (_off, btype, _flags, payload) in enumerate(blocks):
                if btype == MelogFile.TYPE_NAME:
                    nid, pos = _get_varint(payload, 0)
                    id2name[nid] = payload[pos:].decode("utf-8")
                elif btype == MelogFile.TYPE_METRIC:
                    block_prev = prev  # 本 block 首条记录的增量基准
                    try:
                        rows, prev = _decode_metric(payload, id2name, prev)
                    except KeyError:
                        break  # 符号表缺失 = 文件损坏，按已解析部分处理
                    for row in rows:
                        if row[0] >= cut_step:
                            cut_at = i
                            break
                        last_kept = row
                    if cut_at is not None:
                        prev = block_prev  # 区域重扫从本 block 起重新接续
                        break
            if cut_at is None:
                return (last_kept[0], last_kept[1]) if last_kept else (None, None)
            base_count = len(id2name)  # 截断点前已有的符号表项数

            # 受影响区域：过滤保留记录；媒体按 step 判断去留；区域内新
            # 出现的指标名补写符号表项（其原定义在被丢弃区域内）
            name2id = {n: i for i, n in id2name.items()}
            keep_rows: List[Row] = []
            keep_media: List[bytes] = []
            for _off, btype, flags, payload in blocks[cut_at:]:
                if btype == MelogFile.TYPE_MEDIA:
                    rec = _decode_media(btype, flags, payload)
                    if rec is not None and rec.get("step", cut_step) < cut_step:
                        keep_media.append(_block_bytes(btype, flags, payload))
                elif btype == MelogFile.TYPE_NAME:
                    nid, pos = _get_varint(payload, 0)
                    name = payload[pos:].decode("utf-8")
                    if name not in name2id:
                        name2id[name] = nid
                        id2name[nid] = name
                elif btype == MelogFile.TYPE_METRIC:
                    try:
                        rows, prev = _decode_metric(payload, id2name, prev)
                    except KeyError:
                        break  # 符号表缺失 = 文件损坏，按已解析部分处理
                    for row in rows:
                        if row[0] < cut_step:
                            keep_rows.append(row)

            # 原位重写：区域内新增符号表项 + 保留媒体 + 保留指标
            f.seek(blocks[cut_at][0])
            f.truncate()
            for name, nid in name2id.items():
                if nid >= base_count:  # 原定义在被丢弃区域内，重写符号表项
                    buf = bytearray()
                    _put_varint(buf, nid)
                    buf.extend(name.encode("utf-8"))
                    f.write(_block_bytes(MelogFile.TYPE_NAME, 0, bytes(buf)))
            for raw in keep_media:
                f.write(raw)
            prev = last_kept[0] if last_kept else 0
            f.write(_encode_rows(keep_rows, name2id, prev))
            f.flush()

        if keep_rows:
            return (keep_rows[-1][0], keep_rows[-1][1])
        return (last_kept[0], last_kept[1]) if last_kept else (None, None)


# ------------------------------------------------------------------ 读取器
class MelogFileReader:
    """metrics.melog 读取器：按顺序产出指标记录与媒体记录。

    残缺尾部 / 未知 block / 符号表缺失都视为文件损坏，读到即停
    （之前的完整数据不受影响）。
    """

    def __init__(self, path: _PATH):
        self._path = Path(path)

    def records(self) -> Iterator[Row]:
        """按提交顺序产出 (step, epoch, {指标名: 值})。"""
        if not self._path.exists():
            return
        with open(self._path, "rb") as f:
            if f.read(len(MelogFile.MAGIC)) != MelogFile.MAGIC:
                return
            id2name: Dict[int, str] = {}
            prev = 0
            try:
                for btype, _flags, payload in _iter_blocks(f):
                    if btype == MelogFile.TYPE_NAME:
                        nid, pos = _get_varint(payload, 0)
                        id2name[nid] = payload[pos:].decode("utf-8")
                    elif btype == MelogFile.TYPE_METRIC:
                        rows, prev = _decode_metric(payload, id2name, prev)
                        yield from rows
            except (IndexError, struct.error, KeyError):
                return  # 残缺 / 损坏 block：到此为止

    def media(self) -> Iterator[Dict]:
        """按写入顺序产出媒体记录（原始 JSON 字典）。"""
        if not self._path.exists():
            return
        with open(self._path, "rb") as f:
            if f.read(len(MelogFile.MAGIC)) != MelogFile.MAGIC:
                return
            try:
                for btype, flags, payload in _iter_blocks(f):
                    if btype == MelogFile.TYPE_MEDIA:
                        rec = _decode_media(btype, flags, payload)
                        if rec is not None:
                            yield rec
            except (IndexError, struct.error, zlib.error, ValueError):
                return  # 残缺 / 损坏 block：到此为止
