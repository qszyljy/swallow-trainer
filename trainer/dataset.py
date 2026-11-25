"""数据扫描与窗口切片。

两种数据来源，扫描时自动判断：

**1. 手机端导出的片段包（推荐）**
解压 zip 后目录里有 ``manifest.csv``，里面的 ``start_sec/end_sec`` 是人工标注过的
精确区间，``kind=clip`` 的文件本身就是裁好的单个事件。这种数据不需要再猜事件在哪。

**2. 原始录音目录**
直接读 App 录出来的文件名：::

    data/p01_swallow_20261125_101010.wav
    data/p01_cough_20261125_101100.wav

文件名规范 ``[被试编号_]类别_时间戳.wav``，类别取 swallow / cough / interfere / rest。
没有被试编号时归到 ``p00`` 并给出警告——LOSO 划分必须有被试编号。
没有标注时事件区间靠短时能量峰值自动定位，准确度不如人工标注。
"""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

CLASSES = ("swallow", "cough", "interfere")
REST = "rest"
ALL_LABELS = (*CLASSES, REST)

DEFAULT_SR = 48000
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_HOP_SEC = 0.25

_NAME_RE = re.compile(
    r"^(?:(?P<subject>[pP]\d{1,3})_)?(?P<label>[a-zA-Z]+)_"
    r"(?P<stamp>\d{8}_\d{6})(?:_\d{1,3})?\.wav$"
)


@dataclass
class Record:
    path: Path
    subject: str
    label: str
    # 手机端导出的精确事件区间（秒）。None 表示没有标注，只能靠能量峰值定位。
    event_start: float | None = None
    event_end: float | None = None
    # 'clip' 表示这个文件本身就是一个事件片段，整段都算该类
    kind: str = "recording"


@dataclass
class Sample:
    signal: np.ndarray
    label: str
    subject: str
    source: str
    start_sec: float


def scan(root: str | Path) -> list[Record]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    # 手机端导出的 zip 解压后带 manifest.csv，里面是人工标注过的精确区间，
    # 优先用它——比靠能量峰值猜可靠得多。
    for manifest in (root / "manifest.csv", *root.glob("*/manifest.csv")):
        if manifest.exists():
            return scan_manifest(manifest)

    records: list[Record] = []
    missing_subject = 0

    for path in sorted(root.rglob("*.wav")):
        m = _NAME_RE.match(path.name)
        if m:
            label = m.group("label").lower()
            subject = (m.group("subject") or "").lower()
        else:
            # 兜底：类别取父目录名
            label = path.parent.name.lower()
            subject = ""

        if label not in ALL_LABELS:
            continue
        if not subject:
            # 尝试父目录名当被试
            parent = path.parent.name.lower()
            if _NAME_RE.match(f"{parent}_x_20000101_000000.wav"):
                subject = parent
        if not subject:
            subject = "p00"
            missing_subject += 1
        records.append(Record(path=path, subject=subject, label=label))

    if not records:
        raise ValueError(f"{root} 下没有找到可用的 wav 文件")
    if missing_subject:
        warnings.warn(
            f"{missing_subject} 个文件没有被试编号，已归到 p00。"
            "LOSO 会因此失真，请把文件名改成 p<编号>_类别_时间戳.wav",
            stacklevel=2,
        )
    elif "p00" in {r.subject for r in records}:
        warnings.warn(
            "有录音的被试编号是 p00（App 的默认值），说明录的时候没选被试。"
            "这些样本会被当成同一个人，LOSO 结果不可信。",
            stacklevel=2,
        )
    return records


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def scan_manifest(manifest: Path) -> list[Record]:
    """读手机端导出的 manifest.csv。

    列：file,subject,label,source,start_sec,end_sec,kind
    ``file`` 相对 manifest 所在目录。``kind=clip`` 表示文件本身就是事件片段。
    """
    records: list[Record] = []
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rel = (row.get("file") or "").strip()
            if not rel:
                continue
            path = (manifest.parent / rel).resolve()
            if not path.exists():
                continue
            label = (row.get("label") or "").strip().lower()
            if label not in ALL_LABELS:
                continue
            subject = (row.get("subject") or "").strip().lower() or "p00"
            records.append(
                Record(
                    path=path,
                    subject=subject,
                    label=label,
                    event_start=_to_float(row.get("start_sec")),
                    event_end=_to_float(row.get("end_sec")),
                    kind=(row.get("kind") or "recording").strip().lower(),
                )
            )
    if not records:
        raise ValueError(f"{manifest} 里没有可用的记录")
    return records


def load_wav(path: str | Path, target_sr: int = DEFAULT_SR) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, int(sr) // g).astype(np.float32)
        sr = target_sr
    return sr, data


def event_interval(
    signal: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> tuple[float, float]:
    """用短时能量峰值定位事件，返回 ±0.5 s 的区间。

    采集脚本约定动作在中间、前后留静息，所以取能量峰是最稳的自动定位方式；
    真做了人工起止标注的话，这一步可以被标注文件替代。
    """
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(signal)
    if n < frame:
        return 0.0, n / sr

    energies = []
    for start in range(0, n - frame + 1, hop):
        chunk = signal[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk))))
    peak_idx = int(np.argmax(energies))
    peak_t = (peak_idx * hop + frame / 2) / sr
    return max(0.0, peak_t - 0.5), min(n / sr, peak_t + 0.5)


def slice_records(
    records: list[Record],
    sr: int = DEFAULT_SR,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    min_overlap: float = 0.5,
    rest_windows_per_file: int = 4,
) -> list[Sample]:
    """把每条录音切成 1 s 窗，按与事件的叠加重叠率打标签。

    - 非静息录音：窗与事件区间重叠 ≥ min_overlap 才算该类，否则丢弃（边界窗）
    - 静息录音：全部窗都是 rest，但每条录音最多取 rest_windows_per_file 个，
      否则静息样本会压倒三类，训练时权重失控
    """
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    samples: list[Sample] = []

    for rec in records:
        file_sr, signal = load_wav(rec.path, target_sr=sr)
        if file_sr != sr:
            raise ValueError(f"重采样失败：{rec.path}")

        if rec.kind == "clip":
            # 片段已经是裁好的单个事件，整段都要，不做重叠率筛选
            if len(signal) < win:
                padded = np.zeros(win, dtype=np.float32)
                padded[: len(signal)] = signal
                signal = padded
            for start in range(0, len(signal) - win + 1, hop):
                samples.append(
                    Sample(
                        signal=signal[start : start + win].copy(),
                        label=rec.label,
                        subject=rec.subject,
                        source=rec.path.name,
                        start_sec=start / sr,
                    )
                )
            continue

        if len(signal) < win:
            continue

        if rec.label == REST:
            starts = list(range(0, len(signal) - win + 1, hop))
            step = max(1, len(starts) // rest_windows_per_file)
            starts = starts[::step][:rest_windows_per_file]
            for start in starts:
                samples.append(
                    Sample(
                        signal=signal[start : start + win].copy(),
                        label=REST,
                        subject=rec.subject,
                        source=rec.path.name,
                        start_sec=start / sr,
                    )
                )
            continue

        if rec.event_start is not None and rec.event_end is not None:
            t0, t1 = rec.event_start, rec.event_end
        else:
            t0, t1 = event_interval(signal, sr)
        for start in range(0, len(signal) - win + 1, hop):
            w0, w1 = start / sr, (start + win) / sr
            overlap = max(0.0, min(w1, t1) - max(w0, t0))
            if overlap >= min_overlap * window_sec:
                samples.append(
                    Sample(
                        signal=signal[start : start + win].copy(),
                        label=rec.label,
                        subject=rec.subject,
                        source=rec.path.name,
                        start_sec=w0,
                    )
                )
    return samples


def dataset_summary(samples: list[Sample]) -> dict:
    per_label: dict[str, int] = {}
    per_subject: dict[str, int] = {}
    for s in samples:
        per_label[s.label] = per_label.get(s.label, 0) + 1
        per_subject[s.subject] = per_subject.get(s.subject, 0) + 1
    return {
        "total_windows": len(samples),
        "per_label": per_label,
        "per_subject": per_subject,
        "subjects": sorted(per_subject),
    }
