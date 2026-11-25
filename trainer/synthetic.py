"""合成喉麦数据，用于零数据时验证整条管线。

**这是合成信号，只用来验证流程能不能跑通，绝不能写进报告当实验结果。**
真数据一到，把 ``--data`` 指到真实目录即可，其余代码不用改。
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

from .dataset import DEFAULT_SR

LEAD_SEC = 1.0
TAIL_SEC = 1.5


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(order, [max(lo, 1.0) / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return lfilter(b, a, x)


def _lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (sr / 2.0), btype="low")
    return lfilter(b, a, x)


def _swallow(rng: np.random.Generator, sr: int) -> np.ndarray:
    """吞咽：200~500 ms 的低频蠕动，主频 150~350 Hz，包络平滑。"""
    dur = rng.uniform(0.25, 0.5)
    t = np.arange(int(dur * sr)) / sr
    f0 = rng.uniform(150, 350)
    env = np.sin(np.pi * t / dur) ** 2
    sig = np.sin(2 * np.pi * f0 * t) * env
    sig += 0.4 * np.sin(2 * np.pi * f0 * 2.2 * t) * env
    sig += 0.25 * rng.standard_normal(len(t)) * env
    return _lowpass(sig, sr, 900.0) * rng.uniform(0.12, 0.3)


def _cough(rng: np.random.Generator, sr: int) -> np.ndarray:
    """咳嗽：宽频冲击，10~20 ms 起音、200 ms 内衰减。"""
    dur = rng.uniform(0.12, 0.2)
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.exp(-t / rng.uniform(0.03, 0.06))
    burst = rng.standard_normal(n)
    burst = _bandpass(burst, sr, 400.0, 6000.0)
    return burst * env * rng.uniform(0.5, 0.9)


def _interfere(rng: np.random.Generator, sr: int) -> np.ndarray:
    """干扰：说话（谐波 + 振幅调制）、清嗓（短促宽带）。"""
    kind = rng.integers(0, 2)
    if kind == 0:  # 说话
        dur = rng.uniform(0.5, 0.9)
        t = np.arange(int(dur * sr)) / sr
        f0 = rng.uniform(100, 220)
        sig = np.zeros_like(t)
        for h in range(1, 7):
            sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
        am = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t)
        sig = sig * am + 0.2 * rng.standard_normal(len(t))
        sig = _lowpass(sig, sr, 3500.0)
        return sig * rng.uniform(0.1, 0.25)
    # 清嗓
    dur = rng.uniform(0.15, 0.3)
    n = int(dur * sr)
    t = np.arange(n) / sr
    clicks = rng.standard_normal(n) * (rng.random(n) < 0.02)
    clicks = _bandpass(clicks, sr, 800.0, 8000.0)
    return clicks * np.exp(-t / 0.15) * rng.uniform(0.3, 0.6)


def _rest(rng: np.random.Generator, sr: int, dur: float) -> np.ndarray:
    """静息：只有底噪，偶有极轻的自主活动。"""
    return rng.standard_normal(int(dur * sr)) * rng.uniform(0.003, 0.008)


def _subject_color(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """给每个被试不同的增益和频响，让 LOSO 别太好过。"""
    gain = rng.uniform(0.7, 1.4)
    cutoff = rng.uniform(2500.0, 9000.0)
    return _lowpass(x, sr, cutoff) * gain


def generate(
    out_dir: str | Path,
    subjects: int = 6,
    per_class: int = 10,
    rest_files: int = 6,
    sr: int = DEFAULT_SR,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    # 每 1 秒音频约 91 KB（16 bit / 48 kHz），先估一下别把磁盘写满
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    stamp = datetime(2026, 11, 25, 10, 0, 0)

    for s in range(1, subjects + 1):
        subject = f"p{s:02d}"
        (out / subject).mkdir(exist_ok=True)
        for label, maker in (
            ("swallow", _swallow),
            ("cough", _cough),
            ("interfere", _interfere),
        ):
            for i in range(per_class):
                lead = _rest(rng, sr, LEAD_SEC)
                body = maker(rng, sr)
                tail = _rest(rng, sr, TAIL_SEC)
                signal = np.concatenate([lead, body, tail])
                signal = _subject_color(signal, sr, rng)
                signal += _rest(rng, sr, len(signal) / sr) * 0.5
                # 只在快要削顶时才压一下，避免把安静样本放大成爆音
                peak = float(np.max(np.abs(signal)))
                if peak > 0.95:
                    signal = signal / peak * 0.95
                starts = LEAD_SEC
                name = f"{subject}_{label}_{(stamp + timedelta(minutes=len(rows))).strftime('%Y%m%d_%H%M%S')}.wav"
                wavfile.write(
                    out / subject / name,
                    sr,
                    np.clip(signal * 32767, -32768, 32767).astype(np.int16),
                )
                rows.append({"file": f"{subject}/{name}", "subject": subject,
                             "label": label, "event_start_sec": f"{starts:.3f}",
                             "synthetic": "1"})
        for i in range(rest_files):
            signal = _rest(rng, sr, LEAD_SEC + 0.5 + TAIL_SEC)
            signal = _subject_color(signal, sr, rng)
            name = f"{subject}_rest_{(stamp + timedelta(minutes=len(rows))).strftime('%Y%m%d_%H%M%S')}.wav"
            wavfile.write(out / subject / name, sr, (signal * 32767).astype(np.int16))
            rows.append({"file": f"{subject}/{name}", "subject": subject,
                         "label": "rest", "event_start_sec": "", "synthetic": "1"})

    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "subject", "label", "event_start_sec", "synthetic"])
        writer.writeheader()
        writer.writerows(rows)

    return out
