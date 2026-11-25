"""三域特征：时域 / 频域 / 时频（对应任务书评分项 5）。

特征只用于 SVM / 随机森林这条对比线，以及报告里的消融分析；
真正上手机的 1D-CNN 直接吃原始波形，不走这里。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew

N_FFT = 1024
N_MELS = 16
MEL_FMIN = 50.0

_BANDS = (
    ("band_0_250", 0.0, 250.0),
    ("band_250_1k", 250.0, 1000.0),
    ("band_1k_4k", 1000.0, 4000.0),
    ("band_4k_8k", 4000.0, 8000.0),
    ("band_8k_24k", 8000.0, 24000.0),
)

_TF_BANDS = ((0.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0), (8000.0, 24000.0))

_fb_cache: dict[tuple[int, int], np.ndarray] = {}


def _hz2mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    key = (sr, n_fft)
    if key in _fb_cache:
        return _fb_cache[key]

    fmax = sr / 2.0
    mels = np.linspace(_hz2mel(MEL_FMIN), _hz2mel(fmax), n_mels + 2)
    hz = _mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    _fb_cache[key] = fb
    return fb


def feature_names() -> list[str]:
    names = [
        "time_rms",
        "time_peak",
        "time_std",
        "time_zcr",
        "time_kurtosis",
        "time_skew",
        "time_crest",
        "freq_dominant_hz",
        "freq_centroid_hz",
        "freq_spread_hz",
        "freq_rolloff85_hz",
        "freq_flatness",
        "freq_entropy",
    ]
    names += [name for name, _, _ in _BANDS]
    names += [f"tf_mel_mean_{i}" for i in range(N_MELS)]
    names += [f"tf_mel_std_{i}" for i in range(N_MELS)]
    names += [f"tf_entropy_band_{i}" for i in range(len(_TF_BANDS))]
    return names


def extract(signal: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    eps = 1e-12

    # ---------- 时域 ----------
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))) / 2.0)
    feats = [
        rms,
        peak,
        float(np.std(x)),
        zcr,
        float(kurtosis(x, fisher=True)),
        float(skew(x)),
        peak / (rms + eps),
    ]

    # ---------- 频域 ----------
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    power = spectrum**2 + eps
    total = power.sum()

    dominant = float(freqs[int(np.argmax(spectrum))])
    centroid = float((freqs * power).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
    cumulative = np.cumsum(power) / total
    rolloff = float(freqs[int(np.searchsorted(cumulative, 0.85))])

    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + eps))
    prob = power / total
    entropy = float(-(prob * np.log2(prob + eps)).sum() / np.log2(len(prob)))

    feats += [dominant, centroid, spread, rolloff, flatness, entropy]
    for _, lo, hi in _BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(float(power[mask].sum() / total))

    # ---------- 时频 -----------------
    _, _, zxx = stft(x, fs=sr, window="hann", nperseg=512, noverlap=256, boundary=None)
    spec = np.abs(zxx) ** 2
    stft_freqs = np.fft.rfftfreq(512, 1.0 / sr)
    # 注意：滤波器组的 n_fft 必须和上面 STFT 的 nperseg 一致，否则维度对不上
    mel = _mel_filterbank(sr, n_fft=512) @ spec  # (n_mels, frames)
    log_mel = np.log1p(mel * 1e3)
    feats += list(log_mel.mean(axis=1))
    feats += list(log_mel.std(axis=1))

    for lo, hi in _TF_BANDS:
        mask = (stft_freqs >= lo) & (stft_freqs < hi)
        band_energy = spec[mask].sum(axis=0)
        p = band_energy / (band_energy.sum() + eps)
        feats.append(float(-(p * np.log2(p + eps)).sum() / np.log2(len(p) + eps)))

    return np.asarray(feats, dtype=np.float32)
