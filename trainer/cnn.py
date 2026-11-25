"""1D-CNN：吃原始波形，手机端部署路线。

这里只做训练与导出，预处理约定由 preprocess.json 固定：
48 kHz、单声道、1 s 窗（48000 点）、逐窗 z-score 归一化。

注意：TensorFlow 在 Windows 上对 Python 版本挑剔，建议 Python 3.11 + TF 2.16。
本模块在没装 TF 的环境里会给出明确提示，不会影响其它模型的运行。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_LEN = 48000


def _tf():
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 取决于本机环境
        raise RuntimeError(
            "1D-CNN 需要 TensorFlow。建议：Python 3.11 环境里 pip install tensorflow==2.16.*"
        ) from exc
    return tf


def build_cnn(n_classes: int, input_len: int = INPUT_LEN, seed: int = 0):
    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_len, 1)),
            tf.keras.layers.Conv1D(16, 51, strides=8, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(4),
            tf.keras.layers.Conv1D(32, 25, strides=2, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(4),
            tf.keras.layers.Conv1D(64, 11, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


def preprocess(signals: np.ndarray) -> np.ndarray:
    """逐窗 z-score，手机端必须实现同一套归一化。"""
    x = np.asarray(signals, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std)[..., None]


def train(
    signals: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    epochs: int = 30,
    batch_size: int = 16,
    seed: int = 0,
):
    tf = _tf()
    model = build_cnn(n_classes, signals.shape[1], seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        preprocess(signals),
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )
    return model, history


def export_tflite(model, path: str | Path) -> Path:
    tf = _tf()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    out = Path(path)
    out.write_bytes(converter.convert())
    return out
