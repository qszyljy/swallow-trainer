"""1D-CNN 的训练、LOSO 评估与 tflite 导出。

只做两件事：把原始波形喂给 CNN，把评估结果对齐到和 sklearn 那条线一样的
`evaluate.loso` 输出结构——这样混淆矩阵、误触发率、宏 F1 的算法只有一份。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import cnn, evaluate


class KerasAdapter:
    """把 keras 模型包成 evaluate.loso 认识的 fit / predict 接口。

    predict 返回字符串标签而不是下标，这样即使某一折缺某个类别，
    混淆矩阵的行列顺序也还是由 evaluate 统一决定。
    """

    def __init__(self, epochs: int, batch_size: int, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = None
        self._labels: list[str] = []

    def fit(self, X, y):
        y = np.asarray(y)
        self._labels = list(dict.fromkeys(y.tolist()))
        index = {label: i for i, label in enumerate(self._labels)}
        target = np.array([index[value] for value in y], dtype=np.int32)
        self.model, _ = cnn.train(
            X,
            target,
            n_classes=len(self._labels),
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return self

    def predict(self, X):
        probs = self.model.predict(cnn.preprocess(X), verbose=0)
        return np.array([self._labels[int(i)] for i in probs.argmax(axis=1)])


def train_pipeline(
    samples,
    epochs: int = 30,
    batch_size: int = 16,
    seed: int = 0,
):
    """LOSO 评估 + 用全量数据训一个最终模型。"""
    X = np.stack([s.signal for s in samples]).astype(np.float32)
    y = np.array([s.label for s in samples])
    subjects = np.array([s.subject for s in samples])

    result = evaluate.loso(
        X,
        y,
        subjects,
        lambda s: KerasAdapter(epochs, batch_size, s),
        seed=seed,
    )

    final = KerasAdapter(epochs, batch_size, seed).fit(X, y)
    return result, final, X.shape[1]


def export_tflite(model, path: str | Path) -> Path:
    return cnn.export_tflite(model, path)


def measure_latency_ms(tflite_path: str | Path, runs: int = 20) -> float:
    """在电脑上量一次推理时延，仅作参考——报告里的端侧时延必须真机打点。"""
    import tensorflow as tf  # noqa: PLC0415

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    sample = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
    for _ in range(3):  # 预热
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(timings))
