"""模型工厂。

- ``svm`` / ``rf`` / ``mlp``：sklearn 基线，用于报告里的分类器对比（项 5/6）
- ``cnn``：1D-CNN 吃原始波形，手机端部署路线，需要 TensorFlow（见 cnn.py）

注意：sklearn 模型是「用来论证特征有效」的，不是最终上手机的模型——
没有现成的 tflite 转换路径。真要在手机上跑 sklearn 模型，走 m2cgen 导出 Dart。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SKLEARN_KINDS = ("svm", "rf", "mlp")
ALL_KINDS = (*SKLEARN_KINDS, "cnn")


def build_sklearn(kind: str, seed: int = 0):
    if kind == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=4.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=800,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"未知模型：{kind}，可选 {SKLEARN_KINDS}")
