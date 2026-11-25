"""留一被试交叉验证（LOSO）与指标计算。

任务书的硬要求，每一条都在这里兑现：
- 按被试划分，禁止随机打散
- ≥ 5 折，报告均值 ± 标准差
- 3×3 混淆矩阵、每类 precision/recall/F1、宏平均 F1
- 静息不计入三类指标，但单独报误触发率
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from .dataset import CLASSES, REST


def _metrics_from_cm(cm: np.ndarray, labels: list[str]) -> dict:
    """从混淆矩阵手算指标，这样「真值为事件、被误判成静息」也能算成漏检。"""
    index = {label: i for i, label in enumerate(labels)}
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    per_class: dict[str, dict[str, float]] = {}

    for label in CLASSES:
        if label not in index:
            continue
        i = index[label]
        tp = float(cm[i, i])
        fn = float(rows[i] - tp)
        fp = float(cols[i] - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(rows[i]),
        }

    macro_f1 = (
        float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    )

    mis_trigger = None
    if REST in index:
        r = index[REST]
        if rows[r] > 0:
            mis_trigger = float((rows[r] - cm[r, r]) / rows[r])

    return {"macro_f1": macro_f1, "per_class": per_class, "mis_trigger_rate": mis_trigger}


def loso(
    X: np.ndarray,
    y: list[str],
    subjects: np.ndarray,
    build_model,
    seed: int = 0,
) -> dict:
    labels = [*CLASSES, REST] if REST in set(y) else list(CLASSES)
    index = {label: i for i, label in enumerate(labels)}
    cm_total = np.zeros((len(labels), len(labels)), dtype=int)

    y_arr = np.asarray(y)
    folds: list[dict] = []

    for subject in sorted(set(subjects.tolist())):
        test_mask = subjects == subject
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        model = build_model(seed)
        model.fit(X[train_mask], y_arr[train_mask])
        pred = model.predict(X[test_mask])

        cm = confusion_matrix(
            y_arr[test_mask], pred, labels=labels
        )
        cm_total += cm
        fold_metrics = _metrics_from_cm(cm, labels)
        fold_metrics["held_out"] = subject
        fold_metrics["n_test"] = int(test_mask.sum())
        folds.append(fold_metrics)

    overall = _metrics_from_cm(cm_total, labels)
    f1s = [f["macro_f1"] for f in folds]
    mis = [f["mis_trigger_rate"] for f in folds if f["mis_trigger_rate"] is not None]

    per_class_mean = {}
    for label in CLASSES:
        vals = [f["per_class"][label] for f in folds if label in f["per_class"]]
        if not vals:
            continue
        per_class_mean[label] = {
            key: float(np.mean([v[key] for v in vals]))
            for key in ("precision", "recall", "f1")
        }

    return {
        "labels": labels,
        "folds": folds,
        "n_folds": len(folds),
        "macro_f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "macro_f1_std": float(np.std(f1s)) if f1s else 0.0,
        "per_class_mean": per_class_mean,
        "mis_trigger_rate_mean": float(np.mean(mis)) if mis else None,
        "confusion_matrix": cm_total.tolist(),
        "overall": overall,
    }


def plot_confusion(result: dict, path: str | Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = result["labels"]
    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    if title:
        ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > cm.max() / 2 else "#333333",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
