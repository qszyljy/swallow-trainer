"""导出包：手机端唯一接口。

导出目录结构（缺一件，手机端就没法「直接接入」）：

    export/
      labels.txt         类别顺序，手机端按它解释输出
      preprocess.json    采样率、窗长、归一化、输入形状
      model.joblib       模型本体（cnn 路线是 model.tflite）
      eval.json          LOSO 指标、混淆矩阵、模型体积
      confusion.png      混淆矩阵图，直接贴报告
      config.json        这次运行的全部参数，供复现
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from joblib import dump

from .dataset import CLASSES, REST


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_package(
    out_dir: str | Path,
    model,
    result: dict,
    dataset_summary: dict,
    config: dict,
    *,
    labels: list[str] | None = None,
    dart: bool = False,
) -> Path:
    pkg = Path(out_dir) / "export"
    pkg.mkdir(parents=True, exist_ok=True)

    class_order = list(labels or CLASSES) + [REST]
    (pkg / "labels.txt").write_text("\n".join(class_order) + "\n", encoding="utf-8")

    model_path = pkg / "model.joblib"
    dump(model, model_path)

    preprocess = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate": config["sample_rate"],
        "window_seconds": config["window_seconds"],
        "hop_seconds": config["hop_seconds"],
        "normalize": {"mode": "per_window_zscore"},
        "labels": class_order,
        "model": {
            "format": "joblib",
            "file": model_path.name,
            "input": "features",
            "feature_count": config.get("feature_count"),
            "feature_names": config.get("feature_names", []),
        },
        "deploy_note": (
            "sklearn 模型用于实验对比；端侧部署建议走 1D-CNN + tflite，"
            "或 m2cgen 生成 Dart 代码"
        ),
    }
    _dump_json(pkg / "preprocess.json", preprocess)

    metrics = {
        "created_at": preprocess["created_at"],
        "model_kind": config.get("model_kind"),
        "window_seconds": config["window_seconds"],
        "hop_seconds": config["hop_seconds"],
        "n_folds": result["n_folds"],
        "macro_f1_mean": result["macro_f1_mean"],
        "macro_f1_std": result["macro_f1_std"],
        "per_class_mean": result["per_class_mean"],
        "mis_trigger_rate_mean": result["mis_trigger_rate_mean"],
        "confusion_matrix": result["confusion_matrix"],
        "labels": result["labels"],
        "folds": [
            {
                "held_out": f["held_out"],
                "macro_f1": f["macro_f1"],
                "mis_trigger_rate": f["mis_trigger_rate"],
            }
            for f in result["folds"]
        ],
        "dataset": dataset_summary,
        "model_size_bytes": model_path.stat().st_size,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    _dump_json(pkg / "eval.json", metrics)
    _dump_json(pkg / "config.json", config)

    if dart:
        try:
            import m2cgen

            (pkg / "model.dart").write_text(
                m2cgen.export_to_dart(model), encoding="utf-8"
            )
            preprocess["model"]["dart_file"] = "model.dart"
            _dump_json(pkg / "preprocess.json", preprocess)
        except Exception as exc:  # noqa: BLE001 - 导出失败不该中断整条流水线
            print(f"[warn] m2cgen 导出 Dart 失败：{exc}")

    return pkg
