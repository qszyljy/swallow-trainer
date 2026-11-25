"""命令行入口。

    python -m trainer.cli synthetic --out data/synthetic
    python -m trainer.cli run       --data data/synthetic --out runs
    python -m trainer.cli experiments --data data/synthetic --out runs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import dataset, evaluate, export, features, models, synthetic
from .dataset import CLASSES, DEFAULT_HOP_SEC, DEFAULT_SR, DEFAULT_WINDOW_SEC, REST


def _print_summary(result: dict, samples: list) -> None:
    summary = dataset.dataset_summary(samples)
    print(f"窗口 {summary['total_windows']} 个，被试 {len(summary['subjects'])} 人，"
          f"每类 {summary['per_label']}")
    print(f"LOSO {result['n_folds']} 折 | 宏平均 F1 = "
          f"{result['macro_f1_mean']:.3f} ± {result['macro_f1_std']:.3f}")
    print("每类指标：")
    for label in CLASSES:
        m = result["per_class_mean"].get(label)
        if not m:
            continue
        print(f"  {label:9s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    mis = result["mis_trigger_rate_mean"]
    if mis is not None:
        print(f"静息段误触发率 = {mis:.3f}")


def _extract(samples: list, sr: int) -> np.ndarray:
    return np.stack([features.extract(s.signal, sr) for s in samples])


def cmd_synthetic(args: argparse.Namespace) -> int:
    out = synthetic.generate(
        args.out,
        subjects=args.subjects,
        per_class=args.per_class,
        rest_files=args.rest,
        seed=args.seed,
    )
    print(f"合成数据 -> {out}")
    print("注意：合成信号只用于验证流程，不能当实验数据写进报告。")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.out) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    records = dataset.scan(args.data)
    samples = dataset.slice_records(
        records,
        sr=args.sample_rate,
        window_sec=args.window,
        hop_sec=args.hop,
    )
    if not samples:
        print("没有切出任何窗口，检查录音长度是否大于一个窗")
        return 1

    if args.model == "cnn":
        return _run_cnn(args, samples, run_dir, started)

    X = _extract(samples, args.sample_rate)
    y = [s.label for s in samples]
    subjects = np.asarray([s.subject for s in samples])

    result = evaluate.loso(
        X, y, subjects, lambda seed: models.build_sklearn(args.model, seed), seed=args.seed
    )
    _print_summary(result, samples)

    evaluate.plot_confusion(
        result,
        run_dir / "confusion.png",
        title=f"{args.model} | window {args.window:.2f}s | LOSO",
    )

    final_model = models.build_sklearn(args.model, args.seed)
    final_model.fit(X, y)

    config = {
        "model_kind": args.model,
        "data": str(Path(args.data).resolve()),
        "sample_rate": args.sample_rate,
        "window_seconds": args.window,
        "hop_seconds": args.hop,
        "seed": args.seed,
        "feature_count": int(X.shape[1]),
        "feature_names": features.feature_names(),
        "fold_count": result["n_folds"],
        "synthetic": args.synthetic,
    }
    pkg = export.export_package(
        run_dir,
        final_model,
        result,
        dataset.dataset_summary(samples),
        config,
        labels=list(CLASSES),
        dart=args.dart,
    )

    elapsed = time.time() - started
    print(f"导出包 -> {pkg}")
    print(f"耗时 {elapsed:.1f} s")
    if args.synthetic:
        print("警告：本次用的是合成数据，指标没有意义，只证明管线跑通。")
    return 0


def _run_cnn(args: argparse.Namespace, samples: list, run_dir: Path, started: float) -> int:
    from . import runner_cnn  # 延迟导入：没装 TensorFlow 时其它模型照常可用

    try:
        result, adapter, input_len = runner_cnn.train_pipeline(
            samples,
            epochs=args.epochs,
            batch_size=args.batch,
            seed=args.seed,
        )
    except RuntimeError as exc:
        print(exc)
        return 2

    _print_summary(result, samples)
    evaluate.plot_confusion(
        result,
        run_dir / "confusion.png",
        title=f"1d-cnn | window {args.window:.2f}s | LOSO",
    )

    tflite_path = run_dir / "model.tflite"
    runner_cnn.export_tflite(adapter.model, tflite_path)
    size_kb = tflite_path.stat().st_size / 1024

    latency = None
    try:
        latency = runner_cnn.measure_latency_ms(tflite_path)
    except Exception as exc:  # noqa: BLE001 - 测不到时延不该让整次训练失败
        print(f"[warn] 电脑端时延测量失败：{exc}")

    config = {
        "model_kind": "cnn",
        "data": str(Path(args.data).resolve()),
        "sample_rate": args.sample_rate,
        "window_seconds": args.window,
        "hop_seconds": args.hop,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "input_samples": int(input_len),
        "feature_count": None,
        "feature_names": [],
        "fold_count": result["n_folds"],
        "synthetic": args.synthetic,
    }
    pkg = export.export_package(
        run_dir,
        adapter.model,
        result,
        dataset.dataset_summary(samples),
        config,
        labels=list(CLASSES),
        tflite_path=tflite_path,
        latency_ms=latency,
    )

    print(f"模型 {size_kb:.0f} KB" + (f" · 电脑端推理 {latency:.1f} ms" if latency else ""))
    print(f"导出包 -> {pkg}")
    print(f"耗时 {time.time() - started:.1f} s")
    if args.synthetic:
        print("警告：本次用的是合成数据，指标没有意义，只证明管线跑通。")
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    """模型对比 × 窗长敏感性，一次跑完，结果写成 markdown 和 csv。"""
    out_dir = Path(args.out) / datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [float(w) for w in args.windows.split(",")]
    kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    records = dataset.scan(args.data)

    rows: list[dict] = []
    for window in windows:
        samples = dataset.slice_records(
            records,
            sr=args.sample_rate,
            window_sec=window,
            hop_sec=window / 4.0,
        )
        if not samples:
            print(f"窗长 {window}s 切不出窗口，跳过")
            continue
        X = _extract(samples, args.sample_rate)
        y = [s.label for s in samples]
        subjects = np.asarray([s.subject for s in samples])

        for kind in kinds:
            result = evaluate.loso(
                X, y, subjects,
                lambda seed: models.build_sklearn(kind, seed),
                seed=args.seed,
            )
            row = {
                "window_seconds": window,
                "model": kind,
                "n_windows": len(samples),
                "macro_f1_mean": round(result["macro_f1_mean"], 4),
                "macro_f1_std": round(result["macro_f1_std"], 4),
                "mis_trigger_rate": (
                    round(result["mis_trigger_rate_mean"], 4)
                    if result["mis_trigger_rate_mean"] is not None
                    else ""
                ),
            }
            for label in CLASSES:
                m = result["per_class_mean"].get(label)
                row[f"recall_{label}"] = round(m["recall"], 4) if m else ""
            rows.append(row)
            print(
                f"窗 {window:.2f}s | {kind:4s} | F1 {result['macro_f1_mean']:.3f} "
                f"± {result['macro_f1_std']:.3f}"
            )

    if not rows:
        print("没有产生任何结果")
        return 1

    with (out_dir / "experiments.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = list(rows[0].keys())
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in header) + " |")
    (out_dir / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"实验矩阵 -> {out_dir / 'experiments.md'}")
    if args.synthetic:
        print("警告：合成数据，数字只说明管线跑通，不能进报告。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainer", description="喉麦吞咽事件分类 · 训练内核"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="生成合成数据（仅用于验证管线）")
    syn.add_argument("--out", required=True)
    syn.add_argument("--subjects", type=int, default=6)
    syn.add_argument("--per-class", type=int, default=10, dest="per_class")
    syn.add_argument("--rest", type=int, default=6)
    syn.add_argument("--seed", type=int, default=0)
    syn.set_defaults(func=cmd_synthetic)

    run = sub.add_parser("run", help="训练 + LOSO 评估 + 导出")
    run.add_argument("--data", required=True)
    run.add_argument("--out", default="runs")
    run.add_argument("--model", default="svm", choices=[*models.ALL_KINDS])
    run.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC)
    run.add_argument("--hop", type=float, default=DEFAULT_HOP_SEC)
    run.add_argument("--sample-rate", type=int, default=DEFAULT_SR, dest="sample_rate")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--epochs", type=int, default=30, help="仅 cnn 用")
    run.add_argument("--batch", type=int, default=16, help="仅 cnn 用")
    run.add_argument("--dart", action="store_true", help="额外用 m2cgen 导出 Dart 代码")
    run.add_argument("--synthetic", action="store_true", help="标记：本次数据是合成的")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("experiments", help="模型对比 × 窗长敏感性")
    exp.add_argument("--data", required=True)
    exp.add_argument("--out", default="runs")
    exp.add_argument("--models", default="svm,rf")
    exp.add_argument("--windows", default="0.5,1.0,1.5")
    exp.add_argument("--sample-rate", type=int, default=DEFAULT_SR, dest="sample_rate")
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--synthetic", action="store_true")
    exp.set_defaults(func=cmd_experiments)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
