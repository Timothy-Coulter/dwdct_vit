#!/usr/bin/env python
"""Convenience entrypoint to train or benchmark EfficientNet-B0."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarks import BenchmarkConfig, TrainingConfig, run_benchmark, train_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EfficientNet-B0 training or benchmarking.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train EfficientNet-B0.")
    train.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "mnist", "stl10"], required=True
    )
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--limit", type=int, default=4096, help="Cap training set size.")
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--device", type=str, default=None)
    train.add_argument("--use-fake-data", action="store_true", default=False)
    train.add_argument("--no-pretrained", action="store_true", default=True)
    train.add_argument("--unfreeze-backbone", action="store_true", default=True)
    train.add_argument("--log-dir", type=Path, default=None)
    train.add_argument("--checkpoint-dir", type=Path, default=None)
    train.add_argument("--seed", type=int, default=42)

    bench = subparsers.add_parser("benchmark", help="Benchmark EfficientNet-B0.")
    bench.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "mnist", "stl10"], required=True
    )
    bench.add_argument("--batch-size", type=int, default=32)
    bench.add_argument("--num-workers", type=int, default=2)
    bench.add_argument("--limit", type=int, default=512, help="Cap eval set size.")
    bench.add_argument("--eval-fraction", type=float, default=0.2)
    bench.add_argument("--device", type=str, default=None)
    bench.add_argument("--use-fake-data", action="store_true")
    bench.add_argument("--no-pretrained", action="store_true")
    bench.add_argument("--checkpoint", type=Path, default=None, help="Path to a saved checkpoint.")
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--finetune-steps", type=int, default=320)

    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainingConfig(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            val_fraction=args.val_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            use_fake_data=bool(args.use_fake_data),
            download=not args.use_fake_data,
            use_pretrained=not args.no_pretrained,
            freeze_backbone=not args.unfreeze_backbone,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
        )
        checkpoint_path = train_model(cfg)
        print(f"Training complete. Best checkpoint: {checkpoint_path}")
    else:
        bench_cfg = BenchmarkConfig(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            eval_fraction=args.eval_fraction,
            device=args.device,
            use_fake_data=bool(args.use_fake_data),
            download=not args.use_fake_data,
            use_pretrained=not args.no_pretrained,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            finetune_steps=args.finetune_steps,
        )
        result = run_benchmark(bench_cfg)
        print(
            f"Benchmark dataset={result.dataset} acc={result.accuracy:.4f} "
            f"throughput={result.throughput:.2f}/s samples={result.num_eval_samples} "
            f"checkpoint={result.checkpoint_path}"
        )


if __name__ == "__main__":
    _main()
