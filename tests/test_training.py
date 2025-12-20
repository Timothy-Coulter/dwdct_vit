"""Training and checkpoint reuse tests for EfficientNet benchmark."""

from pathlib import Path

from src.benchmarks import BenchmarkConfig, TrainingConfig, run_benchmark, train_model


def test_training_and_benchmark_with_checkpoint(tmp_path: Path) -> None:
    """Train on fake data, save checkpoint, and reuse it for benchmarking."""
    checkpoint_dir = tmp_path / "ckpts"
    log_dir = tmp_path / "logs"
    train_cfg = TrainingConfig(
        dataset="cifar10",
        use_fake_data=True,
        limit=64,
        batch_size=8,
        val_fraction=0.25,
        epochs=1,
        use_pretrained=False,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )
    checkpoint_path = train_model(train_cfg)
    assert checkpoint_path.exists()

    bench_cfg = BenchmarkConfig(
        dataset="cifar10",
        use_fake_data=True,
        limit=32,
        batch_size=8,
        eval_fraction=0.25,
        use_pretrained=False,
        checkpoint_path=checkpoint_path,
        device="cpu",
        finetune_steps=0,
    )
    result = run_benchmark(bench_cfg)
    assert result.checkpoint_path is not None
    assert result.num_eval_samples > 0
    assert 0.0 <= result.accuracy <= 1.0
