"""Tests for the EfficientNet benchmark."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.benchmarks import BenchmarkConfig, run_benchmark


def test_run_benchmark_fake_data_cpu() -> None:
    """Benchmark runs on fake data without downloading pretrained weights."""
    config = BenchmarkConfig(
        dataset="cifar10",
        use_fake_data=True,
        limit=32,
        batch_size=8,
        finetune_steps=2,
        download=False,
        use_pretrained=False,
    )
    result = run_benchmark(config)

    assert result.dataset == "cifar10"
    assert result.num_eval_samples > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.throughput >= 0.0
