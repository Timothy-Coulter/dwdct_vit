"""Run EfficientNet-B0 benchmarks on CIFAR10/100 and MNIST using MMPreTrain."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

DatasetName = Literal["cifar10", "cifar100", "mnist", "stl10"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "efficientnet-b0_3rdparty_8xb32_in1k"


class BenchmarkConfig:
    """Configuration for the EfficientNet benchmark."""

    __slots__ = (
        "dataset",
        "data_root",
        "batch_size",
        "num_workers",
        "limit",
        "finetune_steps",
        "eval_fraction",
        "device",
        "use_fake_data",
        "download",
        "use_pretrained",
        "checkpoint_path",
        "seed",
    )

    def __init__(
        self,
        dataset: DatasetName,
        data_root: Path | str = Path("datasets"),
        batch_size: int = 32,
        num_workers: int = 2,
        limit: int | None = 512,
        finetune_steps: int = 320,
        eval_fraction: float = 0.2,
        device: str | torch.device | None = None,
        use_fake_data: bool = False,
        download: bool = True,
        use_pretrained: bool = True,
        checkpoint_path: Path | str | None = None,
        seed: int = 42,
    ) -> None:
        """Create a benchmark configuration."""
        self.dataset: DatasetName = dataset
        self.data_root: Path = Path(data_root)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.limit: int | None = limit
        self.finetune_steps: int = finetune_steps
        self.eval_fraction: float = eval_fraction
        self.device: str | torch.device | None = device
        self.use_fake_data: bool = use_fake_data
        self.download: bool = download
        self.use_pretrained: bool = use_pretrained
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.seed: int = seed

    def resolved_device(self) -> torch.device:
        """Return the torch device to use."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BenchmarkResult:
    """Result of a benchmark run."""

    __slots__ = (
        "dataset",
        "num_eval_samples",
        "accuracy",
        "throughput",
        "duration_sec",
        "finetune_steps",
        "device",
        "checkpoint_path",
    )

    def __init__(
        self,
        dataset: DatasetName,
        num_eval_samples: int,
        accuracy: float,
        throughput: float,
        duration_sec: float,
        finetune_steps: int,
        device: str,
        checkpoint_path: Path | None,
    ) -> None:
        """Initialize an immutable benchmark result container."""
        self.dataset = dataset
        self.num_eval_samples = num_eval_samples
        self.accuracy = accuracy
        self.throughput = throughput
        self.duration_sec = duration_sec
        self.finetune_steps = finetune_steps
        self.device = device
        self.checkpoint_path = checkpoint_path


class TrainingConfig:
    """Configuration for training EfficientNet before benchmarking."""

    __slots__ = (
        "dataset",
        "data_root",
        "batch_size",
        "num_workers",
        "limit",
        "val_fraction",
        "epochs",
        "learning_rate",
        "weight_decay",
        "device",
        "use_fake_data",
        "download",
        "use_pretrained",
        "freeze_backbone",
        "log_dir",
        "checkpoint_dir",
        "seed",
    )

    def __init__(
        self,
        dataset: DatasetName,
        data_root: Path | str = Path("datasets"),
        batch_size: int = 64,
        num_workers: int = 2,
        limit: int | None = 4096,
        val_fraction: float = 0.2,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device | None = None,
        use_fake_data: bool = False,
        download: bool = True,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
        log_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize training configuration values."""
        self.dataset: DatasetName = dataset
        self.data_root: Path = Path(data_root)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.limit: int | None = limit
        self.val_fraction: float = val_fraction
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.device: str | torch.device | None = device
        self.use_fake_data: bool = use_fake_data
        self.download: bool = download
        self.use_pretrained: bool = use_pretrained
        self.freeze_backbone: bool = freeze_backbone
        self.log_dir: Path = (
            Path(log_dir) if log_dir is not None else Path("runs") / "efficientnet" / dataset
        )
        self.checkpoint_dir: Path = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else Path("checkpoints") / "efficientnet" / dataset
        )
        self.seed: int = seed

    def resolved_device(self) -> torch.device:
        """Return the torch device to use."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_transformers_for_mmpretrain() -> None:
    """Backfill APIs removed from transformers that MMPreTrain still imports."""
    try:
        from transformers import modeling_utils, pytorch_utils
        from transformers.generation import utils as generation_utils
    except Exception:
        return

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = (
            pytorch_utils.find_pruneable_heads_and_indices
        )
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "GenerationMixin"):
        modeling_utils.GenerationMixin = getattr(generation_utils, "GenerationMixin", None)


def _build_transform(dataset: DatasetName) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if dataset == "mnist":
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def _materialize_dataset(
    dataset_name: DatasetName,
    data_root: Path,
    use_fake_data: bool,
    limit: int | None,
    download: bool,
) -> tuple[Dataset[tuple[torch.Tensor, int]], int]:
    transform = _build_transform(dataset_name)
    if use_fake_data:
        num_classes = 100 if dataset_name == "cifar100" else 10
        size = limit or 256
        dataset: Dataset[tuple[torch.Tensor, int]] = datasets.FakeData(
            size=size,
            image_size=(3, 224, 224),
            num_classes=num_classes,
            transform=transform,
        )
        return dataset, num_classes

    root = str(data_root)
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=True, download=download, transform=transform)
        num_classes = 100
    elif dataset_name == "stl10":
        dataset = datasets.STL10(root=root, split="train", download=download, transform=transform)
        num_classes = 10
    else:
        dataset = datasets.MNIST(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
        num_classes = 10

    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = Subset(dataset, list(range(limit)))
    return dataset, num_classes


def _split_dataset(
    dataset: Dataset[tuple[torch.Tensor, int]],
    eval_fraction: float,
    seed: int | None,
) -> tuple[Dataset[tuple[torch.Tensor, int]], Dataset[tuple[torch.Tensor, int]]]:
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty, cannot run benchmark.")
    eval_size = max(1, int(total * eval_fraction)) if total > 1 else 1
    train_size = max(1, total - eval_size)
    eval_size = total - train_size

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    indices = permutation
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    train_split: Dataset[tuple[torch.Tensor, int]] = Subset(dataset, train_indices)
    eval_split: Dataset[tuple[torch.Tensor, int]] = Subset(dataset, eval_indices)
    return train_split, eval_split


def _prepare_model(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool,
    freeze_backbone: bool = True,
) -> nn.Module:
    _patch_transformers_for_mmpretrain()
    from mmpretrain import get_model
    from mmpretrain.models.heads import LinearClsHead

    model: nn.Module = get_model(MODEL_NAME, pretrained=use_pretrained, device=device)

    in_features: int | None = None
    if hasattr(model, "head") and hasattr(model.head, "fc"):
        head_fc = model.head.fc
        if isinstance(head_fc, nn.Linear):
            in_features = head_fc.in_features
    if in_features is None and hasattr(model, "head") and hasattr(model.head, "in_channels"):
        in_features = int(model.head.in_channels)
    if in_features is None:
        raise RuntimeError("Unable to infer classifier head input features for EfficientNet.")

    model.head = LinearClsHead(
        num_classes=num_classes,
        in_channels=in_features,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
    )
    if freeze_backbone and hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(device)
    return model


def _finetune_head(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_steps: int,
) -> int:
    if max_steps <= 0:
        return 0
    model.train()
    head_params = getattr(model, "head", model).parameters()
    optimizer = torch.optim.Adam(head_params, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    processed = 0
    for step, (images, targets) in enumerate(loader):
        if step >= max_steps:
            break
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images, mode="tensor")
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        processed += targets.numel()
    return processed


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    correct = 0
    total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images, mode="tensor")
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
    duration = time.perf_counter() - start
    throughput = total / duration if duration > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, throughput, duration


def _save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": metadata}, path)


def _load_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint state in {path}")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return state_dict, meta


def _build_metadata(config_dataset: DatasetName, num_classes: int, seed: int) -> dict[str, object]:
    return {
        "dataset": config_dataset,
        "num_classes": num_classes,
        "seed": seed,
        "model_name": MODEL_NAME,
    }


def train_model(config: TrainingConfig) -> Path:
    """Train EfficientNet and return the path to the best checkpoint."""
    torch.manual_seed(config.seed)
    device = config.resolved_device()
    dataset, num_classes = _materialize_dataset(
        dataset_name=config.dataset,
        data_root=config.data_root,
        use_fake_data=config.use_fake_data,
        limit=config.limit,
        download=config.download,
    )
    train_split, val_split = _split_dataset(
        dataset, eval_fraction=config.val_fraction, seed=config.seed
    )

    model = _prepare_model(
        num_classes=num_classes,
        device=device,
        use_pretrained=config.use_pretrained,
        freeze_backbone=config.freeze_backbone,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_split,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_split,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_acc = -1.0
    best_path = config.checkpoint_dir / "best.pth"
    last_path = config.checkpoint_dir / "last.pth"
    writer = SummaryWriter(log_dir=str(config.log_dir))
    global_step = 0
    try:
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            seen = 0
            progress = enumerate(train_loader)
            model.train()
            for batch_idx, (images, targets) in progress:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = model(images, mode="tensor")
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                batch_size = targets.numel()
                epoch_loss += float(loss.item()) * batch_size
                seen += batch_size
                if batch_idx % 10 == 0:
                    print(
                        f"[epoch {epoch + 1}/{config.epochs}] "
                        f"step {batch_idx + 1}/{len(train_loader)} "
                        f"loss={loss.item():.4f}"
                    )
                global_step += 1

            val_acc, val_throughput, val_duration = _evaluate(
                model=model, loader=val_loader, device=device
            )
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("val/throughput", val_throughput, epoch)
            writer.add_scalar("val/duration_sec", val_duration, epoch)
            if seen > 0:
                avg_loss = epoch_loss / seen
                print(
                    f"[epoch {epoch + 1}/{config.epochs}] "
                    f"avg_loss={avg_loss:.4f} val_acc={val_acc:.4f} "
                    f"val_throughput={val_throughput:.2f}/s"
                )

            metadata = _build_metadata(
                config_dataset=config.dataset,
                num_classes=num_classes,
                seed=config.seed,
            )
            _save_checkpoint(model, last_path, metadata)
            if val_acc > best_acc:
                best_acc = val_acc
                _save_checkpoint(model, best_path, metadata)
    finally:
        writer.flush()
        writer.close()

    return best_path if best_acc >= 0 else last_path


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the EfficientNet benchmark and return metrics."""
    dataset, num_classes = _materialize_dataset(
        dataset_name=config.dataset,
        data_root=config.data_root,
        use_fake_data=config.use_fake_data,
        limit=config.limit,
        download=config.download,
    )
    train_split, eval_split = _split_dataset(
        dataset, eval_fraction=config.eval_fraction, seed=config.seed
    )
    device = config.resolved_device()
    checkpoint_meta: dict[str, object] | None = None
    if config.checkpoint_path is not None:
        state_dict, checkpoint_meta = _load_checkpoint(config.checkpoint_path, device=device)
        num_classes_meta = checkpoint_meta.get("num_classes", num_classes)
        if isinstance(num_classes_meta, int):
            ckpt_num_classes = num_classes_meta
        elif isinstance(num_classes_meta, (str, float)):
            ckpt_num_classes = int(num_classes_meta)
        else:
            ckpt_num_classes = num_classes
        ckpt_dataset = checkpoint_meta.get("dataset")
        if ckpt_dataset is not None and ckpt_dataset != config.dataset:
            raise ValueError(
                f"Checkpoint dataset '{ckpt_dataset}' does not match requested '{config.dataset}'."
            )
        model = _prepare_model(
            num_classes=ckpt_num_classes,
            device=device,
            use_pretrained=False,
            freeze_backbone=True,
        )
        model.load_state_dict(state_dict, strict=False)
        finetune_steps = 0
    else:
        model = _prepare_model(
            num_classes=num_classes,
            device=device,
            use_pretrained=config.use_pretrained,
            freeze_backbone=True,
        )
        finetune_steps = config.finetune_steps

    train_loader = DataLoader(
        train_split,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_split,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    _finetune_head(model, train_loader, device=device, max_steps=finetune_steps)
    accuracy, throughput, duration = _evaluate(model, eval_loader, device=device)
    return BenchmarkResult(
        dataset=config.dataset,
        num_eval_samples=len(eval_split),
        accuracy=accuracy,
        throughput=throughput,
        duration_sec=duration,
        finetune_steps=finetune_steps,
        device=str(device),
        checkpoint_path=config.checkpoint_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EfficientNet-B0 training and benchmarking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train EfficientNet-B0 head/backbone")
    train_parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "mnist", "stl10"], required=True
    )
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--limit", type=int, default=4096)
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--use-fake-data", action="store_true")
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--unfreeze-backbone", action="store_true")
    train_parser.add_argument("--log-dir", type=Path, default=None)
    train_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=42)

    bench_parser = subparsers.add_parser("benchmark", help="Benchmark EfficientNet-B0")
    bench_parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "mnist", "stl10"], required=True
    )
    bench_parser.add_argument("--batch-size", type=int, default=32)
    bench_parser.add_argument("--num-workers", type=int, default=2)
    bench_parser.add_argument("--limit", type=int, default=512)
    bench_parser.add_argument("--eval-fraction", type=float, default=0.2)
    bench_parser.add_argument("--device", type=str, default=None)
    bench_parser.add_argument("--use-fake-data", action="store_true")
    bench_parser.add_argument("--no-pretrained", action="store_true")
    bench_parser.add_argument("--checkpoint", type=Path, default=None)
    bench_parser.add_argument("--seed", type=int, default=42)
    bench_parser.add_argument("--finetune-steps", type=int, default=320)

    return parser


def _main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_cfg = TrainingConfig(
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
        checkpoint_path = train_model(train_cfg)
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
