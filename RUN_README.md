Run Quickstart
==============

This project ships a helper script `run.py` to train or benchmark EfficientNet-B0 across CIFAR10, CIFAR100, MNIST, and STL10 using the `TrainingConfig` and `BenchmarkConfig` in `src/benchmarks/efficientnet_mmx.py`.

Prereqs
-------
- Dependencies synced (`uv sync --extra dev` if needed).
- Datasets are downloaded automatically unless `--use-fake-data` is set.
- Checkpoints/logs default to `checkpoints/efficientnet/<dataset>` and `runs/efficientnet/<dataset>`.

Train
-----
Examples:
- Real CIFAR-10 (1 epoch, pretrained backbone frozen):
  ```
  python run.py train --dataset cifar10 --epochs 1 --batch-size 64 --limit 4096
  ```
- Fake data smoke:
  ```
  python run.py train --dataset cifar10 --use-fake-data --epochs 1 --limit 64 --no-pretrained --device cpu
  ```
Key flags:
- `--dataset {cifar10,cifar100,mnist,stl10}`
- `--limit <int>`: cap train set size
- `--val-fraction <float>`: train/val split
- `--learning-rate`, `--weight-decay`, `--epochs`
- `--no-pretrained`: skip hub weights
- `--unfreeze-backbone`: allow backbone finetuning
- `--log-dir`, `--checkpoint-dir`: override defaults

Benchmark
---------
Use a saved checkpoint (skips finetune when provided):
```
python run.py benchmark --dataset cifar10 \
  --checkpoint checkpoints/efficientnet/cifar10/best.pth \
  --batch-size 64 --limit 512 --finetune-steps 0
```
Without a checkpoint, it will optionally finetune the head for `--finetune-steps` after hub init.

Copy/paste command gallery
--------------------------
- Train CIFAR-10 (pretrained, frozen backbone):
  ```
  python run.py train --dataset cifar10 --epochs 1 --batch-size 64 --limit 4096
  ```
- Train CIFAR-100 with unfrozen backbone:
  ```
  python run.py train --dataset cifar100 --epochs 2 --batch-size 64 --limit 4096 --unfreeze-backbone
  ```
- Train MNIST on fake data (CPU, quick smoke):
  ```
  python run.py train --dataset mnist --use-fake-data --epochs 1 --limit 64 --no-pretrained --device cpu
  ```
- Train STL10 (full download, may take time):
  ```
  python run.py train --dataset stl10 --epochs 1 --batch-size 64 --limit 4096
  ```
- Benchmark using a saved checkpoint (CIFAR-10):
  ```
  python run.py benchmark --dataset cifar10 --checkpoint checkpoints/efficientnet/cifar10/best.pth --batch-size 64 --limit 512 --finetune-steps 0
  ```
- Benchmark without checkpoint (use hub weights + short finetune):
  ```
  python run.py benchmark --dataset cifar100 --batch-size 64 --limit 512 --finetune-steps 160
  ```
- Inspect TensorBoard logs (from repo root):
  ```
  tensorboard --logdir runs/efficientnet
  ```

General flags
-------------
- `--use-fake-data`: avoid downloads
- `--device <str>`: e.g., `cuda`, `cpu`
- `--num-workers`: dataloader workers
- `--seed`: deterministic splits/shuffling

Notes
-----
- STL10 is large (~2.6 GB); first run will download and extract.
- TensorBoard logs are written during training; open with `tensorboard --logdir runs/efficientnet`.
