from __future__ import annotations

from typing import Any

from torch import Module

class LinearClsHead(Module):
    def __init__(
        self, num_classes: int, in_channels: int, loss: Any, topk: tuple[int, ...]
    ) -> None: ...
