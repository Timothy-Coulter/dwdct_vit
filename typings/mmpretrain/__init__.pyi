from __future__ import annotations

from typing import Any

from torch import Module

# ruff: noqa: N801

def get_model(name: str, pretrained: bool = ..., device: Any = ..., **kwargs: Any) -> Module: ...

class _HeadsNamespace:
    class LinearClsHead(Module):
        def __init__(
            self, num_classes: int, in_channels: int, loss: Any, topk: tuple[int, ...]
        ) -> None: ...

class models:
    heads = _HeadsNamespace()
