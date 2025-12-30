from __future__ import annotations

from typing import Any

from torch import Module, Tensor

class Linear(Module):
    in_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = ...) -> None: ...

class CrossEntropyLoss(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

functional: Any
mse_loss: Any
