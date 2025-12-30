from __future__ import annotations

from typing import Any, Protocol

from transformers.generation import utils as generation_utils  # noqa: F401

class _ModelingUtils(Protocol):
    apply_chunking_to_forward: Any
    find_pruneable_heads_and_indices: Any
    prune_linear_layer: Any
    GenerationMixin: Any

class _PytorchUtils(Protocol):
    apply_chunking_to_forward: Any
    find_pruneable_heads_and_indices: Any
    prune_linear_layer: Any

modeling_utils: _ModelingUtils
pytorch_utils: _PytorchUtils
