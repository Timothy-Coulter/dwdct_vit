from __future__ import annotations

from collections.abc import Callable
from typing import Any

class _ParametrizeDecorator:
    def __call__(self, argnames: str, argvalues: Any, **kwargs: Any) -> Callable[..., Any]: ...

class _MarkNamespace:
    parametrize: _ParametrizeDecorator

mark: _MarkNamespace
