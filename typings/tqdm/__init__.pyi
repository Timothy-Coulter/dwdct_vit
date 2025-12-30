from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar, overload

# ruff: noqa: UP047

T = TypeVar("T")

@overload
def tqdm(iterable: Iterable[T], desc: str | None = ..., total: int | None = ...) -> Iterator[T]: ...
@overload
def tqdm(
    iterable: None = ..., desc: str | None = ..., total: int | None = ...
) -> Iterator[int]: ...
