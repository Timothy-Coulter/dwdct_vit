from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Generic, TypeVar

# ruff: noqa: UP046,UP047

T_co = TypeVar("T_co", covariant=True)

class Dataset(Generic[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...

class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None: ...

class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int = ...,
        shuffle: bool = ...,
        num_workers: int = ...,
        pin_memory: bool = ...,
        **kwargs: object,
    ) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...

class _RandomSampler:
    def __iter__(self) -> Iterator[int]: ...

def random_split(
    dataset: Dataset[T_co], lengths: Iterable[int], **kwargs: object
) -> list[Dataset[T_co]]: ...
