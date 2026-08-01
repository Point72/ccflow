from typing import Generic, TypeVar

from ..base import ResultBase

__all__ = ("DictResult",)


K = TypeVar("K")
V = TypeVar("V")


class DictResult(ResultBase, Generic[K, V]):
    value: dict[K, V]
