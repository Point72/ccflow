from typing import Dict, Generic, TypeVar

from ..base import ResultBase

__all__ = ("DictResult",)


K = TypeVar("K")
V = TypeVar("V")


class DictResult(ResultBase, Generic[K, V]):
    """Holds a Dict."""

    value: Dict[K, V]
