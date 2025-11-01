from typing import Generic, List, TypeVar

from ..base import ResultBase

__all__ = ("ListResult",)


V = TypeVar("V")


class ListResult(ResultBase, Generic[V]):
    value: List[V]
