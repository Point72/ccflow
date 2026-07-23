from typing import Generic, TypeVar

from ..base import ResultBase

__all__ = ("ListResult",)


V = TypeVar("V")


class ListResult(ResultBase, Generic[V]):
    value: list[V]
