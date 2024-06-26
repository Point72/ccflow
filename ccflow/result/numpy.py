"""This module defines re-usable result types for the "Callable Model" framework
defined in flow.callable.py.
"""

from typing import Generic, TypeVar

from ..base import ResultBase
from ..exttypes import NDArray

__all__ = ("NumpyResult",)


T = TypeVar("T")


class NumpyResult(ResultBase, Generic[T]):
    array: NDArray[T]

    def __eq__(self, other):
        return type(self) is type(other) and (self.array == other.array).all()
