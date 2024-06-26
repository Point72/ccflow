"""This module defines re-usable result types for the "Callable Model" framework
defined in flow.callable.py.
"""

import pydantic
from packaging import version
from typing import Dict, Generic, TypeVar

from ..base import ResultBase

__all__ = (
    "GenericResult",
    "DictResult",
)


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

if version.parse(pydantic.__version__) < version.parse("2"):
    from pydantic.generics import GenericModel

    class GenericResult(ResultBase, GenericModel, Generic[T]):
        """Holds anything."""

        value: T

        @classmethod
        def validate(cls, v, field=None):
            if not isinstance(v, GenericResult) and not (isinstance(v, dict) and "value" in v):
                v = {"value": v}
            return super(GenericResult, cls).validate(v)

else:
    from pydantic import model_validator

    class GenericResult(ResultBase, Generic[T]):
        """Holds anything."""

        value: T

        @model_validator(mode="wrap")
        def _validate_generic_result(cls, v, handler, info):
            if isinstance(v, GenericResult) and not isinstance(v, cls):
                v = {"value": v.value}
            elif not isinstance(v, GenericResult) and not (isinstance(v, dict) and "value" in v):
                v = {"value": v}
            return handler(v)


class DictResult(ResultBase, Generic[K, V]):
    """Holds a Dict."""

    value: Dict[K, V]
