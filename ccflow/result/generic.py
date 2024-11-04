from typing import Generic, TypeVar

from pydantic import model_validator

from ..base import ResultBase

__all__ = ("GenericResult",)

T = TypeVar("T")


class GenericResult(ResultBase, Generic[T]):
    """Holds anything."""

    value: T

    @model_validator(mode="wrap")
    def _validate_generic_result(cls, v, handler, info):
        if isinstance(v, GenericResult) and not isinstance(v, cls):
            v = {"value": v.value}
        elif not isinstance(v, GenericResult) and not (isinstance(v, dict) and "value" in v):
            v = {"value": v}
        if isinstance(v, dict) and "value" in v:
            from ..context import GenericContext

            if isinstance(v["value"], GenericContext):
                v["value"] = v["value"].value
        return handler(v)
