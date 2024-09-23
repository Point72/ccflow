from typing import Generic, Hashable, Sequence, Set, TypeVar

from pydantic import model_validator

from .base import ContextBase, ResultBase

__all__ = (
    "GenericContext",
    "GenericResult",
)

C = TypeVar("C", bound=Hashable)
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
            if isinstance(v["value"], GenericContext):
                v["value"] = v["value"].value
        return handler(v)


class GenericContext(ContextBase, Generic[C]):
    """Holds anything."""

    value: C

    @model_validator(mode="wrap")
    def _validate_generic_context(cls, v, handler, info):
        if isinstance(v, GenericContext) and not isinstance(v, cls):
            v = {"value": v.value}
        elif not isinstance(v, GenericContext) and not (isinstance(v, dict) and "value" in v):
            v = {"value": v}
        if isinstance(v, dict) and "value" in v:
            if isinstance(v["value"], GenericResult):
                v["value"] = v["value"].value
            if isinstance(v["value"], Sequence) and not isinstance(v["value"], Hashable):
                v["value"] = tuple(v["value"])
            if isinstance(v["value"], Set) and not isinstance(v["value"], Hashable):
                v["value"] = frozenset(v["value"])
        return handler(v)
