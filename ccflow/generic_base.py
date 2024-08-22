import pydantic
from packaging import version
from typing import Generic, Hashable, Sequence, Set, TypeVar

from .base import ContextBase, ResultBase

__all__ = (
    "GenericContext",
    "GenericResult",
)

C = TypeVar("C", bound=Hashable)
T = TypeVar("T")

if version.parse(pydantic.__version__) < version.parse("2"):
    from pydantic.generics import GenericModel

    class GenericResult(ResultBase, GenericModel, Generic[T]):
        """Holds anything."""

        value: T

        @classmethod
        def validate(cls, v, field=None):
            if not isinstance(v, GenericResult) and not (isinstance(v, dict) and "value" in v):
                v = {"value": v}
            if isinstance(v, dict) and "value" in v:
                if isinstance(v["value"], GenericContext):
                    v["value"] = v["value"].value
            return super(GenericResult, cls).validate(v)

    class GenericContext(ContextBase, GenericModel, Generic[C]):
        """Holds anything."""

        value: C

        @classmethod
        def validate(cls, v, field=None):
            if not isinstance(v, GenericContext) and not (isinstance(v, dict) and "value" in v):
                v = {"value": v}
            if isinstance(v, dict) and "value" in v:
                if isinstance(v["value"], GenericResult):
                    v["value"] = v["value"].value
                if isinstance(v["value"], Sequence) and not isinstance(v["value"], Hashable):
                    v["value"] = tuple(v["value"])
                if isinstance(v["value"], Set) and not isinstance(v["value"], Hashable):
                    v["value"] = frozenset(v["value"])
            return super(GenericContext, cls).validate(v)

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
