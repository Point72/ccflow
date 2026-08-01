import math
from io import StringIO
from typing import Annotated, Any

import numpy as np
import orjson
import polars as pl
from packaging import version
from polars import selectors as _cs
from typing_extensions import Self

__all__ = ("PolarsExpression", "PolarsSelector")


class _PolarsExprPydanticAnnotation:
    """Provides a polars expressions from a string"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(function=cls._decode),
            python_schema=core_schema.no_info_plain_validator_function(function=cls._validate),
            serialization=core_schema.plain_serializer_function_ser_schema(cls._encode, return_schema=core_schema.dict_schema()),
        )

    @classmethod
    def _decode(cls, obj):
        # We embed polars expressions as a dict, so we need to convert to a full json string first
        json_str = orjson.dumps(obj).decode("utf-8", "ignore")
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            return pl.Expr.deserialize(StringIO(json_str))
        else:
            # polars deserializes from a binary format by default.
            return pl.Expr.deserialize(StringIO(json_str), format="json")

    @classmethod
    def _encode(cls, obj, info=None):
        # obj.meta.serialize produces a string containing a dict, but we just want to return the dict.
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            return orjson.loads(obj.meta.serialize())
        else:
            # polars serializes into a binary format by default.
            return orjson.loads(obj.meta.serialize(format="json"))

    @classmethod
    def _eval_locals(cls) -> dict:
        local_vars = {"col": pl.col, "c": pl.col, "np": np, "numpy": np, "pl": pl, "polars": pl, "math": math}
        try:
            import scipy as sp  # Optional dependency.

            local_vars.update({"scipy": sp, "sp": sp, "sc": sp})
        except ImportError:
            pass
        return local_vars

    @classmethod
    def _coerce_from_expr(cls, expr: pl.Expr) -> pl.Expr:
        return expr

    @classmethod
    def _validate(cls, value: Any) -> Self:
        if isinstance(value, pl.Expr):
            return cls._coerce_from_expr(value)

        if isinstance(value, str):
            try:
                expression = eval(value, cls._eval_locals(), {})
            except Exception as ex:
                raise ValueError(f"Error encountered constructing expression - {ex!s}") from ex

            if not isinstance(expression, pl.Expr):
                raise ValueError(f"Supplied value '{value}' does not evaluate to a Polars expression")  # noqa: TRY004
            return cls._coerce_from_expr(expression)

        raise ValueError(f"Supplied value '{value}' cannot be converted to a Polars expression")


class _PolarsSelectorPydanticAnnotation(_PolarsExprPydanticAnnotation):
    """Provides a polars column selector from a string or expression.

    Polars column selectors (``polars.selectors``) subclass ``pl.Expr``, so they serialize via ``Expr.meta.serialize`` like any
    other expression. However, ``Expr.deserialize`` returns a plain ``Expr`` rather than a ``Selector``, which breaks the selector
    set-operation overloads (``|``, ``&``, ``-``, ``~``). This annotation promotes the deserialized expression back to a
    ``Selector`` via ``meta.as_selector()`` and restricts inputs to actual column selectors.
    """

    @classmethod
    def _decode(cls, obj):
        return super()._decode(obj).meta.as_selector()

    @classmethod
    def _eval_locals(cls) -> dict:
        return {**super()._eval_locals(), "cs": _cs, "selectors": _cs}

    @classmethod
    def _coerce_from_expr(cls, expr: pl.Expr) -> pl.Expr:
        if _cs.is_selector(expr):
            return expr
        if expr.meta.is_column_selection():
            return expr.meta.as_selector()
        raise ValueError(f"Supplied polars expression {expr!r} is not a column selector")


PolarsExpression = Annotated[pl.Expr, _PolarsExprPydanticAnnotation]

PolarsSelector = Annotated[pl.Expr, _PolarsSelectorPydanticAnnotation]
