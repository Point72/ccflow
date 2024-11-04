import math
from io import StringIO
from typing import Any

import numpy as np
import orjson
import polars as pl
from packaging import version
from pydantic import TypeAdapter
from typing_extensions import Self

__all__ = ("PolarsExpression",)


class PolarsExpression(pl.Expr):
    """Provides a polars expressions from a string"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(function=cls._decode),
            python_schema=core_schema.no_info_plain_validator_function(function=cls._validate),
            serialization=core_schema.plain_serializer_function_ser_schema(cls._encode, return_schema=core_schema.dict_schema()),
        )

    @staticmethod
    def _decode(obj):
        # We embed polars expressions as a dict, so we need to convert to a full json string first
        json_str = orjson.dumps(obj).decode("utf-8", "ignore")
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            return pl.Expr.deserialize(StringIO(json_str))
        else:
            # polars deserializes from a binary format by default.
            return pl.Expr.deserialize(StringIO(json_str), format="json")

    @staticmethod
    def _encode(obj, info=None):
        # obj.meta.serialize produces a string containing a dict, but we just want to return the dict.
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            return orjson.loads(obj.meta.serialize())
        else:
            # polars serializes into a binary format by default.
            return orjson.loads(obj.meta.serialize(format="json"))

    @classmethod
    def _validate(cls, value: Any) -> Self:
        if isinstance(value, pl.Expr):
            return value

        if isinstance(value, str):
            try:
                local_vars = {"col": pl.col, "c": pl.col, "np": np, "numpy": np, "pl": pl, "polars": pl, "math": math}
                try:
                    import scipy as sp  # Optional dependency.

                    local_vars.update({"scipy": sp, "sp": sp, "sc": sp})
                except ImportError:
                    pass
                expression = eval(value, local_vars, {})
            except Exception as ex:
                raise ValueError(f"Error encountered constructing expression - {str(ex)}")

            if not issubclass(type(expression), pl.Expr):
                raise ValueError(f"Supplied value '{value}' does not evaluate to a Polars expression")
            return expression

        raise ValueError(f"Supplied value '{value}' cannot be converted to a Polars expression")

    @classmethod
    def validate(cls, value: Any) -> Self:
        """Try to convert/validate an arbitrary value to a PolarsExpression."""
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(PolarsExpression)
