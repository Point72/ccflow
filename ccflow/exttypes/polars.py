import math
import numpy as np
import orjson
import polars as pl
import scipy as sp
from io import StringIO
from packaging import version
from typing import Any

__all__ = ("PolarsExpression",)


class PolarsExpression(pl.Expr):
    """Provides a polars expressions from a string"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation for pydantic v2"""
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(function=cls._decode),
            python_schema=core_schema.no_info_plain_validator_function(function=cls.validate),
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
    def validate(cls, value: Any, field=None) -> Any:
        if isinstance(value, pl.Expr):
            return value

        if isinstance(value, str):
            try:
                expression = eval(
                    value,
                    {"col": pl.col, "c": pl.col, "np": np, "numpy": np, "pl": pl, "polars": pl, "scipy": sp, "sp": sp, "sc": sp, "math": math},
                    {},
                )
            except Exception as ex:
                raise ValueError(f"Error encountered constructing expression - {str(ex)}")

            if not issubclass(type(expression), pl.Expr):
                raise ValueError(f"Supplied value '{value}' does not evaluate to a Polars expression")
            return expression

        raise ValueError(f"Supplied value '{value}' cannot be converted to a Polars expression")
