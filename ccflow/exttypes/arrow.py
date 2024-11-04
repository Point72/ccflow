from typing import Generic, Type, TypeVar, Union

import pandas as pd
import pyarrow as pa
from pydantic import TypeAdapter
from pydantic_core import core_schema
from typing_extensions import Literal, Self, get_args


class ArrowSchema(type):
    """A metaclass for creating Arrow schema-specific types that can be used with Generic classes"""

    @classmethod
    def make(
        cls,
        schema: pa.Schema,
        strict: Union[bool, Literal["filter"]] = "filter",
        clsname: str = "_ArrowSchema",
    ) -> Self:
        """Take the schema and a strict flag to define the type.
        The strict flag follows the same conventions as used by pandera.
        Schemas are order-dependent.
        """
        if strict not in [True, False, "filter"]:
            raise ValueError("strict must be True, False or 'filter'")
        return cls(clsname, (cls,), {"schema": schema, "strict": strict})

    def __new__(mcs, clsname, bases, dct):
        newclass = super(ArrowSchema, mcs).__new__(mcs, clsname, bases, dct)

        err_msg = "Cannot instantiate an instance of ArrowSchema directly."

        def __init__(self, *args, **kwargs):
            raise TypeError(err_msg)

        newclass.__init__ = __init__

        return newclass


S = TypeVar("S", bound=ArrowSchema)


class ArrowTable(pa.Table, Generic[S]):
    """Pydantic compatible wrapper around Arrow tables, with optional schema validation."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def _validate(v):
            subtypes = get_args(source_type)
            if subtypes:
                return cls._validate(v, subtypes[0].schema, subtypes[0].strict)
            else:
                return cls._validate(v, None, None)

        return core_schema.no_info_plain_validator_function(_validate)

    @classmethod
    def _validate(cls, v, schema, strict):
        """Helper function for validation with common functionality between v1 and v2"""
        if isinstance(v, list):
            v = pa.Table.from_batches(v)
        elif hasattr(v, "to_arrow"):  # For polars, but without importing it
            v = v.to_arrow()
        elif isinstance(v, pd.DataFrame):
            v = pa.Table.from_pandas(v)
        elif isinstance(v, dict):
            v = pa.Table.from_pydict(v)
        elif not isinstance(v, pa.Table):
            raise ValueError(f"Value of type {type(v)} cannot be converted to pyarrow.Table")

        if schema:
            if strict is True:
                return v.cast(schema)
            elif strict == "filter":
                v = v.drop([c for c in v.schema.names if c not in schema.names])
                return v.cast(schema)
            elif strict is False:
                extra_cols = [c for c in v.schema.names if c not in schema.names]
                v_checked = v.drop(extra_cols).cast(schema)
                for extra in extra_cols:
                    v_checked = v_checked.append_column(extra, v[extra])
                return v_checked
        return v


class PyArrowDatatype(str):
    """Custom datatype represents a string validated as a PyarrowDatatype."""

    @property
    def datatype(self) -> Type:
        """Return the underlying PyarrowDatatype"""
        try:
            value = eval(self)
            if not isinstance(value, pa.lib.DataType):
                raise ValueError(f"ensure this value contains a valid PyarrowDatatype string: {value}")
            return value
        except Exception as e:
            raise ValueError(f"ensure this value contains a valid PyarrowDatatype string: {e}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value) -> Self:
        if isinstance(value, pa.lib.DataType):
            return value

        if isinstance(value, str):
            value = cls(value)
            value.datatype
            return value

        raise ValueError(f"ensure this value contains a valid PyarrowDatatype string: {value}")

    @classmethod
    def validate(cls, value) -> Self:
        """Try to convert/validate an arbitrary value to a PyArrowDatatype."""
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(PyArrowDatatype)
