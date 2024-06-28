import pandas as pd
import pyarrow
import pyarrow as pa
import pyarrow.lib
from typing import Any, Generic, Type, TypeVar, Union
from typing_extensions import Literal, get_args


class ArrowSchema(type):
    """A metaclass for creating Arrow schema-specific types that can be used in Generic"""

    @classmethod
    def make(
        cls,
        schema: pa.Schema,
        strict: Union[bool, Literal["filter"]] = "filter",
        clsname: str = "_ArrowSchema",
    ):
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

        def __get_validators__(cls):
            yield cls.validate

        def validate(cls, v, field=None):
            raise ValueError(err_msg)

        newclass.__init__ = __init__
        newclass.__get_validators__ = classmethod(__get_validators__)
        newclass.validate = classmethod(validate)

        return newclass


S = TypeVar("S", bound=ArrowSchema)


class ArrowTable(pyarrow.Table, Generic[S]):
    """Pydantic compatible wrapper around Arrow tables, with optional schema validation."""

    @classmethod
    def __get_validators__(cls):
        """Validation for pydantic v1"""
        yield cls.validate

    @classmethod
    def validate(cls, v, field):
        if field.sub_fields:
            schema = field.sub_fields[0].type_.schema
            strict = field.sub_fields[0].type_.strict
        else:
            schema = None
            strict = None
        return cls._validate(v, schema, strict)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation for pydantic v2"""
        from pydantic_core import core_schema

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
            v = pyarrow.Table.from_batches(v)
        elif hasattr(v, "to_arrow"):  # For polars, but without importing it
            v = v.to_arrow()
        elif isinstance(v, pd.DataFrame):
            v = pyarrow.Table.from_pandas(v)
        elif isinstance(v, dict):
            v = pyarrow.Table.from_pydict(v)
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
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation for pydantic v2"""
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value, field=None) -> Any:
        if isinstance(value, pa.lib.DataType):
            return value

        if isinstance(value, str):
            value = cls(value)
            value.datatype
            return value

        raise ValueError(f"ensure this value contains a valid PyarrowDatatype string: {value}")
