import difflib
import inspect
import pprint
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Any, Optional, Union, get_args, get_origin

import narwhals.stable.v1 as nw
from pydantic import AfterValidator, ConfigDict, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import core_schema

__all__ = (
    "FrameValidator",
    "DataFrameValidator",
    "LazyFrameValidator",
    "FrameT",
    "DataFrameT",
    "LazyFrameT",
    "DType",
    "Schema",
    "SchemaValidator",
)


class FrameValidator:
    """Pydantic validator for Narwhals Frames.

    When used directly (with lazy=None), it will raise a validation error if an eager frame is passed to a lazy frame source or vice versa.
    """

    lazy: Optional[bool] = None

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        source_origin = get_origin(source_type) or source_type
        if source_origin == nw.DataFrame:
            if cls.lazy is True:
                raise TypeError(f"Must apply {cls.__name__} to a LazyFrame. Received {source_origin}.")
        elif source_origin == nw.LazyFrame:
            if cls.lazy is False:
                raise TypeError(f"Must apply {cls.__name__} to a DataFrame. Received {source_origin}")
        else:
            raise TypeError(f"Must apply {cls.__name__} to a narwhals DataFrame or LazyFrame. Received {source_origin}")

        # Prepare a TypeAdapter for validation of the generic argument, if present
        source_args = get_args(source_type)
        if source_args and source_args[0]:
            source_arg_adapter = TypeAdapter(source_args[0], config=ConfigDict(arbitrary_types_allowed=True))
        else:
            source_arg_adapter = TypeAdapter(Any)

        def validate_from_any(value: Any):
            if isinstance(value, dict):
                backend = None
                if source_args and source_args[0] and source_args[0] is not Any:
                    backend = source_args[0].__module__.split(".", 1)[0]

                try:
                    try:
                        value = nw.from_dict(value, backend=backend)
                    except AssertionError:  # Not implemented backend
                        backend = None
                        value = nw.from_dict(value, backend=backend)
                except TypeError as e:
                    raise ValueError(f"Fail to validate dict using nw.from_dict with backend {backend}") from e
            else:
                try:
                    value = nw.from_native(value)
                except TypeError as e:
                    # pydantic validators should throw value errors
                    raise ValueError("Fail to validate frame using nw.from_native") from e

            if cls.lazy is True and isinstance(value, nw.DataFrame):
                value = value.lazy()
            if cls.lazy is False and isinstance(value, nw.LazyFrame):
                value = value.collect()
            if cls.lazy is None and not isinstance(value, source_origin):
                raise ValueError(f"Expected {source_origin} but got {type(value)}")

            try:
                source_arg_adapter.validate_python(value.to_native())
            except ValueError:
                raise ValueError(f"Failed to match the generic argument ({source_args[0]}) of the Frame with {value.implementation}.") from None
            return value

        def serialize(value: Any):
            if isinstance(value, nw.DataFrame):
                return value.to_dict(as_series=False)
            else:
                raise ValueError("Cannot serialize a LazyFrame to JSON. Please use the collect() method to convert it to a DataFrame first.")

        from_any_schema = core_schema.no_info_plain_validator_function(validate_from_any)
        return core_schema.json_or_python_schema(
            json_schema=from_any_schema,
            python_schema=from_any_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )


class DataFrameValidator(FrameValidator):
    """Pydantic validator for Narwhals DataFrame.

    Will collect lazy frames and convert them to eager frames.
    """

    lazy: Optional[bool] = False


class LazyFrameValidator(FrameValidator):
    """Pydantic validator for Narwhals LazyFrame.

    Will create lazy frames out of eager data frames.
    """

    lazy: Optional[bool] = True


# Mirror nw.DataFrameT and nw.FrameT (but with validation) for consistency. LazyFrameT added for convenience and symmetry.
DataFrameT = Annotated[nw.DataFrame, DataFrameValidator()]
LazyFrameT = Annotated[nw.LazyFrame, LazyFrameValidator()]
FrameT = Union[Annotated[nw.DataFrame, FrameValidator()], Annotated[nw.LazyFrame, FrameValidator()]]


class _DTypeValidator:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        if not inspect.isclass(source_type) or not issubclass(source_type, nw.dtypes.DType):
            raise TypeError(f"Must apply DTypeValidator to a Narwhals DType. Received {source_type}")

        def validate_from_any(value: Any):
            if isinstance(value, str):
                try:
                    value = getattr(nw, value)
                except AttributeError:
                    raise ValueError(f"Fail to find a narwhals DType with name {value}.") from None
            if inspect.isclass(value) and issubclass(value, nw.dtypes.DType):
                value = value()
            if not isinstance(value, nw.dtypes.DType):
                raise ValueError(f"Expected a Narwhals DType but got {type(value)}")
            return value

        def serialize(value: Any):
            if inspect.isclass(value) and issubclass(value, nw.dtypes.DType):
                value = value()
            if not isinstance(value, nw.dtypes.DType):
                raise ValueError(f"Expected a Narwhals DType but got {type(value)}")
            return str(value)

        from_any_schema = core_schema.no_info_plain_validator_function(validate_from_any)
        return core_schema.json_or_python_schema(
            json_schema=from_any_schema,
            python_schema=from_any_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )


# Mirror nw.dtypes.DType (but with validation) for consistency
DType = Annotated[nw.dtypes.DType, _DTypeValidator()]

# Mirror nw.Schema (but with validation) for consistency
Schema = Annotated[OrderedDict[str, DType], AfterValidator(func=nw.Schema)]


@dataclass(frozen=True)
class SchemaValidator:
    """Validator to check the schema of a Narwhals Frame"""

    schema: Schema
    allow_subset: bool = False
    check_ordering: bool = False
    cast: bool = False

    def __post_init__(self):
        try:
            super().__setattr__("schema", TypeAdapter(Schema).validate_python(self.schema))
        except ValueError as e:
            raise TypeError(f"Schema must be a valid Narwhals Schema. Received {self.schema}") from e

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def validate_schema(value: Any):
            if not isinstance(value, (nw.DataFrame, nw.LazyFrame)):
                raise ValueError(f"Expected a Narwhals DataFrame or LazyFrame but got {type(value)}")
            schema = value.collect_schema()
            if self.allow_subset:
                schema = nw.Schema({k: v for k, v in schema.items() if k in self.schema})
            if self.cast:
                # Casting is not supported on lazy frames because we don't have per-column slicing or to_dict
                if isinstance(value, nw.LazyFrame):
                    raise ValueError("Casting is not supported for LazyFrames. Please collect the LazyFrame first.")
                # Preserve ordering of input frame by inserting each column back into an empty dictionary
                new_values = {}
                new_schema = {}
                for col, dtype in schema.items():
                    if col in self.schema and self.schema[col] != dtype:
                        try:
                            new_values[col] = value[col].cast(self.schema[col])
                        except Exception:
                            raise ValueError(f"Failed to cast column {col} from {dtype} to {self.schema[col]}") from None
                        new_schema[col] = self.schema[col]
                    else:
                        new_values[col] = value[col]
                        new_schema[col] = dtype
                value = nw.from_dict(new_values)
                schema = nw.Schema(new_schema)

            d1 = dict(schema)
            d2 = dict(self.schema)
            if d1 != d2:
                diff = "\n" + "\n".join(difflib.ndiff(pprint.pformat(d1).splitlines(), pprint.pformat(d2).splitlines())) + "\n"
                raise ValueError(f"Schema mismatch in column names or types: {diff}")
            if self.check_ordering and schema != self.schema:
                diff = "\n" + "\n".join(difflib.ndiff(pprint.pformat(schema).splitlines(), pprint.pformat(self.schema).splitlines())) + "\n"
                raise ValueError(f"Schema mismatch in column ordering: {diff}")

            return value

        return core_schema.no_info_after_validator_function(validate_schema, handler(source_type))
