from typing import Annotated, Any

import narwhals.stable.v1 as nw
import pandas as pd
import polars as pl
import pytest
from pydantic import TypeAdapter, ValidationError

from ccflow.exttypes.narwhals import (
    DataFrameT,
    DataFrameValidator,
    DType,
    FrameT,
    FrameValidator,
    LazyFrameT,
    LazyFrameValidator,
    Schema,
    SchemaValidator,
)


@pytest.fixture
def data():
    return {
        "a": [1.0, 2.0, 3.0],
        "b": [4, 5, 6],
        "c": ["foo", "bar", "baz"],
        "d": [0, 0, 0],
    }


@pytest.fixture
def schema():
    return {
        "a": nw.Float64,
        "b": nw.Int64,
        "c": nw.String,
        "d": nw.Int64,
    }


@pytest.mark.parametrize(
    "test_type",
    [
        DataFrameT,
        Annotated[nw.DataFrame, DataFrameValidator()],
        Annotated[nw.DataFrame[Any], DataFrameValidator()],
        Annotated[nw.DataFrame[pl.DataFrame], DataFrameValidator()],
    ],
)
def test_dataframe_validation(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data)
    assert isinstance(ta.validate_python(pl_df), nw.DataFrame)
    assert isinstance(ta.validate_python(pl_df.lazy()), nw.DataFrame)

    df = ta.validate_python(pl_df)
    assert isinstance(ta.validate_python(df), nw.DataFrame)
    assert isinstance(ta.validate_python(df.to_dict()), nw.DataFrame)


@pytest.mark.parametrize(
    "test_type",
    [
        LazyFrameT,
        Annotated[nw.LazyFrame, LazyFrameValidator()],
        Annotated[nw.LazyFrame[Any], LazyFrameValidator()],
        Annotated[nw.LazyFrame[pl.LazyFrame], LazyFrameValidator()],
    ],
)
def test_lazyframe_validation(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data).lazy()
    assert isinstance(ta.validate_python(pl_df), nw.LazyFrame)
    assert isinstance(ta.validate_python(pl_df.collect()), nw.LazyFrame)

    df = ta.validate_python(pl_df)
    assert isinstance(ta.validate_python(df), nw.LazyFrame)
    assert isinstance(ta.validate_python(df.collect().to_dict()), nw.LazyFrame)


@pytest.mark.parametrize(
    "test_type",
    [
        FrameT,
    ],
)
def test_frame_validation(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data)
    assert isinstance(ta.validate_python(pl_df), nw.DataFrame)
    assert isinstance(ta.validate_python(pl_df.lazy()), nw.LazyFrame)

    df = ta.validate_python(pl_df)
    assert isinstance(ta.validate_python(df), nw.DataFrame)
    assert isinstance(ta.validate_python(df.to_dict()), nw.DataFrame)

    df_lazy = ta.validate_python(pl_df.lazy())
    assert isinstance(ta.validate_python(df_lazy), nw.LazyFrame)


@pytest.mark.parametrize(
    "test_type",
    [
        Annotated[nw.DataFrame, FrameValidator()],
        Annotated[nw.DataFrame[Any], FrameValidator()],
        Annotated[nw.DataFrame[pl.DataFrame], FrameValidator()],
    ],
)
def test_frame_validation_eager(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data)
    assert isinstance(ta.validate_python(pl_df), nw.DataFrame)

    df = ta.validate_python(pl_df)
    assert isinstance(ta.validate_python(df), nw.DataFrame)
    assert isinstance(ta.validate_python(df.to_dict()), nw.DataFrame)

    # FrameValidator will error on the wrong type of frame
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df.lazy())


@pytest.mark.parametrize(
    "test_type",
    [
        Annotated[nw.LazyFrame, FrameValidator()],
        Annotated[nw.LazyFrame[Any], FrameValidator()],
        Annotated[nw.LazyFrame[pl.LazyFrame], FrameValidator()],
    ],
)
def test_frame_validation_lazy(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df_lazy = pl.DataFrame(data).lazy()
    assert isinstance(ta.validate_python(pl_df_lazy), nw.LazyFrame)

    df = ta.validate_python(pl_df_lazy)
    assert isinstance(ta.validate_python(df), nw.LazyFrame)

    # FrameValidator will error on the wrong type of frame
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df_lazy.collect())


def test_dataframe_validation_wrong_backend(data):
    """Polars-backed dataframe will fail to validate as a pandas backed data frame"""
    # Set backend to pandas
    ta = TypeAdapter(Annotated[nw.DataFrame[pd.DataFrame], DataFrameValidator()])
    pl_df = pl.DataFrame(data)
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df)

    ta = TypeAdapter(Annotated[nw.DataFrame[pl.LazyFrame], DataFrameValidator()])
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df)

    ta = TypeAdapter(Annotated[nw.LazyFrame[pl.DataFrame], LazyFrameValidator()])
    pl_df_lazy = pl.DataFrame(data).lazy()
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df_lazy)


def test_validation_from_dict(data):
    ta = TypeAdapter(DataFrameT)
    with pytest.raises(ValidationError):
        ta.validate_python(data)

    ta = TypeAdapter(Annotated[nw.DataFrame[pl.DataFrame], DataFrameValidator()])
    df = ta.validate_python(data)
    assert isinstance(df, nw.DataFrame)
    assert df.implementation == nw.Implementation.POLARS

    ta = TypeAdapter(Annotated[nw.LazyFrame[pl.LazyFrame], LazyFrameValidator()])
    df = ta.validate_python(data)
    assert isinstance(df, nw.LazyFrame)
    assert df.implementation == nw.Implementation.POLARS

    ta = TypeAdapter(Annotated[nw.DataFrame[pl.DataFrame], FrameValidator()])
    df = ta.validate_python(data)
    assert isinstance(df, nw.DataFrame)
    assert df.implementation == nw.Implementation.POLARS


@pytest.mark.parametrize(
    "test_type",
    [
        DataFrameT,
        Annotated[nw.DataFrame, DataFrameValidator()],
        Annotated[nw.DataFrame[Any], DataFrameValidator()],
        Annotated[nw.DataFrame[pl.DataFrame], DataFrameValidator()],
    ],
)
def test_dataframe_serialization(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data)
    df = ta.validate_python(pl_df)
    json = ta.dump_json(df)
    assert isinstance(json, bytes)

    # Need to specify a specific backend for deserialization
    ta2 = TypeAdapter(Annotated[nw.DataFrame[pl.DataFrame], DataFrameValidator()])
    df2 = ta2.validate_json(json)
    assert isinstance(df2, nw.DataFrame)


@pytest.mark.parametrize(
    "test_type",
    [
        LazyFrameT,
        Annotated[nw.LazyFrame, LazyFrameValidator()],
        Annotated[nw.LazyFrame[Any], LazyFrameValidator()],
        Annotated[nw.LazyFrame[pl.LazyFrame], LazyFrameValidator()],
    ],
)
def test_lazyframe_serialization(data, test_type):
    ta = TypeAdapter(test_type)
    pl_df = pl.DataFrame(data)
    df = ta.validate_python(pl_df)
    with pytest.raises(ValueError):
        ta.dump_json(df)


@pytest.mark.parametrize(
    "test_type",
    [
        Annotated[nw.DataFrame, LazyFrameValidator()],
        Annotated[nw.LazyFrame, DataFrameValidator()],
        Annotated[pl.DataFrame, DataFrameValidator()],
        Annotated[pl.DataFrame, FrameValidator()],
        Annotated[pl.LazyFrame, LazyFrameValidator()],
        Annotated[pl.LazyFrame, FrameValidator()],
    ],
)
def test_invalid_annotations(data, test_type):
    with pytest.raises(TypeError):
        TypeAdapter(test_type)


def test_dtype_validation():
    ta = TypeAdapter(DType)
    assert isinstance(ta.validate_python(nw.Float64()), nw.dtypes.DType)
    assert isinstance(ta.validate_python(nw.Float64), nw.dtypes.DType)
    assert ta.validate_python(nw.Float64) == nw.Float64()
    assert ta.validate_python(nw.String) == nw.String()

    assert ta.validate_python("Float64") == nw.Float64()
    assert ta.validate_python("String") == nw.String()

    with pytest.raises(ValidationError):
        ta.validate_python(None)

    with pytest.raises(ValidationError):
        ta.validate_python("foo")

    with pytest.raises(ValidationError):
        ta.validate_python("DataFrame")


def test_dtype_serialization():
    ta = TypeAdapter(DType)
    assert isinstance(ta.dump_json(nw.Float64()), bytes)
    assert isinstance(ta.validate_json(ta.dump_json(nw.Float64())), nw.dtypes.DType)
    assert isinstance(ta.validate_json(ta.dump_json(nw.Float64)), nw.dtypes.DType)
    assert ta.validate_json(ta.dump_json(nw.Float64())) == nw.Float64()
    assert ta.validate_json(ta.dump_json(nw.Float64)) == nw.Float64()
    assert ta.validate_json(ta.dump_json(nw.String())) == nw.String()
    assert ta.validate_json(ta.dump_json(nw.String)) == nw.String()

    with pytest.raises(ValueError):
        ta.dump_json(None)

    with pytest.raises(ValueError):
        ta.dump_json("foo")


def test_schema():
    ta = TypeAdapter(Schema)
    schema = nw.Schema({"a": nw.Float64(), "b": nw.Int64()})

    result = ta.validate_python(schema)
    assert isinstance(result, nw.Schema)
    assert isinstance(result["a"], nw.dtypes.DType)
    assert result == schema

    result = ta.validate_python(dict(schema))
    assert isinstance(result, nw.Schema)
    assert isinstance(result["a"], nw.dtypes.DType)
    assert result == schema

    result = ta.validate_python({"a": "Float64", "b": "Int64"})
    assert isinstance(result, nw.Schema)
    assert isinstance(result["a"], nw.dtypes.DType)
    assert result == schema

    assert isinstance(ta.dump_json(schema), bytes)
    result = ta.validate_json(ta.dump_json(schema))
    assert isinstance(result, nw.Schema)
    assert isinstance(result["a"], nw.dtypes.DType)
    assert result == schema


def test_schema_validator(schema):
    # Make sure that validation of the schema itself is applied when constructing the SchemaValidator
    validator = SchemaValidator(schema=schema)
    assert isinstance(validator.schema, nw.Schema)
    assert isinstance(validator.schema["a"], nw.dtypes.DType)

    # Make sure it's frozen
    with pytest.raises(AttributeError):
        validator.allow_subset = True


@pytest.mark.parametrize("test_type", [DataFrameT, LazyFrameT])
def test_schema_validation(test_type, data, schema):
    pl_df = pl.DataFrame(data)

    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema)])
    assert isinstance(ta.validate_python(pl_df), (nw.DataFrame, nw.LazyFrame))


@pytest.mark.parametrize("test_type", [DataFrameT, LazyFrameT])
def test_schema_validation_subset(test_type, data, schema):
    pl_df = pl.DataFrame(data)
    # Remove a column from the schema
    schema.pop("c")
    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema)])
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df)

    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema, allow_subset=True)])
    assert isinstance(ta.validate_python(pl_df), (nw.DataFrame, nw.LazyFrame))


@pytest.mark.parametrize("test_type", [DataFrameT, LazyFrameT])
def test_schema_validation_ordering(test_type, data, schema):
    pl_df = pl.DataFrame(data)
    # Remove a column from the schema
    schema_b = schema.pop("b")
    schema_c = schema.pop("c")

    schema["b"] = schema_b
    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema, allow_subset=True, check_ordering=True)])
    with pytest.raises(ValidationError):
        ta.validate_python(pl_df)

    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema, allow_subset=True)])
    assert isinstance(ta.validate_python(pl_df), (nw.DataFrame, nw.LazyFrame))

    schema["c"] = schema_c
    ta = TypeAdapter(Annotated[test_type, SchemaValidator(schema=schema)])
    assert isinstance(ta.validate_python(pl_df), (nw.DataFrame, nw.LazyFrame))


def test_schema_cast(data, schema):
    pl_df = pl.DataFrame(data)
    # Change the type of a column in the schema
    schema["b"] = nw.Float64
    ta = TypeAdapter(Annotated[DataFrameT, SchemaValidator(schema=schema, cast=True)])
    df = ta.validate_python(pl_df)
    assert isinstance(df, nw.DataFrame)
    assert df.collect_schema()["b"] == nw.Float64


def test_schema_cast_fails(data, schema):
    pl_df = pl.DataFrame(data)
    # Change the type of a column in the schema that will fail to cast
    schema["c"] = nw.Float64
    ta = TypeAdapter(Annotated[DataFrameT, SchemaValidator(schema=schema, cast=True)])
    with pytest.raises(ValueError):
        ta.validate_python(pl_df)


def test_schema_cast_lazy_raises(data, schema):
    pl_df = pl.DataFrame(data).lazy()
    with pytest.raises(ValueError):
        ta = TypeAdapter(Annotated[nw.LazyFrame, LazyFrameValidator(), SchemaValidator(schema=schema, cast=True)])
        ta.validate_python(pl_df)


def test_invalid_schema(schema):
    schema["a"] = "float64"
    with pytest.raises(TypeError):
        TypeAdapter(Annotated[DataFrameT, SchemaValidator(schema=schema)])
