from typing import Annotated

import narwhals.stable.v1 as nw
import polars as pl
import pytest

from ccflow.exttypes.narwhals import (
    DataFrameT,
    SchemaValidator,
)
from ccflow.result.narwhals import NarwhalsDataFrameResult, NarwhalsFrameResult


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
        "d": nw.Float64,
    }


def test_narwhals_frame_result(data):
    df = pl.DataFrame(data)
    result = NarwhalsFrameResult(df=df)
    assert isinstance(result.df, nw.DataFrame)
    assert result.df.to_native() is df

    df = pl.DataFrame(data).lazy()
    result = NarwhalsFrameResult(df=df)
    assert isinstance(result.df, nw.LazyFrame)
    assert result.df.to_native() is df


def test_narwhals_dataframe_result(data):
    df = pl.DataFrame(data)
    result = NarwhalsDataFrameResult(df=df)
    assert isinstance(result.df, nw.DataFrame)
    assert result.df.to_native() is df

    df = pl.DataFrame(data).lazy()
    result = NarwhalsDataFrameResult(df=df)
    assert isinstance(result.df, nw.DataFrame)


def test_collect(data):
    df = pl.DataFrame(data)
    result = NarwhalsFrameResult(df=df)
    result2 = result.collect()
    assert isinstance(result2, NarwhalsDataFrameResult)
    assert isinstance(result2.df, nw.DataFrame)


def test_custom(data, schema):
    class MyNarwhalsResult(NarwhalsDataFrameResult):
        df: Annotated[DataFrameT, SchemaValidator(schema, cast=True)]

    df = pl.DataFrame(data)
    result = MyNarwhalsResult(df=df)
    assert result.df.schema["d"] == nw.Float64()
