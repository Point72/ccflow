import math

import numpy as np
import polars as pl
import pytest
import scipy
from packaging import version
from pydantic import TypeAdapter, ValidationError

from ccflow import BaseModel
from ccflow.exttypes.polars import PolarsExpression


def test_expression_passthrough():
    adapter = TypeAdapter(PolarsExpression)
    expression = pl.col("Col1") + pl.col("Col2")
    result = adapter.validate_python(expression)
    assert result.meta.serialize() == expression.meta.serialize()


def test_expression_from_string():
    adapter = TypeAdapter(PolarsExpression)
    expected_result = pl.col("Col1") + pl.col("Col2")
    expression = adapter.validate_python("pl.col('Col1') + pl.col('Col2')")
    assert expression.meta.serialize() == expected_result.meta.serialize()


def test_expression_complex():
    adapter = TypeAdapter(PolarsExpression)
    expected_result = pl.col("Col1") + (scipy.linalg.det(np.eye(2, dtype=int)) - 1) * math.pi * pl.col("Col2") + pl.col("Col2")
    expression = adapter.validate_python("col('Col1') + (sp.linalg.det(numpy.eye(2, dtype=int)) - 1 ) * math.pi * c('Col2') + polars.col('Col2')")
    assert expression.meta.serialize() == expected_result.meta.serialize()


def test_validation_failure():
    adapter = TypeAdapter(PolarsExpression)
    with pytest.raises(ValidationError):
        adapter.validate_python(None)
    with pytest.raises(ValidationError):
        adapter.validate_python("pl.DataFrame()")


def test_validation_eval_failure():
    adapter = TypeAdapter(PolarsExpression)
    with pytest.raises(ValidationError):
        adapter.validate_python("invalid_statement")


def test_json_serialization_roundtrip():
    adapter = TypeAdapter(PolarsExpression)
    expression = pl.col("Col1") + pl.col("Col2")
    json_result = adapter.dump_json(expression)
    if version.parse(pl.__version__) < version.parse("1.0.0"):
        assert json_result.decode("utf-8") == expression.meta.serialize()
    else:
        assert json_result.decode("utf-8") == expression.meta.serialize(format="json")

    expected_result = adapter.validate_json(json_result)
    assert expected_result.meta.serialize() == expression.meta.serialize()


def test_model_field_and_dataframe_filter():
    class DummyExprModel(BaseModel):
        expr: PolarsExpression

    m = DummyExprModel(expr="pl.col('x') > 10")
    assert isinstance(m.expr, pl.Expr)

    df = pl.DataFrame({"x": [5, 10, 11, 20], "y": [1, 2, 3, 4]})
    filtered = df.filter(m.expr)
    assert filtered.select("x").to_series().to_list() == [11, 20]


# Explicitly test the legacy classmethod validator for backwards compatibility
def test_polars_expression_validate_passthrough():
    adapter = TypeAdapter(PolarsExpression)
    expression = pl.col("Col1") + pl.col("Col2")
    result = adapter.validate_python(expression)
    assert result.meta.serialize() == expression.meta.serialize()


def test_polars_expression_validate_from_string():
    adapter = TypeAdapter(PolarsExpression)
    result = adapter.validate_python("pl.col('Col1') + pl.col('Col2')")
    expected_result = pl.col("Col1") + pl.col("Col2")
    assert result.meta.serialize() == expected_result.meta.serialize()


def test_polars_expression_validate_errors():
    adapter = TypeAdapter(PolarsExpression)
    with pytest.raises(ValidationError):
        adapter.validate_python(None)
    with pytest.raises(ValidationError):
        adapter.validate_python("pl.DataFrame()")
    with pytest.raises(ValidationError):
        adapter.validate_python("invalid_statement")
