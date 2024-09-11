import math
from unittest import TestCase

import numpy as np
import polars as pl
import scipy
from packaging import version
from pydantic import TypeAdapter

from ccflow.exttypes.polars import PolarsExpression


class TestPolarsExpression(TestCase):
    def test_expression(self):
        expression = pl.col("Col1") + pl.col("Col2")
        ta = TypeAdapter(PolarsExpression)
        self.assertEqual(ta.validate_python(expression).meta.serialize(), expression.meta.serialize())

    def test_expression_deserialization(self):
        ta = TypeAdapter(PolarsExpression)
        expression = ta.validate_python("pl.col('Col1') + pl.col('Col2')")
        expected_result = pl.col("Col1") + pl.col("Col2")

        self.assertEqual(expression.meta.serialize(), expected_result.meta.serialize())

    def test_expression_complex(self):
        ta = TypeAdapter(PolarsExpression)
        expression = ta.validate_python(
            "col('Col1') " "+ (sp.linalg.det(numpy.eye(2, dtype=int)) - 1 ) * math.pi * c('Col2') " "+ polars.col('Col2')"
        )
        expected_result = pl.col("Col1") + (scipy.linalg.det(np.eye(2, dtype=int)) - 1) * math.pi * pl.col("Col2") + pl.col("Col2")

        self.assertEqual(
            ta.validate_python(expression).meta.serialize(),
            expected_result.meta.serialize(),
        )

    def test_validation_failure(self):
        with self.assertRaises(ValueError):
            TypeAdapter(PolarsExpression).validate_python(None)

        with self.assertRaises(ValueError):
            TypeAdapter(PolarsExpression).validate_python("pl.DataFrame()")

    def test_validation_eval_failure(self):
        with self.assertRaises(ValueError):
            TypeAdapter(PolarsExpression).validate_python("invalid_statement")

    def test_json_serialization(self):
        expression = pl.col("Col1") + pl.col("Col2")
        json_result = TypeAdapter(PolarsExpression).dump_json(expression)
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            self.assertEqual(json_result.decode("utf-8"), expression.meta.serialize())
        else:
            # polars serializes into a binary format by default.
            self.assertEqual(json_result.decode("utf-8"), expression.meta.serialize(format="json"))

        expected_result = TypeAdapter(PolarsExpression).validate_json(json_result)
        self.assertEqual(expected_result.meta.serialize(), expression.meta.serialize())
