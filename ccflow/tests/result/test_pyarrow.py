from datetime import date
from unittest import TestCase

import pandas as pd
import polars as pl
import pyarrow as pa

from ccflow.context import DateRangeContext
from ccflow.result.pyarrow import ArrowDateRangeResult, ArrowResult


class TestResult(TestCase):
    def test_arrow_from_pandas(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        r = ArrowResult.model_validate({"table": df})
        self.assertIsInstance(r.table, pa.Table)

        r = ArrowResult(table=df)
        self.assertIsInstance(r.table, pa.Table)

        r = ArrowResult.model_validate(df)
        self.assertIsInstance(r.table, pa.Table)

    def test_arrow_from_polars(self):
        df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        r = ArrowResult.model_validate({"table": df})
        self.assertIsInstance(r.table, pa.Table)

        r = ArrowResult(table=df)
        self.assertIsInstance(r.table, pa.Table)

        r = ArrowResult.model_validate(df)
        self.assertIsInstance(r.table, pa.Table)

    def test_arrow_date_range(self):
        df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "D": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]})
        context = DateRangeContext(start_date=date(2020, 1, 1), end_date=date(2020, 1, 3))
        r = ArrowDateRangeResult.model_validate({"table": df, "date_col": "D", "context": context})
        self.assertIsInstance(r.table, pa.Table)

        self.assertRaises(ValueError, ArrowDateRangeResult.model_validate, {"table": df, "date_col": "B", "context": context})
        self.assertRaises(ValueError, ArrowDateRangeResult.model_validate, {"table": df, "date_col": "E", "context": context})
        context = DateRangeContext(start_date=date(2020, 1, 1), end_date=date(2020, 1, 2))
        self.assertRaises(ValueError, ArrowDateRangeResult.model_validate, {"table": df, "date_col": "E", "context": context})
