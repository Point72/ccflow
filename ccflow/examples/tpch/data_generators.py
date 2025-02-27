"""
Some of the logic in this file courtesy of https://github.com/narwhals-dev/narwhals/blob/main/tpch/

MIT License

Copyright (c) 2024, Marco Gorelli

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import io
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.csv as pc
from pydantic import model_validator

from ccflow import CallableModel, Flow
from ccflow.result.narwhals import NarwhalsDataFrameResult

from .base import TPCHQueryContext, TPCHTableContext

__all__ = ("TPCHAnswerGenerator", "TPCHDataGenerator")


class TPCHAnswerGenerator(CallableModel):
    """Generates data for the TPC-H benchmark."""

    scale_factor: float
    _conn: Any = None

    @model_validator(mode="after")
    def _validate(self):
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            self._conn.execute("INSTALL tpch; LOAD tpch")
        return self

    def get_query(self, context) -> str:
        return f"""
            SELECT answer FROM tpch_answers()
            WHERE scale_factor={self.scale_factor} AND query_nr={context.query_id}
            """

    @Flow.call()
    def __call__(self, context: TPCHQueryContext) -> NarwhalsDataFrameResult:
        """Generates data for the TPC-H benchmark."""
        results = self._conn.query(self.get_query(context))
        row = results.fetchone()
        if row:
            answer = row[0]
            tbl_answer = pc.read_csv(io.BytesIO(answer.encode("utf-8")), parse_options=pc.ParseOptions(delimiter="|"))
            return NarwhalsDataFrameResult(df=tbl_answer)
        else:
            raise ValueError(f"No TPCH answers found for the given scale factor ({self.scale_factor}) and query number ({context.query_id}).")


class TPCHDataGenerator(CallableModel):
    scale_factor: float
    _conn: Any = None
    _generated: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            self._conn.execute("INSTALL tpch; LOAD tpch")
        return self

    def _generate_if_needed(self):
        if self._generated:
            return
        self._conn.execute(f"CALL dbgen(sf={self.scale_factor})")
        self._generated = True

    @Flow.call
    def __call__(self, context: TPCHTableContext) -> NarwhalsDataFrameResult:
        """Generates data for the TPC-H benchmark."""
        self._generate_if_needed()
        tbl = self._conn.query(f"SELECT * FROM {context.table}")
        tbl_arrow = tbl.to_arrow_table()
        new_schema = []
        for field in tbl_arrow.schema:
            if isinstance(field.type, type(pa.decimal128(1))):
                new_schema.append(pa.field(field.name, pa.float64()))
            elif field.type == pa.date32():
                new_schema.append(pa.field(field.name, pa.timestamp("ns")))
            else:
                new_schema.append(field)
        tbl_arrow = tbl_arrow.cast(pa.schema(new_schema))
        # Convert to Polars DataFrame to use the polars backend by default for downstream calculations (it's faster)
        return NarwhalsDataFrameResult(df=pl.from_arrow(tbl_arrow))
