"""
Some of the logic in this file courtesy of https://github.com/narwhals-dev/narwhals/blob/main/tpch/

MIT License

Copyright (c) 2024, Marco Gorelli

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import io
from typing import Any, Literal

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.csv as pc
from pydantic import conint, model_validator

from ccflow import BaseModel, CallableModel, Flow, NullContext
from ccflow.result.narwhals import NarwhalsDataFrameResult

__all__ = ("TPCHTable", "TPCHDuckDBBackend", "TPCHTableProvider", "TPCHAnswerProvider")


TPCHTable = Literal["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def _convert_schema(schema: pa.Schema) -> pa.Schema:
    """Cast decimal columns to float64 and date32 to ns timestamp.

    Narwhals' polars/pandas/etc. backends prefer these dtypes over the
    DuckDB-native decimal/date32 representations.
    """
    new_fields = []
    for field in schema:
        if pa.types.is_decimal(field.type):
            new_fields.append(pa.field(field.name, pa.float64()))
        elif field.type == pa.date32():
            new_fields.append(pa.field(field.name, pa.timestamp("ns")))
        else:
            new_fields.append(field)
    return pa.schema(new_fields)


class TPCHDuckDBBackend(BaseModel):
    """Shared DuckDB connection that runs ``dbgen`` once for a given scale factor.

    This is a plain ``ccflow.BaseModel``, not a ``CallableModel``. The
    distinction matters in ccflow:

    * ``CallableModel`` subclasses are the only models the framework invokes
      as workflow steps (via ``@Flow.call``). They represent *something to
      run*.
    * ``BaseModel`` subclasses live in the ``ModelRegistry`` as plain
      configured Python objects — useful for shared state, connections,
      configuration that other models depend on. They are not themselves
      callable as workflow steps.

    This backend is shared state: it owns one DuckDB connection and ensures
    ``dbgen(sf=...)`` runs exactly once. By registering it under
    ``/tpch/backend`` and having every ``TPCHTableProvider`` /
    ``TPCHAnswerProvider`` reference that same path, the whole example uses a
    single connection regardless of how many providers are instantiated.

    Note on ``_conn`` / ``_generated``: leading-underscore annotated fields
    on a ``BaseModel`` become Pydantic ``PrivateAttr``s — they are not part
    of the model's public schema, and *they are not preserved by
    ``model_copy()``*. The ``model_validator`` below re-initialises the
    connection on every fresh instance, so a copied backend would simply
    create its own connection (and re-run ``dbgen`` lazily on first use)
    rather than share the original's state.
    """

    scale_factor: float
    _conn: Any = None
    _generated: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            self._conn.execute("INSTALL tpch; LOAD tpch")
        return self

    def _ensure_generated(self) -> None:
        if not self._generated:
            self._conn.execute(f"CALL dbgen(sf={self.scale_factor})")
            self._generated = True

    def get_table(self, table: TPCHTable) -> pl.DataFrame:
        self._ensure_generated()
        tbl_arrow = self._conn.query(f"SELECT * FROM {table}").to_arrow_table()
        tbl_arrow = tbl_arrow.cast(_convert_schema(tbl_arrow.schema))
        # Use the polars backend by default; it's the fastest narwhals backend
        # for the downstream query bodies.
        return pl.from_arrow(tbl_arrow)

    def get_answer(self, query_id: int) -> pa.Table:
        row = self._conn.query(f"SELECT answer FROM tpch_answers() WHERE scale_factor={self.scale_factor} AND query_nr={query_id}").fetchone()
        if not row:
            raise ValueError(f"No TPC-H answer found for scale_factor={self.scale_factor}, query_nr={query_id}")
        return pc.read_csv(io.BytesIO(row[0].encode("utf-8")), parse_options=pc.ParseOptions(delimiter="|"))


class TPCHTableProvider(CallableModel):
    """Provides a single TPC-H table as a Narwhals frame.

    One instance per table; the output schema is fixed by the ``table`` field.
    The call takes a ``NullContext`` because the provider has no runtime
    parameters — everything it needs is already on the model itself. The
    ``= NullContext()`` default lets callers (such as ``TPCHQuery``) invoke
    the provider with no arguments; ``@Flow.call`` reads the default from the
    signature in that case.
    """

    backend: TPCHDuckDBBackend
    table: TPCHTable

    @Flow.call
    def __call__(self, context: NullContext = NullContext()) -> NarwhalsDataFrameResult:
        return NarwhalsDataFrameResult(df=self.backend.get_table(self.table))


class TPCHAnswerProvider(CallableModel):
    """Provides the canonical reference answer for a single TPC-H query."""

    backend: TPCHDuckDBBackend
    query_id: conint(ge=1, le=22)

    @Flow.call
    def __call__(self, context: NullContext = NullContext()) -> NarwhalsDataFrameResult:
        return NarwhalsDataFrameResult(df=self.backend.get_answer(self.query_id))
