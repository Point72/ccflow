from importlib import import_module
from typing import Dict, Tuple

from pydantic import Field

from ccflow import CallableModel, CallableModelGenericType, Flow
from ccflow.result.narwhals import NarwhalsFrameResult

from .base import TPCHQueryContext, TPCHTable, TPCHTableContext

__all__ = ("TPCHQueryRunner",)


_QUERY_TABLE_MAP: Dict[int, Tuple[TPCHTable, ...]] = {
    1: ("lineitem",),
    2: ("region", "nation", "supplier", "part", "partsupp"),
    3: ("customer", "lineitem", "orders"),
    4: ("lineitem", "orders"),
    5: ("region", "nation", "customer", "lineitem", "orders", "supplier"),
    6: ("lineitem",),
    7: ("nation", "customer", "lineitem", "orders", "supplier"),
    8: ("part", "supplier", "lineitem", "orders", "customer", "nation", "region"),
    9: ("part", "partsupp", "nation", "lineitem", "orders", "supplier"),
    10: ("customer", "nation", "lineitem", "orders"),
    11: ("nation", "partsupp", "supplier"),
    12: ("lineitem", "orders"),
    13: ("customer", "orders"),
    14: ("lineitem", "part"),
    15: ("lineitem", "supplier"),
    16: ("part", "partsupp", "supplier"),
    17: ("lineitem", "part"),
    18: ("customer", "lineitem", "orders"),
    19: ("lineitem", "part"),
    20: ("part", "partsupp", "nation", "lineitem", "supplier"),
    21: ("lineitem", "nation", "orders", "supplier"),
    22: ("customer", "orders"),
}


class TPCHQueryRunner(CallableModel):
    """Generically runs TPC-H queries from a pre-packaged repository of queries (courtesy of narwhals)."""

    table_provider: CallableModelGenericType[TPCHTableContext, NarwhalsFrameResult]
    query_table_map: Dict[int, Tuple[TPCHTable, ...]] = Field(_QUERY_TABLE_MAP, validate_default=True)

    @Flow.call
    def __call__(self, context: TPCHQueryContext) -> NarwhalsFrameResult:
        query_module = import_module(f"ccflow.examples.tpch.queries.q{context.query_id}")
        inputs = (self.table_provider(TPCHTableContext(table=table)).df for table in self.query_table_map[context.query_id])
        result = query_module.query(*inputs)
        return NarwhalsFrameResult(df=result)
