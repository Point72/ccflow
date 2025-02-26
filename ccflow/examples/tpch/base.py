from typing import Literal

from pydantic import conint

from ccflow import ContextBase

__all__ = (
    "TPCHTable",
    "TPCHTableContext",
    "TPCHQueryContext",
)


TPCHTable = Literal["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


class TPCHTableContext(ContextBase):
    table: TPCHTable


class TPCHQueryContext(ContextBase):
    query_id: conint(ge=1, le=22)
