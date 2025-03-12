"""Custom log formatters for result types."""

import logging
import pprint

import narwhals as nw
import polars as pl
import pyarrow as pa
from pydantic import BaseModel


class PolarsTableFormatter(logging.Formatter):
    """Formats Arrow Tables and Narwhals eager Dataframes as polars tables.
    Leaves Narwhals LazyFrame and other types as-is.
    """

    def __init__(self, *args, **kwargs):
        self.polars_config = kwargs.pop("polars_config", {})
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record and converts Arrow Tables and Narwhals eager Dataframes to polars tables."""
        if hasattr(record, "result") and isinstance(record.result, BaseModel):
            out = record.result.model_dump(by_alias=True)
            for k, v in out.items():
                if isinstance(v, pa.Table):
                    out[k] = pl.from_arrow(v)
                elif isinstance(v, nw.DataFrame):
                    out[k] = v.to_polars()
            with pl.Config(**self.polars_config):
                record.msg = f"{record.msg}\n{pprint.pformat(out, width=120)}"
        return super().format(record)
