"""This module defines re-usable result types for the "Callable Model" framework
defined in flow.callable.py.
"""

import pandas as pd
import pyarrow as pa
import xarray as xr
from pydantic import field_validator

from ..base import ResultBase

__all__ = ("XArrayResult",)


class XArrayResult(ResultBase):
    array: xr.DataArray

    @field_validator("array", mode="before")
    def _from_pandas(cls, v):
        if isinstance(v, pd.DataFrame):
            return xr.DataArray(v)
        return v

    @field_validator("array", mode="before")
    def _from_arrow(cls, v):
        if isinstance(v, pa.Table):
            return xr.DataArray(v.to_pandas())
        return v
