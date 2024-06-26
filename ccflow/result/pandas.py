"""This module defines re-usable result types for the "Callable Model" framework
defined in flow.callable.py.
"""

import pandas as pd
import pyarrow as pa
from pydantic import validator

from ..base import ResultBase

__all__ = ("PandasResult",)


class PandasResult(ResultBase):
    df: pd.DataFrame

    @validator("df", pre=True)
    def _from_arrow(cls, v):
        if isinstance(v, pa.Table):
            return v.to_pandas()
        return v

    @validator("df", pre=True)
    def _from_series(cls, v):
        if isinstance(v, pd.Series):
            return pd.DataFrame(v)
        return v
