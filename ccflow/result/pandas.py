import pandas as pd
import pyarrow as pa
from pydantic import field_validator

from ..base import ResultBase

__all__ = ("PandasResult",)


class PandasResult(ResultBase):
    df: pd.DataFrame

    @field_validator("df", mode="before")
    def _from_arrow(cls, v):
        if isinstance(v, pa.Table):
            return v.to_pandas()
        return v

    @field_validator("df", mode="before")
    def _from_series(cls, v):
        if isinstance(v, pd.Series):
            return pd.DataFrame(v)
        return v
