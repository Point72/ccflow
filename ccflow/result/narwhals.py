import narwhals.stable.v1 as nw
from pydantic import Field, model_validator

from ..base import ResultBase
from ..exttypes.narwhals import DataFrameT, FrameT

__all__ = (
    "NarwhalsFrameResult",
    "NarwhalsDataFrameResult",
)


class NarwhalsFrameResult(ResultBase):
    """Result that holds a Narwhals DataFrame or LazyFrame."""

    df: FrameT = Field(description="Narwhals DataFrame or LazyFrame")

    def collect(self) -> "NarwhalsDataFrameResult":
        """Collects the result into a NarwhalsDataFrameResult."""
        if isinstance(self.df, nw.LazyFrame):
            return NarwhalsDataFrameResult(df=self.df.collect(), **self.model_dump(exclude={"df", "type_"}))
        return NarwhalsDataFrameResult(df=self.df, **self.model_dump(exclude={"df", "type_"}))

    @model_validator(mode="wrap")
    def _validate(cls, v, handler, info):
        if not isinstance(v, NarwhalsFrameResult) and not (isinstance(v, dict) and "df" in v):
            v = {"df": v}
        return handler(v)


class NarwhalsDataFrameResult(NarwhalsFrameResult):
    df: DataFrameT = Field(description="Narwhals eager Dataframe")

    def collect(self) -> "NarwhalsDataFrameResult":
        """Collects the result into a NarwhalsDataFrameResult."""
        return self
