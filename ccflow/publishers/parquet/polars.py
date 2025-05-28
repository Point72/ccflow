import abc
from typing import Any, Dict

from pydantic import Field
from typing_extensions import override

from ccflow.exttypes.narwhals import DataFrameT
from ccflow.publisher import BasePublisher

__all__ = ("PolarsParquetPublisher",)


class PolarsParquetPublisher(BasePublisher, abc.ABC):
    """Write data from a single narwhals DataFrame into parquet using polars.write_parquet."""

    data: DataFrameT = None
    kwargs: Dict[str, Any] = Field({}, description="Additional kwargs to pass to the arrow parquet writer. ")
    # mkdir: bool = Field(False, description="Whether to create the directory if it does not exist, on local filesystem. Do not use with cloud object storage.")

    @override
    def __call__(self) -> str:
        if self.data is None:
            raise ValueError("'data' field must be set before publishing")
        value = self.data.to_polars()
        path = self.get_name()
        # if self.mkdir:
        #     import os
        #     os.mkdir(str(Path(path).parent))
        value.write_parquet(path, **self.kwargs.copy())
        return path
