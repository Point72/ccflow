"""
We include this file as wrappers over pandas objects for pydantic validation.

Pandera offers this ability but it is currently not a requirement for ccflow
and importing the package provides overhead
(it is for ccflow)

Based on:
https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
"""

from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Union

import numpy as np
import orjson
import pandas as pd

from ..serialization import make_ndarray_orjson_valid, orjson_dumps


class GenericPandasWrapper(ABC):
    @classmethod
    @abstractmethod
    def validate(cls, val: Any) -> Any: ...

    @classmethod
    @abstractmethod
    def encode(cls, x: Any, info=None) -> str:
        # Note that we supply an "info" argument so that it can be used as a serializer in pydantic v2
        ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls.encode,
                info_arg=False,
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )


class SeriesWrapper(pd.Series, GenericPandasWrapper):
    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v

        if isinstance(v, pd.Series):
            return cls(v)

        if isinstance(v, (dict, str)):
            return cls.decode(v)

        if isinstance(v, (list, tuple, np.ndarray, pd.Index, pd.DataFrame)):
            return cls(pd.Series(v))

        raise ValueError(f"Unable to validate {v}")

    @classmethod
    def encode(cls, x: "SeriesWrapper", info=None) -> str:
        series_dict = {
            "name": x.name,
            "index": make_ndarray_orjson_valid(x.index.to_numpy()),
            "values": make_ndarray_orjson_valid(x.to_numpy()),
            "dtype": str(x.dtype),
        }
        return orjson_dumps(series_dict)

    @classmethod
    def decode(cls, x: Union[str, dict]) -> "SeriesWrapper":
        if isinstance(x, str):
            x = orjson.loads(x)
            index = x["index"]
            # apparently this is what empty indices are
            index = pd.RangeIndex(start=0, stop=0, step=1) if not index else index
            series = pd.Series(x["values"], index=index, name=x["name"], dtype=x["dtype"])
            return cls(series)
        return cls(pd.Series(x))


class DataFrameWrapper(pd.DataFrame, GenericPandasWrapper):
    """Wrapper around a dataframe that can be used as a field in a pydantic model
    without needing to enable arbitrary_types_allowed.
    """

    @classmethod
    def validate(cls, v):
        if isinstance(v, cls):
            return v

        if isinstance(v, pd.DataFrame):
            return cls(v)

        if isinstance(v, (str, dict)):
            return cls.decode(v)

        if isinstance(v, (list, tuple)):
            return cls(pd.DataFrame(v))

        raise ValueError(f"Unable to validate {v}")

    @classmethod
    def encode(cls, x: "DataFrameWrapper", info=None) -> str:
        return x.to_json(orient="table")

    @classmethod
    def decode(cls, x: Union[str, dict]) -> "DataFrameWrapper":
        if isinstance(x, str):
            return cls(pd.read_json(StringIO(x), orient="table"))
        return cls(pd.DataFrame.from_dict(x))


class SparseNumericDataFrame(DataFrameWrapper):
    """A DataFrame wrapper that serializes efficiently for sparse matrices.
    Everything must be convertible to float64, as that will be the datatype returned
    when decoding."""

    def get_coordinates_and_data(self):
        """Get a list of tuples of the coordinates for non-zero data and the non-zero data."""
        data_i, data_j = np.nonzero(self.values)
        data_v = self.values[data_i, data_j]
        return [(i, j, v) for i, j, v in zip(self.index[data_i], self.columns[data_j], data_v)]

    @classmethod
    def encode(cls, x: "SparseNumericDataFrame", info=None) -> str:
        data_i, data_j = np.nonzero(x.values)
        data_v = x.values[data_i, data_j]
        index_vals = x.index.to_numpy()
        col_vals = x.columns.to_numpy()
        output = {
            "index": make_ndarray_orjson_valid(index_vals),
            "columns": make_ndarray_orjson_valid(col_vals),
            "data_i": data_i.astype(int),
            "data_j": data_j.astype(int),
            "data_v": make_ndarray_orjson_valid(data_v),
        }
        return orjson_dumps(output)

    @classmethod
    def decode(cls, x: Union[str, dict]) -> "SparseNumericDataFrame":
        if isinstance(x, str):
            x = orjson.loads(x)

        columns = x["columns"]
        index = x["index"]
        data_i = x["data_i"]
        data_j = x["data_j"]
        data_v = x["data_v"]
        vals = np.zeros((len(index), len(columns)))
        vals[data_i, data_j] = data_v
        # df = pd.DataFrame(vals, index=index, columns=columns).astype(dtypes)
        df = pd.DataFrame(vals, index=index, columns=columns)
        return cls(df)
