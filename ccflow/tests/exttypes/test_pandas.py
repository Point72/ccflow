import numpy as np
import pandas as pd
import pytest

from ccflow import BaseModel
from ccflow.exttypes import DataFrameWrapper, SeriesWrapper, SparseNumericDataFrame


class MyModel(BaseModel):
    series: SeriesWrapper = None
    df: DataFrameWrapper = None


class MyModelSparse(BaseModel):
    series: SeriesWrapper = None
    df: SparseNumericDataFrame = None


# Using a MultiIndex
arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
multi_index_df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])

series_data = [
    pd.Series([1, 2, 3], name="We have a name"),
    pd.Series([1.1, 2.2, 3.3]),
    pd.Series(["a", "b", "c"]),
    pd.Series(pd.date_range("2021-01-01", periods=3)),
    pd.Series(pd.Categorical(["a", "b", "a"])),
    pd.Series({"a": 11, "b": 12}),
    pd.Series(["asdhasd"]),
    pd.Series(["a", "b", 9, 10.1], index=["1", "2", "3", "4"]),
    pd.Series([], dtype=np.object_),
]

dataframe_data = [
    pd.DataFrame({"A": ["a", "b", "c"], "B": ["d", "e", "f"]}),
    pd.DataFrame({"A": ["a", "9", "c"], "B": [-99.3214, "e", 100]}),
    pd.DataFrame(
        {
            "A": pd.date_range("2021-01-01", periods=3),
            "B": pd.date_range("2021-01-04", periods=3),
            "C": [0, 1, 2],
        }
    ),
    pd.DataFrame({"A": pd.Categorical(["a", "b", "a"]), "B": pd.Categorical(["d", "e", "d"])}),
    multi_index_df,
]

# we force the dtypes to be float64 for the input dataframes
sparse_numeric_dataframe_data = [
    pd.DataFrame(
        {
            "A": pd.Series([1, 2.0, 3]),
            "B": pd.Series([4, 5, -6.0]),
            "C": pd.Series([7.0, 8, 9]),
        }
    ),
    pd.DataFrame({"A": [0, 0, 0], "B": [0, 0, 0]}).astype(np.float64),
    pd.DataFrame([{"A": 1, "B": 4, "C": 7}, {"A": 2, "B": 5, "C": 8}, {"A": 3, "B": 6, "C": 9}]).astype(np.float64),
    pd.DataFrame({"A": [-1, 0, 3.0], "B": [0, 5.0, 0]}),
    pd.DataFrame(
        np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=np.float64),
        columns=["A", "B", "C"],
    ),
    pd.DataFrame({"A": [0.08]}),
    pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, -6.5]}).astype(np.float64),
    pd.DataFrame({"A": [1.1, 2.2, 3.3], "B": [4.4, 5.5, 6.6]}),
    pd.DataFrame({"A": [1.1, 2.2, 3.3], "B": [4.4, 5.5, 6.6], "C": [np.nan, np.nan, -129]}),
]


@pytest.mark.parametrize("series", series_data)
def test_series_wrapper(series):
    wrapped_series = SeriesWrapper.validate(series)
    assert isinstance(wrapped_series, SeriesWrapper)
    encoded_series = SeriesWrapper.encode(wrapped_series)
    decoded_series = SeriesWrapper.decode(encoded_series)
    pd.testing.assert_series_equal(series, decoded_series, check_series_type=False)


@pytest.mark.parametrize("dataframe", dataframe_data + sparse_numeric_dataframe_data)
def test_dataframe_wrapper(dataframe):
    wrapped_dataframe = DataFrameWrapper.validate(dataframe)
    assert isinstance(wrapped_dataframe, DataFrameWrapper)
    encoded_dataframe = DataFrameWrapper.encode(wrapped_dataframe)
    decoded_dataframe = DataFrameWrapper.decode(encoded_dataframe)
    pd.testing.assert_frame_equal(dataframe, decoded_dataframe, check_frame_type=False)


@pytest.mark.parametrize("sparse_dataframe", sparse_numeric_dataframe_data)
def test_sparse_dataframe_wrapper(sparse_dataframe):
    wrapped_sparse_dataframe = SparseNumericDataFrame.validate(sparse_dataframe)
    assert isinstance(wrapped_sparse_dataframe, SparseNumericDataFrame)
    encoded_sparse_dataframe = SparseNumericDataFrame.encode(wrapped_sparse_dataframe)
    decoded_sparse_dataframe = SparseNumericDataFrame.decode(encoded_sparse_dataframe)
    pd.testing.assert_frame_equal(sparse_dataframe, decoded_sparse_dataframe, check_frame_type=False)


@pytest.mark.parametrize("ModelClass", [MyModel, MyModelSparse])
def test_model_with_wrappers(ModelClass):
    for my_series, my_df in zip(series_data, sparse_numeric_dataframe_data):
        model = ModelClass(series=my_series.copy(), df=my_df)
        assert model.series.equals(my_series)
        assert isinstance(model.series, (SeriesWrapper, DataFrameWrapper))

        # we perform a change
        model.df *= 3
        assert model.df.equals(my_df * 3)

        # Object serialization
        serialized = model.dict()
        deserialized = type(model).parse_obj(serialized)
        assert model == deserialized

        # JSON serialization
        serialized = model.json()
        deserialized = type(model).parse_raw(serialized)
        for field, value in model:
            other_value = getattr(deserialized, field)
            if isinstance(other_value, pd.Series):
                pd.testing.assert_series_equal(value, other_value, check_series_type=False)
            elif isinstance(other_value, pd.DataFrame):
                pd.testing.assert_frame_equal(value, other_value, check_frame_type=False)
            else:
                assert value == other_value
