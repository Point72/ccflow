from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ccflow.publishers.parquet.polars import PolarsParquetPublisher


@pytest.fixture()
def df():
    np.random.seed(123)
    date_idx = pl.date_range(date(2022, 1, 1), date(2022, 1, 10), eager=True).alias("date")
    symbols = ["x", "y", "z"]
    columns = ["A", "B", "C"]
    df = date_idx.to_frame().join(pl.DataFrame({"symbols": symbols}), how="cross")
    data = pl.DataFrame(np.random.randint(0, 3, (len(df), len(columns))), schema=columns)
    df = pl.concat([df, data], how="horizontal")
    return df


@pytest.mark.parametrize("compression", ["snappy", "lz4"])
def test_basic(df, tmp_path, compression):
    p = PolarsParquetPublisher(
        data=df,
        name_params={"date": date(2022, 1, 10)},
        name=str(tmp_path / "table_{{date}}.parquet"),
        kwargs={"compression": compression},
    )
    out = p()

    assert Path(out) == tmp_path / "table_2022-01-10.parquet"
    assert Path(out).exists()
    out_df = pl.read_parquet(out)
    assert_frame_equal(out_df, df)
