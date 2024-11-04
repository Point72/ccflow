from unittest import TestCase

import pandas as pd
import pyarrow as pa
import xarray as xr

from ccflow.result.xarray import XArrayResult


class TestResult(TestCase):
    def test_xarray(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        da = xr.DataArray(df)

        r = XArrayResult(array=df)
        self.assertIsInstance(r.array, xr.DataArray)
        self.assertTrue(r.array.equals(da))

        t = pa.Table.from_pandas(df)
        r = XArrayResult(array=t)
        self.assertIsInstance(r.array, xr.DataArray)
        self.assertTrue(r.array.equals(da))

        r = XArrayResult.model_validate({"array": df})
        self.assertIsInstance(r.array, xr.DataArray)
        self.assertTrue(r.array.equals(da))

        r = XArrayResult.model_validate({"array": t})
        self.assertIsInstance(r.array, xr.DataArray)
        self.assertTrue(r.array.equals(da))
