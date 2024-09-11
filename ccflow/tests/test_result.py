from unittest import TestCase

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from ccflow.generic_base import GenericResult
from ccflow.result.numpy import NumpyResult
from ccflow.result.pandas import PandasResult
from ccflow.result.pyarrow import ArrowResult
from ccflow.result.xarray import XArrayResult


class TestGenericResult(TestCase):
    def test_generic(self):
        v = {"a": 1, "b": [2, 3]}
        result = GenericResult(value=v)
        self.assertEqual(GenericResult.model_validate(v), result)
        self.assertIs(GenericResult.model_validate(result), result)

        v = {"value": 5}
        self.assertEqual(GenericResult.model_validate(v), GenericResult(value=5))
        self.assertEqual(GenericResult[int].model_validate(v), GenericResult[int](value=5))
        self.assertEqual(GenericResult[str].model_validate(v), GenericResult[str](value="5"))

        self.assertEqual(GenericResult.model_validate("foo"), GenericResult(value="foo"))
        self.assertEqual(GenericResult[str].model_validate(5), GenericResult[str](value="5"))

        result = GenericResult(value=5)
        # Note that this will work, even though GenericResult is not a subclass of GenericResult[str]
        self.assertEqual(GenericResult[str].model_validate(result), GenericResult[str](value="5"))


class TestResult(TestCase):
    def test_numpy(self):
        x = np.array([1.0, 3.0])
        r = NumpyResult[np.float64](array=x)
        np.testing.assert_equal(r.array, x)

        # Check you can also construct from list
        r = NumpyResult[np.float64](array=x.tolist())
        np.testing.assert_equal(r.array, x)

        self.assertRaises(TypeError, NumpyResult[np.float64], np.array(["foo"]))

        # Test generic
        r = NumpyResult[object](array=x)
        np.testing.assert_equal(r.array, x)
        r = NumpyResult[object](array=[None, "foo", 4.0])
        np.testing.assert_equal(r.array, np.array([None, "foo", 4.0]))

    def test_pandas(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        t = pa.Table.from_pandas(df)

        r = PandasResult(df=t)
        self.assertIsInstance(r.df, pd.DataFrame)

        r = PandasResult.model_validate({"df": t})
        self.assertIsInstance(r.df, pd.DataFrame)

        r = PandasResult(df=df["A"])
        self.assertIsInstance(r.df, pd.DataFrame)
        self.assertEqual(r.df.columns, ["A"])

    def test_arrow(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        r = ArrowResult.model_validate({"table": df})
        self.assertIsInstance(r.table, pa.Table)

        r = ArrowResult(table=df)
        self.assertIsInstance(r.table, pa.Table)

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
