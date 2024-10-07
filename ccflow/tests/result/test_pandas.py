from unittest import TestCase

import pandas as pd
import pyarrow as pa

from ccflow.result.pandas import PandasResult


class TestResult(TestCase):
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
