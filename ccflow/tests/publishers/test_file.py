import os
import pickle
import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase

import narwhals.stable.v1 as nw
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel

from ccflow.exttypes import NDArray
from ccflow.publishers import (
    DictTemplateFilePublisher,
    GenericFilePublisher,
    JSONPublisher,
    NarwhalsFilePublisher,
    PandasFilePublisher,
    PicklePublisher,
    YAMLPublisher,
)


class MyTestModel(PydanticBaseModel):
    foo: int
    bar: date
    baz: NDArray[float]


class TestFilePublishers(TestCase):
    def setUp(self) -> None:
        self.cwd = Path.cwd()

    def tearDown(self) -> None:
        os.chdir(self.cwd)

    def test_generic(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = GenericFilePublisher(name="directory/test_generic", suffix=".txt")
            p.data = "foo"
            path = p()
            self.assertEqual(path, Path("directory/test_generic.txt"))
            with open(path, "r") as f:
                self.assertEqual(f.read(), "foo")

            # Test that we can call it again
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), "foo")

    def test_generic_param(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = GenericFilePublisher(
                name="directory/test_{{param}}",
                name_params={"param": "generic"},
                suffix=".txt",
            )
            p.data = "foo"
            path = p()
            self.assertEqual(path, Path("directory/test_generic.txt"))
            with open(path, "r") as f:
                self.assertEqual(f.read(), "foo")

            # Test that we can call it again
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), "foo")

    def test_json(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = JSONPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
                kwargs=dict(default=str),
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            path = p()
            self.assertEqual(path, Path("test_JSON.json"))
            with open(path, "r") as f:
                self.assertEqual(f.read(), r'{"foo":5,"bar":"2020-01-01"}')

    def test_yaml(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = YAMLPublisher(name="test_{{param}}", name_params={"param": "yaml"})
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), "bar: 2020-01-01\nfoo: 5\n")

    def test_pickle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = PicklePublisher(name="test_{{param}}", name_params={"param": "Pickle"})
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            path = p()
            self.assertEqual(path, Path("test_Pickle.pickle"))
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.assertEqual(data, p.data)

    def test_dict_template(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            template = "The value of foo is {{foo}} and the value of bar is {{bar}}"
            p = DictTemplateFilePublisher(
                name="test_{{params}}",
                name_params={"param": "dict"},
                suffix=".txt",
                template=template,
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), "The value of foo is 5 and the value of bar is 2020-01-01")

    def test_pandas_html(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = PandasFilePublisher(
                name="test_{{param}}",
                name_params={"param": "pandas"},
                kwargs={"border": 0},  # Remove ugly HTML border!
            )
            p.data = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
            path = p()
            self.assertEqual(path, Path("test_pandas.html"))
            with open(path, "r") as f:
                out = f.read()
                self.assertTrue(out.startswith("<table"))
                self.assertTrue(out.endswith("</table>"))

    def test_pandas_string(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = PandasFilePublisher(
                name="test_{{param}}",
                name_params={"param": "pandas"},
                func="to_string",
                suffix=".txt",
            )
            p.data = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
            path = p()
            self.assertEqual(path, Path("test_pandas.txt"))
            with open(path, "r") as f:
                out = f.read()
                self.assertEqual(out, "   a    b\n0  1  foo\n1  2  bar\n2  3  baz")

    def test_pandas_feather(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = PandasFilePublisher(
                name="test_{{param}}",
                name_params={"param": "pandas"},
                func="to_feather",
                suffix=".f",
                mode="wb",
            )
            p.data = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
            path = p()
            self.assertEqual(path, Path("test_pandas.f"))
            df = pd.read_feather(path)
            pd.testing.assert_frame_equal(df, p.data)

    def test_narwhals_csv(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = NarwhalsFilePublisher(
                name="test_{{param}}",
                name_params={"param": "narwhals"},
                func="write_csv",
                suffix=".csv",
            )
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
            p.data = nw.from_native(df)
            path = p()
            self.assertEqual(path, Path("test_narwhals.csv"))
            df2 = pd.read_csv(path)
            pd.testing.assert_frame_equal(df, df2)

    def test_narwhals_parquet(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = NarwhalsFilePublisher(
                name="test_{{param}}",
                name_params={"param": "narwhals"},
                func="write_parquet",
                suffix=".parquet",
                mode="wb",
            )
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
            p.data = nw.from_native(df)
            path = p()
            self.assertEqual(path, Path("test_narwhals.parquet"))
            df2 = pd.read_parquet(path)
            pd.testing.assert_frame_equal(df, df2)
