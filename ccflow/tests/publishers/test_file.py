import os
import pickle
import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import pydantic
import pydantic.json
import pytest
from pydantic import BaseModel as PydanticBaseModel

from ccflow.exttypes import NDArray
from ccflow.publishers import (
    DictTemplateFilePublisher,
    GenericFilePublisher,
    JSONPublisher,
    PandasFilePublisher,
    PicklePublisher,
    PydanticJSONPublisher,
    YAMLPublisher,
)


class MyTestModel(PydanticBaseModel):
    foo: int
    bar: date
    baz: NDArray[float]

    class Config:
        json_encoders = {np.ndarray: lambda arr: arr.tolist()}


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

    @pytest.mark.skipif(
        pydantic.__version__.startswith("2"),
        reason="Not supported in v2, but it seems like a bug in pydantic!",
    )
    def test_pydantic_json(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)

            # Use generic type
            p = PydanticJSONPublisher[MyTestModel](name="test_pydantic_json")
            p.data = {"foo": 5, "bar": date(2020, 1, 1), "baz": np.array([])}
            self.assertIsInstance(p.data, MyTestModel)
            p.options.include = {"foo", "bar"}
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), '{"foo": 5, "bar": "2020-01-01"}')

            # Don't use the generic type, and pass the numpy value
            # Because MyTestModel knows how to serialize JSON, it works
            p2 = PydanticJSONPublisher(name="test_pydantic_json_numpy")
            model = MyTestModel.validate({"foo": 5, "bar": date(2020, 1, 1), "baz": np.array([1.0, 2.0, 3.0])})
            p2.data = model
            p2.options.exclude = {"foo"}
            path2 = p2()
            with open(path2, "r") as f:
                self.assertEqual(f.read(), '{"bar": "2020-01-01", "baz": [1.0, 2.0, 3.0]}')

    @pytest.mark.skipif(
        pydantic.__version__.startswith("2"),
        reason="Not supported in v2, because you can't pass encoders at json serialization time",
    )
    def test_pydantic_json_custom(self):
        """Similar to the above test, but without specifying the pydantic model upfront,
        and passing it a numpy array."""
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)

            p = PydanticJSONPublisher(name="test_pydantic_json_numpy")
            p.data = {
                "foo": 5,
                "bar": date(2020, 1, 1),
                "baz": np.array([1.0, 2.0, 3.0]),
            }
            p.options.exclude = {"foo"}
            # Because the dynamically created model does *not* have a config for numpy json encoding,
            # we need to pass a custom encoder

            def custom_encoder(v):
                if isinstance(v, np.ndarray):
                    return v.tolist()
                return pydantic.json.pydantic_encoder(v)

            p.kwargs["encoder"] = custom_encoder
            path = p()
            with open(path, "r") as f:
                self.assertEqual(f.read(), '{"bar": "2020-01-01", "baz": [1.0, 2.0, 3.0]}')

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
