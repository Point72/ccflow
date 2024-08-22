import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict
from unittest import TestCase

import numpy as np
import pydantic.json
from pydantic import BaseModel as PydanticBaseModel

from ccflow.exttypes import NDArray
from ccflow.publisher import BasePublisher
from ccflow.publishers import CompositePublisher, DictTemplateFilePublisher, GenericFilePublisher, JSONPublisher


class DummyPublisher(BasePublisher):
    data: tuple = None

    def __call__(self):
        raise ValueError


class MyTestModel(PydanticBaseModel):
    foo: int
    bar: date
    baz: NDArray[float]


class ComplexTestModel(PydanticBaseModel):
    a: str = "foo"
    b: str = "bar"
    c: MyTestModel = MyTestModel(foo=5, bar=date(2020, 1, 1), baz=np.array([]))
    d: Dict[str, float] = {"x": 0.0, "y": 1.0}
    e: date = date(2022, 1, 1)


class TestCompositePublishers(TestCase):
    def setUp(self) -> None:
        self.cwd = Path.cwd()

    def tearDown(self) -> None:
        os.chdir(self.cwd)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_pydantic", sep="/")
            data = {"a": "foo", "b": {"c": "bar", "d": 10.0}}
            p.data = data
            self.assertIsInstance(p, PydanticBaseModel)
            self.assertEqual(p.data.dict(), data)

    def test_field_publishers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_{{param}}", name_params={"param": "pydantic"}, sep="/")
            p.models_as_dict = True
            p.options.exclude = {"e"}
            p.field_publishers["b"] = GenericFilePublisher(suffix=".txt")
            p.field_publishers["c"] = DictTemplateFilePublisher(suffix=".html", template="<html>{{foo}} on {{bar}}</html>")
            p.field_publishers["d"] = JSONPublisher(name="custom_d")
            p.field_publishers["e"] = GenericFilePublisher(name="", suffix=".txt")

            p.data = ComplexTestModel()
            paths = p()
            target = {
                "b": Path("test_pydantic/b.txt"),
                "c": Path("test_pydantic/c.html"),
                "d": Path("custom_d.json"),
            }
            self.assertDictEqual(paths, target)
            for path in paths.values():
                self.assertTrue(path.exists())

            # Test actual values to make sure empty files aren't written
            with open(paths["b"], "r") as f:
                self.assertEqual(f.read(), "bar")
            with open(paths["c"], "r") as f:
                self.assertEqual(f.read(), "<html>5 on 2020-01-01</html>")
            with open(paths["d"], "r") as f:
                self.assertEqual(f.read(), '{"x":0.0,"y":1.0}')

    def test_default_publishers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_{{param}}", name_params={"param": "pydantic"}, sep="/")
            p.models_as_dict = True
            p.options.exclude = {"c", "d"}
            p.field_publishers["b"] = GenericFilePublisher(name="test_pydantic/custom_b", suffix=".txt")
            p.default_publishers.append(DummyPublisher())
            p.default_publishers.append(GenericFilePublisher(suffix=".txt"))

            p.data = ComplexTestModel()
            paths = p()
            target = {
                "a": Path("test_pydantic/a.txt"),
                "b": Path("test_pydantic/custom_b.txt"),
                "e": Path("test_pydantic/e.txt"),
            }
            self.assertDictEqual(paths, target)
            for path in paths.values():
                self.assertTrue(path.exists())

            # Test actual values to make sure empty files aren't written
            with open(paths["a"], "r") as f:
                self.assertEqual(f.read(), "foo")
            with open(paths["b"], "r") as f:
                self.assertEqual(f.read(), "bar")
            with open(paths["e"], "r") as f:
                self.assertEqual(f.read(), "2022-01-01")

    def test_root_publisher(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_pydantic", sep="/")
            p.models_as_dict = True
            p.field_publishers["c"] = DictTemplateFilePublisher(suffix=".html", template="<html>{{foo}} on {{bar}}</html>")
            p.root_publisher = JSONPublisher(kwargs={"default": pydantic.json.pydantic_encoder})
            p.options.include = {"c"}
            p.data = ComplexTestModel()
            paths = p()
            target = {
                "c": Path("test_pydantic/c.html"),
            }
            self.assertDictEqual(paths, target)

    def test_skip_root_publisher(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_pydantic", sep="/")
            p.models_as_dict = True
            p.field_publishers["c"] = DictTemplateFilePublisher(suffix=".html", template="<html>{{foo}} on {{bar}}</html>")
            p.root_publisher = JSONPublisher(kwargs={"default": pydantic.json.pydantic_encoder})

            p.data = ComplexTestModel()
            paths = p()
            target = {
                "c": Path("test_pydantic/c.html"),
                "__root__": Path("test_pydantic.json"),
            }
            self.assertDictEqual(paths, target)
            for path in paths.values():
                self.assertTrue(path.exists())
            with open(paths["__root__"], "r") as f:
                self.assertEqual(
                    f.read(),
                    '{"a":"foo","b":"bar","d":{"x":0.0,"y":1.0},"e":"2022-01-01"}',
                )

    def test_dict(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_composite_dict", sep="/")
            p.field_publishers["b"] = GenericFilePublisher(name="", suffix=".txt")
            p.field_publishers["c"] = DictTemplateFilePublisher(name="", suffix=".html", template="<html>{{foo}} on {{bar}}</html>")
            p.field_publishers["d"] = JSONPublisher(name="custom_d")
            p.field_publishers["e"] = GenericFilePublisher(name="", suffix=".txt")

            p.data = ComplexTestModel().dict()
            paths = p()
            target = {
                "b": Path("test_composite_dict/b.txt"),
                "c": Path("test_composite_dict/c.html"),
                "d": Path("custom_d.json"),
                "e": Path("test_composite_dict/e.txt"),
            }
            self.assertDictEqual(paths, target)
            for path in paths.values():
                self.assertTrue(path.exists())

    def test_dict_recursive(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_recursive", sep="/")
            p.default_publishers.append(p)  # Recursive!! If it finds another dict, it will step into it
            p.default_publishers.append(GenericFilePublisher(name="", suffix=".txt"))

            p.data = {"a": "foo", "b": {"c": "bar", "d": "baz"}}
            paths = p()
            target = {
                "a": Path("test_recursive/a.txt"),
                "b": {
                    "c": Path("test_recursive/b/c.txt"),
                    "d": Path("test_recursive/b/d.txt"),
                },
            }
            self.assertDictEqual(paths, target)

            with open(paths["a"], "r") as f:
                self.assertEqual(f.read(), "foo")
            with open(paths["b"]["c"], "r") as f:
                self.assertEqual(f.read(), "bar")
            with open(paths["b"]["d"], "r") as f:
                self.assertEqual(f.read(), "baz")

    def test_no_publisher_found(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_pydantic", sep="/")
            p.models_as_dict = True
            p.data = ComplexTestModel()
            with self.assertLogs(level=logging.INFO) as captured:
                paths = p()
            self.assertEqual(paths, {})
            self.assertEqual(len(captured.records), 5)

    def test_publish_errors(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            p = CompositePublisher(name="test_pydantic", sep="/")
            p.models_as_dict = True
            p.field_publishers["c"] = JSONPublisher(name="")
            p.field_publishers["d"] = DictTemplateFilePublisher(name="", suffix=".html", template="<html>{{z}}</html>")
            p.default_publishers.append(GenericFilePublisher(name="generic", suffix=".txt"))
            # field 'c' is not JSON-serializable by orjson
            # specifically because orjson currently only supports
            # dict keys that are serializable without a default

            class InnerComplexTestModel(PydanticBaseModel):
                a: str = "foo"
                b: str = "bar"
                c: MyTestModel = MyTestModel(foo=5, bar=date(2020, 1, 1), baz=np.array([]))
                d: Dict[timedelta, float] = {timedelta(seconds=1): 0.0}
                e: date = date(2022, 1, 1)

            p.data = InnerComplexTestModel()
            self.assertRaises(Exception, p)

            # Test that other results are still written!
            paths = {
                "a": Path("test_pydantic/a.txt"),
                "b": Path("test_pydantic/b.txt"),
                "e": Path("test_pydantic/e.txt"),
            }
            for path in paths.values():
                self.assertTrue(path.exists())

            # Test actual values to make sure empty files aren't written
            with open(paths["a"], "r") as f:
                self.assertEqual(f.read(), "foo")
            with open(paths["b"], "r") as f:
                self.assertEqual(f.read(), "bar")
            with open(paths["e"], "r") as f:
                self.assertEqual(f.read(), "2022-01-01")
