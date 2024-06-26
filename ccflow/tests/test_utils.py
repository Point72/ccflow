import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from typing import Dict, List
from unittest import TestCase

from ccflow.utils import PydanticDictOptions, models_to_pandas


class SubModel(PydanticBaseModel):
    i: int = 1
    j: int = 2


class MyTestModel(PydanticBaseModel):
    a: str
    b: float
    c: List[str] = []
    d: Dict[str, float] = {}
    e: SubModel


class TestModelsToPandas(TestCase):
    def test_single(self):
        model = MyTestModel(a="foo", b=1.0, c=["x", "y"], d={"p": 0.2, "q": 0.8}, e=SubModel())
        df = models_to_pandas(model, sep=" ")
        target = pd.DataFrame.from_records(
            [
                {
                    "a": "foo",
                    "b": 1.0,
                    "c": ["x", "y"],
                    "d p": 0.2,
                    "d q": 0.8,
                    "e i": 1,
                    "e j": 2,
                }
            ]
        )
        pd.testing.assert_frame_equal(df, target)

    def test_multiple(self):
        model = MyTestModel(a="foo", b=1.0, c=["x", "y"], d={"p": 0.2, "q": 0.8}, e=SubModel())
        model2 = MyTestModel(a="bar", b=2.0, c=[], d={"p": 0.4, "q": 0.6}, e=SubModel(i=2, j=3))
        df = models_to_pandas([model, model2], sep=" ")
        target = pd.DataFrame.from_records(
            [
                {
                    "a": "foo",
                    "b": 1.0,
                    "c": ["x", "y"],
                    "d p": 0.2,
                    "d q": 0.8,
                    "e i": 1,
                    "e j": 2,
                },
                {
                    "a": "bar",
                    "b": 2.0,
                    "c": [],
                    "d p": 0.4,
                    "d q": 0.6,
                    "e i": 2,
                    "e j": 3,
                },
            ]
        )
        pd.testing.assert_frame_equal(df, target)

    def test_options(self):
        model = MyTestModel(a="foo", b=1.0, c=["x", "y"], d={"p": 0.2, "q": 0.8}, e=SubModel())
        model2 = MyTestModel(a="bar", b=2.0, c=[], d={"p": 0.4, "q": 0.6}, e=SubModel(i=2, j=3))
        options = PydanticDictOptions(exclude={"a", "b"}, exclude_defaults=True)

        df = models_to_pandas([model, model2], options, sep=" ")
        target = pd.DataFrame.from_records(
            [
                {"c": ["x", "y"], "d p": 0.2, "d q": 0.8},
                {"d p": 0.4, "d q": 0.6, "e i": 2, "e j": 3},
            ]
        )
        pd.testing.assert_frame_equal(df, target)
