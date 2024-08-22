from typing import Any, Dict, List
from unittest import TestCase

import pydantic
import pytest
from pydantic import ValidationError

from ccflow import BaseModel, PyObjectPath


class ModelA(BaseModel):
    x: str


class ModelB(BaseModel):
    x: str


class ModelC(BaseModel):
    x: Any
    y: str = None


class MyTestModel(BaseModel):
    a: str
    b: float
    c: List[str] = []
    d: Dict[str, float] = {}


class MyTestModelSubclass(MyTestModel):
    pass


class MyClass:
    def __init__(self, p="p", q=10.0):
        self.p = p
        self.q = q


class MyNestedModel(BaseModel):
    x: MyTestModel
    y: MyTestModel
    z: MyClass = MyClass()

    class Config:
        arbitrary_types_allowed = True  # To allow z = MyClass, even though there is no validator


class DoubleNestedModel(BaseModel):
    a: Dict[str, MyNestedModel] = {}
    b: List[MyTestModel] = []


class CopyModel(BaseModel):
    x: str

    class Config:
        copy_on_validate = "deep"


class TestBaseModel(TestCase):
    def test_extra_fields(self):
        self.assertRaises(ValidationError, MyTestModel, a="foo", b=0.0, extra=None)

    def test_validate_assignment(self):
        m = MyTestModel(a="foo", b=0.0)

        def f():
            m.b = "bar"

        self.assertRaises(ValidationError, f)

        # There is one other odd pydantic bug where successful validation would create a new attribute
        # for _target_ due to confusion about the aliases
        m.b = 1.0
        self.assertFalse(hasattr(m, "_target_"))

    def test_dict(self):
        m = MyTestModel(a="foo", b=0.0)
        self.assertEqual(m.parse_obj(m.dict(by_alias=False)), m)
        self.assertEqual(m.parse_obj(m.dict(by_alias=True)), m)

        # Make sure parsing doesn't impact original dict (i.e. by popping element out of it)
        d = m.dict(by_alias=False)
        self.assertEqual(m.parse_obj(d).dict(by_alias=False), d)
        d = m.dict(by_alias=True)
        self.assertEqual(m.parse_obj(d).dict(by_alias=True), d)

    def test_coerce_from_other_type(self):
        # This test would have broken when we originally turned on extra field validation,
        # due to the fact that ModelA and ModelB are unrelated types, and so coercion attempts
        # to take dict(ModelA(x="foo")), which contains the "type_" field, which is considered as
        # an "extra" field due to the aliasing, which then causes a failure.
        # It was fixed by popping this out in the pre-root validation.
        self.assertEqual(ModelB.validate(ModelA(x="foo")), ModelB(x="foo"))

        self.assertRaises(ValidationError, ModelA.validate, ModelC(x="foo", y="bar"))

    def test_type_after_assignment(self):
        # This test catches an original bug where the type_ validator didn't originally return
        # an object of type PyObjectPath
        m = ModelA(x="foo")
        path = "ccflow.tests.test_base.ModelA"
        self.assertIsInstance(m.type_, PyObjectPath)
        self.assertEqual(m.type_, path)
        m.x = "bar"
        self.assertIsInstance(m.type_, PyObjectPath)
        self.assertEqual(m.type_, path)

    @pytest.mark.skipif(
        pydantic.__version__.startswith("2"),
        reason="copy_on_validate only relevant in v1",
    )
    def test_copy_on_validate(self):
        # Make sure that this functionality works (and isn't broken by our validation function)
        m = CopyModel(x="foo")
        self.assertEqual(CopyModel.validate(m), m)
        self.assertIsNot(CopyModel.validate(m), m)

    def test_validate(self):
        self.assertEqual(ModelA.validate({"x": "foo"}), ModelA(x="foo"))
        type_ = "ccflow.tests.test_base.ModelA"
        self.assertEqual(ModelA.validate({"_target_": type_, "x": "foo"}), ModelA(x="foo"))
        self.assertEqual(BaseModel.validate({"_target_": type_, "x": "foo"}), ModelA(x="foo"))
        self.assertEqual(BaseModel.validate(ModelA(x="foo")), ModelA(x="foo"))

    def test_widget(self):
        obj = object()
        m = ModelC(x=obj)
        d1 = m.get_widget(
            json_kwargs={"by_alias": False},
        ).data
        self.assertEqual(
            d1,
            {
                "type_": "ccflow.tests.test_base.ModelC",
                "x": str(obj),
                "y": None,
            },
        )
        d2 = m.get_widget(
            json_kwargs={"exclude_none": True, "by_alias": False},
            widget_kwargs={"expanded": True, "root": "bar"},
        )._data_and_metadata()
        self.assertEqual(
            d2,
            (
                {
                    "type_": "ccflow.tests.test_base.ModelC",
                    "x": str(obj),
                },
                {"expanded": True, "root": "bar"},
            ),
        )
