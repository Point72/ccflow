from typing import Any, Dict, List
from unittest import TestCase, mock

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, ValidationError

from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, NullContext, PyObjectPath
from ccflow.local_persistence import LOCAL_ARTIFACTS_MODULE_NAME


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
    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow z = MyClass, even though there is no validator

    x: MyTestModel
    y: MyTestModel
    z: MyClass = MyClass()


class DoubleNestedModel(BaseModel):
    a: Dict[str, MyNestedModel] = {}
    b: List[MyTestModel] = []


class CopyModel(BaseModel):
    x: str


class GenericAliasAdjust(BaseModel):
    x: list[str]


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
        self.assertEqual(m.model_validate(m.model_dump(by_alias=False)), m)
        self.assertEqual(m.model_validate(m.model_dump(by_alias=True)), m)

        # Make sure parsing doesn't impact original dict (i.e. by popping element out of it)
        d = m.model_dump(by_alias=False)
        self.assertEqual(m.model_validate(d).model_dump(by_alias=False), d)
        d = m.model_dump(by_alias=True)
        self.assertEqual(m.model_validate(d).model_dump(by_alias=True), d)

    def test_coerce_from_other_type(self):
        # This test would have broken when we originally turned on extra field validation,
        # due to the fact that ModelA and ModelB are unrelated types, and so coercion attempts
        # to take dict(ModelA(x="foo")), which contains the "type_" field, which is considered as
        # an "extra" field due to the aliasing, which then causes a failure.
        # It was fixed by popping this out in the pre-root validation.
        self.assertEqual(ModelB.model_validate(ModelA(x="foo")), ModelB(x="foo"))

        self.assertRaises(ValidationError, ModelA.model_validate, ModelC(x="foo", y="bar"))

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

    def test_pyobjectpath_requires_ccflow_local_registration(self):
        class PlainLocalModel(PydanticBaseModel):
            value: int

        with self.assertRaises(ValueError):
            PyObjectPath.validate(PlainLocalModel)

        class LocalCcflowModel(BaseModel):
            value: int

        path = PyObjectPath.validate(LocalCcflowModel)
        self.assertEqual(path.object, LocalCcflowModel)
        self.assertTrue(str(path).startswith(f"{LOCAL_ARTIFACTS_MODULE_NAME}."))

    def test_validate(self):
        self.assertEqual(ModelA.model_validate({"x": "foo"}), ModelA(x="foo"))
        type_ = "ccflow.tests.test_base.ModelA"
        self.assertEqual(ModelA.model_validate({"_target_": type_, "x": "foo"}), ModelA(x="foo"))
        self.assertEqual(BaseModel.model_validate({"_target_": type_, "x": "foo"}), ModelA(x="foo"))
        self.assertEqual(BaseModel.model_validate(ModelA(x="foo")), ModelA(x="foo"))

    def test_deferred_build_serialization(self):
        # When defer_build was originally switched on, this didn't play nicely with SerializeAsAny
        # Because the Child model is not explicitly instantiated, the Parent model does not have a serialization schema for it

        # Define Child and Parent at module-level so that "type_" will validate properly
        global Child, Parent

        class Child(BaseModel):
            a: str

        class Parent(BaseModel):
            c: Child  # Note that under the hood, ccflow turns this into SerializeAsAny[Child] so that child classes of Child will be fully serialized

        p = Parent(c=dict(a=""))
        self.assertIsInstance(p.model_dump(), dict)

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


class TestLocalRegistrationKind(TestCase):
    def test_base_model_defaults_to_model_kind(self):
        with mock.patch("ccflow.base.register_local_subclass") as register:

            class LocalModel(BaseModel):
                value: int

        register.assert_called_once()
        args, kwargs = register.call_args
        self.assertIs(args[0], LocalModel)
        self.assertEqual(kwargs["kind"], "model")

    def test_context_defaults_to_context_kind(self):
        with mock.patch("ccflow.base.register_local_subclass") as register:

            class LocalContext(ContextBase):
                value: int

        register.assert_called_once()
        args, kwargs = register.call_args
        self.assertIs(args[0], LocalContext)
        self.assertEqual(kwargs["kind"], "context")

    def test_callable_defaults_to_callable_kind(self):
        with mock.patch("ccflow.base.register_local_subclass") as register:

            class LocalCallable(CallableModel):
                @Flow.call
                def __call__(self, context: NullContext) -> GenericResult:
                    return GenericResult(value="ok")

        register.assert_called_once()
        args, kwargs = register.call_args
        self.assertIs(args[0], LocalCallable)
        self.assertEqual(kwargs["kind"], "callable_model")

    def test_explicit_override_respected(self):
        with mock.patch("ccflow.base.register_local_subclass") as register:

            class CustomKind(BaseModel):
                __ccflow_local_registration_kind__ = "custom"
                value: int

        register.assert_called_once()
        args, kwargs = register.call_args
        self.assertIs(args[0], CustomKind)
        self.assertEqual(kwargs["kind"], "custom")
