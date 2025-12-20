import sys
from pickle import dumps as pdumps, loads as ploads
from typing import ClassVar, Generic, List, Optional, Tuple, Type, TypeVar, Union
from unittest import TestCase

import ray
from pydantic import ValidationError
from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

from ccflow import (
    CallableModel,
    CallableModelGenericType,
    ContextBase,
    ContextType,
    Flow,
    GenericResult,
    GraphDepList,
    MetaData,
    ModelRegistry,
    NullContext,
    ResultBase,
    ResultType,
    WrapperModel,
)
from ccflow.local_persistence import LOCAL_ARTIFACTS_MODULE_NAME
from ccflow.tests.local_helpers import build_local_callable, build_local_context


def _find_registered_name(module, cls):
    for name, value in vars(module).items():
        if value is cls:
            return name
    raise AssertionError(f"{cls} not found in {module.__name__}")


class MyContext(ContextBase):
    a: str


class MyExtendedContext(MyContext):
    b: float
    c: bool


class MyOtherContext(ContextBase):
    a: int


class ListContext(ContextBase):
    ll: List[str] = []


class MyResult(ResultBase):
    x: int
    y: str


class MyCallable(CallableModel):
    i: int
    ll: List[int] = []

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
        return MyResult(x=self.i, y=context.a)


class MyCallableOptionalContext(CallableModel):
    @Flow.call
    def __call__(self, context: Optional[MyContext] = None) -> MyResult:
        context = context or MyContext(a="default")
        return MyResult(x=1, y=context.a)


class MyCallableChild(MyCallable):
    pass


class MyCallableParent(CallableModel):
    my_callable: MyCallable

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
        return self.my_callable(context)


class MyCallableParent_basic(MyCallableParent):
    @Flow.deps
    def __deps__(self, context: MyContext) -> GraphDepList:
        return [(self.my_callable, [MyContext(a="goodbye")])]


class MyCallableParent_multi(MyCallableParent):
    @Flow.deps
    def __deps__(self, context: MyContext) -> GraphDepList:
        return [(self.my_callable, [MyContext(a="goodbye"), MyContext(a="hello")])]


class MyCallableParent_bad_deps_sig(MyCallableParent):
    @Flow.deps
    def __deps__(self, context: MyContext, val: int):
        return []


class MyCallableParent_bad_annotation(MyCallableParent):
    @Flow.deps
    def __deps__(self, context: ContextType):
        return []


class MyCallableParent_bad_context(MyCallableParent):
    @Flow.deps
    def __deps__(self, context: MyContext) -> GraphDepList:
        return [(self.my_callable, [ListContext(ll=[1, 2, 3])])]


class IdentityCallable(CallableModel):
    @Flow.call
    def __call__(self, context: MyContext) -> MyContext:
        return context


class BadModelNoContextNoResult(CallableModel):
    @Flow.call
    def __call__(self, context):
        return None


class BadModelNeedRealTypes(CallableModel):
    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        return None


class BadModelMissingFlowCallDecorator(CallableModel):
    """Model missing Flow.call decorator"""

    def __call__(self, context: MyContext) -> MyResult:
        return MyResult(x=1, y="foo")


class BadModelMissingFlowDepsDecorator(CallableModel):
    """Model missing Flow.deps decorator"""

    def __deps__(self, context: MyContext) -> GraphDepList:
        return []

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
        return MyResult(x=1, y="foo")


class BadModelMissingContextArg(CallableModel):
    @Flow.call
    def __call__(self, custom_arg: MyContext) -> MyResult:
        return custom_arg


class BadModelDoubleContextArg(CallableModel):
    @Flow.call
    def __call__(self, context: MyContext, context2: MyContext) -> MyResult:
        return context


class BadModelMismatchedContextAndCall(CallableModel):
    """Model with mismatched context_type and __call__ annotation"""

    @property
    def context_type(self):
        return MyOtherContext

    @property
    def result_type(self):
        return MyResult

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
        return context


class BadModelGenericMismatchedContextAndCall(CallableModelGenericType[MyOtherContext, MyResult]):
    """Model with mismatched context_type and __call__ annotation"""

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
        return context


class BadModelMismatchedResultAndCall(CallableModel):
    """Model with mismatched result_type and __call__ annotation"""

    @property
    def context_type(self):
        return NullContext

    @property
    def result_type(self):
        return GenericResult

    @Flow.call
    def __call__(self, context: NullContext) -> MyResult:
        return context


class BadModelGenericMismatchedResultAndCall(CallableModelGenericType[NullContext, GenericResult]):
    """Model with mismatched result_type and __call__ annotation"""

    @Flow.call
    def __call__(self, context: NullContext) -> MyResult:
        return context


class MyWrapper(WrapperModel[MyCallable]):
    """This wrapper model specifically takes a MyCallable instance for 'model'"""

    @Flow.call
    def __call__(self, context):
        return self.model(context)


class MyWrapperGeneric(WrapperModel):
    """This wrapper takes any callable model that takes a MyContext and returns a MyResult"""

    model: CallableModelGenericType[MyContext, MyResult]

    @Flow.call
    def __call__(self, context):
        return self.model(context)


class MyTest(ContextBase):
    c: MyContext


TContext = TypeVar("TContext", bound=ContextBase)
TResult = TypeVar("TResult", bound=ResultBase)


class MyCallableBase(CallableModelGenericType[TContext, TResult]):
    pass


class MyCallableImpl(MyCallableBase[NullContext, GenericResult[int]]):
    pass


class MyCallableFromGeneric(MyCallableImpl):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult[int]:
        return GenericResult[int](value=42)


class MyNullContext(NullContext): ...


class MyCallableFromGenericNullContext(MyCallableImpl):
    @Flow.call
    def __call__(self, context: MyNullContext) -> GenericResult[int]:
        return GenericResult[int](value=42)


class BaseGeneric(CallableModelGenericType[ContextType, ResultType], Generic[ContextType, ResultType]): ...


class NextGeneric(BaseGeneric[ContextType, ResultType], Generic[ContextType, ResultType]): ...


class PartialGeneric(NextGeneric[NullContext, ResultType], Generic[ResultType]): ...


class LastGeneric(PartialGeneric[GenericResult[int]]):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult[int]:
        return GenericResult[int](value=42)


class PartialGenericReversed(NextGeneric[ContextType, GenericResult[int]], Generic[ContextType]): ...


class LastGenericReversed(PartialGenericReversed[NullContext]):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult[int]:
        return GenericResult[int](value=42)


class AResult(ResultBase):
    a: int


class BResult(ResultBase):
    b: str


class UnionReturn(CallableModel):
    @property
    def result_type(self) -> Type[ResultType]:
        return AResult

    @Flow.call
    def __call__(self, context: NullContext) -> Union[AResult, BResult]:
        # Return one branch of the Union
        return AResult(a=1)


class BadModelUnionReturnNoProperty(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> Union[AResult, BResult]:
        # Return one branch of the Union
        return AResult(a=1)


class UnionReturnGeneric(CallableModelGenericType[NullContext, Union[AResult, BResult]]):
    @property
    def result_type(self) -> Type[ResultType]:
        return AResult

    @Flow.call
    def __call__(self, context: NullContext) -> Union[AResult, BResult]:
        # Return one branch of the Union
        return AResult(a=1)


class BadModelUnionReturnGeneric(CallableModelGenericType[NullContext, Union[AResult, BResult]]):
    @Flow.call
    def __call__(self, context: NullContext) -> Union[AResult, BResult]:
        # Return one branch of the Union
        return AResult(a=1)


class MyGenericContext(ContextBase, Generic[TContext]):
    value: TContext


class ModelMixedGenericsEnforceContextMatch(CallableModel, Generic[TContext, TResult]):
    model: CallableModelGenericType[TContext, TResult]

    @property
    def context_type(self) -> Type[ContextType]:
        return MyGenericContext[self.model.context_type]

    @property
    def result_type(self) -> Type[ResultType]:
        return GenericResult[self.model.result_type]

    @Flow.deps
    def __deps__(self, context: MyGenericContext[TContext]) -> List[Tuple[CallableModelGenericType[TContext, TResult], List[ContextType]]]:
        return []

    @Flow.call
    def __call__(self, context: MyGenericContext[TContext]) -> TResult:
        return GenericResult(value=None)


class TestContext(TestCase):
    def test_immutable(self):
        x = MyContext(a="foo")

        def f():
            x.a = "bar"

        self.assertRaises(Exception, f)  # v2 raises a ValidationError instead of a TypeError

    def test_hashable(self):
        x = MyContext(a="foo")
        self.assertTrue(hash(x))

    def test_copy_on_validate(self):
        ll = []
        c = ListContext(ll=ll)
        ll.append("foo")
        self.assertEqual(c.ll, [])
        c2 = ListContext.model_validate(c)
        c.ll.append("bar")
        # c2 context does not share the same list as c1 context
        self.assertEqual(c2.ll, [])

    def test_parse(self):
        out = MyContext.model_validate("foo")
        self.assertEqual(out, MyContext(a="foo"))

        out = MyExtendedContext.model_validate("foo,5,True")
        self.assertEqual(out, MyExtendedContext(a="foo", b=5, c=True))
        out2 = MyExtendedContext.model_validate(("foo", 5, True))
        self.assertEqual(out2, out)
        out3 = MyExtendedContext.model_validate(["foo", 5, True])
        self.assertEqual(out3, out)

    def test_registration(self):
        r = ModelRegistry.root()
        r.add("bar", MyContext(a="foo"))
        self.assertEqual(MyContext.model_validate("bar"), MyContext(a="foo"))
        self.assertEqual(MyContext.model_validate("baz"), MyContext(a="baz"))


class TestCallableModel(TestCase):
    def test_callable(self):
        m = MyCallable(i=5)
        self.assertEqual(m(MyContext(a="foo")), MyResult(x=5, y="foo"))
        self.assertEqual(m.context_type, MyContext)
        self.assertEqual(m.result_type, MyResult)
        out = m.model_dump(mode="python")
        self.assertIn("meta", out)
        self.assertIn("i", out)
        self.assertIn("type_", out)
        self.assertNotIn("context_type", out)

    def test_signature(self):
        m = MyCallable(i=5)
        context = MyContext(a="foo")
        target = m(context)
        self.assertEqual(m(context=context), m(context))
        # Validate from dict
        self.assertEqual(m(dict(a="foo")), target)
        self.assertEqual(m(context=dict(a="foo")), target)
        # Kwargs passed in
        self.assertEqual(m(a="foo"), target)
        # No argument
        self.assertRaises(TypeError, m)
        # context and kwargs
        self.assertRaises(TypeError, m, context, a="foo")
        self.assertRaises(TypeError, m, context=context, a="foo")

    def test_signature_optional_context(self):
        m = MyCallableOptionalContext()
        context = MyContext(a="foo")
        target = m(context)
        self.assertEqual(m(context=context), target)
        self.assertEqual(m().y, "default")

    def test_inheritance(self):
        m = MyCallableChild(i=5)
        self.assertEqual(m(MyContext(a="foo")), MyResult(x=5, y="foo"))
        self.assertEqual(m.context_type, MyContext)
        self.assertEqual(m.result_type, MyResult)
        out = m.model_dump(mode="python")
        self.assertIn("meta", out)
        self.assertIn("i", out)
        self.assertIn("type_", out)
        self.assertNotIn("context_type", out)

    def test_meta(self):
        m = MyCallable(i=1)
        self.assertEqual(m.meta, MetaData())
        md = MetaData(name="foo", description="My Foo")
        m = MyCallable(i=1, meta=md)
        self.assertEqual(m.meta, md)

    def test_copy_on_validate(self):
        ll = []
        m = MyCallable(i=5, ll=ll)
        ll.append("foo")
        # List is copied on construction
        self.assertEqual(m.ll, [])
        m2 = MyCallable.model_validate(m)
        m.ll.append("bar")
        # When m2 is validated, it still shares same list with m1
        self.assertEqual(m.ll, m2.ll)

    def test_types(self):
        error = "Must either define a type annotation for context on __call__ or implement 'context_type'"
        self.assertRaisesRegex(TypeError, error, BadModelNoContextNoResult)

        error = "Context type declared in signature of __call__ must be a subclass of ContextBase. Received ~ContextType"
        self.assertRaisesRegex(TypeError, error, BadModelNeedRealTypes)

        error = "__call__ function of CallableModel must be wrapped with the Flow.call decorator"
        self.assertRaisesRegex(ValueError, error, BadModelMissingFlowCallDecorator)

        error = "__deps__ function of CallableModel must be wrapped with the Flow.deps decorator"
        self.assertRaisesRegex(ValueError, error, BadModelMissingFlowDepsDecorator)

        error = "__call__ method must take a single argument, named 'context'"
        self.assertRaisesRegex(ValueError, error, BadModelMissingContextArg)

        error = "__call__ method must take a single argument, named 'context'"
        self.assertRaisesRegex(ValueError, error, BadModelDoubleContextArg)

        error = "The context_type <class 'ccflow.tests.test_callable.MyOtherContext'> must match the type of the context accepted by __call__ <class 'ccflow.tests.test_callable.MyContext'>"
        self.assertRaisesRegex(ValueError, error, BadModelMismatchedContextAndCall)

        error = "The result_type <class 'ccflow.result.generic.GenericResult'> must match the return type of __call__ <class 'ccflow.tests.test_callable.MyResult'>"
        self.assertRaisesRegex(ValueError, error, BadModelMismatchedResultAndCall)

        error = "Model __call__ signature result type cannot be a Union type without a concrete property. Please define a property 'result_type' on the model."
        self.assertRaisesRegex(TypeError, error, BadModelUnionReturnNoProperty)

    def test_identity(self):
        # Make sure that an "identity" mapping works
        ident = IdentityCallable()
        context = MyContext(a="foo")
        # Note that because contexts copy on validate, and the decorator does validation,
        # the return value is a copy of the input.
        self.assertEqual(ident(context), context)
        self.assertIsNot(ident(context), context)

    def test_context_call_match_enforcement_generic_base(self):
        # This should not raise
        _ = ModelMixedGenericsEnforceContextMatch(model=IdentityCallable())

    def test_union_return(self):
        m = UnionReturn()
        result = m(NullContext())
        self.assertIsInstance(result, AResult)
        self.assertEqual(result.a, 1)


class TestCallableModelRegistration(TestCase):
    def test_module_level_class_retains_module(self):
        self.assertEqual(MyCallable.__module__, __name__)
        dynamic_module = sys.modules.get(LOCAL_ARTIFACTS_MODULE_NAME)
        if dynamic_module:
            self.assertFalse(any(value is MyCallable for value in vars(dynamic_module).values()))

    def test_module_level_context_retains_module(self):
        self.assertEqual(MyContext.__module__, __name__)
        dynamic_module = sys.modules.get(LOCAL_ARTIFACTS_MODULE_NAME)
        if dynamic_module:
            self.assertFalse(any(value is MyContext for value in vars(dynamic_module).values()))

    def test_local_class_moves_under_dynamic_namespace(self):
        LocalCallable = build_local_callable()
        module_name = LocalCallable.__module__
        self.assertEqual(module_name, LOCAL_ARTIFACTS_MODULE_NAME)
        dynamic_module = sys.modules[module_name]
        self.assertIs(getattr(dynamic_module, LocalCallable.__qualname__), LocalCallable)
        self.assertIn("<locals>", getattr(LocalCallable, "__ccflow_dynamic_origin__"))
        result = LocalCallable()(NullContext())
        self.assertEqual(result.value, "local")

    def test_multiple_local_definitions_have_unique_identifiers(self):
        first = build_local_callable()
        second = build_local_callable()
        self.assertNotEqual(first.__qualname__, second.__qualname__)
        dynamic_module = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]
        self.assertIs(getattr(dynamic_module, first.__qualname__), first)
        self.assertIs(getattr(dynamic_module, second.__qualname__), second)

    def test_context_and_callable_same_name_do_not_collide(self):
        def build_conflicting():
            class LocalThing(ContextBase):
                value: int

            context_cls = LocalThing

            class LocalThing(CallableModel):
                @Flow.call
                def __call__(self, context: context_cls) -> GenericResult:
                    return GenericResult(value=context.value)

            callable_cls = LocalThing
            return context_cls, callable_cls

        LocalContext, LocalCallable = build_conflicting()
        locals_module = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]
        ctx_attr = _find_registered_name(locals_module, LocalContext)
        model_attr = _find_registered_name(locals_module, LocalCallable)
        self.assertTrue(ctx_attr.startswith("context__"))
        self.assertTrue(model_attr.startswith("callable_model__"))
        ctx_hint = ctx_attr.partition("__")[2].rsplit("__", 1)[0]
        model_hint = model_attr.partition("__")[2].rsplit("__", 1)[0]
        self.assertEqual(ctx_hint, model_hint)
        self.assertEqual(getattr(locals_module, ctx_attr), LocalContext)
        self.assertEqual(getattr(locals_module, model_attr), LocalCallable)
        self.assertNotEqual(ctx_attr, model_attr, "Kind-prefixed names keep contexts and callables distinct.")

    def test_local_callable_type_path_roundtrip(self):
        LocalCallable = build_local_callable()
        instance = LocalCallable()
        path = instance.type_
        self.assertEqual(path.object, LocalCallable)
        self.assertTrue(str(path).startswith(f"{LOCAL_ARTIFACTS_MODULE_NAME}."))

    def test_local_context_type_path_roundtrip(self):
        LocalContext = build_local_context()
        ctx = LocalContext(value=10)
        path = ctx.type_
        self.assertEqual(path.object, LocalContext)
        self.assertTrue(str(path).startswith(f"{LOCAL_ARTIFACTS_MODULE_NAME}."))

    def test_local_context_and_model_serialization_roundtrip(self):
        class LocalContext(ContextBase):
            value: int

        class LocalModel(CallableModel):
            factor: int = 2

            @Flow.call
            def __call__(self, context: LocalContext) -> GenericResult:
                return GenericResult(value=context.value * self.factor)

        instance = LocalModel(factor=5)
        context = LocalContext(value=7)
        result = instance(context)
        self.assertEqual(result.value, 35)
        serialized_model = instance.model_dump(mode="python")
        restored_model = LocalModel.model_validate(serialized_model)
        self.assertEqual(restored_model, instance)
        serialized_context = context.model_dump(mode="python")
        restored_context = LocalContext.model_validate(serialized_context)
        self.assertEqual(restored_context, context)
        restored_result = restored_model(restored_context)
        self.assertEqual(restored_result.value, 35)

    def test_multiple_nested_levels_unique_paths(self):
        created = []

        def layer(depth: int):
            class LocalContext(ContextBase):
                value: int

            class LocalModel(CallableModel):
                multiplier: int = depth + 1
                call_count: ClassVar[int] = 0

                @Flow.call
                def __call__(self, context: LocalContext) -> GenericResult:
                    type(self).call_count += 1
                    return GenericResult(value=context.value * self.multiplier)

            created.append((depth, LocalContext, LocalModel))

            if depth < 2:

                def inner():
                    layer(depth + 1)

                inner()

        def sibling_group():
            class LocalContext(ContextBase):
                value: int

            class LocalModel(CallableModel):
                @Flow.call
                def __call__(self, context: LocalContext) -> GenericResult:
                    return GenericResult(value=context.value)

            created.append(("sibling", LocalContext, LocalModel))

        layer(0)
        sibling_group()
        sibling_group()

        locals_module = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]

        context_names = {ctx.__qualname__ for _, ctx, _ in created}
        model_names = {model.__qualname__ for _, _, model in created}
        self.assertEqual(len(context_names), len(created))
        self.assertEqual(len(model_names), len(created))

        for label, ctx_cls, model_cls in created:
            self.assertIs(getattr(locals_module, ctx_cls.__qualname__), ctx_cls)
            self.assertIs(getattr(locals_module, model_cls.__qualname__), model_cls)
            self.assertIn("<locals>", getattr(ctx_cls, "__ccflow_dynamic_origin__"))
            self.assertIn("<locals>", getattr(model_cls, "__ccflow_dynamic_origin__"))
            ctx_instance = ctx_cls(value=4)
            result = model_cls()(ctx_instance)
            if isinstance(label, str):  # sibling group
                self.assertEqual(result.value, ctx_instance.value)
            else:
                self.assertEqual(result.value, ctx_instance.value * (label + 1))


class TestWrapperModel(TestCase):
    def test_wrapper(self):
        md = MetaData(name="foo", description="My Foo")
        m = MyCallable(i=1, meta=md)
        w = MyWrapper(model=m)
        self.assertEqual(w.context_type, m.context_type)
        self.assertEqual(w.result_type, m.result_type)

    def test_validate(self):
        data = {"model": {"i": 1, "meta": {"name": "bar"}}}
        w = MyWrapper.model_validate(data)
        self.assertIsInstance(w.model, MyCallable)
        self.assertEqual(w.model.i, 1)

        # Make sure this works if model is in the registry (and is just passed in as a string)
        # This has tripped up the validation in the past
        md = MetaData(name="foo", description="My Foo")
        m = MyCallable(i=1, meta=md)
        r = ModelRegistry.root().clear()
        r.add("foo", m)
        data = {"model": "foo"}
        w = MyWrapper.model_validate(data)
        self.assertEqual(w.model, m)


class TestCallableModelGenericType(TestCase):
    def test_wrapper(self):
        m = MyCallable(i=1)
        w = MyWrapperGeneric(model=m)
        self.assertEqual(w.context_type, m.context_type)
        self.assertEqual(w.result_type, m.result_type)

    def test_wrapper_bad(self):
        m = IdentityCallable()
        self.assertRaises(ValueError, MyWrapperGeneric, model=m)

    def test_wrapper_reference(self):
        m = MyCallable(i=1)
        r = ModelRegistry.root().clear()
        r.add("foo", m)
        w = MyWrapperGeneric(model="foo")
        self.assertEqual(w.model, m)
        self.assertEqual(w.context_type, m.context_type)
        self.assertEqual(w.result_type, m.result_type)

    def test_use_as_base_class(self):
        class MyCallable(CallableModelGenericType[NullContext, GenericResult[int]]):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=42)

        m = MyCallable()
        self.assertEqual(m.context_type, NullContext)
        self.assertEqual(m.result_type, GenericResult[int])
        self.assertEqual(m(NullContext()).value, 42)

    def test_use_as_base_class_no_call_annotations(self):
        class MyCallable(CallableModelGenericType[NullContext, GenericResult[int]]):
            @Flow.call
            def __call__(self, context):
                return GenericResult[int](value=42)

        m = MyCallable()
        self.assertEqual(m.context_type, NullContext)
        self.assertEqual(m.result_type, GenericResult[int])
        self.assertEqual(m(NullContext()).value, 42)

    def test_use_as_base_class_inheritance(self):
        m2 = MyCallableFromGeneric()
        self.assertEqual(m2.context_type, NullContext)
        self.assertEqual(m2.result_type, GenericResult[int])
        res2 = m2(NullContext())
        self.assertEqual(res2.value, 42)

        # test pickling
        for dump, load in [(pdumps, ploads), (rcpdumps, rcploads)]:
            dumped = dump(m2, protocol=5)
            m3 = load(dumped)
            self.assertEqual(m3.context_type, NullContext)
            self.assertEqual(m3.result_type, GenericResult[int])
            res3 = m3(NullContext())
            self.assertEqual(res3.value, 42)

    def test_align_annotation_and_context_class(self):
        m2 = MyCallableFromGenericNullContext()
        self.assertEqual(m2.context_type, MyNullContext)
        self.assertEqual(m2.result_type, GenericResult[int])
        res2 = m2(NullContext())
        self.assertEqual(res2.value, 42)

        # test pickling
        for dump, load in [(pdumps, ploads), (rcpdumps, rcploads)]:
            dumped = dump(m2, protocol=5)
            m3 = load(dumped)
            self.assertEqual(m3.context_type, MyNullContext)
            self.assertEqual(m3.result_type, GenericResult[int])
            res3 = m3(NullContext())
            self.assertEqual(res3.value, 42)

    def test_use_as_base_class_mixed_annotations(self):
        m = LastGeneric()

        # test pickling
        for dump, load in [(pdumps, ploads), (rcpdumps, rcploads)]:
            dumped = dump(m, protocol=5)
            m2 = load(dumped)
            self.assertEqual(m2.context_type, NullContext)
            self.assertEqual(m2.result_type, GenericResult[int])
            res2 = m2(NullContext())
            self.assertEqual(res2.value, 42)

        # test ray
        @ray.remote
        def calc(x) -> int:
            return x(NullContext()).value

        with ray.init(num_cpus=1):
            res = ray.get(calc.remote(m))

        self.assertEqual(res, 42)

    def test_use_as_base_class_mixed_annotations_reversed(self):
        m = LastGenericReversed()

        # test pickling
        for dump, load in [(pdumps, ploads), (rcpdumps, rcploads)]:
            dumped = dump(m, protocol=5)
            m2 = load(dumped)
            self.assertEqual(m2.context_type, NullContext)
            self.assertEqual(m2.result_type, GenericResult[int])
            res2 = m2(NullContext())
            self.assertEqual(res2.value, 42)

        # test ray
        @ray.remote
        def calc(x) -> int:
            return x(NullContext()).value

        with ray.init(num_cpus=1):
            res = ray.get(calc.remote(m))

        self.assertEqual(res, 42)

    def test_use_as_base_class_conflict(self):
        class MyCallable(CallableModelGenericType[NullContext, GenericResult[int]]):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[float]:
                return GenericResult[float](value=42.0)

        with self.assertRaises(TypeError):
            MyCallable()

    def test_types_generic(self):
        error = "Context type annotation <class 'ccflow.tests.test_callable.MyContext'> on __call__ does not match context_type <class 'ccflow.tests.test_callable.MyOtherContext'> defined by CallableModelGenericType"
        self.assertRaisesRegex(TypeError, error, BadModelGenericMismatchedContextAndCall)

        error = "Return type annotation <class 'ccflow.tests.test_callable.MyResult'> on __call__ does not match result_type <class 'ccflow.result.generic.GenericResult'> defined by CallableModelGenericType"
        self.assertRaisesRegex(TypeError, error, BadModelGenericMismatchedResultAndCall)

        error = "Model __call__ signature result type cannot be a Union type without a concrete property. Please define a property 'result_type' on the model."
        self.assertRaisesRegex(TypeError, error, BadModelUnionReturnGeneric)

    def test_union_return_generic(self):
        m = UnionReturnGeneric()
        result = m(NullContext())
        self.assertIsInstance(result, AResult)
        self.assertEqual(result.a, 1)

    def test_generic_validates_assignment(self):
        class MyCallable(CallableModelGenericType[NullContext, GenericResult[int]]):
            x: int = 1

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                self.x = 5
                assert self.x == 5
                return GenericResult[float](value=self.x)

        m = MyCallable()
        self.assertEqual(m(NullContext()).value, 5)


class TestCallableModelDeps(TestCase):
    def test_basic(self):
        m = MyCallable(i=1)
        n = MyCallableParent_basic(my_callable=m)
        context = MyContext(a="hello")

        self.assertEqual(m.__deps__(context), [])
        self.assertEqual(n.__deps__(context), [(m, [MyContext(a="goodbye")])])

        result = n(context)
        self.assertEqual(result, MyResult(x=1, y="hello"))

    def test_multiple(self):
        m = MyCallable(i=1)
        n = MyCallableParent_multi(my_callable=m)
        context = MyContext(a="hello")

        self.assertEqual(m.__deps__(context), [])
        self.assertEqual(
            n.__deps__(context),
            [(m, [MyContext(a="goodbye"), MyContext(a="hello")])],
        )

        result = n(context)
        self.assertEqual(result, MyResult(x=1, y="hello"))

    def test_empty(self):
        m = MyCallable(i=1)
        n = MyCallableParent(my_callable=m)
        context = MyContext(a="hello")

        self.assertEqual(m.__deps__(context), [])
        self.assertEqual(n.__deps__(context), [])

        result = n(context)
        self.assertEqual(result, MyResult(x=1, y="hello"))

    def test_dep_context_validation(self):
        m = MyCallable(i=1)
        n = MyCallableParent_bad_context(my_callable=m)
        context = NullContext()  # Wrong context type
        with self.assertRaises(ValueError) as e:
            n.__deps__(context)

        self.assertIn("validation error for MyContext", str(e.exception))

    def test_bad_deps_sig(self):
        m = MyCallable(i=1)
        with self.assertRaises(ValueError) as e:
            MyCallableParent_bad_deps_sig(my_callable=m)

        msg = e.exception.errors()[0]["msg"]
        self.assertEqual(
            msg,
            "Value error, __deps__ method must take a single argument, named 'context'",
        )

    def test_bad_annotation(self):
        m = MyCallable(i=1)
        with self.assertRaises(ValidationError) as e:
            MyCallableParent_bad_annotation(my_callable=m)

        msg = e.exception.errors()[0]["msg"]
        target = "Value error, The type of the context accepted by __deps__ ~ContextType must match that accepted by __call__ <class 'ccflow.tests.test_callable.MyContext'>"

        self.assertEqual(msg, target)

    def test_bad_decorator(self):
        """Test that we can't apply Flow.deps to a function other than __deps__."""
        with self.assertRaises(ValueError):

            class MyCallableParent_bad_decorator(MyCallableParent):
                @Flow.deps
                def foo(self, context):
                    return []
