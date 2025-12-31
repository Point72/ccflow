from pickle import dumps as pdumps, loads as ploads
from typing import Generic, List, Optional, Tuple, Type, TypeVar, Union
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
    dynamic_context,
)
from ccflow.local_persistence import LOCAL_ARTIFACTS_MODULE_NAME


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
    """Smoke test verifying CallableModel inherits registration from BaseModel.

    NOTE: Registration behavior is thoroughly tested at the BaseModel level in
    test_local_persistence.py. This single test verifies inheritance works.
    """

    def test_local_callable_smoke_test(self):
        """Verify that local CallableModel classes inherit registration from BaseModel."""

        class LocalContext(ContextBase):
            value: int

        class LocalCallable(CallableModel):
            @Flow.call
            def __call__(self, context: LocalContext) -> GenericResult:
                return GenericResult(value=context.value * 2)

        # Basic registration should work (inherits from BaseModel)
        self.assertIn("<locals>", LocalCallable.__qualname__)
        self.assertTrue(hasattr(LocalCallable, "__ccflow_import_path__"))
        self.assertTrue(LocalCallable.__ccflow_import_path__.startswith(LOCAL_ARTIFACTS_MODULE_NAME))

        # type_ should work
        instance = LocalCallable()
        self.assertEqual(instance.type_.object, LocalCallable)

        # Callable should execute correctly
        result = instance(LocalContext(value=21))
        self.assertEqual(result.value, 42)


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


# =============================================================================
# Tests for dynamic_context decorator
# =============================================================================


class TestDynamicContext(TestCase):
    """Tests for the @dynamic_context decorator."""

    def test_basic_usage_with_kwargs(self):
        """Test basic dynamic_context usage with keyword arguments."""

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, x: int, y: str = "default") -> GenericResult:
                return GenericResult(value=f"{x}-{y}")

        model = DynamicCallable()

        # Call with kwargs
        result = model(x=42, y="hello")
        self.assertEqual(result.value, "42-hello")

        # Call with default
        result = model(x=10)
        self.assertEqual(result.value, "10-default")

    def test_dynamic_context_attribute(self):
        """Test that __dynamic_context__ attribute is set."""

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, a: int, b: str) -> GenericResult:
                return GenericResult(value=f"{a}-{b}")

        # The __call__ method should have __dynamic_context__
        call_method = DynamicCallable.__call__
        self.assertTrue(hasattr(call_method, "__wrapped__"))
        # Access the inner function's __dynamic_context__
        inner = call_method.__wrapped__
        self.assertTrue(hasattr(inner, "__dynamic_context__"))

        dyn_ctx = inner.__dynamic_context__
        self.assertTrue(issubclass(dyn_ctx, ContextBase))
        self.assertIn("a", dyn_ctx.model_fields)
        self.assertIn("b", dyn_ctx.model_fields)

    def test_dynamic_context_is_registered(self):
        """Test that the dynamic context is registered for serialization."""

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, value: int) -> GenericResult:
                return GenericResult(value=value)

        inner = DynamicCallable.__call__.__wrapped__
        dyn_ctx = inner.__dynamic_context__

        # Should have __ccflow_import_path__ set
        self.assertTrue(hasattr(dyn_ctx, "__ccflow_import_path__"))
        self.assertTrue(dyn_ctx.__ccflow_import_path__.startswith(LOCAL_ARTIFACTS_MODULE_NAME))

    def test_call_with_context_object(self):
        """Test calling with a context object instead of kwargs."""

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, x: int, y: str = "default") -> GenericResult:
                return GenericResult(value=f"{x}-{y}")

        model = DynamicCallable()

        # Get the dynamic context class
        dyn_ctx = DynamicCallable.__call__.__wrapped__.__dynamic_context__

        # Create a context object
        ctx = dyn_ctx(x=99, y="context")
        result = model(ctx)
        self.assertEqual(result.value, "99-context")

    def test_with_parent_context(self):
        """Test dynamic_context with parent context class."""

        class ParentContext(ContextBase):
            base_value: str = "base"

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context(parent=ParentContext)
            def __call__(self, *, x: int, base_value: str) -> GenericResult:
                return GenericResult(value=f"{x}-{base_value}")

        # Get dynamic context
        dyn_ctx = DynamicCallable.__call__.__wrapped__.__dynamic_context__

        # Should inherit from ParentContext
        self.assertTrue(issubclass(dyn_ctx, ParentContext))

        # Should have both fields
        self.assertIn("base_value", dyn_ctx.model_fields)
        self.assertIn("x", dyn_ctx.model_fields)

        # Create context with parent field
        ctx = dyn_ctx(x=42, base_value="custom")
        self.assertEqual(ctx.base_value, "custom")
        self.assertEqual(ctx.x, 42)

    def test_parent_fields_must_be_in_signature(self):
        """Test that parent fields must be included in function signature."""

        class ParentContext(ContextBase):
            required_field: str

        with self.assertRaises(TypeError) as cm:

            class DynamicCallable(CallableModel):
                @Flow.call
                @dynamic_context(parent=ParentContext)
                def __call__(self, *, x: int) -> GenericResult:
                    return GenericResult(value=x)

        self.assertIn("required_field", str(cm.exception))

    def test_cloudpickle_roundtrip(self):
        """Test cloudpickle roundtrip for dynamic context callable."""

        class DynamicCallable(CallableModel):
            multiplier: int = 2

            @Flow.call
            @dynamic_context
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x * self.multiplier)

        model = DynamicCallable(multiplier=3)

        # Test roundtrip
        restored = rcploads(rcpdumps(model))

        result = restored(x=10)
        self.assertEqual(result.value, 30)

    def test_ray_task_execution(self):
        """Test dynamic context callable in Ray task."""

        class DynamicCallable(CallableModel):
            factor: int = 2

            @Flow.call
            @dynamic_context
            def __call__(self, *, x: int, y: int = 1) -> GenericResult:
                return GenericResult(value=(x + y) * self.factor)

        @ray.remote
        def run_callable(model, **kwargs):
            return model(**kwargs).value

        model = DynamicCallable(factor=5)

        with ray.init(num_cpus=1):
            result = ray.get(run_callable.remote(model, x=10, y=2))

        self.assertEqual(result, 60)  # (10 + 2) * 5

    def test_multiple_dynamic_context_methods(self):
        """Test callable with multiple dynamic_context decorated methods."""

        class MultiMethodCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, a: int) -> GenericResult:
                return GenericResult(value=a)

            @dynamic_context
            def other_method(self, *, b: str, c: float = 1.0) -> GenericResult:
                return GenericResult(value=f"{b}-{c}")

        model = MultiMethodCallable()

        # Test __call__
        result1 = model(a=42)
        self.assertEqual(result1.value, 42)

        # Test other_method (without Flow.call, just the dynamic_context wrapper)
        # Need to create the context manually
        other_ctx = model.other_method.__dynamic_context__
        ctx = other_ctx(b="hello", c=2.5)
        result2 = model.other_method(ctx)
        self.assertEqual(result2.value, "hello-2.5")

    def test_context_type_property_works(self):
        """Test that type_ property works on the dynamic context."""

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x)

        dyn_ctx = DynamicCallable.__call__.__wrapped__.__dynamic_context__
        ctx = dyn_ctx(x=42)

        # type_ should work and be importable
        type_path = str(ctx.type_)
        self.assertIn("_Local_", type_path)
        self.assertEqual(ctx.type_.object, dyn_ctx)

    def test_complex_field_types(self):
        """Test dynamic_context with complex field types."""
        from typing import List, Optional

        class DynamicCallable(CallableModel):
            @Flow.call
            @dynamic_context
            def __call__(
                self,
                *,
                items: List[int],
                name: Optional[str] = None,
                count: int = 0,
            ) -> GenericResult:
                total = sum(items) + count
                return GenericResult(value=f"{name}:{total}" if name else str(total))

        model = DynamicCallable()

        result = model(items=[1, 2, 3], name="test", count=10)
        self.assertEqual(result.value, "test:16")

        result = model(items=[5, 5])
        self.assertEqual(result.value, "10")


class TestFlowDynamicCall(TestCase):
    """Tests for @Flow.dynamic_call decorator."""

    def test_basic_usage(self):
        """Test basic @Flow.dynamic_call usage."""

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, x: int, y: str = "default") -> GenericResult:
                return GenericResult(value=f"{x}-{y}")

        model = DynamicCallable()

        result = model(x=42, y="hello")
        self.assertEqual(result.value, "42-hello")

        result = model(x=10)
        self.assertEqual(result.value, "10-default")

    def test_dynamic_context_attributes_preserved(self):
        """Test that __dynamic_context__ and __result_type__ are directly accessible."""

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x)

        # Should be directly accessible without traversing __wrapped__ chain
        method = DynamicCallable.__call__
        self.assertTrue(hasattr(method, "__dynamic_context__"))
        self.assertTrue(hasattr(method, "__result_type__"))
        self.assertTrue(issubclass(method.__dynamic_context__, ContextBase))
        self.assertEqual(method.__result_type__, GenericResult)

    def test_model_result_type_property(self):
        """Test that model.result_type returns correct type for dynamic contexts."""

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x)

        model = DynamicCallable()
        self.assertEqual(model.result_type, GenericResult)

    def test_with_parent_context(self):
        """Test @Flow.dynamic_call with parent context."""

        class ParentContext(ContextBase):
            base_value: str = "base"

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call(parent=ParentContext)
            def __call__(self, *, x: int, base_value: str) -> GenericResult:
                return GenericResult(value=f"{x}-{base_value}")

        model = DynamicCallable()

        # Get dynamic context by traversing __wrapped__ chain
        dyn_ctx = _find_dynamic_context(DynamicCallable.__call__)

        # Should inherit from ParentContext
        self.assertTrue(issubclass(dyn_ctx, ParentContext))

        # Call should work, uses parent default
        result = model(x=42, base_value="custom")
        self.assertEqual(result.value, "42-custom")

    def test_with_flow_options(self):
        """Test @Flow.dynamic_call with FlowOptions parameters."""

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call(validate_result=False)
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x)

        model = DynamicCallable()
        result = model(x=42)
        self.assertEqual(result.value, 42)

    def test_cloudpickle_roundtrip(self):
        """Test cloudpickle roundtrip with @Flow.dynamic_call."""

        class DynamicCallable(CallableModel):
            multiplier: int = 2

            @Flow.dynamic_call
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x * self.multiplier)

        model = DynamicCallable(multiplier=3)
        restored = rcploads(rcpdumps(model))

        result = restored(x=10)
        self.assertEqual(result.value, 30)

    def test_ray_task(self):
        """Test @Flow.dynamic_call in Ray task."""

        class DynamicCallable(CallableModel):
            factor: int = 2

            @Flow.dynamic_call
            def __call__(self, *, x: int, y: int = 1) -> GenericResult:
                return GenericResult(value=(x + y) * self.factor)

        @ray.remote
        def run_callable(model, **kwargs):
            return model(**kwargs).value

        model = DynamicCallable(factor=5)

        with ray.init(num_cpus=1):
            result = ray.get(run_callable.remote(model, x=10, y=2))

        self.assertEqual(result, 60)

    def test_dynamic_context_is_registered(self):
        """Test that the dynamic context from @Flow.dynamic_call is registered."""

        class DynamicCallable(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, value: int) -> GenericResult:
                return GenericResult(value=value)

        # Find dynamic context by traversing __wrapped__ chain
        dyn_ctx = _find_dynamic_context(DynamicCallable.__call__)

        self.assertTrue(hasattr(dyn_ctx, "__ccflow_import_path__"))
        self.assertTrue(dyn_ctx.__ccflow_import_path__.startswith(LOCAL_ARTIFACTS_MODULE_NAME))


def _find_dynamic_context(func):
    """Helper to find __dynamic_context__ by traversing the __wrapped__ chain."""
    visited = set()
    current = func
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "__dynamic_context__"):
            return current.__dynamic_context__
        current = getattr(current, "__wrapped__", None)
    return None
