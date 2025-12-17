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


class TestDynamicContext(TestCase):
    """Test the dynamic_context decorator and Flow.dynamic_call functionality."""

    def test_basic_dynamic_call(self):
        """Test basic usage of Flow.dynamic_call decorator."""

        class DynamicModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int, b: str = "default") -> GenericResult:
                return GenericResult(value={"a": a, "b": b})

        m = DynamicModel()

        # Test calling with kwargs
        result = m(a=42, b="test")
        self.assertEqual(result.value, {"a": 42, "b": "test"})

        # Test with default value
        result2 = m(a=100)
        self.assertEqual(result2.value, {"a": 100, "b": "default"})

    def test_context_type_from_dynamic_context(self):
        """Test that context_type is correctly derived from the dynamic context."""

        class DynamicModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, x: float, y: int = 0) -> GenericResult:
                return GenericResult(value=x + y)

        m = DynamicModel()

        # Check context_type
        ctx_type = m.context_type
        self.assertTrue(issubclass(ctx_type, ContextBase))
        self.assertIn("x", ctx_type.model_fields)
        self.assertIn("y", ctx_type.model_fields)

    def test_pass_context_object_directly(self):
        """Test passing a context object directly instead of kwargs."""

        class DynamicModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int, b: str) -> GenericResult:
                return GenericResult(value=f"{a}:{b}")

        m = DynamicModel()

        # Get the dynamic context class
        ctx_class = m.__call__.__dynamic_context__
        ctx = ctx_class(a=42, b="test")

        result = m(ctx)
        self.assertEqual(result.value, "42:test")

    def test_parent_context_class(self):
        """Test using a parent context class with additional shared fields."""

        class ParentContext(ContextBase):
            shared_field: str = "shared_default"

        class DynamicModel(CallableModel):
            @Flow.dynamic_call(parent=ParentContext)
            def __call__(self, *, value: int) -> GenericResult:
                return GenericResult(value=value)

        m = DynamicModel()

        # Check that the dynamic context inherits from ParentContext
        ctx_class = m.__call__.__dynamic_context__
        self.assertTrue(issubclass(ctx_class, ParentContext))
        self.assertIn("shared_field", ctx_class.model_fields)
        self.assertIn("value", ctx_class.model_fields)

    def test_multiple_dynamic_methods(self):
        """Test multiple methods with different dynamic contexts on the same model."""

        class MultiMethodModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int) -> GenericResult:
                return GenericResult(value=a)

            @Flow.dynamic_call
            def other_method(self, *, x: float, y: float) -> GenericResult:
                return GenericResult(value=x + y)

        m = MultiMethodModel()

        # Test __call__
        result1 = m(a=42)
        self.assertEqual(result1.value, 42)

        # Test other_method
        result2 = m.other_method(x=1.5, y=2.5)
        self.assertEqual(result2.value, 4.0)

        # Check that each method has its own context type
        call_ctx = m.__call__.__dynamic_context__
        other_ctx = m.other_method.__dynamic_context__
        self.assertIsNot(call_ctx, other_ctx)
        self.assertIn("a", call_ctx.model_fields)
        self.assertIn("x", other_ctx.model_fields)

    def test_mix_with_regular_flow_call(self):
        """Test mixing Flow.dynamic_call with regular Flow.call on the same model."""

        class MixedModel(CallableModel):
            @Flow.call
            def __call__(self, context: MyContext) -> MyResult:
                return MyResult(x=1, y=context.a)

            @Flow.dynamic_call
            def dynamic_method(self, *, value: float) -> GenericResult:
                return GenericResult(value=value * 2)

        m = MixedModel()

        # Test regular __call__
        result1 = m(a="hello")
        self.assertEqual(result1.y, "hello")

        # Test dynamic_method
        result2 = m.dynamic_method(value=3.14)
        self.assertEqual(result2.value, 6.28)

    def test_flow_options_with_dynamic_call(self):
        """Test that FlowOptions parameters work with Flow.dynamic_call."""
        import logging

        class OptionsModel(CallableModel):
            @Flow.dynamic_call(log_level=logging.WARNING, validate_result=False)
            def __call__(self, *, val: str) -> GenericResult:
                # Return a dict instead of GenericResult - should work with validate_result=False
                return {"value": val}

        m = OptionsModel()
        result = m(val="test")

        # With validate_result=False, should get the dict back
        self.assertEqual(result, {"value": "test"})
        self.assertIsInstance(result, dict)

    def test_error_missing_return_annotation(self):
        """Test that missing return annotation raises an error."""
        with self.assertRaises(ValueError) as ctx:

            class BadModel(CallableModel):
                @Flow.dynamic_call
                def __call__(self, *, a: int):
                    pass

        self.assertIn("return type annotation", str(ctx.exception))

    def test_error_missing_param_annotation(self):
        """Test that missing parameter annotation raises an error."""
        with self.assertRaises(ValueError) as ctx:

            class BadModel(CallableModel):
                @Flow.dynamic_call
                def __call__(self, *, a, b: str = "default") -> GenericResult:
                    pass

        self.assertIn("type annotation", str(ctx.exception))

    def test_error_kwargs_not_allowed(self):
        """Test that **kwargs parameter raises an error."""
        with self.assertRaises(ValueError) as ctx:

            class BadModel(CallableModel):
                @Flow.dynamic_call
                def __call__(self, *, a: int, **kwargs) -> GenericResult:
                    pass

        self.assertIn("kwargs", str(ctx.exception).lower())

    def test_error_args_not_allowed(self):
        """Test that *args parameter raises an error."""
        with self.assertRaises(ValueError) as ctx:

            class BadModel(CallableModel):
                @Flow.dynamic_call
                def __call__(self, *args, a: int) -> GenericResult:
                    pass

        self.assertIn("args", str(ctx.exception).lower())

    def test_error_missing_required_arg_at_call_time(self):
        """Test that missing required arguments at call time raises ValidationError."""

        class Model(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int, b: str) -> GenericResult:
                return GenericResult(value=a)

        m = Model()

        with self.assertRaises(ValidationError) as ctx:
            m(a=42)  # Missing 'b'

        self.assertIn("b", str(ctx.exception))
        self.assertIn("required", str(ctx.exception).lower())

    def test_dynamic_context_decorator_standalone(self):
        """Test the dynamic_context decorator can be used standalone."""

        @dynamic_context
        def my_func(self, *, a: int, b: str = "default") -> GenericResult:
            return GenericResult(value={"a": a, "b": b})

        # Check that __dynamic_context__ is set
        self.assertTrue(hasattr(my_func, "__dynamic_context__"))

        # Check that the context class has the right fields
        ctx_class = my_func.__dynamic_context__
        self.assertTrue(issubclass(ctx_class, ContextBase))
        self.assertIn("a", ctx_class.model_fields)
        self.assertIn("b", ctx_class.model_fields)

    def test_complex_types(self):
        """Test dynamic context with complex types like List, Optional, etc."""
        from typing import List, Optional

        class ComplexModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, items: List[int], name: Optional[str] = None) -> GenericResult:
                return GenericResult(value={"items": items, "name": name})

        m = ComplexModel()

        result = m(items=[1, 2, 3], name="test")
        self.assertEqual(result.value["items"], [1, 2, 3])
        self.assertEqual(result.value["name"], "test")

        result2 = m(items=[4, 5])
        self.assertEqual(result2.value["items"], [4, 5])
        self.assertIsNone(result2.value["name"])

    def test_error_parent_field_collision(self):
        """Test that parameters colliding with parent context fields raise an error."""

        class ParentContext(ContextBase):
            shared_field: str = "default"
            another_field: int = 0

        with self.assertRaises(ValueError) as ctx:

            class BadModel(CallableModel):
                @Flow.dynamic_call(parent=ParentContext)
                def __call__(self, *, shared_field: int) -> GenericResult:  # Collision with parent
                    return GenericResult(value=shared_field)

        self.assertIn("shared_field", str(ctx.exception))
        self.assertIn("collide", str(ctx.exception).lower())
        self.assertIn("ParentContext", str(ctx.exception))

    def test_result_type_attribute(self):
        """Test that __result_type__ is set on dynamic context functions."""

        class Model(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int) -> GenericResult:
                return GenericResult(value=a)

        m = Model()

        # Check __result_type__ is set
        self.assertTrue(hasattr(m.__call__, "__result_type__"))
        self.assertEqual(m.__call__.__result_type__, GenericResult)

    def test_context_name_uses_qualname(self):
        """Test that dynamic context class names use __qualname__ for better debuggability."""

        class MyModel(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, x: int) -> GenericResult:
                return GenericResult(value=x)

        m = MyModel()
        ctx_class = m.__call__.__dynamic_context__

        # The name should include the class name (from __qualname__)
        self.assertIn("MyModel", ctx_class.__name__)
        self.assertIn("__call__", ctx_class.__name__)
        self.assertIn("DynamicContext", ctx_class.__name__)

    def test_result_type_validation_with_dynamic_context(self):
        """Test that result type is validated even with dynamic context."""

        class Model(CallableModel):
            @Flow.dynamic_call
            def __call__(self, *, a: int) -> MyResult:
                # Return wrong type - should be caught by validation
                # MyResult requires x: int and y: str fields, so a string can't be coerced
                return "not a valid result"

        m = Model()

        # This should raise because we're returning a string that can't be coerced to MyResult
        # and validate_result defaults to True
        with self.assertRaises(ValidationError):
            m(a=42)

    def test_result_type_validation_can_be_disabled(self):
        """Test that result validation can be disabled with validate_result=False."""

        class Model(CallableModel):
            @Flow.dynamic_call(validate_result=False)
            def __call__(self, *, a: int) -> GenericResult:
                return {"value": a}  # Return dict instead of GenericResult

        m = Model()

        # With validate_result=False, this should work
        result = m(a=42)
        self.assertEqual(result, {"value": 42})
        self.assertIsInstance(result, dict)

    def test_dynamic_context_standalone_with_parent(self):
        """Test dynamic_context decorator standalone with parent parameter."""

        class ParentContext(ContextBase):
            shared: str = "default"

        @dynamic_context(parent=ParentContext)
        def my_method(self, *, value: int) -> GenericResult:
            return GenericResult(value=value)

        # Check inheritance
        ctx_class = my_method.__dynamic_context__
        self.assertTrue(issubclass(ctx_class, ParentContext))
        self.assertIn("shared", ctx_class.model_fields)
        self.assertIn("value", ctx_class.model_fields)

        # Check __result_type__ is set
        self.assertTrue(hasattr(my_method, "__result_type__"))
        self.assertEqual(my_method.__result_type__, GenericResult)
