from typing import Generic, List, Optional, TypeVar
from unittest import TestCase

from pydantic import ValidationError

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


class MyContext(ContextBase):
    a: str


class MyExtendedContext(MyContext):
    b: float
    c: bool


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


class BadModel1(CallableModel):
    @Flow.call
    def __call__(self, context):
        return None


class BadModel2(CallableModel):
    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        return None


class BadModel3(CallableModel):
    def __call__(self, context: MyContext) -> MyResult:
        return MyResult(x=1, y="foo")


class BadModel4(CallableModel):
    @Flow.call
    def __call__(self, custom_arg: MyContext) -> MyResult:
        return custom_arg


class BadModel5(CallableModel):
    @Flow.call
    def __call__(self, context: MyContext, context2: MyContext) -> MyResult:
        return context


class BadModel6(CallableModel):
    @property
    def context_type(self):
        return NullContext

    @property
    def result_type(self):
        return GenericResult

    @Flow.call
    def __call__(self, context: MyContext) -> MyResult:
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
        self.assertRaisesRegex(TypeError, error, BadModel1)

        error = "Context type declared in signature of __call__ must be a subclass of ContextBase. Received ~ContextType"
        self.assertRaisesRegex(TypeError, error, BadModel2)

        error = "__call__ function of CallableModel must be wrapped with the Flow.call decorator"
        self.assertRaisesRegex(ValueError, error, BadModel3)

        error = "__call__ method must take a single argument, named 'context'"
        self.assertRaisesRegex(ValueError, error, BadModel4)

        error = "__call__ method must take a single argument, named 'context'"
        self.assertRaisesRegex(ValueError, error, BadModel5)

        error = "The context_type <class 'ccflow.context.NullContext'> must match the type of the context accepted by __call__ <class 'ccflow.tests.test_callable.MyContext'>"
        self.assertRaisesRegex(ValueError, error, BadModel6)

    def test_identity(self):
        # Make sure that an "identity" mapping works
        ident = IdentityCallable()
        context = MyContext(a="foo")
        # Note that because contexts copy on validate, and the decorator does validation,
        # the return value is a copy of the input.
        self.assertEqual(ident(context), context)
        self.assertIsNot(ident(context), context)


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
        TContext = TypeVar("TContext", bound=ContextBase)
        TResult = TypeVar("TResult", bound=ResultBase)

        class MyCallableBase(CallableModelGenericType[TContext, TResult]):
            pass

        class MyCallableImpl(MyCallableBase[NullContext, GenericResult[int]]):
            pass

        class MyCallable(MyCallableImpl):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=42)

        m2 = MyCallable()
        self.assertEqual(m2.context_type, NullContext)
        self.assertEqual(m2.result_type, GenericResult[int])
        res2 = m2(NullContext())
        self.assertEqual(res2.value, 42)

    def test_align_annotation_and_context_class(self):
        TContext = TypeVar("TContext", bound=ContextBase)
        TResult = TypeVar("TResult", bound=ResultBase)

        class MyCallableBase(CallableModelGenericType[TContext, TResult]):
            pass

        class MyCallableImpl(MyCallableBase[NullContext, GenericResult[int]]):
            pass

        class MyNullContext(NullContext): ...

        class MyCallable(MyCallableImpl):
            @Flow.call
            def __call__(self, context: MyNullContext) -> GenericResult[int]:
                return GenericResult[int](value=42)

        m2 = MyCallable()
        self.assertEqual(m2.context_type, MyNullContext)
        self.assertEqual(m2.result_type, GenericResult[int])
        res2 = m2(NullContext())
        self.assertEqual(res2.value, 42)

    def test_use_as_base_class_mixed_annotations(self):
        class Base(CallableModelGenericType[ContextType, ResultType], Generic[ContextType, ResultType]): ...

        class Next(Base[ContextType, ResultType], Generic[ContextType, ResultType]): ...

        class Partial(Next[NullContext, ResultType], Generic[ResultType]): ...

        class Last(Partial[GenericResult[int]]):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=42)

        Last()

    def test_use_as_base_class_mixed_annotations_reversed(self):
        class Base(CallableModelGenericType[ContextType, ResultType], Generic[ContextType, ResultType]): ...

        class Next(Base[ContextType, ResultType], Generic[ContextType, ResultType]): ...

        class Partial(Next[ContextType, GenericResult[int]], Generic[ContextType]): ...

        class Last(Partial[NullContext]):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=42)

        Last()

    def test_use_as_base_class_conflict(self):
        class MyCallable(CallableModelGenericType[NullContext, GenericResult[int]]):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[float]:
                return GenericResult[float](value=42.0)

        with self.assertRaises(TypeError):
            MyCallable()


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
