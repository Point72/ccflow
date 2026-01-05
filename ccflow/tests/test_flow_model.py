"""Tests for Flow.model decorator."""

from datetime import date, timedelta
from typing import Annotated
from unittest import TestCase

from pydantic import ValidationError
from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

from ccflow import (
    CallableModel,
    ContextBase,
    DateRangeContext,
    Dep,
    DepOf,
    Flow,
    GenericResult,
    ModelRegistry,
    ResultBase,
)


class SimpleContext(ContextBase):
    """Simple context for testing."""

    value: int


class ExtendedContext(ContextBase):
    """Extended context with multiple fields."""

    x: int
    y: str = "default"


class MyResult(ResultBase):
    """Custom result type for testing."""

    data: str


# =============================================================================
# Basic Flow.model Tests
# =============================================================================


class TestFlowModelBasic(TestCase):
    """Basic Flow.model functionality tests."""

    def test_simple_model_explicit_context(self):
        """Test Flow.model with explicit context parameter."""

        @Flow.model
        def simple_loader(context: SimpleContext, multiplier: int) -> GenericResult[int]:
            return GenericResult(value=context.value * multiplier)

        # Create model instance
        loader = simple_loader(multiplier=3)

        # Should be a CallableModel
        self.assertIsInstance(loader, CallableModel)

        # Execute
        ctx = SimpleContext(value=10)
        result = loader(ctx)

        self.assertIsInstance(result, GenericResult)
        self.assertEqual(result.value, 30)

    def test_model_with_default_params(self):
        """Test Flow.model with default parameter values."""

        @Flow.model
        def loader_with_defaults(context: SimpleContext, multiplier: int = 2, prefix: str = "result") -> GenericResult[str]:
            return GenericResult(value=f"{prefix}:{context.value * multiplier}")

        # Create with defaults
        loader = loader_with_defaults()
        result = loader(SimpleContext(value=5))
        self.assertEqual(result.value, "result:10")

        # Create with custom values
        loader2 = loader_with_defaults(multiplier=3, prefix="custom")
        result2 = loader2(SimpleContext(value=5))
        self.assertEqual(result2.value, "custom:15")

    def test_model_context_type_property(self):
        """Test that generated model has correct context_type."""

        @Flow.model
        def typed_model(context: ExtendedContext, factor: int) -> GenericResult[int]:
            return GenericResult(value=context.x * factor)

        model = typed_model(factor=2)
        self.assertEqual(model.context_type, ExtendedContext)

    def test_model_result_type_property(self):
        """Test that generated model has correct result_type."""

        @Flow.model
        def custom_result_model(context: SimpleContext) -> MyResult:
            return MyResult(data=f"value={context.value}")

        model = custom_result_model()
        self.assertEqual(model.result_type, MyResult)

    def test_model_with_no_extra_params(self):
        """Test Flow.model with only context parameter."""

        @Flow.model
        def identity_model(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        model = identity_model()
        result = model(SimpleContext(value=42))
        self.assertEqual(result.value, 42)

    def test_model_with_flow_options(self):
        """Test Flow.model with Flow.call options."""

        @Flow.model(cacheable=True, validate_result=True)
        def cached_model(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=value + context.value)

        model = cached_model(value=10)
        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 15)

    def test_model_with_underscore_context(self):
        """Test Flow.model with '_' as context parameter (unused context convention)."""

        @Flow.model
        def loader(context: SimpleContext, base: int) -> GenericResult[int]:
            return GenericResult(value=context.value + base)

        @Flow.model
        def consumer(_: SimpleContext, data: DepOf[..., GenericResult[int]]) -> GenericResult[int]:
            # Context not used directly, just passed to dependency
            return GenericResult(value=data.value * 2)

        load = loader(base=100)
        consume = consumer(data=load)

        result = consume(SimpleContext(value=10))
        # loader: 10 + 100 = 110, consumer: 110 * 2 = 220
        self.assertEqual(result.value, 220)

        # Verify context_type is still correct
        self.assertEqual(consume.context_type, SimpleContext)


# =============================================================================
# context_args Mode Tests
# =============================================================================


class TestFlowModelContextArgs(TestCase):
    """Tests for Flow.model with context_args (unpacked context)."""

    def test_context_args_basic(self):
        """Test basic context_args usage."""

        @Flow.model(context_args=["start_date", "end_date"])
        def date_range_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
            return GenericResult(value=f"{source}:{start_date} to {end_date}")

        loader = date_range_loader(source="db")

        # Should use DateRangeContext
        self.assertEqual(loader.context_type, DateRangeContext)

        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        result = loader(ctx)
        self.assertEqual(result.value, "db:2024-01-01 to 2024-01-31")

    def test_context_args_custom_context(self):
        """Test context_args with custom context type."""

        @Flow.model(context_args=["x", "y"])
        def unpacked_model(x: int, y: str, multiplier: int = 1) -> GenericResult[str]:
            return GenericResult(value=f"{y}:{x * multiplier}")

        model = unpacked_model(multiplier=2)

        # Create context with generated type
        ctx_type = model.context_type
        ctx = ctx_type(x=5, y="test")

        result = model(ctx)
        self.assertEqual(result.value, "test:10")

    def test_context_args_with_defaults(self):
        """Test context_args where context fields have defaults."""

        @Flow.model(context_args=["value"])
        def model_with_ctx_default(value: int = 42, extra: str = "foo") -> GenericResult[str]:
            return GenericResult(value=f"{extra}:{value}")

        model = model_with_ctx_default()

        # Create context - the generated context should allow default
        ctx_type = model.context_type
        ctx = ctx_type(value=100)

        result = model(ctx)
        self.assertEqual(result.value, "foo:100")


# =============================================================================
# Dependency Tests
# =============================================================================


class TestFlowModelDependencies(TestCase):
    """Tests for Flow.model with dependencies."""

    def test_simple_dependency_with_depof(self):
        """Test simple dependency using DepOf shorthand."""

        @Flow.model
        def loader(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=value + context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
            multiplier: int = 1,
        ) -> GenericResult[int]:
            return GenericResult(value=data.value * multiplier)

        # Create pipeline
        load = loader(value=10)
        consume = consumer(data=load, multiplier=2)

        ctx = SimpleContext(value=5)
        result = consume(ctx)

        # loader returns 10 + 5 = 15, consumer multiplies by 2 = 30
        self.assertEqual(result.value, 30)

    def test_dependency_with_explicit_dep(self):
        """Test dependency using explicit Dep() annotation."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 2)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Annotated[GenericResult[int], Dep()],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value + 100)

        load = loader()
        consume = consumer(data=load)

        result = consume(SimpleContext(value=10))
        # loader: 10 * 2 = 20, consumer: 20 + 100 = 120
        self.assertEqual(result.value, 120)

    def test_dependency_with_direct_value(self):
        """Test that Dep fields can accept direct values (not CallableModel)."""

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value + context.value)

        # Pass direct value instead of CallableModel
        consume = consumer(data=GenericResult(value=100))

        result = consume(SimpleContext(value=5))
        self.assertEqual(result.value, 105)

    def test_deps_method_generation(self):
        """Test that __deps__ method is correctly generated."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value)

        load = loader()
        consume = consumer(data=load)

        ctx = SimpleContext(value=10)
        deps = consume.__deps__(ctx)

        # Should have one dependency
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0][0], load)
        self.assertEqual(deps[0][1], [ctx])

    def test_no_deps_when_direct_value(self):
        """Test that __deps__ returns empty when direct values used."""

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value)

        consume = consumer(data=GenericResult(value=100))

        deps = consume.__deps__(SimpleContext(value=10))
        self.assertEqual(len(deps), 0)


# =============================================================================
# Transform Tests
# =============================================================================


class TestFlowModelTransforms(TestCase):
    """Tests for Flow.model with context transforms."""

    def test_transform_in_dep(self):
        """Test dependency with context transform."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Annotated[
                GenericResult[int],
                Dep(transform=lambda ctx: ctx.model_copy(update={"value": ctx.value + 10})),
            ],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value * 2)

        load = loader()
        consume = consumer(data=load)

        ctx = SimpleContext(value=5)
        result = consume(ctx)

        # Transform adds 10 to context.value: 5 + 10 = 15
        # Loader returns that: 15
        # Consumer multiplies by 2: 30
        self.assertEqual(result.value, 30)

    def test_transform_in_deps_method(self):
        """Test that transform is applied in __deps__ method."""

        def transform_fn(ctx):
            return ctx.model_copy(update={"value": ctx.value * 3})

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Annotated[GenericResult[int], Dep(transform=transform_fn)],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value)

        load = loader()
        consume = consumer(data=load)

        ctx = SimpleContext(value=7)
        deps = consume.__deps__(ctx)

        # Transform should be applied
        self.assertEqual(len(deps), 1)
        transformed_ctx = deps[0][1][0]
        self.assertEqual(transformed_ctx.value, 21)  # 7 * 3

    def test_date_range_transform(self):
        """Test transform pattern with date ranges using context_args."""

        @Flow.model(context_args=["start_date", "end_date"])
        def range_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
            return GenericResult(value=f"{source}:{start_date}")

        def lookback_transform(ctx: DateRangeContext) -> DateRangeContext:
            return ctx.model_copy(update={"start_date": ctx.start_date - timedelta(days=1)})

        @Flow.model(context_args=["start_date", "end_date"])
        def range_processor(
            start_date: date,
            end_date: date,
            data: Annotated[GenericResult[str], Dep(transform=lookback_transform)],
        ) -> GenericResult[str]:
            return GenericResult(value=f"processed:{data.value}")

        loader = range_loader(source="db")
        processor = range_processor(data=loader)

        ctx = DateRangeContext(start_date=date(2024, 1, 10), end_date=date(2024, 1, 31))
        result = processor(ctx)

        # Transform should shift start_date back by 1 day
        self.assertEqual(result.value, "processed:db:2024-01-09")


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestFlowModelPipeline(TestCase):
    """Tests for multi-stage pipelines with Flow.model."""

    def test_three_stage_pipeline(self):
        """Test a three-stage computation pipeline."""

        @Flow.model
        def stage1(context: SimpleContext, base: int) -> GenericResult[int]:
            return GenericResult(value=context.value + base)

        @Flow.model
        def stage2(
            context: SimpleContext,
            input_data: DepOf[..., GenericResult[int]],
            multiplier: int,
        ) -> GenericResult[int]:
            return GenericResult(value=input_data.value * multiplier)

        @Flow.model
        def stage3(
            context: SimpleContext,
            input_data: DepOf[..., GenericResult[int]],
            offset: int = 0,
        ) -> GenericResult[int]:
            return GenericResult(value=input_data.value + offset)

        # Build pipeline
        s1 = stage1(base=100)
        s2 = stage2(input_data=s1, multiplier=2)
        s3 = stage3(input_data=s2, offset=50)

        ctx = SimpleContext(value=10)
        result = s3(ctx)

        # s1: 10 + 100 = 110
        # s2: 110 * 2 = 220
        # s3: 220 + 50 = 270
        self.assertEqual(result.value, 270)

    def test_diamond_dependency_pattern(self):
        """Test diamond-shaped dependency pattern."""

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def branch_a(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value * 2)

        @Flow.model
        def branch_b(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value + 100)

        @Flow.model
        def merger(
            context: SimpleContext,
            a: DepOf[..., GenericResult[int]],
            b: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=a.value + b.value)

        src = source()
        a = branch_a(data=src)
        b = branch_b(data=src)
        merge = merger(a=a, b=b)

        ctx = SimpleContext(value=10)
        result = merge(ctx)

        # source: 10
        # branch_a: 10 * 2 = 20
        # branch_b: 10 + 100 = 110
        # merger: 20 + 110 = 130
        self.assertEqual(result.value, 130)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFlowModelIntegration(TestCase):
    """Integration tests for Flow.model with ccflow infrastructure."""

    def test_registry_integration(self):
        """Test that Flow.model models work with ModelRegistry."""

        @Flow.model
        def registrable_model(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=context.value + value)

        model = registrable_model(value=100)

        registry = ModelRegistry.root().clear()
        registry.add("test_model", model)

        retrieved = registry["test_model"]
        self.assertEqual(retrieved, model)

        result = retrieved(SimpleContext(value=10))
        self.assertEqual(result.value, 110)

    def test_serialization_dump(self):
        """Test that generated models can be serialized."""

        @Flow.model
        def serializable_model(context: SimpleContext, value: int = 42) -> GenericResult[int]:
            return GenericResult(value=value)

        model = serializable_model(value=100)
        dumped = model.model_dump(mode="python")

        self.assertIn("value", dumped)
        self.assertEqual(dumped["value"], 100)
        self.assertIn("type_", dumped)

    def test_pickle_roundtrip(self):
        """Test cloudpickle serialization of generated models."""

        @Flow.model
        def pickleable_model(context: SimpleContext, factor: int) -> GenericResult[int]:
            return GenericResult(value=context.value * factor)

        model = pickleable_model(factor=3)

        # Cloudpickle roundtrip (standard pickle won't work for local classes)
        pickled = rcpdumps(model, protocol=5)
        restored = rcploads(pickled)

        result = restored(SimpleContext(value=10))
        self.assertEqual(result.value, 30)

    def test_mix_with_manual_callable_model(self):
        """Test mixing Flow.model with manually defined CallableModel."""

        class ManualModel(CallableModel):
            offset: int

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value + self.offset)

        @Flow.model
        def generated_consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
            multiplier: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data.value * multiplier)

        manual = ManualModel(offset=50)
        generated = generated_consumer(data=manual, multiplier=2)

        result = generated(SimpleContext(value=10))
        # manual: 10 + 50 = 60
        # generated: 60 * 2 = 120
        self.assertEqual(result.value, 120)


# =============================================================================
# Error Case Tests
# =============================================================================


class TestFlowModelErrors(TestCase):
    """Error case tests for Flow.model."""

    def test_missing_return_type(self):
        """Test error when return type annotation is missing."""
        with self.assertRaises(TypeError) as cm:

            @Flow.model
            def no_return(context: SimpleContext):
                return GenericResult(value=1)

        self.assertIn("return type annotation", str(cm.exception))

    def test_non_result_return_type(self):
        """Test error when return type is not ResultBase subclass."""
        with self.assertRaises(TypeError) as cm:

            @Flow.model
            def bad_return(context: SimpleContext) -> int:
                return 42

        self.assertIn("ResultBase", str(cm.exception))

    def test_missing_context_and_context_args(self):
        """Test error when neither context param nor context_args provided."""
        with self.assertRaises(TypeError) as cm:

            @Flow.model
            def no_context(value: int) -> GenericResult[int]:
                return GenericResult(value=value)

        self.assertIn("context", str(cm.exception))

    def test_invalid_context_arg(self):
        """Test error when context_args refers to non-existent parameter."""
        with self.assertRaises(ValueError) as cm:

            @Flow.model(context_args=["nonexistent"])
            def bad_context_args(x: int) -> GenericResult[int]:
                return GenericResult(value=x)

        self.assertIn("nonexistent", str(cm.exception))

    def test_context_arg_without_annotation(self):
        """Test error when context_arg parameter lacks type annotation."""
        with self.assertRaises(ValueError) as cm:

            @Flow.model(context_args=["x"])
            def untyped_context_arg(x) -> GenericResult[int]:
                return GenericResult(value=x)

        self.assertIn("type annotation", str(cm.exception))


# =============================================================================
# Dep and DepOf Tests
# =============================================================================


class TestDepAndDepOf(TestCase):
    """Tests for Dep and DepOf classes."""

    def test_depof_creates_annotated(self):
        """Test that DepOf[..., T] creates Annotated[Union[T, CallableModel], Dep()]."""
        from typing import Union as TypingUnion, get_args, get_origin

        annotation = DepOf[..., GenericResult[int]]
        self.assertEqual(get_origin(annotation), Annotated)

        args = get_args(annotation)
        # First arg is Union[ResultType, CallableModel]
        self.assertEqual(get_origin(args[0]), TypingUnion)
        union_args = get_args(args[0])
        self.assertIn(GenericResult[int], union_args)
        self.assertIn(CallableModel, union_args)
        # Second arg is Dep()
        self.assertIsInstance(args[1], Dep)
        self.assertIsNone(args[1].context_type)  # ... means inherit from parent

    def test_depof_with_generic_type(self):
        """Test DepOf with nested generic types."""
        from typing import List as TypingList, Union as TypingUnion, get_args, get_origin

        annotation = DepOf[..., GenericResult[TypingList[str]]]
        self.assertEqual(get_origin(annotation), Annotated)

        args = get_args(annotation)
        # First arg is Union[ResultType, CallableModel]
        self.assertEqual(get_origin(args[0]), TypingUnion)
        union_args = get_args(args[0])
        self.assertIn(GenericResult[TypingList[str]], union_args)
        self.assertIn(CallableModel, union_args)

    def test_depof_with_context_type(self):
        """Test DepOf[ContextType, ResultType] syntax."""
        from typing import Union as TypingUnion, get_args, get_origin

        annotation = DepOf[SimpleContext, GenericResult[int]]
        self.assertEqual(get_origin(annotation), Annotated)

        args = get_args(annotation)
        # First arg is Union[ResultType, CallableModel]
        self.assertEqual(get_origin(args[0]), TypingUnion)
        union_args = get_args(args[0])
        self.assertIn(GenericResult[int], union_args)
        self.assertIn(CallableModel, union_args)
        # Second arg is Dep with context_type
        self.assertIsInstance(args[1], Dep)
        self.assertEqual(args[1].context_type, SimpleContext)

    def test_extract_dep_with_annotated(self):
        """Test extract_dep with Annotated type."""
        from ccflow.dep import extract_dep

        dep = Dep(context_type=SimpleContext)
        annotation = Annotated[GenericResult[int], dep]

        base_type, extracted_dep = extract_dep(annotation)
        self.assertEqual(base_type, GenericResult[int])
        self.assertEqual(extracted_dep, dep)

    def test_extract_dep_with_depof(self):
        """Test extract_dep with DepOf type."""
        from typing import Union as TypingUnion, get_args, get_origin

        from ccflow.dep import extract_dep

        annotation = DepOf[..., GenericResult[str]]
        base_type, extracted_dep = extract_dep(annotation)

        # base_type is Union[ResultType, CallableModel]
        self.assertEqual(get_origin(base_type), TypingUnion)
        union_args = get_args(base_type)
        self.assertIn(GenericResult[str], union_args)
        self.assertIn(CallableModel, union_args)
        self.assertIsInstance(extracted_dep, Dep)

    def test_extract_dep_without_dep(self):
        """Test extract_dep with regular type (no Dep)."""
        from ccflow.dep import extract_dep

        base_type, extracted_dep = extract_dep(int)
        self.assertEqual(base_type, int)
        self.assertIsNone(extracted_dep)

    def test_extract_dep_annotated_without_dep(self):
        """Test extract_dep with Annotated but no Dep marker."""
        from ccflow.dep import extract_dep

        annotation = Annotated[int, "some metadata"]
        base_type, extracted_dep = extract_dep(annotation)

        # When no Dep marker is found, returns original annotation unchanged
        self.assertEqual(base_type, annotation)
        self.assertIsNone(extracted_dep)

    def test_is_compatible_type_simple(self):
        """Test _is_compatible_type with simple types."""
        from ccflow.dep import _is_compatible_type

        self.assertTrue(_is_compatible_type(int, int))
        self.assertFalse(_is_compatible_type(int, str))
        self.assertTrue(_is_compatible_type(bool, int))  # bool is subclass of int

    def test_is_compatible_type_generic(self):
        """Test _is_compatible_type with generic types."""
        from ccflow.dep import _is_compatible_type

        self.assertTrue(_is_compatible_type(GenericResult[int], GenericResult[int]))
        self.assertFalse(_is_compatible_type(GenericResult[int], GenericResult[str]))
        self.assertTrue(_is_compatible_type(GenericResult, GenericResult))

    def test_is_compatible_type_none(self):
        """Test _is_compatible_type with None."""
        from ccflow.dep import _is_compatible_type

        self.assertTrue(_is_compatible_type(None, None))
        self.assertFalse(_is_compatible_type(None, int))
        self.assertFalse(_is_compatible_type(int, None))

    def test_is_compatible_type_subclass(self):
        """Test _is_compatible_type with subclasses."""
        from ccflow.dep import _is_compatible_type

        self.assertTrue(_is_compatible_type(MyResult, ResultBase))
        self.assertFalse(_is_compatible_type(ResultBase, MyResult))

    def test_dep_validate_dependency_success(self):
        """Test Dep.validate_dependency with valid dependency."""

        @Flow.model
        def valid_dep(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        dep = Dep()
        model = valid_dep()

        # Should not raise
        dep.validate_dependency(model, GenericResult[int], SimpleContext, "data")

    def test_dep_validate_dependency_context_mismatch(self):
        """Test Dep.validate_dependency with context type mismatch."""

        class OtherContext(ContextBase):
            other: str

        @Flow.model
        def other_dep(context: OtherContext) -> GenericResult[int]:
            return GenericResult(value=42)

        dep = Dep(context_type=SimpleContext)
        model = other_dep()

        with self.assertRaises(TypeError) as cm:
            dep.validate_dependency(model, GenericResult[int], SimpleContext, "data")

        self.assertIn("context_type", str(cm.exception))

    def test_dep_validate_dependency_result_mismatch(self):
        """Test Dep.validate_dependency with result type mismatch."""

        @Flow.model
        def wrong_result(context: SimpleContext) -> MyResult:
            return MyResult(data="test")

        dep = Dep()
        model = wrong_result()

        with self.assertRaises(TypeError) as cm:
            dep.validate_dependency(model, GenericResult[int], SimpleContext, "data")

        self.assertIn("result_type", str(cm.exception))

    def test_dep_validate_dependency_non_callable(self):
        """Test Dep.validate_dependency with non-CallableModel value."""
        dep = Dep()
        # Should not raise for non-CallableModel values
        dep.validate_dependency(GenericResult(value=42), GenericResult[int], SimpleContext, "data")
        dep.validate_dependency("string", GenericResult[int], SimpleContext, "data")
        dep.validate_dependency(123, GenericResult[int], SimpleContext, "data")

    def test_dep_hash(self):
        """Test Dep is hashable for use in sets/dicts."""
        dep1 = Dep()
        dep2 = Dep(context_type=SimpleContext)

        # Should be hashable
        dep_set = {dep1, dep2}
        self.assertEqual(len(dep_set), 2)

        dep_dict = {dep1: "value1", dep2: "value2"}
        self.assertEqual(dep_dict[dep1], "value1")
        self.assertEqual(dep_dict[dep2], "value2")

    def test_dep_apply_with_transform(self):
        """Test Dep.apply with transform function."""

        def transform(ctx):
            return ctx.model_copy(update={"value": ctx.value * 2})

        dep = Dep(transform=transform)

        ctx = SimpleContext(value=10)
        result = dep.apply(ctx)

        self.assertEqual(result.value, 20)

    def test_dep_apply_without_transform(self):
        """Test Dep.apply without transform (identity)."""
        dep = Dep()

        ctx = SimpleContext(value=10)
        result = dep.apply(ctx)

        self.assertIs(result, ctx)

    def test_dep_repr(self):
        """Test Dep string representation."""
        dep1 = Dep()
        self.assertEqual(repr(dep1), "Dep()")

        dep2 = Dep(context_type=SimpleContext)
        self.assertIn("SimpleContext", repr(dep2))

        dep3 = Dep(transform=lambda x: x)
        self.assertIn("transform=", repr(dep3))

    def test_dep_equality(self):
        """Test Dep equality comparison."""
        dep1 = Dep()
        dep2 = Dep()
        dep3 = Dep(context_type=SimpleContext)

        # Note: Two Dep() instances with no arguments are equal
        self.assertEqual(dep1, dep2)
        self.assertNotEqual(dep1, dep3)


# =============================================================================
# Validation Tests
# =============================================================================


class TestFlowModelValidation(TestCase):
    """Tests for dependency validation in Flow.model."""

    def test_context_type_validation(self):
        """Test that context_type mismatch is detected."""

        class OtherContext(ContextBase):
            other: str

        @Flow.model
        def simple_loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def other_loader(context: OtherContext) -> GenericResult[int]:
            return GenericResult(value=42)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Annotated[GenericResult[int], Dep(context_type=SimpleContext)],
        ) -> GenericResult[int]:
            return GenericResult(value=data.value)

        # Should work with matching context
        load1 = simple_loader()
        consume1 = consumer(data=load1)
        self.assertIsNotNone(consume1)

        # Should fail with mismatched context
        load2 = other_loader()
        with self.assertRaises((TypeError, ValidationError)):
            consumer(data=load2)


# =============================================================================
# Hydra Integration Tests
# =============================================================================


# Define Flow.model functions at module level for Hydra to find them
@Flow.model
def hydra_basic_model(context: SimpleContext, value: int, name: str = "default") -> GenericResult[str]:
    """Module-level model for Hydra testing."""
    return GenericResult(value=f"{name}:{context.value + value}")


# --- Additional module-level fixtures for Hydra YAML tests ---


@Flow.model
def basic_loader(context: SimpleContext, source: str, multiplier: int = 1) -> GenericResult[int]:
    """Basic loader that multiplies context value by multiplier."""
    return GenericResult(value=context.value * multiplier)


@Flow.model
def string_processor(context: SimpleContext, prefix: str, suffix: str = "") -> GenericResult[str]:
    """Process context value into a string with prefix and suffix."""
    return GenericResult(value=f"{prefix}{context.value}{suffix}")


@Flow.model
def data_source(context: SimpleContext, base_value: int) -> GenericResult[int]:
    """Source that provides base data."""
    return GenericResult(value=context.value + base_value)


@Flow.model
def data_transformer(
    context: SimpleContext,
    source: DepOf[..., GenericResult[int]],
    factor: int = 2,
) -> GenericResult[int]:
    """Transform data by multiplying with factor."""
    return GenericResult(value=source.value * factor)


@Flow.model
def data_aggregator(
    context: SimpleContext,
    input_a: DepOf[..., GenericResult[int]],
    input_b: DepOf[..., GenericResult[int]],
    operation: str = "add",
) -> GenericResult[int]:
    """Aggregate two inputs."""
    if operation == "add":
        return GenericResult(value=input_a.value + input_b.value)
    elif operation == "multiply":
        return GenericResult(value=input_a.value * input_b.value)
    else:
        return GenericResult(value=input_a.value - input_b.value)


@Flow.model
def pipeline_stage1(context: SimpleContext, initial: int) -> GenericResult[int]:
    """First stage of pipeline."""
    return GenericResult(value=context.value + initial)


@Flow.model
def pipeline_stage2(
    context: SimpleContext,
    stage1_output: DepOf[..., GenericResult[int]],
    multiplier: int = 2,
) -> GenericResult[int]:
    """Second stage of pipeline."""
    return GenericResult(value=stage1_output.value * multiplier)


@Flow.model
def pipeline_stage3(
    context: SimpleContext,
    stage2_output: DepOf[..., GenericResult[int]],
    offset: int = 0,
) -> GenericResult[int]:
    """Third stage of pipeline."""
    return GenericResult(value=stage2_output.value + offset)


def lookback_one_day(ctx: DateRangeContext) -> DateRangeContext:
    """Transform that extends start_date back by one day."""
    return ctx.model_copy(update={"start_date": ctx.start_date - timedelta(days=1)})


@Flow.model
def date_range_loader(
    context: DateRangeContext,
    source: str,
    include_weekends: bool = True,
) -> GenericResult[str]:
    """Load data for a date range."""
    return GenericResult(value=f"{source}:{context.start_date} to {context.end_date}")


@Flow.model
def date_range_processor(
    context: DateRangeContext,
    raw_data: Annotated[GenericResult[str], Dep(transform=lookback_one_day)],
    normalize: bool = False,
) -> GenericResult[str]:
    """Process date range data with lookback."""
    prefix = "normalized:" if normalize else "raw:"
    return GenericResult(value=f"{prefix}{raw_data.value}")


@Flow.model
def hydra_default_model(context: SimpleContext, value: int = 42) -> GenericResult[int]:
    """Module-level model with defaults for Hydra testing."""
    return GenericResult(value=context.value + value)


@Flow.model
def hydra_source_model(context: SimpleContext, base: int) -> GenericResult[int]:
    """Source model for dependency testing."""
    return GenericResult(value=context.value * base)


@Flow.model
def hydra_consumer_model(
    context: SimpleContext,
    source: DepOf[..., GenericResult[int]],
    factor: int = 1,
) -> GenericResult[int]:
    """Consumer model for dependency testing."""
    return GenericResult(value=source.value * factor)


# --- context_args fixtures for Hydra testing ---


@Flow.model(context_args=["start_date", "end_date"])
def context_args_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
    """Loader using context_args with DateRangeContext."""
    return GenericResult(value=f"{source}:{start_date} to {end_date}")


@Flow.model(context_args=["start_date", "end_date"])
def context_args_processor(
    start_date: date,
    end_date: date,
    data: DepOf[..., GenericResult[str]],
    prefix: str = "processed",
) -> GenericResult[str]:
    """Processor using context_args with dependency."""
    return GenericResult(value=f"{prefix}:{data.value}")


class TestFlowModelHydra(TestCase):
    """Tests for Flow.model with Hydra configuration."""

    def test_hydra_instantiate_basic(self):
        """Test that Flow.model factory can be instantiated via Hydra."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Create config that references the factory function by module path
        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_basic_model",
                "value": 100,
                "name": "test",
            }
        )

        # Instantiate via Hydra
        model = instantiate(cfg)

        self.assertIsInstance(model, CallableModel)
        result = model(SimpleContext(value=10))
        self.assertEqual(result.value, "test:110")

    def test_hydra_instantiate_with_defaults(self):
        """Test Hydra instantiation using default parameter values."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_default_model",
                # Not specifying value, should use default
            }
        )

        model = instantiate(cfg)
        result = model(SimpleContext(value=8))
        self.assertEqual(result.value, 50)

    def test_hydra_instantiate_with_dependency(self):
        """Test Hydra instantiation with dependencies."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Create nested config
        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_consumer_model",
                "source": {
                    "_target_": "ccflow.tests.test_flow_model.hydra_source_model",
                    "base": 10,
                },
                "factor": 2,
            }
        )

        model = instantiate(cfg)

        result = model(SimpleContext(value=5))
        # source: 5 * 10 = 50, consumer: 50 * 2 = 100
        self.assertEqual(result.value, 100)


# =============================================================================
# Class-based CallableModel with Auto-Resolution Tests
# =============================================================================


class TestClassBasedDepResolution(TestCase):
    """Tests for auto-resolution of DepOf fields in class-based CallableModels.

    Key pattern: Fields use DepOf annotation, __call__ only takes context,
    and resolved values are accessed via self.field_name during __call__.
    """

    def test_class_based_auto_resolve_basic(self):
        """Test that DepOf fields are auto-resolved and accessible via self."""

        @Flow.model
        def data_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        class Consumer(CallableModel):
            # DepOf expands to Annotated[Union[ResultType, CallableModel], Dep()]
            source: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                # Access resolved value via self.source
                return GenericResult(value=self.source.value + 1)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.source, [context])]

        src = data_source()
        consumer = Consumer(source=src)

        result = consumer(SimpleContext(value=5))
        # source: 5 * 10 = 50, consumer: 50 + 1 = 51
        self.assertEqual(result.value, 51)

    def test_class_based_with_custom_transform(self):
        """Test that custom __deps__ transform is used."""

        @Flow.model
        def data_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        class Consumer(CallableModel):
            source: DepOf[..., GenericResult[int]]
            offset: int = 100

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.source.value + self.offset)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                # Apply custom transform
                transformed_ctx = SimpleContext(value=context.value + 5)
                return [(self.source, [transformed_ctx])]

        src = data_source()
        consumer = Consumer(source=src, offset=1)

        result = consumer(SimpleContext(value=5))
        # transformed context: 5 + 5 = 10
        # source: 10 * 10 = 100
        # consumer: 100 + 1 = 101
        self.assertEqual(result.value, 101)

    def test_class_based_with_annotated_transform(self):
        """Test that Dep transform is used when field not in __deps__."""

        @Flow.model
        def data_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        def double_value(ctx: SimpleContext) -> SimpleContext:
            return SimpleContext(value=ctx.value * 2)

        class Consumer(CallableModel):
            source: Annotated[DepOf[..., GenericResult[int]], Dep(transform=double_value)]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.source.value + 1)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return []  # Empty - uses Dep annotation transform from field

        src = data_source()
        consumer = Consumer(source=src)

        result = consumer(SimpleContext(value=5))
        # transform: 5 * 2 = 10
        # source: 10 * 10 = 100
        # consumer: 100 + 1 = 101
        self.assertEqual(result.value, 101)

    def test_class_based_multiple_deps(self):
        """Test auto-resolution with multiple dependencies."""

        @Flow.model
        def source_a(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def source_b(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 2)

        class Aggregator(CallableModel):
            a: DepOf[..., GenericResult[int]]
            b: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.a.value + self.b.value)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.a, [context]), (self.b, [context])]

        agg = Aggregator(a=source_a(), b=source_b())

        result = agg(SimpleContext(value=10))
        # a: 10, b: 20, aggregator: 30
        self.assertEqual(result.value, 30)

    def test_class_based_deps_with_instance_field_access(self):
        """Test that __deps__ can access instance fields for configurable transforms.

        This is the key advantage of class-based models over @Flow.model:
        transforms can use instance fields like window size.
        """

        @Flow.model
        def data_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        class Consumer(CallableModel):
            source: DepOf[..., GenericResult[int]]
            lookback: int = 5  # Configurable instance field

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.source.value * 2)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                # Access self.lookback in transform - this is why we use class-based!
                transformed = SimpleContext(value=context.value + self.lookback)
                return [(self.source, [transformed])]

        src = data_source()
        consumer = Consumer(source=src, lookback=10)

        result = consumer(SimpleContext(value=5))
        # transformed: 5 + 10 = 15
        # source: 15
        # consumer: 15 * 2 = 30
        self.assertEqual(result.value, 30)

    def test_class_based_with_direct_value(self):
        """Test that DepOf fields can accept pre-resolved values."""

        class Consumer(CallableModel):
            source: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.source.value + context.value)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                # No deps when source is already resolved
                return []

        # Pass direct value instead of CallableModel
        consumer = Consumer(source=GenericResult(value=100))

        result = consumer(SimpleContext(value=5))
        self.assertEqual(result.value, 105)

    def test_class_based_no_double_call(self):
        """Test that dependencies are not called twice during DepOf resolution.

        This verifies that the auto-resolution mechanism doesn't accidentally
        evaluate the same dependency multiple times.
        """
        call_counts = {"source": 0}

        @Flow.model
        def counting_source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 10)

        class Consumer(CallableModel):
            data: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=self.data.value + 1)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.data, [context])]

        src = counting_source()
        consumer = Consumer(data=src)

        # Call consumer - source should only be called once
        result = consumer(SimpleContext(value=5))

        self.assertEqual(result.value, 51)  # 5 * 10 + 1
        self.assertEqual(call_counts["source"], 1, "Source should only be called once")

    def test_class_based_nested_depof_no_double_call(self):
        """Test nested DepOf chain (A -> B -> C) has no double-calls at any layer.

        This tests a 3-layer dependency chain where:
        - layer_c is the leaf (no dependencies)
        - layer_b depends on layer_c
        - layer_a depends on layer_b

        Each layer should be called exactly once.
        """
        call_counts = {"layer_a": 0, "layer_b": 0, "layer_c": 0}

        # Layer C: leaf node (no dependencies)
        @Flow.model
        def layer_c(context: SimpleContext) -> GenericResult[int]:
            call_counts["layer_c"] += 1
            return GenericResult(value=context.value)

        # Layer B: depends on layer_c
        class LayerB(CallableModel):
            source: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                call_counts["layer_b"] += 1
                return GenericResult(value=self.source.value * 10)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.source, [context])]

        # Layer A: depends on layer_b
        class LayerA(CallableModel):
            source: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                call_counts["layer_a"] += 1
                return GenericResult(value=self.source.value + 1)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.source, [context])]

        # Build the chain: A -> B -> C
        c = layer_c()
        b = LayerB(source=c)
        a = LayerA(source=b)

        # Call layer_a - each layer should be called exactly once
        result = a(SimpleContext(value=5))

        # Verify result: C returns 5, B returns 5*10=50, A returns 50+1=51
        self.assertEqual(result.value, 51)

        # Verify each layer called exactly once
        self.assertEqual(call_counts["layer_c"], 1, "layer_c should be called exactly once")
        self.assertEqual(call_counts["layer_b"], 1, "layer_b should be called exactly once")
        self.assertEqual(call_counts["layer_a"], 1, "layer_a should be called exactly once")

    def test_flow_model_uses_unified_resolution_path(self):
        """Test that @Flow.model uses the same resolution path as class-based CallableModel.

        This verifies the consolidation of resolution logic - both @Flow.model and
        class-based models should use _resolve_deps_and_call in callable.py.
        """
        call_counts = {"source": 0, "decorator_model": 0, "class_model": 0}

        @Flow.model
        def shared_source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 2)

        # @Flow.model consumer
        @Flow.model
        def decorator_consumer(
            context: SimpleContext,
            data: DepOf[..., GenericResult[int]],
        ) -> GenericResult[int]:
            call_counts["decorator_model"] += 1
            return GenericResult(value=data.value + 100)

        # Class-based consumer (same logic)
        class ClassConsumer(CallableModel):
            data: DepOf[..., GenericResult[int]]

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                call_counts["class_model"] += 1
                return GenericResult(value=self.data.value + 100)

            @Flow.deps
            def __deps__(self, context: SimpleContext):
                return [(self.data, [context])]

        # Test both consumers with the same source
        src = shared_source()
        dec_consumer = decorator_consumer(data=src)
        cls_consumer = ClassConsumer(data=src)

        ctx = SimpleContext(value=10)

        # Both should produce the same result
        dec_result = dec_consumer(ctx)
        cls_result = cls_consumer(ctx)

        self.assertEqual(dec_result.value, cls_result.value)
        self.assertEqual(dec_result.value, 120)  # 10 * 2 + 100

        # Source should be called exactly twice (once per consumer)
        self.assertEqual(call_counts["source"], 2)
        self.assertEqual(call_counts["decorator_model"], 1)
        self.assertEqual(call_counts["class_model"], 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
