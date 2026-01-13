"""Tests for FlowContext, FlowAPI, and TypedDict-based context validation.

These tests verify the new deferred computation API that uses:
- FlowContext: Universal context carrier with extra="allow"
- TypedDict + TypeAdapter: Schema validation without dynamic class registration
- FlowAPI: The .flow namespace for compute/with_inputs/etc.
"""

import pickle
from datetime import date, timedelta

import cloudpickle
import pytest

from ccflow import Flow, FlowAPI, FlowContext, GenericResult
from ccflow.context import DateRangeContext


class TestFlowContext:
    """Tests for the FlowContext universal carrier."""

    def test_flow_context_basic(self):
        """FlowContext accepts arbitrary fields."""
        ctx = FlowContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        assert ctx.start_date == date(2024, 1, 1)
        assert ctx.end_date == date(2024, 1, 31)

    def test_flow_context_extra_fields(self):
        """FlowContext stores fields in __pydantic_extra__."""
        ctx = FlowContext(x=1, y="hello", z=[1, 2, 3])
        assert ctx.x == 1
        assert ctx.y == "hello"
        assert ctx.z == [1, 2, 3]
        assert ctx.__pydantic_extra__ == {"x": 1, "y": "hello", "z": [1, 2, 3]}

    def test_flow_context_frozen(self):
        """FlowContext is immutable (frozen)."""
        ctx = FlowContext(value=42)
        with pytest.raises(Exception):  # ValidationError for frozen model
            ctx.value = 100

    def test_flow_context_repr(self):
        """FlowContext has a useful repr."""
        ctx = FlowContext(a=1, b=2)
        repr_str = repr(ctx)
        assert "FlowContext" in repr_str
        assert "a=1" in repr_str
        assert "b=2" in repr_str

    def test_flow_context_attribute_error(self):
        """FlowContext raises AttributeError for missing fields."""
        ctx = FlowContext(x=1)
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            _ = ctx.missing

    def test_flow_context_model_dump(self):
        """FlowContext can be dumped (includes extra fields)."""
        ctx = FlowContext(start_date=date(2024, 1, 1), value=42)
        dumped = ctx.model_dump()
        assert dumped["start_date"] == date(2024, 1, 1)
        assert dumped["value"] == 42

    def test_flow_context_pickle(self):
        """FlowContext pickles cleanly."""
        ctx = FlowContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        pickled = pickle.dumps(ctx)
        unpickled = pickle.loads(pickled)
        assert unpickled.start_date == date(2024, 1, 1)
        assert unpickled.end_date == date(2024, 1, 31)

    def test_flow_context_cloudpickle(self):
        """FlowContext works with cloudpickle (for Ray)."""
        ctx = FlowContext(data=[1, 2, 3], name="test")
        pickled = cloudpickle.dumps(ctx)
        unpickled = cloudpickle.loads(pickled)
        assert unpickled.data == [1, 2, 3]
        assert unpickled.name == "test"


class TestFlowAPI:
    """Tests for the FlowAPI (.flow namespace)."""

    def test_flow_compute_basic(self):
        """FlowAPI.compute() validates and executes."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date, source: str = "db") -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date, "source": source})

        model = load_data(source="api")
        result = model.flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        assert result["start"] == date(2024, 1, 1)
        assert result["end"] == date(2024, 1, 31)
        assert result["source"] == "api"

    def test_flow_compute_type_coercion(self):
        """FlowAPI.compute() coerces types via TypeAdapter."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date})

        model = load_data()
        # Pass strings - should be coerced to dates
        result = model.flow.compute(start_date="2024-01-01", end_date="2024-01-31")

        assert result["start"] == date(2024, 1, 1)
        assert result["end"] == date(2024, 1, 31)

    def test_flow_compute_validation_error(self):
        """FlowAPI.compute() raises on missing required args."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={})

        model = load_data()
        with pytest.raises(Exception):  # ValidationError
            model.flow.compute(start_date=date(2024, 1, 1))  # Missing end_date

    def test_flow_unbound_inputs(self):
        """FlowAPI.unbound_inputs returns the context schema."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date, source: str = "db") -> GenericResult[dict]:
            return GenericResult(value={})

        model = load_data(source="api")
        unbound = model.flow.unbound_inputs

        assert "start_date" in unbound
        assert "end_date" in unbound
        assert unbound["start_date"] == date
        assert unbound["end_date"] == date
        # source is not unbound (it has a default/is bound)
        assert "source" not in unbound

    def test_flow_bound_inputs(self):
        """FlowAPI.bound_inputs returns config values."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date, source: str = "db") -> GenericResult[dict]:
            return GenericResult(value={})

        model = load_data(source="api")
        bound = model.flow.bound_inputs

        assert "source" in bound
        assert bound["source"] == "api"
        # Context args are not in bound_inputs
        assert "start_date" not in bound
        assert "end_date" not in bound


class TestBoundModel:
    """Tests for BoundModel (created via .flow.with_inputs())."""

    def test_with_inputs_static_value(self):
        """with_inputs can bind static values."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date})

        model = load_data()
        bound = model.flow.with_inputs(start_date=date(2024, 1, 1))

        # Call with just end_date (start_date is bound)
        ctx = FlowContext(end_date=date(2024, 1, 31))
        result = bound(ctx)
        assert result.value["start"] == date(2024, 1, 1)
        assert result.value["end"] == date(2024, 1, 31)

    def test_with_inputs_transform_function(self):
        """with_inputs can use transform functions."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date})

        model = load_data()
        # Lookback: start_date is 7 days before the context's start_date
        bound = model.flow.with_inputs(start_date=lambda ctx: ctx.start_date - timedelta(days=7))

        ctx = FlowContext(start_date=date(2024, 1, 8), end_date=date(2024, 1, 31))
        result = bound(ctx)
        assert result.value["start"] == date(2024, 1, 1)  # 7 days before
        assert result.value["end"] == date(2024, 1, 31)

    def test_with_inputs_multiple_transforms(self):
        """with_inputs can apply multiple transforms."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date})

        model = load_data()
        bound = model.flow.with_inputs(
            start_date=lambda ctx: ctx.start_date - timedelta(days=7),
            end_date=lambda ctx: ctx.end_date + timedelta(days=1),
        )

        ctx = FlowContext(start_date=date(2024, 1, 8), end_date=date(2024, 1, 30))
        result = bound(ctx)
        assert result.value["start"] == date(2024, 1, 1)
        assert result.value["end"] == date(2024, 1, 31)

    def test_bound_model_has_flow_property(self):
        """BoundModel has a .flow property."""

        @Flow.model(context_args=["x"])
        def compute(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 2)

        model = compute()
        bound = model.flow.with_inputs(x=42)
        assert isinstance(bound.flow, FlowAPI)


class TestTypedDictValidation:
    """Tests for TypedDict-based context validation."""

    def test_schema_stored_on_model(self):
        """Model stores _context_schema for validation."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={})

        model = load_data()
        assert hasattr(model, "_context_schema")
        assert model._context_schema == {"start_date": date, "end_date": date}

    def test_validator_created_lazily(self):
        """TypeAdapter validator is created lazily."""

        @Flow.model(context_args=["x"])
        def compute(x: int) -> GenericResult[int]:
            return GenericResult(value=x)

        model = compute()
        # Initially None
        assert model.__class__._cached_context_validator is None

        # After getting validator, it's cached
        validator = model._get_context_validator()
        assert validator is not None
        assert model.__class__._cached_context_validator is validator

    def test_matched_context_type(self):
        """DateRangeContext pattern is matched for compatibility."""

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={})

        model = load_data()
        # Should match DateRangeContext
        assert model.context_type == DateRangeContext


class TestPicklingSupport:
    """Tests for pickling support (important for Ray).

    Note: Regular pickle cannot pickle locally-defined classes (functions decorated
    inside test methods). cloudpickle CAN handle this, which is why Ray uses it.
    All tests here use cloudpickle to match Ray's behavior.
    """

    def test_model_cloudpickle_roundtrip(self):
        """Model works with cloudpickle (for Ray)."""

        @Flow.model(context_args=["x", "y"])
        def compute(x: int, y: int, multiplier: int = 2) -> GenericResult[int]:
            return GenericResult(value=(x + y) * multiplier)

        model = compute(multiplier=3)

        # cloudpickle roundtrip (what Ray uses)
        pickled = cloudpickle.dumps(model)
        unpickled = cloudpickle.loads(pickled)

        # Should work after unpickling
        result = unpickled.flow.compute(x=1, y=2)
        assert result == 9  # (1 + 2) * 3

    def test_model_cloudpickle_simple(self):
        """Simple model cloudpickle test."""

        @Flow.model(context_args=["value"])
        def double(value: int) -> GenericResult[int]:
            return GenericResult(value=value * 2)

        model = double()

        pickled = cloudpickle.dumps(model)
        unpickled = cloudpickle.loads(pickled)

        result = unpickled.flow.compute(value=21)
        assert result == 42

    def test_validator_recreated_after_cloudpickle(self):
        """TypeAdapter validator is recreated after cloudpickling."""

        @Flow.model(context_args=["x"])
        def compute(x: int) -> GenericResult[int]:
            return GenericResult(value=x)

        model = compute()
        # Warm up the validator cache
        _ = model._get_context_validator()
        assert model.__class__._cached_context_validator is not None

        # cloudpickle and unpickle
        pickled = cloudpickle.dumps(model)
        unpickled = cloudpickle.loads(pickled)

        # Validator should still work (may be lazily recreated)
        result = unpickled.flow.compute(x=42)
        assert result == 42

    def test_flow_context_pickle_standard(self):
        """FlowContext works with standard pickle."""
        ctx = FlowContext(x=1, y=2, z="test")

        pickled = pickle.dumps(ctx)
        unpickled = pickle.loads(pickled)

        assert unpickled.x == 1
        assert unpickled.y == 2
        assert unpickled.z == "test"


class TestIntegrationWithExistingContextTypes:
    """Tests for integration with existing ContextBase subclasses."""

    def test_explicit_context_still_works(self):
        """Explicit context parameter mode still works."""

        @Flow.model
        def load_data(context: DateRangeContext, source: str = "db") -> GenericResult[dict]:
            return GenericResult(value={"start": context.start_date, "end": context.end_date, "source": source})

        model = load_data(source="api")
        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        result = model(ctx)

        assert result.value["start"] == date(2024, 1, 1)
        assert result.value["source"] == "api"

    def test_flow_context_coerces_to_date_range(self):
        """FlowContext can be used with models expecting DateRangeContext."""

        @Flow.model
        def load_data(context: DateRangeContext) -> GenericResult[dict]:
            return GenericResult(value={"start": context.start_date, "end": context.end_date})

        model = load_data()
        # Use FlowContext - should coerce to DateRangeContext
        ctx = FlowContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        result = model(ctx)

        assert result.value["start"] == date(2024, 1, 1)
        assert result.value["end"] == date(2024, 1, 31)

    def test_flow_api_with_explicit_context(self):
        """FlowAPI.compute works with explicit context mode."""

        @Flow.model
        def load_data(context: DateRangeContext, source: str = "db") -> GenericResult[dict]:
            return GenericResult(value={"start": context.start_date, "end": context.end_date})

        model = load_data(source="api")
        result = model.flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        assert result["start"] == date(2024, 1, 1)
        assert result["end"] == date(2024, 1, 31)


class TestLazy:
    """Tests for Lazy (deferred execution with context overrides)."""

    def test_lazy_basic(self):
        """Lazy wraps a model for deferred execution."""
        from ccflow import Lazy

        @Flow.model(context_args=["value"])
        def compute(value: int, multiplier: int = 2) -> GenericResult[int]:
            return GenericResult(value=value * multiplier)

        model = compute(multiplier=3)
        lazy = Lazy(model)

        assert lazy.model is model

    def test_lazy_call_with_static_override(self):
        """Lazy.__call__ with static override values."""
        from ccflow import Lazy

        @Flow.model(context_args=["x", "y"])
        def add(x: int, y: int) -> GenericResult[int]:
            return GenericResult(value=x + y)

        model = add()
        lazy_fn = Lazy(model)(y=100)  # Override y to 100

        ctx = FlowContext(x=5, y=10)  # Original y=10
        result = lazy_fn(ctx)
        assert result.value == 105  # x=5 + y=100 (overridden)

    def test_lazy_call_with_callable_override(self):
        """Lazy.__call__ with callable override (computed at runtime)."""
        from ccflow import Lazy

        @Flow.model(context_args=["value"])
        def double(value: int) -> GenericResult[int]:
            return GenericResult(value=value * 2)

        model = double()
        # Override value to be original value + 10
        lazy_fn = Lazy(model)(value=lambda ctx: ctx.value + 10)

        ctx = FlowContext(value=5)
        result = lazy_fn(ctx)
        assert result.value == 30  # (5 + 10) * 2 = 30

    def test_lazy_with_date_transforms(self):
        """Lazy works with date transforms."""
        from ccflow import Lazy

        @Flow.model(context_args=["start_date", "end_date"])
        def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
            return GenericResult(value={"start": start_date, "end": end_date})

        model = load_data()

        # Use Lazy to create a transform that shifts dates
        lazy_fn = Lazy(model)(
            start_date=lambda ctx: ctx.start_date - timedelta(days=7),
            end_date=lambda ctx: ctx.end_date
        )

        ctx = FlowContext(start_date=date(2024, 1, 15), end_date=date(2024, 1, 31))
        result = lazy_fn(ctx)

        assert result.value["start"] == date(2024, 1, 8)  # 7 days before
        assert result.value["end"] == date(2024, 1, 31)

    def test_lazy_multiple_overrides(self):
        """Lazy supports multiple overrides at once."""
        from ccflow import Lazy

        @Flow.model(context_args=["a", "b", "c"])
        def compute(a: int, b: int, c: int) -> GenericResult[int]:
            return GenericResult(value=a + b + c)

        model = compute()
        lazy_fn = Lazy(model)(
            a=10,  # Static
            b=lambda ctx: ctx.b * 2,  # Transform
            # c not overridden, uses context value
        )

        ctx = FlowContext(a=1, b=5, c=100)
        result = lazy_fn(ctx)
        assert result.value == 10 + 10 + 100  # a=10, b=5*2=10, c=100
