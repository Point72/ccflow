"""Tests for FlowContext, FlowAPI, and BoundModel under the FromContext design."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import cloudpickle
import pytest

from ccflow import BoundModel, CallableModel, ContextBase, Flow, FlowContext, FromContext, GenericResult


class NumberContext(ContextBase):
    x: int


class OffsetModel(CallableModel):
    offset: int

    @Flow.call
    def __call__(self, context: NumberContext) -> GenericResult[int]:
        return GenericResult(value=context.x + self.offset)


def test_flow_context_basic_properties():
    ctx = FlowContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), label="x")
    assert ctx.start_date == date(2024, 1, 1)
    assert ctx.end_date == date(2024, 1, 31)
    assert ctx.label == "x"
    assert dict(ctx) == {"start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), "label": "x"}


def test_flow_context_value_semantics_and_hash():
    first = FlowContext(x=1, values=[1, 2])
    second = FlowContext(x=1, values=[1, 2])
    third = FlowContext(x=2, values=[1, 2])

    assert first == second
    assert first != third
    assert len({first, second, third}) == 2


def test_flow_context_pickle_and_cloudpickle_roundtrip():
    ctx = FlowContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), tags=frozenset({"a", "b"}))
    assert pickle.loads(pickle.dumps(ctx)) == ctx
    assert cloudpickle.loads(cloudpickle.dumps(ctx)) == ctx


def test_flow_api_introspection_for_from_context_model():
    @Flow.model
    def add(a: int, b: FromContext[int], c: FromContext[int] = 5) -> int:
        return a + b + c

    model = add(a=10)
    assert model.flow.context_inputs == {"b": int, "c": int}
    assert model.flow.unbound_inputs == {"b": int}
    assert model.flow.bound_inputs == {"a": 10}
    assert model.flow.compute(b=2).value == 17


def test_flow_api_compute_accepts_single_context_or_kwargs_but_not_both():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    assert model.flow.compute(b=5).value == 15
    assert model.flow.compute(FlowContext(b=6)).value == 16

    with pytest.raises(TypeError, match="either one context object or contextual keyword arguments"):
        model.flow.compute(FlowContext(b=5), b=6)


def test_bound_model_with_inputs_static_and_callable():
    @Flow.model
    def load_window(start_date: FromContext[date], end_date: FromContext[date]) -> GenericResult[dict]:
        return GenericResult(value={"start": start_date, "end": end_date})

    model = load_window()
    shifted = model.flow.with_inputs(
        start_date=lambda ctx: ctx.start_date - timedelta(days=7),
        end_date=date(2024, 1, 31),
    )

    result = shifted(FlowContext(start_date=date(2024, 1, 8), end_date=date(2024, 1, 30)))
    assert result.value == {"start": date(2024, 1, 1), "end": date(2024, 1, 31)}


def test_bound_model_with_inputs_is_branch_local_and_chained():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def combine(left: int, right: int, value: FromContext[int]) -> int:
        return left + right + value

    base = source()
    left = base.flow.with_inputs(value=lambda ctx: ctx.value + 1)
    right = base.flow.with_inputs(value=lambda ctx: ctx.value + 2).flow.with_inputs(value=lambda ctx: ctx.value + 10)
    model = combine(left=left, right=right)

    assert model.flow.compute(value=5).value == (6 + 15 + 5)


def test_bound_model_rejects_regular_field_rewrites():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="only accepts contextual fields"):
        add(a=1).flow.with_inputs(a=3)


def test_bound_model_repr_matches_user_facing_api():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=1)
    bound = model.flow.with_inputs(b=lambda ctx: ctx.b + 1)
    assert repr(bound) == f"{model!r}.flow.with_inputs(b=<lambda>)"


def test_bound_model_serialization_roundtrip_preserves_static_transforms():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_inputs(b=5)
    dumped = bound.model_dump(mode="python")
    restored = type(bound).model_validate(dumped)

    assert restored.flow.compute().value == 15
    assert restored.model.flow.bound_inputs == {"a": 10}


def test_regular_callable_models_still_support_with_inputs():
    model = OffsetModel(offset=10)
    shifted = model.flow.with_inputs(x=lambda ctx: ctx.x * 2)
    assert shifted(NumberContext(x=5)).value == 20


def test_flow_api_for_regular_callable_model():
    model = OffsetModel(offset=10)
    assert model.flow.compute(x=5).value == 15
    assert model.flow.context_inputs == {"x": int}
    assert model.flow.unbound_inputs == {"x": int}
    assert model.flow.bound_inputs == {"offset": 10}


def test_generated_flow_model_compute_is_thread_safe():
    @Flow.model
    def add(a: int, b: FromContext[int], c: FromContext[int]) -> int:
        return a + b + c

    model = add(a=10)

    def worker(n: int) -> int:
        return model.flow.compute(b=n, c=n + 1).value

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(20)))

    assert results == [10 + n + n + 1 for n in range(20)]


def test_bound_model_restore_is_thread_safe():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    dumped = add(a=10).flow.with_inputs(b=5).model_dump(mode="python")

    def worker(_: int) -> int:
        restored = BoundModel.model_validate(dumped)
        return restored.flow.compute().value

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(20)))

    assert results == [15] * 20
