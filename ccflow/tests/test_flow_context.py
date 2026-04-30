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


@Flow.context_transform
def shift_start_date(start_date: FromContext[date], days: int) -> date:
    return start_date - timedelta(days=days)


@Flow.context_transform
def shift_window(start_date: FromContext[date], end_date: FromContext[date], days: int) -> dict[str, object]:
    return {
        "start_date": start_date - timedelta(days=days),
        "end_date": end_date - timedelta(days=days),
    }


@Flow.context_transform
def offset_value(value: FromContext[int], amount: int) -> int:
    return value + amount


@Flow.context_transform
def offset_b(b: FromContext[int], amount: int) -> int:
    return b + amount


@Flow.context_transform
def double_x(x: FromContext[int]) -> int:
    return x * 2


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


def test_bound_model_with_context_static_and_transform():
    @Flow.model
    def load_window(start_date: FromContext[date], end_date: FromContext[date]) -> GenericResult[dict]:
        return GenericResult(value={"start": start_date, "end": end_date})

    model = load_window()
    shifted = model.flow.with_context(
        shift_window(days=7),
        end_date=date(2024, 1, 31),
    )

    result = shifted(FlowContext(start_date=date(2024, 1, 8), end_date=date(2024, 2, 7)))
    assert result.value == {"start": date(2024, 1, 1), "end": date(2024, 1, 31)}


def test_bound_model_with_context_is_branch_local_and_chained():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def combine(left: int, right: int, value: FromContext[int]) -> int:
        return left + right + value

    base = source()
    left = base.flow.with_context(value=offset_value(amount=1))
    right = base.flow.with_context(value=offset_value(amount=2)).flow.with_context(value=offset_value(amount=10))
    model = combine(left=left, right=right)

    assert model.flow.compute(value=5).value == (6 + 15 + 5)


def test_chained_with_context_merges_patch_transforms():
    @Flow.model
    def load(start_date: FromContext[date], end_date: FromContext[date]) -> dict:
        return {"start": start_date, "end": end_date}

    base = load()
    # First with_context applies a patch, second chains another patch on top
    chained = base.flow.with_context(shift_window(days=7)).flow.with_context(shift_window(days=3))

    # Both patches should be present in the merged context spec.
    assert len(chained.context_spec.patches) == 2

    # Patches evaluate against the original context, merge left-to-right.
    # patch1: start - 7, end - 7  =>  Jan 1, Jan 24
    # patch2: start - 3, end - 3  =>  Jan 5, Jan 28  (overwrites patch1 keys)
    result = chained(FlowContext(start_date=date(2024, 1, 8), end_date=date(2024, 1, 31)))
    assert result.value == {"start": date(2024, 1, 5), "end": date(2024, 1, 28)}


def test_compute_kwargs_can_supply_ambient_context_for_upstream_transforms():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def combine(left: int, right: int, bonus: FromContext[int]) -> int:
        return left + right + bonus

    base = source()
    model = combine(
        left=base.flow.with_context(value=offset_value(amount=1)),
        right=base.flow.with_context(value=offset_value(amount=10)),
    )

    assert model.flow.context_inputs == {"bonus": int}
    assert model.flow.compute(value=5, bonus=100).value == (6 + 15 + 100)


def test_bound_model_rejects_regular_field_context_overrides():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="only accepts contextual fields"):
        add(a=1).flow.with_context(a=3)


def test_bound_model_repr_matches_user_facing_api():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=1)
    bound = model.flow.with_context(b=offset_b(amount=1))
    assert repr(bound) == f"{model!r}.flow.with_context(b=offset_b(amount=1))"


def test_bound_model_serialization_roundtrip_preserves_static_transforms():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=5)

    dumped = bound.model_dump(mode="python")
    assert dumped["context_spec"] == {"patches": [], "field_overrides": {"b": {"kind": "static_value", "value": 5}}}

    restored = type(bound).model_validate(dumped)
    assert restored.flow.compute().value == 15
    assert restored.model.flow.bound_inputs == {"a": 10}


def test_bound_model_json_roundtrip_preserves_context_transforms():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=offset_b(amount=1))
    dumped = bound.model_dump(mode="json")
    assert dumped["context_spec"]["field_overrides"]["b"]["binding"]["path"].endswith(".offset_b")

    restored = type(bound).model_validate(dumped)
    assert restored.flow.compute(b=4).value == 15


def test_bound_model_cloudpickle_roundtrip_preserves_context_transforms():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=offset_b(amount=1))
    restored = cloudpickle.loads(cloudpickle.dumps(bound))
    assert restored.flow.compute(b=4).value == 15


def test_bound_model_plain_pickle_roundtrip_preserves_context_transforms():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=offset_b(amount=1))
    restored = pickle.loads(pickle.dumps(bound, protocol=5))
    assert restored.flow.compute(b=4).value == 15


def test_transformed_dag_cloudpickle_roundtrip_preserves_context_transforms():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def combine(left: int, right: int, value: FromContext[int]) -> int:
        return left + right + value

    base = source()
    model = combine(
        left=base.flow.with_context(value=offset_value(amount=1)),
        right=base.flow.with_context(value=offset_value(amount=10)),
    )
    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert restored.flow.compute(value=5).value == (6 + 15 + 5)


def test_bound_model_pydantic_roundtrip_preserves_context_transforms():
    """model_dump(mode='python') + model_validate must preserve serialized context bindings."""

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=offset_b(amount=1))
    assert bound.flow.compute(b=4).value == 15

    dumped = bound.model_dump(mode="python")
    assert dumped["context_spec"]["field_overrides"]["b"]["binding"]["kind"] == "context_transform"

    restored = type(bound).model_validate(dumped)
    assert restored.flow.compute(b=4).value == 15


def test_bound_model_context_spec_dump_contains_patch_and_field_specs():
    """model_dump(mode='json') should emit explicit tagged context-spec objects."""

    @Flow.model
    def load_window(start_date: FromContext[date], end_date: FromContext[date]) -> GenericResult[dict]:
        return GenericResult(value={"start": start_date, "end": end_date})

    dumped = load_window().flow.with_context(shift_window(days=7), start_date=shift_start_date(days=1)).model_dump(mode="json")
    assert dumped["context_spec"]["patches"][0]["kind"] == "context_patch"
    assert dumped["context_spec"]["field_overrides"]["start_date"]["kind"] == "context_value"


def test_regular_callable_models_still_support_with_context():
    model = OffsetModel(offset=10)
    shifted = model.flow.with_context(x=double_x())
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

    dumped = add(a=10).flow.with_context(b=5).model_dump(mode="python")

    def worker(_: int) -> int:
        restored = BoundModel.model_validate(dumped)
        return restored.flow.compute().value

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(20)))

    assert results == [15] * 20
