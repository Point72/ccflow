"""Tests for ``Optional[FromContext[T]]`` vs ``FromContext[Optional[T]]`` semantics.

Two spellings are both contextual and share the same validation base (``Optional[T]``);
they differ only in required-ness:

- ``FromContext[Optional[int]]`` is required-in-context but its value may be ``None``.
- ``Optional[FromContext[int]]`` is optional: when absent from context it is bound to ``None``.
"""

import pytest

from ccflow import Flow, FlowContext, FromContext, Lazy, ModelEvaluationContext
from ccflow.evaluators.common import cache_key


@Flow.model
def required_ctx(a: FromContext[int | None]) -> int:
    return -1 if a is None else a


@Flow.model
def optional_ctx(a: FromContext[int] | None) -> int:
    return -1 if a is None else a


@Flow.model
def explicit_none_default(a: FromContext[int | None] = None) -> int:
    return -1 if a is None else a


@Flow.model
def optional_with_default(a: FromContext[int] | None = 3) -> int:
    return -1 if a is None else a


class TestRequiredNullableContext:
    def test_missing_raises(self):
        with pytest.raises(TypeError, match="Missing contextual input"):
            required_ctx().flow.compute()

    def test_none_is_valid(self):
        assert required_ctx().flow.compute(a=None).value == -1

    def test_value(self):
        assert required_ctx().flow.compute(a=5).value == 5

    def test_inspect_required(self):
        insp = required_ctx().flow.inspect()
        assert "a" in insp.required_inputs


class TestOptionalContext:
    def test_missing_binds_none(self):
        assert optional_ctx().flow.compute().value == -1

    def test_none_explicit(self):
        assert optional_ctx().flow.compute(a=None).value == -1

    def test_value(self):
        assert optional_ctx().flow.compute(a=5).value == 5

    def test_inspect_not_required(self):
        insp = optional_ctx().flow.inspect()
        assert "a" not in insp.required_inputs
        assert "a" in insp.context_inputs


class TestConsistency:
    def test_explicit_none_default_equiv_optional(self):
        # FromContext[Optional[int]] = None is equivalent to Optional[FromContext[int]].
        assert explicit_none_default().flow.compute().value == -1
        assert optional_ctx().flow.compute().value == -1

    def test_optional_with_explicit_default_wins(self):
        # An explicit default overrides the implicit None of Optional[FromContext[int]].
        assert optional_with_default().flow.compute().value == 3

    def test_node_keys_distinguish_required_vs_optional(self):
        req_key = cache_key(ModelEvaluationContext(model=required_ctx(), context=FlowContext(a=5)))
        opt_key = cache_key(ModelEvaluationContext(model=optional_ctx(), context=FlowContext(a=5)))
        # Different required-ness must give distinct logical identities.
        assert req_key != opt_key


class TestRejections:
    def test_nested_lazy_rejected(self):
        with pytest.raises(TypeError, match="Lazy"):

            @Flow.model
            def bad(a: Lazy[int] | None) -> int:
                return 0

    def test_non_optional_union_with_fromcontext_rejected(self):
        with pytest.raises(TypeError, match="only supported as Optional"):

            @Flow.model
            def bad(a: FromContext[int] | str) -> int:
                return 0
