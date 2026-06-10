"""Tests for the ``Flow.model(context_type=, strict=...)`` subset/omnibus behavior."""

from datetime import date

import pytest

from ccflow import ContextBase, Flow, FromContext


class OmnibusContext(ContextBase):
    start_date: date
    end_date: date
    region: str  # extra required field a span model does not consume


class TestSubsetDefault:
    def test_subset_model_builds_and_runs(self):
        # Default strict=False: FromContext fields are a typed subset of the omnibus.
        @Flow.model(context_type=OmnibusContext)
        def span(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return (end_date - start_date).days

        model = span()
        # The omnibus carries `region`, which this model ignores.
        ctx = OmnibusContext(start_date=date(2025, 1, 1), end_date=date(2025, 1, 8), region="us")
        assert model.flow.compute(ctx).value == 7

    def test_subset_compute_with_named_inputs(self):
        @Flow.model(context_type=OmnibusContext)
        def span(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return (end_date - start_date).days

        assert span().flow.compute(start_date="2025-01-01", end_date="2025-01-08").value == 7


class TestStrict:
    def test_strict_rejects_unconsumed_required_field(self):
        with pytest.raises(TypeError, match="has required fields that are not declared as FromContext"):

            @Flow.model(context_type=OmnibusContext, strict=True)
            def span(start_date: FromContext[date], end_date: FromContext[date]) -> int:
                return (end_date - start_date).days

    def test_strict_accepts_full_bijection(self):
        class ExactContext(ContextBase):
            start_date: date
            end_date: date

        @Flow.model(context_type=ExactContext, strict=True)
        def span(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return (end_date - start_date).days

        assert span().flow.compute(start_date="2025-01-01", end_date="2025-01-08").value == 7


class TestSharedChecks:
    def test_missing_field_errors_in_both_modes(self):
        class CtxNoFoo(ContextBase):
            start_date: date

        for strict in (False, True):
            with pytest.raises(TypeError, match="must define fields for all FromContext"):

                @Flow.model(context_type=CtxNoFoo, strict=strict)
                def span(start_date: FromContext[date], foo: FromContext[int]) -> int:
                    return foo

    def test_type_mismatch_errors_in_both_modes(self):
        class CtxBadType(ContextBase):
            start_date: date
            end_date: date

        for strict in (False, True):
            with pytest.raises(TypeError, match="annotates"):

                @Flow.model(context_type=CtxBadType, strict=strict)
                def span(start_date: FromContext[int], end_date: FromContext[date]) -> int:
                    return 0
