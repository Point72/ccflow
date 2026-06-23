"""Tests for positional/string context shorthand on ``@Flow.model(context_type=...)``.

Class-based ``CallableModel`` execution already accepts positional/string context
shorthand (the ordered ``zip(model_fields, v)`` mapping in ``ContextBase``). Generated
``@Flow.model`` instances expose ``FlowContext`` (the open bag) as their runtime context
type, so without a declared ``context_type`` there are no ordered fields to zip against.
When a ``context_type`` is declared, ``compute()`` routes the shorthand through it first.

Scope: this covers the ``compute()`` entry point. The direct-call form (``model([...])``)
is intentionally not supported here because ``Flow.call`` validates against ``FlowContext``
before the generated body runs; supporting it would require reverting the bag-of-types
design.
"""

from datetime import date

from ccflow import DateRangeContext, Flow, FromContext


@Flow.model(context_type=DateRangeContext)
def span(start_date: FromContext[date], end_date: FromContext[date]) -> int:
    return (end_date - start_date).days


class TestComputeShorthand:
    def test_list_shorthand(self):
        assert span().flow.compute(["2025-01-02", "2026-01-01"]).value == 364

    def test_tuple_shorthand(self):
        assert span().flow.compute(("2025-01-02", "2026-01-01")).value == 364

    def test_string_shorthand(self):
        assert span().flow.compute("2025-01-02,2026-01-01").value == 364

    def test_named_inputs_still_work(self):
        assert span().flow.compute(start_date="2025-01-02", end_date="2026-01-01").value == 364

    def test_context_object_still_works(self):
        ctx = DateRangeContext(start_date=date(2025, 1, 2), end_date=date(2026, 1, 1))
        assert span().flow.compute(ctx).value == 364

    def test_shorthand_matches_named(self):
        assert span().flow.compute(["2025-01-02", "2026-01-01"]).value == span().flow.compute(start_date="2025-01-02", end_date="2026-01-01").value


class TestNoDeclaredContextTypeUnaffected:
    def test_bag_model_still_uses_named(self):
        @Flow.model
        def bag(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return (end_date - start_date).days

        # Without a declared context_type there is no field order; named inputs are required.
        assert bag().flow.compute(start_date="2025-01-02", end_date="2026-01-01").value == 364
