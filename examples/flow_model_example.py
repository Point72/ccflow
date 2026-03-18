#!/usr/bin/env python
"""Canonical Flow.model example.

This is the main `@Flow.model` story:

1. define workflow steps as plain Python functions,
2. wire them together by passing upstream models as normal arguments,
3. use a small Python builder for reusable composition,
4. execute either as a normal CallableModel or via `.flow.compute(...)`.

Run with:
    python examples/flow_model_example.py
"""

from datetime import date, timedelta

from ccflow import DateRangeContext, Flow


@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    """Return synthetic revenue for one reporting window."""
    days = (end_date - start_date).days + 1
    region_base = {"us": 1000.0, "eu": 850.0}.get(region, 900.0)
    days_since_2024 = (end_date - date(2024, 1, 1)).days
    trend = days_since_2024 * 2.5
    return round(region_base + days * 8.0 + trend, 2)


@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def revenue_change(
    start_date: date,
    end_date: date,
    current: float,
    previous: float,
    label: str,
    days_back: int,
) -> dict:
    """Compare the current window against a shifted previous window."""
    previous_start = start_date - timedelta(days=days_back)
    previous_end = end_date - timedelta(days=days_back)
    growth_pct = round((current - previous) / previous * 100, 2)
    return {
        "comparison": label,
        "current_window": f"{start_date} -> {end_date}",
        "previous_window": f"{previous_start} -> {previous_end}",
        "current": current,
        "previous": previous,
        "growth_pct": growth_pct,
    }


def shifted_window(model, *, days_back: int):
    """Reuse one upstream model with a shifted runtime window."""
    return model.flow.with_inputs(
        start_date=lambda ctx: ctx.start_date - timedelta(days=days_back),
        end_date=lambda ctx: ctx.end_date - timedelta(days=days_back),
    )


def build_week_over_week_pipeline(region: str):
    """Build one reusable pipeline from plain Flow.model functions."""
    current = load_revenue(region=region)
    previous = shifted_window(current, days_back=7)
    return revenue_change(
        current=current,
        previous=previous,
        label="week_over_week",
        days_back=7,
    )


def main() -> None:
    print("=" * 64)
    print("Flow.model Example")
    print("=" * 64)

    pipeline = build_week_over_week_pipeline(region="us")
    ctx = DateRangeContext(
        start_date=date(2024, 3, 1),
        end_date=date(2024, 3, 31),
    )

    direct = pipeline(ctx)
    computed = pipeline.flow.compute(
        start_date=ctx.start_date,
        end_date=ctx.end_date,
    )

    print("\nPipeline wired from plain functions:")
    print("  current input:", pipeline.current)
    print("  previous input:", pipeline.previous)

    print("\nDirect call and .flow.compute(...) are equivalent:")
    print(f"  direct == computed: {direct == computed}")

    print("\nResult:")
    for key, value in computed.value.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
