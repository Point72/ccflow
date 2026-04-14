#!/usr/bin/env python
"""Main `@Flow.model` example.

Shows how to:

1. define stages as plain Python functions,
2. compose stages by passing upstream models as ordinary arguments,
3. rewrite contextual inputs on one dependency edge with `.flow.with_inputs(...)`,
4. execute either as `model(context)` or `model.flow.compute(...)`.

Run with:
    python examples/flow_model_example.py
"""

from datetime import date, timedelta

from ccflow import DateRangeContext, Flow, FromContext


@Flow.model(context_type=DateRangeContext)
def load_revenue(region: str, start_date: FromContext[date], end_date: FromContext[date]) -> float:
    """Return synthetic revenue for one reporting window."""
    days = (end_date - start_date).days + 1
    region_base = {"us": 1000.0, "eu": 850.0}.get(region, 900.0)
    days_since_2024 = (end_date - date(2024, 1, 1)).days
    trend = days_since_2024 * 2.5
    return round(region_base + days * 8.0 + trend, 2)


@Flow.model(context_type=DateRangeContext)
def revenue_change(
    current: float,
    previous: float,
    label: str,
    days_back: int,
    start_date: FromContext[date],
    end_date: FromContext[date],
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


@Flow.transform
def previous_window(start_date: FromContext[date], end_date: FromContext[date], days_back: int) -> dict[str, object]:
    """Shift both date fields together for a previous reporting window."""
    return {
        "start_date": start_date - timedelta(days=days_back),
        "end_date": end_date - timedelta(days=days_back),
    }


def build_week_over_week_pipeline(region: str):
    """Build one reusable comparison pipeline."""
    current = load_revenue(region=region)
    previous = current.flow.with_inputs(previous_window(days_back=7))
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

    print("\nPipeline:")
    print("  current input:", pipeline.current)
    print("  previous input:", pipeline.previous)

    print("\nExecution:")
    print(f"  direct == computed: {direct == computed}")

    print("\nResult:")
    for key, value in computed.value.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
