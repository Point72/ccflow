#!/usr/bin/env python
"""Hydra + Flow.model builder demo.

This example shows a clean way to mix:

1. ergonomic `@Flow.model` pipeline wiring in Python, and
2. Hydra / ModelRegistry configuration for static pipeline specs.

The pattern is:

- keep runtime context (`start_date`, `end_date`) as runtime inputs,
- use a plain Python builder function for graph construction,
- let Hydra instantiate that builder and register the returned model.

Run with:
    python examples/flow_model_hydra_builder_demo.py
"""

from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from ccflow import BoundModel, CallableModel, DateRangeContext, Flow, FromContext, ModelRegistry

CONFIG_PATH = Path(__file__).with_name("config") / "flow_model_hydra_builder_demo.yaml"
ComparisonName = Literal["week_over_week", "month_over_month"]


@Flow.model(context_type=DateRangeContext)
def load_revenue(region: str, start_date: FromContext[date], end_date: FromContext[date]) -> float:
    """Return synthetic revenue for a date window."""
    days = (end_date - start_date).days + 1
    region_base = {"us": 1000.0, "eu": 850.0, "apac": 920.0}.get(region, 900.0)
    days_since_2024 = (end_date - date(2024, 1, 1)).days
    trend = days_since_2024 * 2.5
    return round(region_base + days * 8.0 + trend, 2)


@Flow.model(context_type=DateRangeContext)
def revenue_change(
    current: float,
    previous: float,
    comparison: ComparisonName,
    start_date: FromContext[date],
    end_date: FromContext[date],
) -> dict:
    """Compare the current window against a shifted previous window."""
    growth = (current - previous) / previous
    previous_start, previous_end = comparison_window(start_date, end_date, comparison)
    return {
        "comparison": comparison,
        "current_window": f"{start_date} -> {end_date}",
        "previous_window": f"{previous_start} -> {previous_end}",
        "current": current,
        "previous": previous,
        "delta": round(current - previous, 2),
        "growth_pct": round(growth * 100, 2),
    }


def comparison_window(start_date: date, end_date: date, comparison: ComparisonName) -> tuple[date, date]:
    """Return the previous window for a named comparison policy."""
    if comparison == "week_over_week":
        return start_date - timedelta(days=7), end_date - timedelta(days=7)

    if start_date.day != 1:
        raise ValueError("month_over_month requires start_date to be the first day of a month")
    if start_date.year != end_date.year or start_date.month != end_date.month:
        raise ValueError("month_over_month requires the current window to stay within one calendar month")
    expected_end = date(end_date.year, end_date.month, monthrange(end_date.year, end_date.month)[1])
    if end_date != expected_end:
        raise ValueError("month_over_month requires end_date to be the last day of that month")

    previous_year = start_date.year if start_date.month > 1 else start_date.year - 1
    previous_month = start_date.month - 1 if start_date.month > 1 else 12
    previous_start = date(previous_year, previous_month, 1)
    previous_end = date(previous_year, previous_month, monthrange(previous_year, previous_month)[1])
    return previous_start, previous_end


def comparison_input(model: CallableModel, comparison: ComparisonName) -> BoundModel:
    """Apply a named comparison policy to one dependency."""
    return model.flow.with_inputs(
        start_date=lambda ctx: comparison_window(ctx.start_date, ctx.end_date, comparison)[0],
        end_date=lambda ctx: comparison_window(ctx.start_date, ctx.end_date, comparison)[1],
    )


def build_comparison(current: CallableModel, *, comparison: ComparisonName):
    """Hydra-friendly builder that returns a configured comparison model."""
    previous = comparison_input(current, comparison)
    return revenue_change(
        current=current,
        previous=previous,
        comparison=comparison,
    )


def main() -> None:
    registry = ModelRegistry.root()
    registry.clear()
    try:
        registry.load_config_from_path(str(CONFIG_PATH), overwrite=True)

        week_over_week = registry["week_over_week"]
        month_over_month = registry["month_over_month"]

        ctx = DateRangeContext(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
        )

        print("=" * 68)
        print("Hydra + Flow.model Builder Demo")
        print("=" * 68)
        print("\nLoaded from config:")
        print("  current_revenue:", registry["current_revenue"])
        print("  week_over_week:", week_over_week)
        print("  month_over_month:", month_over_month)

        week_over_week_result = week_over_week.flow.compute(
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        ).value
        month_over_month_result = month_over_month(ctx).value

        print("\nWeek-over-week:")
        for key, value in week_over_week_result.items():
            print(f"  {key}: {value}")

        print("\nMonth-over-month:")
        for key, value in month_over_month_result.items():
            print(f"  {key}: {value}")
    finally:
        registry.clear()


if __name__ == "__main__":
    main()
