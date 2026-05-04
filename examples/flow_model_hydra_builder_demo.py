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

from datetime import date, timedelta
from pathlib import Path

from ccflow import CallableModel, DateRangeContext, Flow, FromContext, ModelRegistry

CONFIG_PATH = Path(__file__).with_name("config") / "flow_model_hydra_builder_demo.yaml"


@Flow.model(context_type=DateRangeContext)
def count_visitors(location: str, start_date: FromContext[date], end_date: FromContext[date]) -> int:
    """Return a deterministic visitor count for one date window."""
    days = (end_date - start_date).days + 1
    location_offset = sum(ord(ch) for ch in location) % 17
    week_index = (end_date.toordinal() - date(2024, 1, 1).toordinal()) // 7
    return days * 12 + location_offset + week_index


@Flow.model(context_type=DateRangeContext)
def visitor_delta(
    current: int,
    previous: int,
    label: str,
    start_date: FromContext[date],
    end_date: FromContext[date],
) -> dict:
    """Return both visitor counts plus their difference."""
    return {
        "label": label,
        "window": f"{start_date} -> {end_date}",
        "current": current,
        "previous": previous,
        "change": current - previous,
    }


@Flow.context_transform
def shift_window(start_date: FromContext[date], end_date: FromContext[date], days: int) -> dict[str, object]:
    """Shift both date fields together."""
    return {
        "start_date": start_date - timedelta(days=days),
        "end_date": end_date - timedelta(days=days),
    }


def build_visitor_delta(current: CallableModel, *, label: str, days_back: int):
    """Hydra-friendly builder that returns a configured visitor-count model."""
    previous = current.flow.with_context(shift_window(days=days_back))
    return visitor_delta(
        current=current,
        previous=previous,
        label=label,
    )


def main() -> None:
    registry = ModelRegistry.root()
    registry.clear()
    try:
        registry.load_config_from_path(str(CONFIG_PATH), overwrite=True)

        previous_week = registry["previous_week"]
        previous_two_weeks = registry["previous_two_weeks"]

        ctx = DateRangeContext(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 7),
        )

        print("=" * 68)
        print("Hydra + Flow.model Builder Demo")
        print("=" * 68)
        print("\nLoaded from config:")
        print("  library_visitors:", registry["library_visitors"])
        print("  previous_week:", previous_week)
        print("  previous_two_weeks:", previous_two_weeks)

        previous_week_result = previous_week.flow.compute(
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        ).value
        previous_two_weeks_result = previous_two_weeks(ctx).value

        print("\nPrevious week:")
        for key, value in previous_week_result.items():
            print(f"  {key}: {value}")

        print("\nPrevious two weeks:")
        for key, value in previous_two_weeks_result.items():
            print(f"  {key}: {value}")
    finally:
        registry.clear()


if __name__ == "__main__":
    main()
