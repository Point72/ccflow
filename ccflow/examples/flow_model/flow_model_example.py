#!/usr/bin/env python
"""Small `@Flow.model` example.

Shows how to:

1. define stages as plain Python functions,
2. compose stages by passing upstream models as ordinary arguments,
3. rewrite contextual inputs on one dependency edge with `.flow.with_context(...)`,
4. execute the configured graph with `model.flow.compute(...)`.

Run with:
    python ccflow/examples/flow_model/flow_model_example.py
"""

from datetime import date, timedelta

from ccflow import DateRangeContext, Flow, FromContext


def _format_input_names(inputs: dict[str, object]) -> str:
    """Return a compact comma-separated list for example output."""
    return ", ".join(inputs) or "(none)"


def _format_bound_inputs(inputs: dict[str, object]) -> str:
    parts = []
    for name, value in inputs.items():
        display = "model" if hasattr(value, "flow") else repr(value)
        parts.append(f"{name}={display}")
    return ", ".join(parts) or "(none)"


@Flow.model(context_type=DateRangeContext)
def count_visitors(
    location: str,
    start_date: FromContext[date],
    end_date: FromContext[date],
) -> int:
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
) -> dict[str, object]:
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


def build_visitor_pipeline(location: str):
    """Build one reusable visitor-count pipeline."""
    current = count_visitors(location=location)
    previous = current.flow.with_context(shift_window(days=7))
    return visitor_delta(
        current=current,
        previous=previous,
        label="previous_week",
    )


def main() -> None:
    print("=" * 64)
    print("Flow.model Example")
    print("=" * 64)

    pipeline = build_visitor_pipeline(location="library")
    ctx = DateRangeContext(
        start_date=date(2024, 3, 1),
        end_date=date(2024, 3, 7),
    )

    computed_from_context = pipeline.flow.compute(ctx)
    computed_from_kwargs = pipeline.flow.compute(
        start_date=ctx.start_date,
        end_date=ctx.end_date,
    )

    print("\nPipeline:")
    print("  model: visitor_delta")
    pipeline_inspection = pipeline.flow.inspect()
    current_inspection = pipeline.current.flow.inspect()
    previous_inspection = pipeline.previous.flow.inspect()
    print(f"  bound inputs: {_format_bound_inputs(pipeline_inspection.bound_inputs)}")
    print(f"  declared context inputs: {_format_input_names(pipeline_inspection.context_inputs)}")
    print(f"  runtime inputs: {_format_input_names(pipeline_inspection.runtime_inputs)}")
    print(f"  current runtime inputs: {_format_input_names(current_inspection.runtime_inputs)}")
    print(f"  previous runtime inputs: {_format_input_names(previous_inspection.runtime_inputs)}")

    print("\nExecution:")
    print(f"  context object == kwargs: {computed_from_context == computed_from_kwargs}")

    print("\nResult:")
    for key, value in computed_from_kwargs.value.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
