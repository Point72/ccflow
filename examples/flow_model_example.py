#!/usr/bin/env python
"""Example demonstrating the core Flow.model workflow."""

from datetime import date, timedelta

from ccflow import DateRangeContext, Flow


@Flow.model(context_args=["start_date", "end_date"])
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    """Pretend to load revenue for a date window."""
    days = (end_date - start_date).days + 1
    baseline = 1000.0 if region == "us" else 800.0
    return baseline + days * 10.0


@Flow.model(context_args=["start_date", "end_date"])
def summarize_growth(start_date: date, end_date: date, current: float, previous: float) -> dict:
    """Compare the current and previous windows."""
    growth_pct = round((current - previous) / previous * 100, 2)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "current": current,
        "previous": previous,
        "growth_pct": growth_pct,
    }


def main():
    print("=" * 60)
    print("Flow.model Example")
    print("=" * 60)

    current_window = load_revenue(region="us")
    previous_window = current_window.flow.with_inputs(
        start_date=lambda ctx: ctx.start_date - timedelta(days=30),
        end_date=lambda ctx: ctx.end_date - timedelta(days=30),
    )

    growth = summarize_growth(current=current_window, previous=previous_window)

    ctx = DateRangeContext(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    print("\n[1] Execute as a normal CallableModel:")
    print(growth(ctx).value)

    print("\n[2] Execute via .flow.compute(...):")
    print(
        growth.flow.compute(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
    )

    print("\n[3] Inspect bound and unbound inputs:")
    print("  bound_inputs:", growth.flow.bound_inputs)
    print("  unbound_inputs:", growth.flow.unbound_inputs)


if __name__ == "__main__":
    main()
