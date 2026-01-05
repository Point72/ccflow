#!/usr/bin/env python
"""Example demonstrating Flow.model decorator and class-based CallableModel.

This example shows:
- Flow.model for simple functions with minimal boilerplate
- Context transforms with Dep annotations
- Class-based CallableModel for complex cases needing instance field access
"""

from datetime import date, timedelta
from typing import Annotated

from ccflow import CallableModel, DateRangeContext, Dep, DepOf, Flow, GenericResult
from ccflow.callable import resolve


# =============================================================================
# Example 1: Basic Flow.model - No more boilerplate classes!
# =============================================================================

@Flow.model
def load_records(context: DateRangeContext, source: str, limit: int = 100) -> GenericResult[list]:
    """Load records from a data source for the given date range."""
    print(f"  Loading from '{source}' for {context.start_date} to {context.end_date} (limit={limit})")
    return GenericResult(value=[
        {"id": i, "date": str(context.start_date), "value": i * 10}
        for i in range(min(limit, 5))
    ])


# =============================================================================
# Example 2: Dependencies with DepOf - Automatic dependency resolution
# =============================================================================

@Flow.model
def compute_totals(
    _: DateRangeContext,  # Context passed to dependency, not used directly here
    records: DepOf[..., GenericResult[list]],
) -> GenericResult[dict]:
    """Compute totals from loaded records."""
    total = sum(r["value"] for r in records.value)
    count = len(records.value)
    print(f"  Computing totals: {count} records, total={total}")
    return GenericResult(value={"total": total, "count": count})


# =============================================================================
# Example 3: Simple Transform with Flow.model
# When the transform is a fixed function, Flow.model works great
# =============================================================================

def lookback_7_days(ctx: DateRangeContext) -> DateRangeContext:
    """Fixed transform that extends the date range back by 7 days."""
    return ctx.model_copy(update={"start_date": ctx.start_date - timedelta(days=7)})


@Flow.model
def compute_weekly_average(
    _: DateRangeContext,
    records: Annotated[GenericResult[list], Dep(transform=lookback_7_days)],
) -> GenericResult[float]:
    """Compute average using fixed 7-day lookback."""
    values = [r["value"] for r in records.value]
    avg = sum(values) / len(values) if values else 0
    print(f"  Computing weekly average: {avg:.2f} (from {len(values)} records)")
    return GenericResult(value=avg)


# =============================================================================
# Example 4: Class-based CallableModel with Configurable Transform
# When the transform needs access to instance fields (like window size),
# use a class-based approach with auto-resolution
# =============================================================================

class ComputeMovingAverage(CallableModel):
    """Compute moving average with configurable lookback window.

    This demonstrates:
    - Field uses DepOf annotation: accepts either result or CallableModel
    - Instance field (window) accessible in __deps__ for custom transforms
    - resolve() to access resolved dependency values during __call__
    """

    records: DepOf[..., GenericResult[list]]
    window: int = 7  # Configurable lookback window

    @Flow.call
    def __call__(self, context: DateRangeContext) -> GenericResult[float]:
        """Compute the moving average - use resolve() to get resolved value."""
        records = resolve(self.records)  # Get the resolved GenericResult
        values = [r["value"] for r in records.value]
        avg = sum(values) / len(values) if values else 0
        print(f"  Computing {self.window}-day moving average: {avg:.2f} (from {len(values)} records)")
        return GenericResult(value=avg)

    @Flow.deps
    def __deps__(self, context: DateRangeContext):
        """Define dependencies with transform that uses self.window."""
        # This is where we can access instance fields!
        lookback_ctx = context.model_copy(
            update={"start_date": context.start_date - timedelta(days=self.window)}
        )
        return [(self.records, [lookback_ctx])]


# =============================================================================
# Example 5: Multi-stage pipeline - Composing models together
# =============================================================================

@Flow.model
def generate_report(
    context: DateRangeContext,
    totals: DepOf[..., GenericResult[dict]],
    moving_avg: DepOf[..., GenericResult[float]],
    report_name: str = "Daily Report",
) -> GenericResult[str]:
    """Generate a report combining multiple data sources."""
    report = f"""
{report_name}
{'=' * len(report_name)}
Date Range: {context.start_date} to {context.end_date}
Total Value: {totals.value['total']}
Record Count: {totals.value['count']}
Moving Avg: {moving_avg.value:.2f}
"""
    return GenericResult(value=report.strip())


# =============================================================================
# Example 6: Using context_args for cleaner signatures
# =============================================================================

@Flow.model(context_args=["start_date", "end_date"])
def fetch_metadata(start_date: date, end_date: date, category: str) -> GenericResult[dict]:
    """Fetch metadata - note how start_date/end_date are direct parameters."""
    print(f"  Fetching metadata for '{category}' from {start_date} to {end_date}")
    return GenericResult(value={
        "category": category,
        "days": (end_date - start_date).days,
        "generated_at": str(date.today()),
    })


# =============================================================================
# Main: Build and execute the pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Flow.model Example - Simplified CallableModel Creation")
    print("=" * 60)

    ctx = DateRangeContext(
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 31)
    )

    # --- Example 1: Basic model ---
    print("\n[1] Basic Flow.model:")
    loader = load_records(source="main_db", limit=5)
    result = loader(ctx)
    print(f"  Result: {result.value}")

    # --- Example 2: Simple dependency chain ---
    print("\n[2] Dependency chain (loader -> totals):")
    loader = load_records(source="main_db")
    totals = compute_totals(records=loader)
    result = totals(ctx)
    print(f"  Result: {result.value}")

    # --- Example 3: Fixed transform with Flow.model ---
    print("\n[3] Fixed transform (7-day lookback with Flow.model):")
    loader = load_records(source="main_db")
    weekly_avg = compute_weekly_average(records=loader)
    result = weekly_avg(ctx)
    print(f"  Result: {result.value}")

    # --- Example 4: Configurable transform with class-based model ---
    print("\n[4] Configurable transform (class-based with auto-resolution):")
    loader = load_records(source="main_db")

    # 14-day window
    moving_avg_14 = ComputeMovingAverage(records=loader, window=14)
    result = moving_avg_14(ctx)
    print(f"  14-day result: {result.value}")

    # 30-day window - same loader, different window
    moving_avg_30 = ComputeMovingAverage(records=loader, window=30)
    result = moving_avg_30(ctx)
    print(f"  30-day result: {result.value}")

    # --- Example 5: Full pipeline ---
    print("\n[5] Full pipeline (mixing Flow.model and class-based):")
    loader = load_records(source="analytics_db")
    totals = compute_totals(records=loader)
    moving_avg = ComputeMovingAverage(records=loader, window=7)
    report = generate_report(
        totals=totals,
        moving_avg=moving_avg,
        report_name="Analytics Summary"
    )
    result = report(ctx)
    print(result.value)

    # --- Example 6: context_args ---
    print("\n[6] Using context_args (auto-unpacked context):")
    metadata = fetch_metadata(category="sales")
    result = metadata(ctx)
    print(f"  Result: {result.value}")

    # --- Bonus: Inspecting models ---
    print("\n[Bonus] Inspecting models:")
    print(f"  load_records.context_type = {loader.context_type.__name__}")
    print(f"  ComputeMovingAverage uses __deps__ for custom transforms")
    deps = moving_avg.__deps__(ctx)
    for dep_model, dep_contexts in deps:
        print(f"    - Dependency context start: {dep_contexts[0].start_date} (lookback applied)")


if __name__ == "__main__":
    main()
