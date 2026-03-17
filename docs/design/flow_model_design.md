# Flow.model and DepOf: Dependency Injection for CallableModel

## Overview

This document describes the `@Flow.model` decorator and `DepOf` annotation system for reducing boilerplate when creating `CallableModel` pipelines with dependencies.

**Key features:**
- `@Flow.model` - Decorator that generates `CallableModel` classes from plain functions
- `FlowContext` - Universal context carrier for unpacked/deferred execution
- `model.flow.compute(...)` / `model.flow.with_inputs(...)` - Deferred execution helpers
- `DepOf[ContextType, ResultType]` - Type annotation for dependency fields
- `Lazy[T]` - Mark a dependency for lazy, on-demand evaluation
- `FieldExtractor` - Access structured outputs via attribute access on generated models
- `resolve()` - Function to access resolved dependency values in class-based models

## Quick Start

### Pattern 1: `@Flow.model` (Recommended for Declarative Cases)

```python
from datetime import date, timedelta
from typing import Annotated

from ccflow import Flow, DateRangeContext, GenericResult, Dep, DepOf


def previous_window(ctx: DateRangeContext) -> DateRangeContext:
    window = ctx.end_date - ctx.start_date
    return ctx.model_copy(
        update={
            "start_date": ctx.start_date - window - timedelta(days=1),
            "end_date": ctx.start_date - timedelta(days=1),
        }
    )

@Flow.model
def load_revenue(context: DateRangeContext, region: str) -> GenericResult[float]:
    return GenericResult(value=125.0)

@Flow.model
def revenue_growth(
    context: DateRangeContext,
    current: DepOf[..., GenericResult[float]],
    previous: Annotated[GenericResult[float], Dep(transform=previous_window)],
) -> GenericResult[dict]:
    growth = (current.value - previous.value) / previous.value
    return GenericResult(value={"as_of": context.end_date, "growth": growth})

# Build pipeline. The same upstream model is reused twice:
# - once with the original context
# - once with a fixed lookback transform
revenue = load_revenue(region="us")
growth = revenue_growth(current=revenue, previous=revenue)

# Execute
ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = growth(ctx)
```

### Pattern 2: Class-Based (For Complex Cases)

Use class-based when you need **configurable transforms** that depend on instance fields:

```python
from datetime import timedelta

from ccflow import CallableModel, DateRangeContext, Flow, GenericResult, DepOf
from ccflow.callable import resolve  # Import resolve for class-based models

class RevenueAverageWithWindow(CallableModel):
    """Aggregate revenue with a configurable lookback window."""

    revenue: DepOf[..., GenericResult[float]]
    window: int = 7  # Configurable instance field

    @Flow.call
    def __call__(self, context: DateRangeContext) -> GenericResult[float]:
        # Use resolve() to get the resolved value
        revenue = resolve(self.revenue)
        return GenericResult(value=revenue.value / self.window)

    @Flow.deps
    def __deps__(self, context: DateRangeContext):
        # Transform uses self.window - this is why we need class-based!
        lookback_ctx = context.model_copy(
            update={"start_date": context.start_date - timedelta(days=self.window)}
        )
        return [(self.revenue, [lookback_ctx])]

# Usage - different window sizes, same source
loader = load_revenue(region="us")
avg_7 = RevenueAverageWithWindow(revenue=loader, window=7)
avg_30 = RevenueAverageWithWindow(revenue=loader, window=30)
```

## When to Use Which Pattern

| Use `@Flow.model` when...      | Use Class-Based when...               |
|--------------------------------|---------------------------------------|
| The node still reads like a normal function | The main value is custom graph logic |
| Transforms are fixed/declarative | Transforms depend on instance fields |
| Less boilerplate is priority   | You need full control over `__deps__` |
| Dependency wiring fits in the signature | Dependency behavior deserves its own class |

## Core Concepts

### `DepOf[ContextType, ResultType]`

Shorthand for declaring dependency fields that can accept either:
- A pre-computed value of `ResultType`
- A `CallableModel` that produces `ResultType`

```python
# Inherit context type from parent model
data: DepOf[..., GenericResult[dict]]

# Explicit context type
data: DepOf[DateRangeContext, GenericResult[dict]]

# Equivalent to:
data: Annotated[Union[GenericResult[dict], CallableModel], Dep()]
```

For `@Flow.model`, plain non-`DepOf` parameters can also be populated with a
`CallableModel` instance. That lets callers either inject a concrete value or
splice in an upstream computation for the same parameter. Use `Dep`/`DepOf`
when you need explicit dependency metadata such as context transforms or
context-type validation.

That means `DepOf` inside `@Flow.model` is most compelling when the function is
still doing real work and the dependency relationship is simple. If the node is
mostly a vessel for custom dependency graph wiring, a hand-written
`CallableModel` is usually clearer.

### `Dep(transform=..., context_type=...)`

For transforms, use the full `Annotated` form:

```python
from ccflow import Dep

@Flow.model
def compute_stats(
    context: DateRangeContext,
    records: Annotated[GenericResult[dict], Dep(
        transform=lambda ctx: ctx.model_copy(
            update={"start_date": ctx.start_date - timedelta(days=1)}
        )
    )],
) -> GenericResult[float]:
    return GenericResult(value=records.value["count"] * 0.05)
```

### `resolve()` Function

**Only needed for class-based models.** Accesses the resolved value of a `DepOf` field during `__call__`.

```python
from ccflow.callable import resolve

class MyModel(CallableModel):
    data: DepOf[..., GenericResult[int]]

    @Flow.call
    def __call__(self, context: MyContext) -> GenericResult[int]:
        # resolve() returns the GenericResult, not the CallableModel
        result = resolve(self.data)
        return GenericResult(value=result.value + 1)
```

**Behavior:**
- Inside `__call__`: Returns the resolved value
- With direct values (not CallableModel): Returns unchanged (no-op)
- Outside `__call__`: Raises `RuntimeError`
- In `@Flow.model`: Not needed - values are passed as function arguments

**Type inference:**
```python
data: DepOf[..., GenericResult[int]]
resolved = resolve(self.data)  # Type: GenericResult[int]
```

## How Resolution Works

### `@Flow.model` Resolution Flow

1. User calls `model(context)`
2. Generated `__call__` invokes `_resolve_deps_and_call()`
3. For each dependency-bearing field containing a `CallableModel`:
   - Apply transform (if any)
   - Call the dependency
   - Store resolved value in context variable
4. Generated `__call__` reads the resolved values from the dependency store
5. Original function receives resolved values directly as normal function arguments

### Class-Based Resolution Flow

1. User calls `model(context)`
2. `_resolve_deps_and_call()` runs
3. For each `DepOf` field containing a `CallableModel`:
   - Check `__deps__` for custom transforms
   - If not listed in `__deps__`, fall back to the field's `Dep(...)` transform (or the original context)
   - Call the dependency
   - Store resolved value in context variable
4. User's `__call__` accesses values via `resolve(self.field)`

**Important:** Resolution uses a context variable (`contextvars.ContextVar`), making it thread-safe and async-safe.

## Design Decisions

### Decision 1: `resolve()` Instead of Temporary Mutation

**What we chose:** Explicit `resolve()` function with context variables.

**Alternative considered:** Temporarily mutate `self.field` during `__call__` to hold the resolved value, then restore after.

**Why we chose this:**
- No mutation of model state
- Thread/async-safe via contextvars
- Explicit about what's happening
- Easier to debug - `self.field` always shows the original value

**Trade-off:** Slightly more verbose (`resolve(self.data).value` vs `self.data.value`).

### Decision 2: Unified Resolution Path

**What we chose:** Both `@Flow.model` and class-based use the same `_resolve_deps_and_call()` function.

**Why:**
- Single source of truth for resolution logic
- Easier to maintain
- Consistent behavior across patterns

### Decision 3: `resolve()` Not in Top-Level `__all__`

**What we chose:** `resolve` must be imported explicitly: `from ccflow.callable import resolve`

**Why:**
- Only needed for class-based models with `DepOf`
- Keeps top-level namespace clean
- Users who need it can find it easily

### Decision 4: Auto-Wrap Plain Return Values

**What we chose:** If the function's declared return type is not a `ResultBase`
subclass, the generated model wraps the returned value in `GenericResult`.

**Why:**
- Reduces boilerplate for simple scalar / container-returning functions
- Preserves the `CallableModel` contract that runtime results are `ResultBase`
- Still allows explicit `ResultBase` subclasses when you want a precise result type

**Trade-off:** The original Python function may be annotated with a plain value
type while the generated model's runtime `result_type` is `GenericResult`.

### Decision 5: Generated Classes Are Real CallableModels

**What we chose:** Generate actual `CallableModel` subclasses using `type()`.

**Why:**
- Full compatibility with existing infrastructure
- Caching, registry, serialization work unchanged
- Can mix with hand-written classes

## Pitfalls and Limitations

### Pitfall 1: Forgetting `resolve()` in Class-Based Models

```python
class MyModel(CallableModel):
    data: DepOf[..., GenericResult[int]]

    @Flow.call
    def __call__(self, context):
        # WRONG - self.data is still the CallableModel!
        return GenericResult(value=self.data.value + 1)

        # CORRECT
        return GenericResult(value=resolve(self.data).value + 1)
```

**Error you'll see:** `AttributeError: '_SomeModel' object has no attribute 'value'`

### Pitfall 2: Calling `resolve()` Outside `__call__`

```python
model = MyModel(data=some_source())
resolve(model.data)  # RuntimeError!
```

`resolve()` only works during `__call__` execution.

### Pitfall 3: Lambda Transforms Don't Serialize

```python
# Won't serialize - lambdas can't be pickled
Dep(transform=lambda ctx: ctx.model_copy(...))

# Will serialize - use named functions
def shift_start(ctx):
    return ctx.model_copy(update={"start_date": ctx.start_date - timedelta(days=1)})

Dep(transform=shift_start)
```

### Pitfall 4: GraphEvaluator Requires Caching

When using `GraphEvaluator` with `DepOf`, dependencies may be called twice (once by GraphEvaluator, once by resolution) unless caching is enabled.

```python
# Use with caching
from ccflow.evaluators import GraphEvaluator, CachingEvaluator, MultiEvaluator

evaluator = MultiEvaluator(evaluators=[
    CachingEvaluator(),
    GraphEvaluator(),
])
```

### Pitfall 5: Two Mental Models

Users need to remember:
- `@Flow.model`: Use dependency values directly as function arguments
- Class-based: Use `resolve(self.field)` to access values

### Limitation: Custom `__deps__` Is Only Needed for Custom Graph Logic

Class-based models do not need a custom `__deps__` override when the default
field-level `Dep(...)` behavior is sufficient. Override `__deps__` only when
you need instance-dependent transforms or a custom dependency graph:

```python
class Consumer(CallableModel):
    data: DepOf[..., GenericResult[int]]

    @Flow.call
    def __call__(self, context):
        return GenericResult(value=resolve(self.data).value)
```

If you do need to use instance fields in the transform, then `__deps__` is the
right place to do it:

```python
class WindowedConsumer(CallableModel):
    data: DepOf[..., GenericResult[int]]
    window: int = 7

    @Flow.call
    def __call__(self, context):
        return GenericResult(value=resolve(self.data).value)

    @Flow.deps
    def __deps__(self, context):
        shifted = context.model_copy(update={"value": context.value + self.window})
        return [(self.data, [shifted])]
```

### Limitation: `context_args` Type Matching Is Best-Effort

When you use `context_args=[...]`, the framework validates those fields via a
runtime `TypedDict` schema. It only maps to a concrete built-in context type in
special cases such as `DateRangeContext`. Otherwise the generated model's
`context_type` is `FlowContext`, a universal frozen carrier for the validated
context values.

## Complete Example: Multi-Stage Pipeline

```python
from datetime import date, timedelta
from typing import Annotated

from ccflow import (
    CallableModel, DateRangeContext, Dep, DepOf,
    Flow, GenericResult
)
from ccflow.callable import resolve


# Stage 1: Data loader (simple, use @Flow.model)
@Flow.model
def load_events(context: DateRangeContext, source: str) -> GenericResult[list]:
    print(f"Loading from {source} for {context.start_date} to {context.end_date}")
    return GenericResult(value=[
        {"date": str(context.start_date), "count": 100 + i}
        for i in range(5)
    ])


# Stage 2: Transform with fixed lookback (use @Flow.model with Dep transform)
@Flow.model
def compute_daily_totals(
    context: DateRangeContext,
    events: Annotated[GenericResult[list], Dep(
        transform=lambda ctx: ctx.model_copy(
            update={"start_date": ctx.start_date - timedelta(days=1)}
        )
    )],
) -> GenericResult[float]:
    values = [e["count"] for e in events.value]
    total = sum(values) / len(values) if values else 0
    return GenericResult(value=total)


# Stage 3: Configurable window (use class-based)
class ComputeRollingSummary(CallableModel):
    """Summary with configurable lookback window."""

    totals: DepOf[..., GenericResult[float]]
    window: int = 20

    @Flow.call
    def __call__(self, context: DateRangeContext) -> GenericResult[float]:
        totals = resolve(self.totals)
        # Scale by window size
        summary = totals.value * (self.window ** 0.5)
        return GenericResult(value=summary)

    @Flow.deps
    def __deps__(self, context: DateRangeContext):
        lookback = context.model_copy(
            update={"start_date": context.start_date - timedelta(days=self.window)}
        )
        return [(self.totals, [lookback])]


# Build pipeline
events = load_events(source="main_db")
totals = compute_daily_totals(events=events)
summary_20 = ComputeRollingSummary(totals=totals, window=20)
summary_60 = ComputeRollingSummary(totals=totals, window=60)

# Execute
ctx = DateRangeContext(start_date=date(2024, 1, 15), end_date=date(2024, 1, 31))
print(f"20-day summary: {summary_20(ctx).value}")
print(f"60-day summary: {summary_60(ctx).value}")
```

## API Reference

### `@Flow.model`

```python
@Flow.model(
    context_args: list[str] = None,  # Unpack context fields as function args
    cacheable: bool = False,
    volatile: bool = False,
    log_level: int = logging.DEBUG,
    validate_result: bool = True,
    verbose: bool = True,
    evaluator: EvaluatorBase = None,
)
def my_function(context: ContextType, ...) -> ResultType:
    ...
```

If the function is annotated with a plain value type instead of a `ResultBase`
subclass, the generated model will wrap the returned value in `GenericResult`
at runtime.

### `DepOf[ContextType, ResultType]`

```python
# Inherit context from parent
field: DepOf[..., GenericResult[int]]

# Explicit context type
field: DepOf[DateRangeContext, GenericResult[int]]
```

### `Dep(transform=..., context_type=...)`

```python
field: Annotated[GenericResult[int], Dep(
    transform=my_transform_func,      # Optional: (context) -> transformed_context
    context_type=DateRangeContext,    # Optional: Expected context type
)]
```

### `resolve(dep)`

```python
from ccflow.callable import resolve

# Inside __call__ of class-based CallableModel:
resolved_value = resolve(self.dep_field)

# Type signature:
def resolve(dep: Union[T, CallableModel]) -> T: ...
```

## File Structure

```
ccflow/
├── callable.py      # CallableModel, Flow, resolve(), _resolve_deps_and_call()
├── dep.py           # Dep, DepOf, extract_dep()
├── flow_model.py    # @Flow.model implementation
└── tests/
    └── test_flow_model.py  # Comprehensive tests
```
