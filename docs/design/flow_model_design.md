# Flow.model and DepOf: Dependency Injection for CallableModel

## Overview

This document describes the `@Flow.model` decorator and `DepOf` annotation system for reducing boilerplate when creating `CallableModel` pipelines with dependencies.

**Key features:**
- `@Flow.model` - Decorator that generates `CallableModel` classes from plain functions
- `DepOf[ContextType, ResultType]` - Type annotation for dependency fields
- `resolve()` - Function to access resolved dependency values in class-based models

## Quick Start

### Pattern 1: `@Flow.model` (Recommended for Simple Cases)

```python
from datetime import date, timedelta
from typing import Annotated

from ccflow import Flow, DateRangeContext, GenericResult, DepOf

@Flow.model
def load_records(context: DateRangeContext, source: str) -> GenericResult[dict]:
    return GenericResult(value={"count": 100, "date": str(context.start_date)})

@Flow.model
def compute_stats(
    context: DateRangeContext,
    records: DepOf[..., GenericResult[dict]],  # Dependency field
) -> GenericResult[float]:
    # records is already resolved - just use it directly
    return GenericResult(value=records.value["count"] * 0.05)

# Build pipeline
loader = load_records(source="main_db")
stats = compute_stats(records=loader)

# Execute
ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = stats(ctx)
```

### Pattern 2: Class-Based (For Complex Cases)

Use class-based when you need **configurable transforms** that depend on instance fields:

```python
from datetime import timedelta

from ccflow import CallableModel, DateRangeContext, Flow, GenericResult, DepOf
from ccflow.callable import resolve  # Import resolve for class-based models

class AggregateWithWindow(CallableModel):
    """Aggregate records with configurable lookback window."""

    records: DepOf[..., GenericResult[dict]]
    window: int = 7  # Configurable instance field

    @Flow.call
    def __call__(self, context: DateRangeContext) -> GenericResult[float]:
        # Use resolve() to get the resolved value
        records = resolve(self.records)
        return GenericResult(value=records.value["count"] / self.window)

    @Flow.deps
    def __deps__(self, context: DateRangeContext):
        # Transform uses self.window - this is why we need class-based!
        lookback_ctx = context.model_copy(
            update={"start_date": context.start_date - timedelta(days=self.window)}
        )
        return [(self.records, [lookback_ctx])]

# Usage - different window sizes, same source
loader = load_records(source="main_db")
agg_7 = AggregateWithWindow(records=loader, window=7)
agg_30 = AggregateWithWindow(records=loader, window=30)
```

## When to Use Which Pattern

| Use `@Flow.model` when...      | Use Class-Based when...              |
|--------------------------------|--------------------------------------|
| Simple transformations         | Transforms depend on instance fields  |
| Fixed context transforms       | Need `self.field` in `__deps__`       |
| Less boilerplate is priority   | Full control over resolution         |
| No custom `__deps__` logic     | Complex dependency patterns          |

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
3. For each `DepOf` field containing a `CallableModel`:
   - Apply transform (if any)
   - Call the dependency
   - Store resolved value in context variable
4. Generated `__call__` retrieves resolved values via `resolve()`
5. Original function receives resolved values as arguments

### Class-Based Resolution Flow

1. User calls `model(context)`
2. `_resolve_deps_and_call()` runs
3. For each `DepOf` field containing a `CallableModel`:
   - Check `__deps__` for custom transforms
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

### Decision 4: No Auto-Wrapping Return Values

**What we chose:** Functions must explicitly return `ResultBase` subclass.

**Why:**
- Type annotations remain honest
- Consistent with existing `CallableModel` contract
- `GenericResult(value=x)` is minimal overhead

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

### Limitation: `__deps__` Still Required for Class-Based

Even without transforms, class-based models need `__deps__`:

```python
class Consumer(CallableModel):
    data: DepOf[..., GenericResult[int]]

    @Flow.call
    def __call__(self, context):
        return GenericResult(value=resolve(self.data).value)

    @Flow.deps
    def __deps__(self, context):
        return [(self.data, [context])]  # Boilerplate, but required
```

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
