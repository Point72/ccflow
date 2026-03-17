# Flow.model Design

## Overview

`@Flow.model` turns a plain Python function into a real `CallableModel`.

The core goals are:

- keep the authoring model close to an ordinary function,
- preserve the existing evaluator / registry / serialization machinery,
- make deferred execution explicit with `.flow.compute(...)` and `.flow.with_inputs(...)`,
- allow callers to pass either literal values or upstream models for ordinary parameters.

`@Flow.model` is syntactic sugar over the existing ccflow framework. The
generated object is still a standard `CallableModel`, so you can execute it the
same way as any other model by calling it with a context object. The
`.flow.compute(...)` helper is an explicit, ergonomic way to mark the deferred
execution boundary when supplying runtime inputs as keyword arguments.

## Core Patterns

### Default Deferred Style

This is the most ergonomic mode. Bind some parameters up front, then provide
the remaining runtime inputs later.

```python
from ccflow import Flow, FlowContext


@Flow.model
def add(x: int, y: int) -> int:
    return x + y


model = add(x=10)

# Explicit deferred entry point
assert model.flow.compute(y=5) == 15

# Standard CallableModel call path
assert model(FlowContext(y=5)).value == 15

shifted = model.flow.with_inputs(y=lambda ctx: ctx.y * 2)
assert shifted.flow.compute(y=5) == 20
```

In this mode:

- bound parameters are model configuration,
- unbound parameters become runtime inputs for that model instance.

### Explicit Context Parameter

```python
from ccflow import DateRangeContext, Flow


@Flow.model
def load_revenue(context: DateRangeContext, region: str) -> float:
    return 125.0
```

This is the most direct mode. The function receives a normal context object and
returns either a `ResultBase` subclass or a plain value. Plain values are
wrapped into `GenericResult` automatically by the generated model.

### `context_args`

```python
from datetime import date

from ccflow import Flow


@Flow.model(context_args=["start_date", "end_date"])
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    return 125.0
```

This keeps the function signature focused on the inputs it actually uses while
still producing a `CallableModel` that accepts a context at runtime.

Use `context_args` when certain parameters are semantically the execution
context and you want that split to be explicit and stable across model
instances.

When the requested shape matches a built-in context like
`DateRangeContext(start_date, end_date)`, the generated model uses that type.
Otherwise it falls back to `FlowContext`.

### Upstream Models as Normal Arguments

Any non-context parameter can be given either:

- a literal value, or
- another `CallableModel` / `BoundModel`.

If a model is passed, it is evaluated with the current context and its result is
unwrapped before the function is called.

```python
from ccflow import DateRangeContext, Flow


@Flow.model
def load_revenue(context: DateRangeContext, region: str) -> float:
    return 125.0


@Flow.model
def double_revenue(_: DateRangeContext, revenue: float) -> float:
    return revenue * 2


revenue = load_revenue(region="us")
model = double_revenue(revenue=revenue)
result = model.flow.compute(start_date="2024-01-01", end_date="2024-01-31")
```

This is the main composition story for the core API.

### `.flow.with_inputs(...)`

`with_inputs` is how a caller rewires context locally for one upstream model.

```python
from datetime import date, timedelta

from ccflow import DateRangeContext, Flow


@Flow.model(context_args=["start_date", "end_date"])
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    days = (end_date - start_date).days + 1
    return 1000.0 + days * 10.0


@Flow.model(context_args=["start_date", "end_date"])
def revenue_growth(start_date: date, end_date: date, current: float, previous: float) -> dict:
    return {
        "window_end": end_date,
        "growth_pct": round((current - previous) / previous * 100, 2),
    }


current = load_revenue(region="us")
previous = current.flow.with_inputs(
    start_date=lambda ctx: ctx.start_date - timedelta(days=30),
    end_date=lambda ctx: ctx.end_date - timedelta(days=30),
)

model = revenue_growth(current=current, previous=previous)
ctx = DateRangeContext(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)

direct = model(ctx).value
computed = model.flow.compute(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)

assert direct == computed
```

The transform is local to the bound upstream model. The parent model continues
to receive the original context.

### `.flow.compute(...)`

`compute` is the ergonomic entry point for deferred execution:

```python
from ccflow import Flow


@Flow.model
def add(x: int, y: int) -> int:
    return x + y


model = add(x=10)
assert model.flow.compute(y=5) == 15
```

It validates the supplied keyword arguments against the generated context
schema, creates a `FlowContext`, executes the model, and unwraps
`GenericResult.value` if needed.

It is not the only execution path. Because the generated object is still a
standard `CallableModel`, calling `model(context)` remains fully supported.

## FieldExtractor

Accessing an unknown public attribute on a `@Flow.model` instance returns a
`FieldExtractor`. It is itself a `CallableModel` that runs the source model,
then extracts the named field from the result (via `getattr` or dict key
access).

```python
from ccflow import ContextBase, Flow, GenericResult


class TrainingContext(ContextBase):
    seed: int


@Flow.model
def prepare(context: TrainingContext) -> GenericResult[dict]:
    s = context.seed
    return GenericResult(value={"X_train": [s, s * 2], "y_train": [s * 10]})


@Flow.model
def train(context: TrainingContext, X: list, y: list) -> GenericResult[int]:
    return GenericResult(value=sum(X) + sum(y))


prepared = prepare()
model = train(X=prepared.X_train, y=prepared.y_train)
```

Multiple extractors from the same source share the source model instance. If
caching is enabled the source is evaluated only once.

## Lazy Inputs

`Lazy[T]` marks a parameter as on-demand. Instead of eagerly resolving an
upstream model, the generated model passes a zero-argument thunk. The thunk
caches its first result. Lazy dependencies are excluded from the `__deps__`
graph, so they are not pre-evaluated by the evaluator infrastructure.

```python
from ccflow import Flow, Lazy


@Flow.model
def source(value: int) -> int:
    return value * 10


@Flow.model
def maybe_use_source(value: int, data: Lazy[int]) -> int:
    if value > 10:
        return value
    return data()
```

## FlowContext

`FlowContext` is the universal frozen carrier for generated contexts that do
not map to a dedicated built-in context type.

The implementation stays intentionally small:

- context validation is driven by `TypedDict` + `TypeAdapter`,
- runtime execution uses one reusable `FlowContext` type,
- public pydantic iteration (`dict(context)`) is used instead of pydantic
  internals.

## BoundModel

`.flow.with_inputs(...)` returns a `BoundModel`, which is just a thin wrapper
around:

- the original model, and
- a mapping of input transforms.

At call time it:

1. converts the incoming context into a plain dictionary,
1. applies the configured transforms,
1. rebuilds a `FlowContext`,
1. delegates to the wrapped model.

That keeps transformed dependency wiring explicit without adding special
annotation machinery to the core API.

## Flow.call with `auto_context`

Separately from `@Flow.model`, `Flow.call(auto_context=...)` provides a similar
convenience for class-based `CallableModel`s. Instead of defining a separate
`ContextBase` subclass, the decorator generates one from the function's
keyword-only parameters.

```python
from ccflow import CallableModel, Flow, GenericResult


class MyModel(CallableModel):
    @Flow.call(auto_context=True)
    def __call__(self, *, x: int, y: str = "default") -> GenericResult:
        return GenericResult(value=f"{x}-{y}")
```

Passing a `ContextBase` subclass (e.g., `auto_context=DateContext`) makes the
generated context inherit from that class, so it remains compatible with
infrastructure that expects the parent type.

The generated class is registered via `create_ccflow_model` for serialization
support.
