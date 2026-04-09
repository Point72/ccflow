# Flow.model Design

## Overview

`@Flow.model` turns a plain Python function into a real `CallableModel`.

The design is intentionally narrow:

- ordinary unmarked parameters are regular bound inputs,
- `FromContext[T]` marks the only runtime/contextual inputs,
- `.flow.compute(...)` is the execution entry point for the full DAG,
- `.flow.with_inputs(...)` rewires contextual inputs on one dependency edge,
- upstream `CallableModel`s can still be passed as ordinary arguments.

The goal is that a reader can look at one function signature and immediately
see:

1. which values come from runtime context,
2. which values must be bound as regular configuration or dependencies,
3. how to rewrite contextual inputs for one branch of the graph.

## Core Example

```python
from ccflow import Flow, FromContext


@Flow.model
def foo(a: int, b: FromContext[int]) -> int:
    return a + b


# Build an instance with a=11 bound, then supply b=12 at runtime:
configured = foo(a=11)
result = configured.flow.compute(b=12)
assert result.value == 23  # .value unwraps the GenericResult wrapper

# Or create a different instance that stores b=12 as its contextual default:
prefilled = foo(a=11, b=12)
result = prefilled.flow.compute()
assert result.value == 23
```

> **Note:** When the function returns a plain value (like `int` above) instead
> of a `ResultBase` subclass, `@Flow.model` automatically wraps it in
> `GenericResult`. Access the inner value with `.value`.

This is the core contract:

- `a` is a regular parameter — it must be bound at construction time,
- `b` is contextual because it is marked with `FromContext[int]` — it can come
  from runtime context, a contextual default stored on the model instance, or a
  function default,
- `.flow.compute(...)` may carry extra ambient context for upstream graph
  branches, but it never binds regular parameters.

Nothing is being mutated at execution time in the second example.
`prefilled = foo(a=11, b=12)` constructs a different model instance whose
contextual default for `b` is already `12`. Because `b` is still contextual,
incoming runtime context can still override that default.

This means the following is **invalid**:

```python
foo().flow.compute(a=11, b=12)
# TypeError: compute() cannot bind regular parameter(s): a.
# Bind them at construction time.
```

`a` is not contextual, so it must be bound at construction time (`foo(a=11)`).
By contrast, extra ambient fields that are only needed by upstream
`with_inputs(...)` rewrites are allowed on the kwargs entrypoint for
implicit-`FlowContext` graphs.

## Regular Parameters vs Contextual Parameters

### Regular Parameters

Regular parameters are the unmarked ones.

They can be satisfied by:

- a literal value,
- a default value from the function signature,
- an upstream `CallableModel`.

When an upstream model is supplied, `@Flow.model` evaluates it with the current
context and passes the resolved value into the function. This is how you wire
stages together — just pass one model as an argument to another:

```python
from ccflow import Flow, FlowContext, FromContext


@Flow.model
def load_value(value: FromContext[int], offset: int) -> int:
    return value + offset


@Flow.model
def add(a: int, b: FromContext[int]) -> int:
    return a + b


# Wire load_value into add's 'a' parameter:
model = add(a=load_value(offset=5))

# At runtime, load_value runs first (value=7 + offset=5 = 12),
# then add runs (a=12 + b=12 = 24):
assert model.flow.compute(value=7, b=12).value == 24
```

### Contextual Parameters

Contextual parameters are the ones marked with `FromContext[...]`.

They can be satisfied by:

- runtime context,
- contextual defaults stored on the model instance,
- function defaults.

They cannot be satisfied by `CallableModel` values.

A construction-time value for a contextual parameter is still a default, not a
conversion into a regular bound parameter.

Contextual precedence is:

1. branch-local `.flow.with_inputs(...)` rewrites,
2. incoming runtime context,
3. contextual defaults stored on the model instance,
4. function defaults.

## `.flow.compute(...)`

`.flow.compute(...)` is the ergonomic execution entry point for contextual
execution of the whole DAG.

For generated `@Flow.model` stages it accepts either:

- keyword arguments that become the ambient runtime context bag, or
- one context object.

It does not accept both at the same time.

```python
from ccflow import Flow, FlowContext, FromContext


@Flow.model
def add(a: int, b: FromContext[int]) -> int:
    return a + b


model = add(a=10)
assert model.flow.compute(b=5).value == 15
assert model.flow.compute(FlowContext(b=6)).value == 16
```

For implicit-`FlowContext` models, the kwargs form is intentionally a DAG
entrypoint: it can include extra fields needed only by upstream transformed
dependencies. Regular parameters are still never read from runtime context. If
the root model has an unbound regular parameter whose name appears in
`compute(**kwargs)`, `compute()` raises early instead of silently treating that
value as configuration.

```python
from ccflow import Flow, FromContext


@Flow.model
def source(value: FromContext[int]) -> int:
    return value


@Flow.model
def add(left: int, right: int, bonus: FromContext[int]) -> int:
    return left + right + bonus


base = source()
model = add(
    left=base.flow.with_inputs(value=lambda ctx: ctx.value + 1),
    right=base.flow.with_inputs(value=lambda ctx: ctx.value + 10),
)

assert model.flow.context_inputs == {"bonus": int}
assert model.flow.compute(value=5, bonus=100).value == 121
```

If a regular parameter is already bound on the root model, a same-named key in
`compute(**kwargs)` is treated as ambient context for the graph rather than a
rebind of the root parameter:

```python
from ccflow import Flow, FromContext


@Flow.model
def source(a: FromContext[int]) -> int:
    return a


@Flow.model
def combine(a: int, left: int, bonus: FromContext[int]) -> int:
    return a + left + bonus


model = combine(a=100, left=source())

# Root 'a' stays bound to 100. The runtime 'a=7' is still available to
# upstream graph nodes that read it from context.
assert model.flow.compute(a=7, bonus=5).value == 112
```

`compute()` returns the same result object you would get from `model(context)`,
unless `auto_unwrap=True` is enabled for an auto-wrapped plain return type:

```python
from ccflow import Flow, FromContext


@Flow.model(auto_unwrap=True)
def add(a: int, b: FromContext[int]) -> int:
    return a + b


result = add(a=10).flow.compute(b=5)
assert result == 15
```

## `.flow.with_inputs(...)`

`.flow.with_inputs(...)` rewrites contextual inputs locally for one wrapped
dependency.

```python
from datetime import date, timedelta

from ccflow import DateRangeContext, Flow, FromContext


@Flow.model
def load_revenue(region: str, start_date: FromContext[date], end_date: FromContext[date]) -> float:
    days = (end_date - start_date).days + 1
    return 1000.0 + days * 10.0


@Flow.model
def revenue_growth(current: float, previous: float, start_date: FromContext[date], end_date: FromContext[date]) -> dict:
    return {"window_end": end_date, "growth_pct": round((current - previous) / previous * 100, 2)}


current = load_revenue(region="us")
previous = current.flow.with_inputs(
    start_date=lambda ctx: ctx.start_date - timedelta(days=30),
    end_date=lambda ctx: ctx.end_date - timedelta(days=30),
)

growth = revenue_growth(current=current, previous=previous)
result = growth(DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31)))
```

In this example, `current` and `previous` share the same `load_revenue` model
but see different date windows at runtime. The `with_inputs()` call on
`previous` shifts the dates back 30 days without affecting `current`.

Key rules:

- `with_inputs()` only targets contextual fields,
- transforms are branch-local — they only affect the wrapped dependency, not
  the entire pipeline,
- chained `with_inputs()` calls merge, with the newest transform winning for a
  repeated field.

## Explicit Context Interop

`@Flow.model` still supports an explicit context parameter for cases where the
function needs the whole context object:

```python
from ccflow import DateRangeContext, Flow


@Flow.model
def load_revenue(context: DateRangeContext, region: str) -> float:
    days = (context.end_date - context.start_date).days + 1
    return days * 50.0
```

This path is useful when interoperating with existing code that already uses
typed `ContextBase` subclasses, or when the function genuinely needs access to
the full context rather than individual fields.

You can also keep the `FromContext[...]` style while asking ccflow to validate
those contextual fields against an existing nominal context shape:

```python
from ccflow import DateRangeContext, Flow, FromContext


@Flow.model(context_type=DateRangeContext)
def load_revenue(region: str, start_date: FromContext[date], end_date: FromContext[date]) -> float:
    return 125.0
```

That preserves the primary `FromContext[...]` authoring model while letting
callers pass richer context objects whose relevant fields satisfy the declared
`context_type`.

Do not mix both systems in one function signature. A function with an explicit
`context: ContextBase` parameter cannot also declare `FromContext[...]`
parameters.

## Introspection APIs

Generated models expose three useful introspection helpers:

- `model.flow.context_inputs`: the full contextual contract,
- `model.flow.unbound_inputs`: the contextual fields still required at runtime,
- `model.flow.bound_inputs`: regular bound inputs plus any construction-time
  contextual defaults.

Example:

```python
from ccflow import Flow, FromContext


@Flow.model
def add(a: int, b: FromContext[int], c: FromContext[int] = 5) -> int:
    return a + b + c


model = add(a=10)
assert model.flow.context_inputs == {"b": int, "c": int}
assert model.flow.unbound_inputs == {"b": int}
assert model.flow.bound_inputs == {"a": 10}
```

## Lazy Dependencies

`Lazy[T]` defers evaluation of an upstream dependency until the function body
explicitly calls it. This is useful when a dependency is expensive and only
needed conditionally:

```python
from ccflow import Flow, FlowContext, FromContext, Lazy


@Flow.model
def load_value(value: FromContext[int]) -> int:
    return value * 10


@Flow.model
def maybe_use(current: int, fallback: Lazy[int], threshold: FromContext[int]) -> int:
    if current > threshold:
        return current          # fallback is never evaluated
    return fallback()           # evaluate only when needed


model = maybe_use(current=50, fallback=load_value())

# current (50) > threshold (10), so load_value never runs:
assert model.flow.compute(value=3, threshold=10).value == 50

# current (5) <= threshold (10), so load_value runs (3 * 10 = 30):
model2 = maybe_use(current=5, fallback=load_value())
assert model2.flow.compute(value=3, threshold=10).value == 30
```

Without `Lazy[T]`, the upstream model would always run. With it, the function
controls exactly when (and whether) the dependency executes.

## When To Use `@Flow.model`

Use `@Flow.model` when:

- the stage logic is naturally a plain function,
- you want ordinary arguments to look like ordinary Python function parameters,
- the contextual contract is small and explicit,
- the main goal is easy graph authoring on top of existing ccflow machinery.

Use a hand-written class-based `CallableModel` when:

- the model needs custom methods or substantial internal state,
- the full context object is the natural primary interface,
- the stage is no longer best expressed as one function and a small amount of
  wiring.

## Troubleshooting

**`compute()` says a field is not contextual**

That field is a regular parameter. Bind it at construction time. Only
`FromContext[...]` fields belong in `compute()`.

**`with_inputs()` rejects a field**

`with_inputs()` only rewrites contextual inputs. If you are trying to attach one
stage to another, pass the upstream model as a regular argument at construction
time.

**A contextual parameter still shows up in `context_inputs` after I bound it**

That is expected. `context_inputs` reports the full contextual contract.
`unbound_inputs` reports only the contextual values still needed at runtime.

**A shared dependency runs more than once**

`@Flow.model` authors the graph cleanly, but execution still follows the normal
ccflow evaluator path. If you need deduplication or graph scheduling, use the
appropriate evaluators and cache settings just as you would for class-based
`CallableModel`s.
