- [Base Model](#base-model)
- [Callable Model](#callable-model)
- [Model Registry](#model-registry)
- [Models](#models)
- [Publishers](#publishers)
- [Evaluators](#evaluators)
- [Results](#results)

`ccflow` (Composable Configuration Flow) is a collection of tools for workflow
configuration, orchestration, and dependency injection. It is intended to stay
flexible across ETL workflows, model training, report generation, and service
configuration.

## Base Model

`BaseModel` is the core pydantic-based model class used throughout `ccflow`.
Models are regular data objects with validation, serialization, and registry
support.

## Callable Model

`CallableModel` is a `BaseModel` that can be executed with a context
(`ContextBase`) and produces a result (`ResultBase`).

### Flow.model Decorator

`@Flow.model` is the plain-function front door to `CallableModel`.

It generates a real `CallableModel` class with proper `__call__` and `__deps__`
methods, so it still plugs into the normal evaluator, registry, cache, Hydra,
and serialization machinery.

If the function returns a plain value instead of a `ResultBase`, the generated
model wraps it in `GenericResult`.

`@Flow.transform` is the companion API for defining serializable contextual
rewrites used by `.flow.with_inputs(*patches, **field_overrides)`. Transforms
that return a mapping are **patch transforms** (passed positionally); transforms
that return a single value are **field transforms** (passed by keyword).

#### Primary Authoring Model

`FromContext[T]` is the only marker for runtime/contextual inputs.

```python
from ccflow import Flow, FromContext


@Flow.model
def add(a: int, b: FromContext[int]) -> int:
    return a + b


model = add(a=10)
assert model.flow.compute(b=5).value == 15

prefilled = add(a=10, b=7)
assert prefilled.flow.compute().value == 17
```

That means:

- `a` is a regular parameter,
- `b` is contextual,
- `.flow.compute(...)` only accepts contextual inputs.

`prefilled = add(a=10, b=7)` creates a different model instance with a stored
contextual default for `b`. `compute()` does not mutate the model; it resolves
the remaining contextual inputs for that execution.

Regular parameters can be satisfied by:

- literal values,
- function defaults,
- upstream `CallableModel`s.

Contextual parameters can be satisfied by:

- runtime context,
- contextual defaults stored on the model instance,
- function defaults.

Contextual parameters cannot be bound to `CallableModel` values.

#### Nominal Context Validation

You can keep the `FromContext[...]` style while validating those fields against
an existing context type:

```python
from datetime import date
from ccflow import DateRangeContext, Flow, FromContext


@Flow.model(context_type=DateRangeContext)
def load_data(source: str, start_date: FromContext[date], end_date: FromContext[date]) -> float:
    return 125.0
```

This validates/coerces the named `FromContext[...]` fields against
`DateRangeContext`, but generated `@Flow.model` instances still report
`FlowContext` as their runtime `context_type`.

If the function genuinely needs the runtime context object itself inside the
function body on each call, write a normal `CallableModel` subclass instead of
using `@Flow.model`.

#### Composing Dependencies

Passing an upstream model as an ordinary argument is the main composition story.

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


@Flow.transform
def previous_window(start_date: FromContext[date], end_date: FromContext[date], days: int) -> dict[str, object]:
    return {
        "start_date": start_date - timedelta(days=days),
        "end_date": end_date - timedelta(days=days),
    }


current = load_revenue(region="us")

# Reuse the same model with a shifted date window for "previous":
previous = current.flow.with_inputs(previous_window(days=30))

growth = revenue_growth(current=current, previous=previous)

# Execute — current sees Jan 2024, previous sees Dec 2023:
ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = growth(ctx)
```

#### Deferred Execution Helpers

`model.flow.compute(...)` accepts either contextual keyword arguments or one
context object and returns the same result as `model(context)` (unless
`auto_unwrap=True` is set, in which case auto-wrapped `GenericResult` values
are unwrapped to plain values).

`model.flow.with_inputs(*patches, **field_overrides)` rewrites contextual inputs
on one dependency edge. Positional args are patch transforms that return a
mapping of contextual field names. Keyword overrides accept either literal
values or field transforms. The rewrite is branch-local and serializes as data,
not raw callable state.

Generated models also expose introspection helpers:

```python
model = add(a=10)
model.flow.context_inputs   # {"b": int}        — the full contextual contract
model.flow.unbound_inputs   # {"b": int}        — contextual fields still needed at runtime
model.flow.bound_inputs     # {"a": 10}         — all construction-time values
```

#### Lazy Dependencies

`Lazy[T]` is the lazy type-level marker for dependency parameters.

```python
from ccflow import Flow, FromContext, Lazy


@Flow.model
def load_value(value: FromContext[int]) -> int:
    return value * 10


@Flow.model
def choose(current: int, deferred: Lazy[int], threshold: FromContext[int]) -> int:
    if current > threshold:
        return current
    return deferred()
```

Use `Lazy[T]` when a dependency is expensive and the function should decide
whether to execute it. Without `Lazy[T]`, all upstream `CallableModel`
dependencies are evaluated eagerly before the function runs. With it, the
dependency is wrapped in a zero-argument thunk that the function calls
explicitly (or never calls, avoiding the cost entirely).

## Model Registry

The model registry lets you register models by name and resolve them later,
including from config-driven workflows.

- root registry access: `ModelRegistry.root()`
- add and remove models by name
- reuse shared instances through registry references

## Models

The `ccflow.models` package contains concrete model implementations that build
on the framework primitives.

Use these when you want reusable, prebuilt model classes instead of authoring
your own `CallableModel` or `@Flow.model` stage.

## Publishers

Publishers handle result publication and side-effectful output sinks.

They are useful when a workflow result needs to be written to an external
system rather than only returned to the caller.

## Evaluators

Evaluators control how `CallableModel`s execute.

Key point for `@Flow.model`: it does not create a new execution engine. It
authors models that still run through the existing evaluator stack.

Depending on your evaluator setup, you can add logging, caching, graph-aware
execution, or custom execution policies.

## Results

`ResultBase` is the common base class for workflow results.

`GenericResult[T]` is the default wrapper used when:

- a model naturally wants one value payload, or
- a `@Flow.model` function returns a plain Python value instead of a concrete
  `ResultBase` subclass.
