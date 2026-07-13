# Defining Workflows

So far your configuration objects have only held data. This tutorial makes them *run*. You will meet the three ingredients of a workflow step — a **result**, a **context**, and a **callable model** — build one, run it, and see how the `@Flow.call` decorator adds type checking, logging, and other behavior behind the scenes.

Follow along in a Python session. For the full catalog of built-in results, contexts, and evaluators, see the [Reference](Reference); this tutorial teaches the pattern with a few representative examples.

A workflow step in `ccflow` is a `CallableModel`: something you call *with a context* that returns *a result*. Three abstractions make this work:

- a **result** type to hold what a step returns,
- a **context** type to parameterize the step at runtime,
- the **`@Flow.call`** decorator, through which the framework injects type checking, logging, caching, and alternative evaluation.

## Results

Every step returns a `ResultBase`. The simplest is `GenericResult`, which holds anything in its `value`:

```python
from ccflow import GenericResult
print(GenericResult(value="Anything goes here"))
#> GenericResult(value='Anything goes here')
```

You can ask for type safety on the value using Python generics:

```python
result = GenericResult[str](value="Any string")

try:
    GenericResult[str](value={"x": "foo", "y": 5.0})
except ValueError as e:
    print(e)
#> 1 validation error for GenericResult[str] ...
```

Pydantic validation also cuts boilerplate — a bare value is validated into the wrapper:

```python
print(GenericResult.model_validate("Any string"))
#> GenericResult(value='Any string')
```

When you know the shape of your output, define a proper result schema by subclassing `ResultBase`:

```python
from ccflow import ResultBase

class MyResult(ResultBase):
    x: str
    y: float

print(MyResult(x="foo", y=5.0))
#> MyResult(x='foo', y=5.0)
```

`ccflow` ships typed results for common data structures (pandas, numpy, Arrow, xarray, Narwhals). The [Contexts and Results](Contexts-and-Results) reference lists them all.

## Contexts

A context carries the parameters that vary between runs. Some steps need none — use `NullContext`:

```python
from ccflow import NullContext
print(NullContext())
#> NullContext()
```

`GenericContext` mirrors `GenericResult` for ad-hoc parameters:

```python
from ccflow import GenericContext
print(GenericContext[str].model_validate(100))
#> GenericContext[str](value='100')
```

And you define your own when a workflow has specific parameters:

```python
from ccflow import ContextBase
from datetime import datetime

class LocationTimestampContext(ContextBase):
    latitude: float
    longitude: float
    timestamp: datetime
```

Contexts are **frozen** (immutable) and hashable by default, so the framework can use them as cache keys. `ccflow` also provides date-oriented contexts (`DateContext`, `DateRangeContext`, and more) with convenient validation — see the [Contexts and Results](Contexts-and-Results) reference.

## Your first callable model

Put the pieces together with a `CallableModel`. You implement `__call__` as a function of the context and decorate it with `@Flow.call`. Here is the classic [FizzBuzz](https://en.wikipedia.org/wiki/Fizz_buzz) problem as a model:

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext

class FizzBuzzModel(CallableModel):
    fizz: str = "Fizz"
    buzz: str = "Buzz"

    @Flow.call
    def __call__(self, context: GenericContext[int]) -> GenericResult[list[int | str]]:
        n = context.value
        result = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                result.append(f"{self.fizz}{self.buzz}")
            elif i % 3 == 0:
                result.append(self.fizz)
            elif i % 5 == 0:
                result.append(self.buzz)
            else:
                result.append(i)
        return result

model = FizzBuzzModel()
print(model(15))
#> GenericResult[list[Union[int, str]]](value=[1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz', 11, 'Fizz', 13, 14, 'FizzBuzz'])
```

Two things worth noticing. You called `model(15)` with a bare integer, not a `GenericContext[int]` — and you returned a bare list, not a `GenericResult`. The `@Flow.call` decorator did the conversions for you.

The model's fields (`fizz`, `buzz`) are configuration; the context (`15`) is the runtime parameter. That split is the whole idea: configure once, run across many contexts.

## Type checking comes for free

`@Flow.call` runs pydantic validation on the way in and out, so invalid inputs fail clearly:

```python
try:
    model("not an integer")
except ValueError as e:
    print(e)
#> 1 validation error for GenericContext[int] ...
```

By default the decorator infers the context and result types from your `__call__` signature:

```python
print(model.context_type)
#> <class 'ccflow.context.GenericContext[int]'>
print(model.result_type)
#> <class 'ccflow.result.generic.GenericResult[list[Union[int, str]]]'>
```

When the types depend on configuration, override the `context_type` / `result_type` properties instead of annotating:

```python
from typing import Type

class DynamicTypedModel(CallableModel):
    input_type: Type
    output_type: Type

    @property
    def context_type(self):
        return GenericContext[self.input_type]

    @property
    def result_type(self):
        return GenericResult[self.output_type]

    @Flow.call
    def __call__(self, context):
        return context.value

print(DynamicTypedModel(input_type=int, output_type=str)(5))
#> GenericResult[str](value='5')
```

## Controlling the Flow decorator

`@Flow.call` is the seam where framework behavior is layered on, controlled by `FlowOptions`. You can set options four ways: as arguments to the decorator, via the `FlowOptionsOverride` context manager, on the model's `meta.options`, or per call with `_options`.

Turn off result validation for one model:

```python
class NoValidationModel(CallableModel):
    @Flow.call(validate_result=False)
    def __call__(self, context: GenericContext[str]) -> GenericResult[float]:
        return "foo"

print(NoValidationModel()("foo"))
#> foo
```

Raise the log level for a call and every sub-call, scoped by a context manager:

```python
import logging
from ccflow import FlowOptionsOverride

model = FizzBuzzModel()
with FlowOptionsOverride(options={"log_level": logging.WARN}):
    _ = model(15)
#[FizzBuzzModel]: Start evaluation of __call__ on GenericContext[int](value=15).
#[FizzBuzzModel]: End evaluation of __call__ on GenericContext[int](value=15) (time elapsed: ...).
```

Or pass options to a single call:

```python
_ = model(15, _options={"log_level": logging.WARN})
```

The full set of options lives on the `FlowOptions` schema — see [Flow Options](Core-Types#flow-options) in the reference. The `meta` attribute on every `CallableModel` also carries a `name` and `description` (set automatically when models load from Hydra configs) and can hold `options` so they travel with the model in a config file.

## Evaluators run your steps

You have been running models the "standard" way — Python calls `__call__` directly. An **evaluator** changes *how* a model runs. The default logs each evaluation; others cache, evaluate an explicit dependency graph, retry on failure, or distribute work. Because an evaluator is set through `FlowOptions`, adding these behaviors does not change your step at all.

You set an evaluator the same way you set any option:

```python
from ccflow.evaluators import MemoryCacheEvaluator

with FlowOptionsOverride(options={"cacheable": True, "evaluator": MemoryCacheEvaluator()}):
    _ = model(15)
```

That is the whole idea; the practical guides cover the two you will reach for most:

- [Cache Results](Cache-Results) — avoid redundant work with the `MemoryCacheEvaluator`, and evaluate dependency graphs.
- [Retry on Failure](Retry-on-Failure) — make flaky steps resilient.

The full list of evaluators is in the [Built-in Models](Built-in-Models) reference.

## What you learned

- A workflow step is a `CallableModel` you call with a context to get a result.
- `@Flow.call` handles type conversion and validation, and is where framework behavior is injected.
- `FlowOptions` (via the decorator, an override, `meta`, or `_options`) tunes that behavior.
- Evaluators decide *how* steps run, without touching the steps themselves.

## Next steps

- [Building an ETL Pipeline](Building-an-ETL-Pipeline) — chain callable models into an end-to-end pipeline.
- [Bind Logic to Configs](Bind-Logic-to-Configs) — patterns for attaching logic, including publishers.
- [Core Concepts](Core-Concepts) — how contexts, results, and evaluators fit the bigger picture.
