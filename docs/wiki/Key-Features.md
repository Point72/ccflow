- [Base Model](#base-model)
- [Callable Model](#callable-model)
- [Model Registry](#model-registry)
- [Models](#models)
- [Publishers](#publishers)
- [Evaluators](#evaluators)
- [Results](#results)

`ccflow` (Composable Configuration Flow) is a collection of tools for workflow configuration, orchestration, and dependency injection.
It is intended to be flexible enough to handle diverse use cases, including data retrieval, validation, transformation, and loading (i.e. ETL workflows), model training, microservice configuration, and automated report generation.

## Base Model

Central to `ccflow` is the `BaseModel` class.
`BaseModel` is the base class for models in the `ccflow` framework.
A model is basically a data class (class with attributes).
The naming was inspired by the open source library [Pydantic](https://docs.pydantic.dev/latest/) (`BaseModel` actually inherits from the Pydantic base model class).

## Callable Model

`CallableModel` is the base class for a special type of `BaseModel` which can be called.
`CallableModel`'s are called with a context (something that derives from `ContextBase`) and returns a result (something that derives from `ResultBase`).
As an example, you may have a `SQLReader` callable model that when called with a `DateRangeContext` returns a `ArrowResult` (wrapper around a Arrow table) with data in the date range defined by the context by querying some SQL database.

### Flow.model Decorator

The `@Flow.model` decorator provides a simpler way to define `CallableModel`s
using plain Python functions instead of classes. It automatically generates a
standard `CallableModel` class with proper `__call__` and `__deps__` methods,
so it still uses the normal ccflow framework for evaluation, caching,
serialization, and registry loading.

If a `@Flow.model` function returns a plain value instead of a `ResultBase`
subclass, the generated model automatically wraps it in `GenericResult` at
runtime so it still behaves like a normal `CallableModel`.

You can execute a generated model in two equivalent ways:

- call it directly with a context object: `model(ctx)`
- use `.flow.compute(...)` to supply runtime inputs as keyword arguments

`.flow.compute(...)` is mainly an explicit, ergonomic way to mark the deferred
execution point.

#### Context Modes

There are three ways to define how a `@Flow.model` function receives its
runtime context.

**Mode 1 — Explicit context parameter:**

The function takes a `context` parameter (or `_` if unused) annotated with a
`ContextBase` subclass. This is the most direct mode and behaves like a
traditional `CallableModel.__call__`.

```python
from datetime import date
from ccflow import Flow, GenericResult, DateRangeContext

@Flow.model
def load_data(context: DateRangeContext, source: str) -> GenericResult[dict]:
    return GenericResult(value=query_db(source, context.start_date, context.end_date))

loader = load_data(source="my_database")

ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = loader(ctx)
```

**Mode 2 — Unpacked context with `context_args`:**

Instead of receiving a context object, you list which parameters should come
from the context at runtime. The remaining parameters are model configuration.

```python
from datetime import date
from ccflow import Flow, GenericResult, DateRangeContext

@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def load_data(start_date: date, end_date: date, source: str) -> GenericResult[str]:
    return GenericResult(value=f"{source}:{start_date} to {end_date}")

loader = load_data(source="my_database")

# Opt in explicitly when you want compatibility with an existing context type
assert loader.context_type == DateRangeContext

ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = loader(ctx)
```

By default, `context_args` models use `FlowContext`, a universal frozen carrier
for the validated fields. If you want the generated model to advertise and
accept an existing `ContextBase` subclass, pass `context_type=...` explicitly.

Use `context_args` when some parameters are semantically "the execution
context" and you want that split to stay stable and explicit:

- the runtime context should be stable across instances
- the split between config and runtime inputs matters semantically
- the model is naturally "run over a context" such as date windows,
  partitions, or scenarios
- you want the generated model to accept a specific existing context type
  such as `DateRangeContext`

**Mode 3 — Dynamic deferred style (no explicit context):**

When there is no `context` parameter and no `context_args`, all parameters are
potential configuration or runtime inputs. Parameters provided at construction
are bound (configuration); everything else comes from the context at runtime.

```python
from ccflow import Flow

@Flow.model
def add(x: int, y: int) -> int:
    return x + y

model = add(x=10)

# `x` is bound when the model is created.
# `y` is supplied later at execution time.
assert model.flow.compute(y=5).value == 15

# `.flow.with_inputs(...)` rewrites runtime inputs for this call path.
doubled_y = model.flow.with_inputs(y=lambda ctx: ctx.y * 2)
assert doubled_y.flow.compute(y=5).value == 20
```

#### Composing Dependencies

Any non-context parameter can be bound either to a literal value or to another
`CallableModel`. If you pass an upstream model, `@Flow.model` evaluates it with
the current context and passes the resolved value into your function.

```python
from datetime import date, timedelta
from ccflow import DateRangeContext, Flow

@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    days = (end_date - start_date).days + 1
    return 1000.0 + days * 10.0

@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def revenue_growth(
    start_date: date,
    end_date: date,
    current: float,
    previous: float,
) -> dict:
    return {
        "window_end": end_date,
        "growth_pct": round((current - previous) / previous * 100, 2),
    }

current = load_revenue(region="us")
previous = current.flow.with_inputs(
    start_date=lambda ctx: ctx.start_date - timedelta(days=30),
    end_date=lambda ctx: ctx.end_date - timedelta(days=30),
)
growth = revenue_growth(current=current, previous=previous)

ctx = DateRangeContext(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)

# Standard ccflow execution
direct = growth(ctx)

# Equivalent explicit deferred entry point
computed = growth.flow.compute(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)

assert direct == computed
```

#### Deferred Execution Helpers

**`.flow.compute(**kwargs)`** validates the keyword arguments against the
generated context schema, wraps them in a `FlowContext`, and calls the model.
It returns the same result object you would get from `model(context)`.

**`.flow.with_inputs(**transforms)`** returns a `BoundModel` that applies
context transforms before delegating to the underlying model. Each transform
is either a static value or a `(ctx) -> value` callable. Transforms are local
to the wrapped model — upstream models never see them.

```python
from ccflow import Flow, FlowContext

@Flow.model
def add(x: int, y: int) -> int:
    return x + y

model = add(x=10)
assert model.flow.compute(y=5).value == 15

shifted = model.flow.with_inputs(y=lambda ctx: ctx.y * 2)
assert shifted.flow.compute(y=5).value == 20

# You can also call with a context object directly
ctx = FlowContext(y=5)
assert model(ctx).value == 15
assert shifted(ctx).value == 20
```

#### Lazy Dependencies with `Lazy[T]`

Mark a parameter with `Lazy[T]` to defer its evaluation. Instead of eagerly
resolving the upstream model, the generated model passes a zero-argument thunk
that evaluates on first call and caches the result. The thunk unwraps
`GenericResult` automatically, so `T` should be the inner value type.

```python
from ccflow import ContextBase, Flow, GenericResult, Lazy

class SimpleContext(ContextBase):
    value: int

@Flow.model
def fast_path(context: SimpleContext) -> GenericResult[int]:
    return GenericResult(value=context.value)

@Flow.model
def slow_path(context: SimpleContext) -> GenericResult[int]:
    return GenericResult(value=context.value * 100)

@Flow.model
def smart_selector(
    context: SimpleContext,
    fast: int,        # Eagerly resolved and unwrapped
    slow: Lazy[int],  # Deferred — receives a thunk returning unwrapped int
    threshold: int = 10,
) -> GenericResult[int]:
    if fast > threshold:
        return GenericResult(value=fast)
    return GenericResult(value=slow())  # Evaluated only when called

model = smart_selector(
    fast=fast_path(),
    slow=slow_path(),
    threshold=10,
)
```

`Lazy` dependencies are excluded from the model's `__deps__` graph, so they
are not pre-evaluated by the evaluator infrastructure.

#### Decorator Options

`@Flow.model(...)` accepts the same options as `Flow.call` to control execution
behavior:

- `cacheable` — enable caching of results
- `volatile` — mark as volatile (always re-execute)
- `log_level` — logging verbosity
- `validate_result` — validate return type
- `verbose` — verbose logging output
- `evaluator` — custom evaluator

When not explicitly set, these inherit from any active `FlowOptionsOverride`.

#### Hydra / YAML Configuration

`@Flow.model` decorated functions work seamlessly with Hydra configuration and
the `ModelRegistry`:

```yaml
# config.yaml
data:
  _target_: mymodule.load_data
  source: my_database

transformed:
  _target_: mymodule.transform_data
  raw_data: data  # Reference by registry name (same instance is shared)

aggregated:
  _target_: mymodule.aggregate_data
  transformed: transformed  # Reference by registry name
```

```python
from ccflow import ModelRegistry

registry = ModelRegistry.root()
registry.load_config_from_path("config.yaml")

# References by name ensure the same object instance is shared
model = registry["aggregated"]
```

### Flow.call with `auto_context`

For class-based `CallableModel`s, `Flow.call(auto_context=...)` provides a
similar convenience. Instead of defining a separate `ContextBase` subclass, the
decorator generates one from the function's keyword-only parameters.

```python
from ccflow import CallableModel, Flow, GenericResult

class MyModel(CallableModel):
    @Flow.call(auto_context=True)
    def __call__(self, *, x: int, y: str = "default") -> GenericResult:
        return GenericResult(value=f"{x}-{y}")

model = MyModel()
result = model(x=42, y="hello")
assert result.value == "42-hello"
```

You can also pass a parent context class so the generated context inherits
from it:

```python
from datetime import date
from ccflow import CallableModel, DateContext, Flow, GenericResult

class MyModel(CallableModel):
    @Flow.call(auto_context=DateContext)
    def __call__(self, *, date: date, extra: int = 0) -> GenericResult:
        return GenericResult(value=date.day + extra)
```

The generated context class is a proper `ContextBase` subclass, so it works
with all existing evaluator and registry infrastructure.

## Model Registry

A `ModelRegistry` is a named collection of models.
A `ModelRegistry` can be loaded from YAML configuration, which means you can define a collection of models using YAML.
This is really powerful because this gives you a easy way to define a collection of Python objects via configuration.

## Models

Although you are free to define your own models (`BaseModel` implementations) to use in your flow graph,
`ccflow` comes with some models that you can use off the shelf to solve common problems. `ccflow` comes with a range of models for reading data.

The following table summarizes the available models.

> [!NOTE]
>
> Some models are still in the process of being open sourced.

## Publishers

`ccflow` also comes with a range of models for writing data.
These are referred to as publishers.
You can "chain" publishers and callable models using `PublisherModel` to call a `CallableModel` and publish the results in one step.
In fact, `ccflow` comes with several implementations of `PublisherModel` for common publishing use cases.

The following table summarizes the "publisher" models.

> [!NOTE]
>
> Some models are still in the process of being open sourced.

| Name                         | Path                | Description                                                                                                                                                 |
| :--------------------------- | :------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DictTemplateFilePublisher`  | `ccflow.publishers` | Publish data to a file after populating a Jinja template.                                                                                                   |
| `GenericFilePublisher`       | `ccflow.publishers` | Publish data using a generic "dump" Callable. Uses `smart_open` under the hood so that local and cloud paths are supported.                                 |
| `JSONPublisher`              | `ccflow.publishers` | Publish data to file in JSON format.                                                                                                                        |
| `PandasFilePublisher`        | `ccflow.publishers` | Publish a pandas data frame to a file using an appropriate method on pd.DataFrame. For large-scale exporting (using parquet), see `PandasParquetPublisher`. |
| `NarwhalsFilePublisher`      | `ccflow.publishers` | Publish a narwhals data frame to a file using an appropriate method on nw.DataFrame.                                                                        |
| `PicklePublisher`            | `ccflow.publishers` | Publish data to a pickle file.                                                                                                                              |
| `PydanticJSONPublisher`      | `ccflow.publishers` | Publish a pydantic model to a json file. See [Pydantic modeljson](https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump)                 |
| `YAMLPublisher`              | `ccflow.publishers` | Publish data to file in YAML format.                                                                                                                        |
| `CompositePublisher`         | `ccflow.publishers` | Highly configurable, publisher that decomposes a pydantic BaseModel or a dictionary into pieces and publishes each piece separately.                        |
| `PrintPublisher`             | `ccflow.publishers` | Print data using python standard print.                                                                                                                     |
| `LogPublisher`               | `ccflow.publishers` | Print data using python standard logging.                                                                                                                   |
| `PrintJSONPublisher`         | `ccflow.publishers` | Print data in JSON format.                                                                                                                                  |
| `PrintYAMLPublisher`         | `ccflow.publishers` | Print data in YAML format.                                                                                                                                  |
| `PrintPydanticJSONPublisher` | `ccflow.publishers` | Print pydantic model as json. See https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump_json                                             |
| `ArrowDatasetPublisher`      | *Coming Soon!*      |                                                                                                                                                             |
| `PandasDeltaPublisher`       | *Coming Soon!*      |                                                                                                                                                             |
| `EmailPublisher`             | *Coming Soon!*      |                                                                                                                                                             |
| `MatplotlibFilePublisher`    | *Coming Soon!*      |                                                                                                                                                             |
| `MLFlowArtifactPublisher`    | *Coming Soon!*      |                                                                                                                                                             |
| `MLFlowPublisher`            | *Coming Soon!*      |                                                                                                                                                             |
| `PandasParquetPublisher`     | *Coming Soon!*      |                                                                                                                                                             |
| `PlotlyFilePublisher`        | *Coming Soon!*      |                                                                                                                                                             |
| `XArrayPublisher`            | *Coming Soon!*      |                                                                                                                                                             |

## Evaluators

`ccflow` comes with "evaluators" that allows you to evaluate (i.e. run) `CallableModel` s in different ways.

The following table summarizes the "evaluator" models.

> [!NOTE]
>
> Some models are still in the process of being open sourced.

| Name                                | Path                | Description                                                                                                                    |
| :---------------------------------- | :------------------ | :----------------------------------------------------------------------------------------------------------------------------- |
| `LazyEvaluator`                     | `ccflow.evaluators` | Evaluator that only actually runs the callable once an attribute of the result is queried (by hooking into `__getattribute__`) |
| `LoggingEvaluator`                  | `ccflow.evaluators` | Evaluator that logs information about evaluating the callable.                                                                 |
| `MemoryCacheEvaluator`              | `ccflow.evaluators` | Evaluator that caches results in memory.                                                                                       |
| `MultiEvaluator`                    | `ccflow.evaluators` | An evaluator that combines multiple evaluators.                                                                                |
| `GraphEvaluator`                    | `ccflow.evaluators` | Evaluator that evaluates the dependency graph of callable models in topologically sorted order.                                |
| `ChunkedDateRangeEvaluator`         | *Coming Soon!*      |                                                                                                                                |
| `ChunkedDateRangeResultsAggregator` | *Coming Soon!*      |                                                                                                                                |
| `RayChunkedDateRangeEvaluator`      | *Coming Soon!*      |                                                                                                                                |
| `DependencyTrackingEvaluator`       | *Coming Soon!*      |                                                                                                                                |
| `DiskCacheEvaluator`                | *Coming Soon!*      |                                                                                                                                |
| `ParquetCacheEvaluator`             | *Coming Soon!*      |                                                                                                                                |
| `RayCacheEvaluator`                 | *Coming Soon!*      |                                                                                                                                |
| `RayGraphEvaluator`                 | *Coming Soon!*      |                                                                                                                                |
| `RayDelayedDistributedEvaluator`    | *Coming Soon!*      |                                                                                                                                |
| `ParquetCacheEvaluator`             | *Coming Soon!*      |                                                                                                                                |
| `RetryEvaluator`                    | *Coming Soon!*      |                                                                                                                                |

## Results

A Result is an object that holds the results from a callable model. It provides the equivalent of a strongly typed dictionary where the keys and schema are known upfront.

The following table summarizes the "result" models.

| Name                      | Path                     | Description                                                                                                                                                                                                              |
| :------------------------ | :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GenericResult`           | `ccflow.result`          | A generic result (holds anything).                                                                                                                                                                                       |
| `DictResult`              | `ccflow.result`          | A generic dict (key/value) result.                                                                                                                                                                                       |
| `ArrowResult`             | `ccflow.result.pyarrow`  | Holds an arrow table.                                                                                                                                                                                                    |
| `ArrowDateRangeResult`    | `ccflow.result.pyarrow`  | Extension of `ArrowResult` for representing a table over a date range that can be divided by date, such that generation of any sub-range of dates gives the same results as the original table filtered for those dates. |
| `NarwhalsResult`          | `ccflow.result.narwhals` | Holds a narwhals `DataFrame` or `LazyFrame`.                                                                                                                                                                             |
| `NarwhalsDataFrameResult` | `ccflow.result.narwhals` | Holds a narwhals eager `DataFrame`.                                                                                                                                                                                      |
| `NumpyResult`             | `ccflow.result.numpy`    | Holds a numpy array.                                                                                                                                                                                                     |
| `PandasResult`            | `ccflow.result.pandas`   | Holds a pandas dataframe.                                                                                                                                                                                                |
| `XArrayResult`            | `ccflow.result.xarray`   | Holds an xarray.                                                                                                                                                                                                         |
