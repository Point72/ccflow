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

You can execute a generated model in two equivalent ways:

- call it directly with a context object: `model(ctx)`
- use `.flow.compute(...)` to supply runtime inputs as keyword arguments

`.flow.compute(...)` is mainly an explicit, ergonomic way to mark the deferred
execution point.

**Basic Example:**

```python
from datetime import date
from ccflow import Flow, GenericResult, DateRangeContext

@Flow.model
def load_data(context: DateRangeContext, source: str) -> GenericResult[dict]:
    # Your data loading logic here
    return GenericResult(value=query_db(source, context.start_date, context.end_date))

# Create model instance
loader = load_data(source="my_database")

# Execute with context
ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = loader(ctx)
```

**Default `@Flow.model` Style:**

Use this when you want the simplest API and do not need to declare a formal
context shape up front.

```python
from ccflow import Flow

@Flow.model
def add(x: int, y: int) -> int:
    return x + y

model = add(x=10)

# `x` is bound when the model is created.
# `y` is supplied later at execution time.
assert model.flow.compute(y=5) == 15

# `.flow.with_inputs(...)` rewrites runtime inputs for this call path.
doubled_y = model.flow.with_inputs(y=lambda ctx: ctx.y * 2)
assert doubled_y.flow.compute(y=5) == 20
```

In this mode:

- bound parameters are model configuration
- unbound parameters are runtime inputs for that model instance

**Composing Dependencies with Normal Parameters:**

Any non-context parameter can be bound either to a literal value or to another
`CallableModel`. If you pass an upstream model, `@Flow.model` evaluates it with
the current context and passes the resolved value into your function.

```python
from datetime import date, timedelta
from ccflow import DateRangeContext, Flow

@Flow.model(context_args=["start_date", "end_date"])
def load_revenue(start_date: date, end_date: date, region: str) -> float:
    days = (end_date - start_date).days + 1
    return 1000.0 + days * 10.0

@Flow.model(context_args=["start_date", "end_date"])
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
direct = growth(ctx).value

# Equivalent explicit deferred entry point
computed = growth.flow.compute(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)

assert direct == computed
```

This pattern is the main story for transformed dependencies. `@Flow.model`
still produces an ordinary `CallableModel`; `.flow.compute(...)` is just a
clearer way to say "supply the runtime inputs here."

**Why `context_args` Exists:**

Without `context_args`, runtime inputs are inferred from whichever parameters
are still unbound on a particular model instance. That is flexible and
ergonomic.

Use `context_args` when some parameters are semantically "the execution
context" and you want that split to stay stable and explicit:

- the runtime context should be stable across instances
- the split between config and runtime inputs matters semantically
- the model is naturally "run over a context" such as date windows,
  partitions, or scenarios
- you want the generated model to match a built-in context type like
  `DateRangeContext` when possible

**Deferred Execution Helpers:**

```python
from ccflow import Flow

@Flow.model
def add(x: int, y: int) -> int:
    return x + y

model = add(x=10)
assert model.flow.compute(y=5) == 15

shifted = model.flow.with_inputs(y=lambda ctx: ctx.y * 2)
assert shifted.flow.compute(y=5) == 20
```

If you already have a real context object, you can call the model directly
instead:

```python
from ccflow import FlowContext

ctx = FlowContext(y=5)
assert model(ctx).value == 15
assert shifted(ctx).value == 20
```

**Hydra/YAML Configuration:**

`Flow.model` decorated functions work seamlessly with Hydra configuration and the `ModelRegistry`:

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

When loaded via `ModelRegistry.load_config()`, references by name ensure the same object instance is shared across all consumers.

**Auto-Unpacked Context with `context_args`:**

Instead of taking an explicit `context` parameter, you can use `context_args` to automatically unpack context fields as function parameters. This is useful when you want cleaner function signatures:

```python
from datetime import date
from ccflow import Flow, GenericResult, DateRangeContext

# Instead of: def load_data(context: DateRangeContext, source: str)
# Use context_args to unpack the context fields directly:
@Flow.model(context_args=["start_date", "end_date"])
def load_data(start_date: date, end_date: date, source: str) -> GenericResult[str]:
    return GenericResult(value=f"{source}:{start_date} to {end_date}")

# The decorator matches common built-in context types when possible
loader = load_data(source="my_database")
assert loader.context_type == DateRangeContext

# Execute with context as usual
ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
result = loader(ctx)  # "my_database:2024-01-01 to 2024-01-31"
```

The `context_args` parameter specifies which function parameters should be extracted from the context. Those fields are validated through a runtime schema built from the parameter annotations. For well-known shapes such as `start_date` / `end_date`, the generated model uses a concrete built-in context type like `DateRangeContext`; otherwise it uses `FlowContext`, a universal frozen carrier for the validated fields.

If a `@Flow.model` function returns a plain value instead of a `ResultBase`
subclass, the generated model automatically wraps that value in `GenericResult`
at runtime so it still behaves like a normal `CallableModel`.

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
