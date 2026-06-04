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
| `RetryEvaluator`                    | `ccflow.evaluators` | Evaluator that retries the evaluation of a callable model on failure, with exponential backoff and jitter.                     |
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

### Retrying on failure

The `RetryEvaluator` retries a `CallableModel` when evaluation raises an exception. It supports
exponential backoff (`wait_initial`, `wait_multiplier`, `wait_max`), random `wait_jitter` to avoid
thundering-herd retries, and stop conditions via `max_attempts` and `max_delay`. `max_delay` caps
the *cumulative* time spent sleeping between retries (not the total wall-clock runtime of the
evaluation); once the next backoff would push the accumulated wait over the budget, no further
retries are attempted. Use `retry_exceptions` / `no_retry_exceptions` to control which exceptions
are retried.

```python
from ccflow import FlowOptionsOverride
from ccflow.evaluators import RetryEvaluator

retry = RetryEvaluator(
    max_attempts=5,
    wait_initial=0.5,       # first retry waits ~0.5s
    wait_multiplier=2.0,    # then 1s, 2s, 4s, ...
    wait_max=30.0,          # cap any single wait at 30s
    wait_jitter=0.25,       # add up to 0.25s of random jitter
    max_delay=120.0,        # stop retrying once cumulative waiting would exceed 2 minutes
    retry_exceptions=["builtins.TimeoutError", "builtins.ConnectionError"],
)

with FlowOptionsOverride(options={"evaluator": retry}):
    result = my_model(my_context)
```

> [!NOTE]
>
> `retry_exceptions` and `no_retry_exceptions` accept any importable exception path. The paths are
> validated (imported) eagerly, so a third-party exception such as `"httpx.ConnectError"` requires
> that package to be installed.

The evaluator is *transparent* (a successful result is identical to evaluating the model directly,
so caching and dependency graphs are unaffected) and holds no mutable per-call state, so a single
instance can be shared safely across threads and combined with parallel evaluators (e.g. a Ray or
Celery evaluator). Place it *inside* a parallel evaluator to retry each task independently, or
*outside* to retry the whole parallel dispatch as a unit.

#### Choosing which models are retried

An evaluator wraps *every* model evaluated in its scope (including dependencies). There are a few
ways to control which tasks are retried and which are not, from coarse to fine:

- **Scope the override to model types or instances.** `FlowOptionsOverride` can target specific
  models, so only those get the retry evaluator while everything else uses the default:

  ```python
  with FlowOptionsOverride(options={"evaluator": retry}, model_types=(FetchFromApi,)):
      result = pipeline(context)          # only FetchFromApi instances are retried
  ```

  Use `models=(instance,)` to target a single instance instead of a type.

- **Override per call.** Pass `_options` for a one-off retry without any surrounding context:

  ```python
  result = my_model(context, _options={"evaluator": retry})
  ```

- **Pin it on the model.** Set `model.meta.options` so a model always carries its own retry policy.

- **Select inside the evaluator.** When retry is part of a single global evaluator chain (e.g.
  combined with logging/caching), use `include_model_types` / `exclude_model_types` to decide which
  models the *same* evaluator retries. Non-selected models are evaluated once and passed straight
  through (`exclude_model_types` wins over `include_model_types`):

  ```python
  retry = RetryEvaluator(
      max_attempts=5,
      include_model_types=["mypkg.FetchFromApi", "mypkg.CallService"],  # only these are retried
      exclude_model_types=["mypkg.PureTransform"],                      # never retried
  )
  ```

#### `RetryModel`: retrying a single model declaratively

`RetryEvaluator` is the cross-cutting ("how to run") way to add retries: it is configured at
runtime and wraps whatever models fall in its scope. When you instead want a retry policy attached
to one specific model *as part of the graph itself*, use `RetryModel`. It wraps another
`CallableModel` and becomes a first-class node, so the policy can be declared statically in
config/registries and shows up explicitly in serialization and the dependency graph. It shares all
its configuration and retry mechanics with `RetryEvaluator` (both build on the same `RetryPolicy`),
and preserves the wrapped model's `context_type` / `result_type`.

```python
from ccflow.models import RetryModel

flaky = RetryModel(
    model=fetch_from_api,   # any CallableModel
    max_attempts=5,
    wait_initial=0.5,
    retry_exceptions=["builtins.TimeoutError", "builtins.ConnectionError"],
)

result = flaky(my_context)  # same context/result types as fetch_from_api
```

Use `RetryEvaluator` for runtime, cross-cutting retries (including across parallel evaluators), and
`RetryModel` when the retry policy is a declarative, visible part of the model graph.

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
