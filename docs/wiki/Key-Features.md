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
| `ReportingEvaluator`                | `ccflow.evaluators` | Transparent evaluator that reports telemetry (tracing / metrics / alerting) about each evaluation in its scope.                |
| `DryRunEvaluator`                   | `ccflow.evaluators` | Evaluator that walks the dependency graph and reports a plan *without running any model bodies*.                               |
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
    retry_exceptions=[TimeoutError, ConnectionError],
)

with FlowOptionsOverride(options={"evaluator": retry}):
    result = my_model(my_context)
```

> [!NOTE]
>
> `retry_exceptions` and `no_retry_exceptions` accept either a bare exception class (as shown above,
> the natural form in Python) or any importable exception path as a string (e.g.
> `"httpx.ConnectError"`, the form to use in YAML/Hydra config files). The paths are validated
> (imported) eagerly, so a third-party exception requires that package to be installed.

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
    retry_exceptions=[TimeoutError, ConnectionError],
)

result = flaky(my_context)  # same context/result types as fetch_from_api
```

Use `RetryEvaluator` for runtime, cross-cutting retries (including across parallel evaluators), and
`RetryModel` when the retry policy is a declarative, visible part of the model graph.

### Reporting, tracing & dry-run

Reporting is *telemetry about the evaluation itself* — which model ran, on what context, how long it
took, how it relates to other evaluations (parent/child spans), and whether it failed. It is
strictly **transparent**: it never changes the value returned by the wrapped evaluation, so caching
and dependency graphs are unaffected. This is what distinguishes reporting from *publishing*:

| Concern        | Acts on                     | Changes the result? | Example                                |
| :------------- | :-------------------------- | :------------------ | :------------------------------------- |
| **Reporting**  | the *evaluation* (metadata) | No (transparent)    | spans, latency metrics, failure alerts |
| **Publishing** | the *result payload*        | No, but consumes it | write a DataFrame to disk / a database |

Like retries, reporting comes in both the cross-cutting (`ReportingEvaluator`) and structural
(`ReportingModel`) forms, both built on a shared `ReportingPolicy`. Events are delivered to a
pluggable `Reporter` sink. Signal-specific subclasses specialise the policy:

- **Tracing** (`TracingReportingEvaluator` / `TracingReportingModel`) — spans with parent/child
  correlation. `OpenTelemetryTracingReportingEvaluator` (aliased `OpenTelemetryEvaluator`) opens
  real OpenTelemetry spans (install the optional extra: `pip install ccflow[otel]`).
- **Metrics** (`MetricsReportingEvaluator` / `MetricsReportingModel`) — success/error counters and a
  latency histogram. `OpenTelemetryMetricsReportingEvaluator` pushes to OpenTelemetry instruments.
- **Alerting** (`AlertsReportingEvaluator` / `AlertsReportingModel`) — prioritised alerts with
  `P1`–`P5` tags (`AlertPriority`), emitted on failure (and optionally on recovery).

```python
from ccflow import FlowOptionsOverride
from ccflow.evaluators import TracingReportingEvaluator
from ccflow.utils.reporting import InMemoryReporter, ReportPhase

reporter = InMemoryReporter()
tracing = TracingReportingEvaluator(reporter=reporter)

with FlowOptionsOverride(options={"evaluator": tracing}):
    result = my_model(my_context)          # result is unchanged

# reporter.events is a list of structured ReportEvent objects forming a span tree
phases = [e.phase for e in reporter.events]   # e.g. [START, SUCCESS, END, ...]
```

Available reporters (`ccflow.utils.reporting`): `NoOpReporter`, `InMemoryReporter` (testing /
introspection), `LoggingReporter` (writes events to a logger), `CompositeReporter` (fan-out), and
`UIReporter` (a thread-safe, bounded buffer drained by an observability UI). `ReportingStateStore`
folds a stream of events into per-node state (keyed by `span_id`) so a UI can reconstruct the live
span tree, and `run_scope(...)` tags all events emitted within a run with a shared `run_id`.

> [!NOTE]
>
> Current evaluators and models emit **node** lifecycle events (`START`, `SUCCESS`, `ERROR`, `RETRY`,
> `GIVE_UP`, `END`, plus `QUEUED` / `SKIPPED` from `DryRunEvaluator`). The run/graph envelope phases
> `RUN_STARTED`, `RUN_FINISHED` and `GRAPH_DISCOVERED` are **reserved** for a future explicit
> observer layer — `ReportingStateStore` already folds them, but no evaluator emits them today, so
> `run_scope(...)` gives you a shared `run_id` on node events rather than a complete run envelope.

Each `ReportEvent` carries a structured `extra` dict whose keys depend on the signal/source:

| Source                          | `extra` keys          | Meaning                                            |
| :------------------------------ | :-------------------- | :------------------------------------------------- |
| `DryRunEvaluator`               | `node_key`, `dry_run` | Logical node identity (equals `cache_key()`); flag |
| `MetricsReportingEvaluator`     | `metric`, `value`     | Metric name (counter/histogram) and its value      |
| Retry lifecycle (`RetryPolicy`) | `delay`, `reason`     | Backoff before next attempt; give-up reason        |

> [!NOTE]
>
> `LoggingEvaluator` — the default evaluator — is itself a reporting evaluator built on a
> `LoggingPolicy`, so it participates in the same span tree. Configure a `reporter` on it to receive
> structured events in addition to its log lines. A structural `LoggingModel` is also available.

Placeholders for additional backends (`Datadog*`, `Opsgenie*`, `JSMAlerts*`, `NewRelic*`) are defined
so they can be referenced in config and tracked, but raise `NotImplementedError` until implemented.

#### Previewing a run with `DryRunEvaluator`

`DryRunEvaluator` *plans* an evaluation: it walks the dependency graph (which evaluates the cheap
`__deps__` declarations but never a model body), emits `QUEUED` then `SKIPPED` events for every node
with a parent/child span tree mirroring the graph, and returns a synthetic result without running
anything. This is useful for previewing what a run would do and for driving a UI.

```python
from ccflow.evaluators import DryRunEvaluator
from ccflow.utils.reporting import InMemoryReporter

reporter = InMemoryReporter()
with FlowOptionsOverride(options={"evaluator": DryRunEvaluator(reporter=reporter)}):
    pipeline(my_context)                   # no model bodies run

planned = [(e.model_name, e.phase) for e in reporter.events]
```

> [!WARNING]
>
> With the default `synthetic_result=True`, the value returned by a dry run is built with
> `model_construct`: required fields may be **unset** and validators do **not** run. Treat it purely
> as a planning placeholder for previews/UIs — do **not** feed it into downstream computation. Because
> the return value differs from a real run, `DryRunEvaluator` is **not** transparent in this mode, so
> it participates in cache keys and will not contaminate a cached real result.

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
