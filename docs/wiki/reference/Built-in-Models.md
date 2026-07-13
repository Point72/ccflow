# Built-in Models

The models, publishers, and evaluators that ship with `ccflow`. Use these off the shelf or as references for your own. For usage, see the [How-to Guides](How-to-Guides); for the base classes, see [Core Types](Core-Types).

> [!NOTE]
> Some entries are marked *Coming Soon!* — they are part of the design and in the process of being open-sourced.

## Models

Beyond the `CallableModel` and `BaseModel` subclasses you write yourself, `ccflow` provides composition models used throughout the framework, including:

- `PublisherModel` — runs a `CallableModel` and hands its result to a publisher in one step.
- `RetryModel` — wraps a `CallableModel` with a retry policy as a first-class graph node (see [Retry on Failure](Retry-on-Failure)).
- Narwhals-based models for cross-dataframe-library operations.

Additional data-reading models are being open-sourced over time.

## Publishers

Publishers (`ccflow.publishers`) are models that write or send data. A common interface lets one be substituted for another purely through configuration. See [Bind Logic to Configs](Bind-Logic-to-Configs#write-a-custom-publisher) to write your own.

| Name                         | Path                | Description                                                                                |
| :--------------------------- | :------------------ | :----------------------------------------------------------------------------------------- |
| `DictTemplateFilePublisher`  | `ccflow.publishers` | Publish to a file after populating a Jinja template.                                       |
| `GenericFilePublisher`       | `ccflow.publishers` | Publish using a generic "dump" callable. Uses `smart_open`, so local and cloud paths work. |
| `JSONPublisher`              | `ccflow.publishers` | Publish to a file in JSON format.                                                          |
| `PandasFilePublisher`        | `ccflow.publishers` | Publish a pandas DataFrame using an appropriate `pd.DataFrame` method.                     |
| `NarwhalsFilePublisher`      | `ccflow.publishers` | Publish a Narwhals DataFrame using an appropriate `nw.DataFrame` method.                   |
| `PicklePublisher`            | `ccflow.publishers` | Publish data to a pickle file.                                                             |
| `PydanticJSONPublisher`      | `ccflow.publishers` | Publish a pydantic model to a JSON file.                                                   |
| `YAMLPublisher`              | `ccflow.publishers` | Publish to a file in YAML format.                                                          |
| `CompositePublisher`         | `ccflow.publishers` | Decompose a model or dict into pieces and publish each separately.                         |
| `PrintPublisher`             | `ccflow.publishers` | Print data using standard `print`.                                                         |
| `LogPublisher`               | `ccflow.publishers` | Print data using standard logging.                                                         |
| `PrintJSONPublisher`         | `ccflow.publishers` | Print data in JSON format.                                                                 |
| `PrintYAMLPublisher`         | `ccflow.publishers` | Print data in YAML format.                                                                 |
| `PrintPydanticJSONPublisher` | `ccflow.publishers` | Print a pydantic model as JSON.                                                            |
| `ArrowDatasetPublisher`      | *Coming Soon!*      |                                                                                            |
| `PandasParquetPublisher`     | *Coming Soon!*      |                                                                                            |
| `PandasDeltaPublisher`       | *Coming Soon!*      |                                                                                            |
| `EmailPublisher`             | *Coming Soon!*      |                                                                                            |
| `MatplotlibFilePublisher`    | *Coming Soon!*      |                                                                                            |
| `PlotlyFilePublisher`        | *Coming Soon!*      |                                                                                            |
| `XArrayPublisher`            | *Coming Soon!*      |                                                                                            |
| `MLFlowArtifactPublisher`    | *Coming Soon!*      |                                                                                            |
| `MLFlowPublisher`            | *Coming Soon!*      |                                                                                            |

## Evaluators

Evaluators control *how* a `CallableModel` runs. Set one through `FlowOptions` (see [Core Types](Core-Types#flow-options)). Usage is in [Cache Results](Cache-Results) and [Retry on Failure](Retry-on-Failure).

| Name                                | Path                | Description                                                        |
| :---------------------------------- | :------------------ | :----------------------------------------------------------------- |
| `LazyEvaluator`                     | `ccflow.evaluators` | Runs the callable only once an attribute of the result is queried. |
| `LoggingEvaluator`                  | `ccflow.evaluators` | Logs information about evaluating the callable (the default).      |
| `MemoryCacheEvaluator`              | `ccflow.evaluators` | Caches results in memory.                                          |
| `MultiEvaluator`                    | `ccflow.evaluators` | Combines multiple evaluators.                                      |
| `GraphEvaluator`                    | `ccflow.evaluators` | Evaluates the dependency graph in topologically sorted order.      |
| `RetryEvaluator`                    | `ccflow.evaluators` | Retries evaluation on failure with exponential backoff and jitter. |
| `ChunkedDateRangeEvaluator`         | *Coming Soon!*      |                                                                    |
| `ChunkedDateRangeResultsAggregator` | *Coming Soon!*      |                                                                    |
| `DependencyTrackingEvaluator`       | *Coming Soon!*      |                                                                    |
| `DiskCacheEvaluator`                | *Coming Soon!*      |                                                                    |
| `ParquetCacheEvaluator`             | *Coming Soon!*      |                                                                    |
| `RayChunkedDateRangeEvaluator`      | *Coming Soon!*      |                                                                    |
| `RayCacheEvaluator`                 | *Coming Soon!*      |                                                                    |
| `RayGraphEvaluator`                 | *Coming Soon!*      |                                                                    |
| `RayDelayedDistributedEvaluator`    | *Coming Soon!*      |                                                                    |

### How cache keys are built

`MemoryCacheEvaluator` uses `ccflow.evaluators.cache_key()`, which delegates to `ccflow.utils.compute_cache_token()`. A key combines a **data token** (for the serialized model/context payload) and a **behavior token** (for each callable/evaluator class that can affect the result).

- For a direct `CallableModel` call, the key depends on `model.model_dump(mode="python")` and `compute_behavior_token(type(model))`.
- For a `ModelEvaluationContext`, the structural key depends on the context payload plus the function name (`__call__` vs `__deps__`), the behavior token of the model class, and the data/behavior tokens of any **non-transparent** evaluators in the chain.

Transparent evaluators (those whose `is_transparent` returns `True`) are skipped, so wrapping a model in logging or timing does not change its cache identity.

`compute_behavior_token()` hashes the class's Python-defined methods and also consults `__ccflow_tokenizer_deps__` for behavior defined outside the class body:

```python
class MyModel(CallableModel):
    __ccflow_tokenizer_deps__ = [helper, SharedLogic]
```

Entries may be functions or classes; class dependencies are tokenized recursively. Recursive class dependency graphs raise a `TypeError`.

### `RetryEvaluator` parameters

`RetryEvaluator` (and `RetryModel`, which share a `RetryPolicy`) support:

| Parameter             | Meaning                                                                      |
| :-------------------- | :--------------------------------------------------------------------------- |
| `max_attempts`        | Maximum number of attempts.                                                  |
| `wait_initial`        | Wait before the first retry.                                                 |
| `wait_multiplier`     | Backoff multiplier between retries.                                          |
| `wait_max`            | Cap on any single wait.                                                      |
| `wait_jitter`         | Maximum random jitter added to a wait.                                       |
| `max_delay`           | Cap on *cumulative* waiting; no retry once the next backoff would exceed it. |
| `retry_exceptions`    | Exceptions (classes or importable path strings) that trigger a retry.        |
| `no_retry_exceptions` | Exceptions that must not be retried.                                         |

See [Retry on Failure](Retry-on-Failure) for usage.

## See also

- [Core Types](Core-Types) — the base classes these extend.
- [Flow Model](Flow-Model) — the `@Flow.model` API.
