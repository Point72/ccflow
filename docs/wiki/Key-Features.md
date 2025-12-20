- [Base Model](#base-model)
- [Callable Model](#callable-model)
  - [Dynamic Contexts](#dynamic-contexts)
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

### Dynamic Contexts

When building `CallableModel`s, you typically need to define a separate context class for each model:

```python
from ccflow import CallableModel, ContextBase, Flow, GenericResult

# Traditional approach: define a separate context class
class MyContext(ContextBase):
    a: int
    b: str = "default"

class MyModel(CallableModel):
    @Flow.call
    def __call__(self, context: MyContext) -> GenericResult:
        return GenericResult(value={"a": context.a, "b": context.b})

model = MyModel()
model(a=42, b="test")  # Works, but required defining MyContext separately
```

For simple use cases, this boilerplate can be tedious. The `Flow.dynamic_call` decorator provides a more ergonomic alternative by automatically generating the context class from the function signature:

```python
from ccflow import CallableModel, Flow, GenericResult

# Dynamic context approach: context is inferred from the function signature
class MyModel(CallableModel):
    @Flow.dynamic_call
    def __call__(self, *, a: int, b: str = "default") -> GenericResult:
        return GenericResult(value={"a": a, "b": b})

model = MyModel()
model(a=42, b="test")  # Works! No separate context class needed
model(a=100)           # Default value for 'b' is used
```

**When to use `Flow.dynamic_call`:**

- **Rapid prototyping**: When you want to quickly iterate without defining context classes upfront
- **Simple models**: When your context has only a few fields and doesn't need to be reused
- **Internal methods**: For additional methods on a `CallableModel` that need their own context type

**When to use the traditional approach:**

- **Shared contexts**: When multiple models should accept the same context type
- **Complex validation**: When you need custom validators or computed fields on the context
- **Documentation**: When you want the context class to be explicitly documented and discoverable

**Advanced usage:**

**Inheriting from a parent context:**

You can inherit from an existing context class using the `parent` parameter. This is useful when you want to:

- Share common fields (like `user_id`, `timestamp`) across multiple models
- Add validation constraints from a base context class
- Ensure certain fields are always present for auditing or logging

```python
class BaseContext(ContextBase):
    user_id: str
    timestamp: int = 0

class MyModel(CallableModel):
    @Flow.dynamic_call(parent=BaseContext)
    def __call__(self, *, query: str) -> GenericResult:
        # The dynamic context has 'user_id' and 'timestamp' from BaseContext,
        # plus 'query' from the function signature.
        # Only 'query' is passed to this function; parent fields are for validation/storage.
        return GenericResult(value=query)

model = MyModel()
# Must provide 'user_id' (required from parent) and 'query' (required from function)
result = model(user_id="user123", query="SELECT *")
# Can also provide 'timestamp' (has default in parent)
result = model(user_id="user123", timestamp=1000, query="SELECT *")
```

> **Note:** Function parameters cannot have the same name as fields in the parent context. This prevents accidental shadowing and ensures clear ownership of each field.

**Multiple methods with different contexts:**

```python
class MultiMethodModel(CallableModel):
    @Flow.dynamic_call
    def __call__(self, *, x: int) -> GenericResult:
        return GenericResult(value=x)

    @Flow.dynamic_call
    def process(self, *, data: list, threshold: float = 0.5) -> GenericResult:
        return GenericResult(value=[d for d in data if d > threshold])
```

**Using evaluators with dynamic contexts:**

Most evaluators work seamlessly with `Flow.dynamic_call`:

```python
from ccflow import FlowOptionsOverride, FlowOptions
from ccflow.evaluators import LoggingEvaluator, LazyEvaluator

class MyModel(CallableModel):
    @Flow.dynamic_call
    def __call__(self, *, value: int) -> GenericResult:
        return GenericResult(value=value * 2)

model = MyModel()

# LoggingEvaluator - works perfectly
with FlowOptionsOverride(options=FlowOptions(evaluator=LoggingEvaluator())):
    result = model(value=42)

# LazyEvaluator - works perfectly
with FlowOptionsOverride(options=FlowOptions(evaluator=LazyEvaluator())):
    result = model(value=42)
```

Need caching as well? Dynamic contexts now serialize through an internal fallback so in-memory caching works too:

```python
from ccflow import FlowOptions, FlowOptionsOverride
from ccflow.evaluators import GraphEvaluator, MemoryCacheEvaluator, MultiEvaluator

cacheable = MultiEvaluator(evaluators=[GraphEvaluator(), MemoryCacheEvaluator()])

with FlowOptionsOverride(options=FlowOptions(evaluator=cacheable, cacheable=True)):
    first = model(value=42)   # Executes model
    second = model(value=42)  # Served from cache, no re-execution
```

> Dynamic contexts are registered under `ccflow._dynamic_contexts`, so caching/graph evaluators can import them just like regular context classes. Custom evaluators that rely on different serialization schemes should perform a similar registration.

**Standalone decorator:**

The `dynamic_context` decorator can also be used standalone if you need lower-level control:

```python
from ccflow import dynamic_context, Flow

@dynamic_context
def my_method(self, *, a: int, b: str) -> GenericResult:
    return GenericResult(value=a)

# my_method.__dynamic_context__ contains the generated context class
# my_method.__result_type__ contains the return type annotation
```

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
