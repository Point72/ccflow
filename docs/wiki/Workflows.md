- [Workflows with `ccflow`](#workflows-with-ccflow)
  - [Result Type](#result-type)
    - [GenericResult](#genericresult)
    - [Common Result Types](#common-result-types)
    - [Custom Results](#custom-results)
  - [Context](#context)
    - [NullContext](#nullcontext)
    - [GenericContext](#genericcontext)
    - [Common Context Types](#common-context-types)
    - [Custom Contexts](#custom-contexts)
    - [Properties](#properties)
  - [Callable Models](#callable-models)
    - [Type Checking](#type-checking)
    - [Flow Decorator](#flow-decorator)
    - [Metadata](#metadata)
  - [Evaluators](#evaluators)
    - [In-memory Caching](#in-memory-caching)
    - [Graph Evaluation](#graph-evaluation)
    - [User Defined Evaluators](#user-defined-evaluators)

# Workflows with `ccflow`

As described in the [Workflow Design Goals](Design-Goals#workflow-design-goals), `ccflow` wishes to make it easy to define (via configuration) the tasks/steps that make up a workflow. In the [Using Configuration](#Using-Configuration) section and in some of the examples above, we described how to bind functionality to the configuration by adding `__call__` methods to the models.

In `ccflow` we formalize this pattern, by introducing the following

- a "Result" type to hold the results of workflow steps
- a "Context" type to parameterize workflows, allowing them to be templatized and dynamically configured at runtime
- a "Flow" decorator to allow the framework to inject additional functionality (type checking, logging, caching, etc.) behind the scenes

## Result Type

In order to generically operate on the data that communicated between steps of the workflow, `ccflow` defines a base class for the results that a step returns.
These are also child classes of `BaseModel` so one gets all the type validation and serialization features of pydantic models for free. Insisting on a base class for all result types also allows
the framework to perform additional magic (such as delayed evaluation).

Some result types are provided with `ccflow`, but it is straightforward to define your own.

### `GenericResult`

The `GenericResult` can hold anything in the value field. It is useful if the result type is not known upfront, not important, or for ad-hoc experimentation:

```python
from ccflow import GenericResult
result = GenericResult(value="Anything goes here")
print(result)
#> GenericResult(value='Anything goes here')
```

However, it could also hold a dictionary:

```python
result = GenericResult(value={"x": "foo", "y": 5.0})
print(result)
#> GenericResult(value={'x': 'foo', 'y': 5.0})
```

The `GenericResult` uses python Generics to provide some amount of type checking if desired on the single `value` field. For example, if you want to ensure that the value is always a string, you can specify this when creating the instance:

```python
result = GenericResult[str](value="Any string")
print(result)
#> GenericResult(value='Any string')
```

Now, if you try to pass a dictionary, it will raise an exception

```python
try:
    result = GenericResult[str](value={"x": "foo", "y": 5.0})
except ValueError as e:
    print(e)
#> 1 validation error for GenericResult[str]
#> value
#>   Input should be a valid string [type=string_type, input_value={'x': 'foo', 'y': 5.0}, input_type=dict]
#>     For further information visit https://errors.pydantic.dev/2.11/v/string_type
```

We also leverage pydantic to reduce the amount of boilerplate needed to instantiate the `GenericResult`. It will automatically validate arbitrary values as needed:

```python
print(GenericResult.model_validate("Any string"))
#> GenericResult(value='Any string')
```

### Common Result Types

`ccflow` provides some other result types to make it easy to work with common data structures.
These include `PandasResult`, `NDArrayResult`, `ArrowResult`, `XArrayResult` and `NarwhalsFrameResult` (for generic compatibility across dataframe libraries using the excellent [`narwhals`](https://narwhals-dev.github.io/narwhals/) library).
Each of these types is designed to hold a specific type of data structure (which will be validated via pydantic), and they all inherit from `ResultBase` to ensure compatibility with the workflow system.

### Custom Results

If you have any information about the return types, it makes the code more readable and more robust to define a proper schema by defining your own subclass of `ResultBase`:

```python
class MyResult(ResultBase):
    x: str
    y: float

result = MyResult(x="foo", y=5.0)
print(result)
#> MyResult(x='foo', y=5.0)
```

## Context

The context is a collection of parameters that templetize a workflow, given all the other configuration parameters. `ccflow` provides a base class for all contexts, `ContextBase`.
Since we also wish to integrate with the configuration framework described above, `ContextBase` is a child class of `BaseModel` so that it is configurable and serializable, and furthermore it is
a child class of `ResultBase` so that contexts can be used as the return value from a step in a workflow. This provides a convenient way to introduce dynamism in the workflows.

Contexts are helpful when a typical usage pattern might be to run many similar workflows where only one or two configuration parameters are different between them (i.e. date, region, product/user id, etc.).
Thus, the writer of a complex workflow with many configuration option can choose which parameters to expose to users of the workflow via the context, while keeping the rest of the parameters hidden in the configuration.
This allows for a more streamlined interface for the more common use cases, while all configuration options can still be modified via the `ccflow` configuration framework.

### `NullContext`

Some workflows may not require a context at all, in which case they can use the `NullContext`, which takes no parameters; these workflows are completely defined by their configuration.

```python
from ccflow import NullContext
print(NullContext())
#> NullContext()
```

Custom pydantic validation can create such a context directly from `None` or from an empty container:

```python
from ccflow import NullContext
print(NullContext.model_validate(None))
#> NullContext()
print(NullContext.model_validate([]))
#> NullContext()
```

### `GenericContext`

Similar to the `GenericResult` described above, the `GenericContext` is a flexible context type that can hold any type of value. This is useful when the context is not known upfront or when you want to pass arbitrary data to a workflow step.

```python
from ccflow import GenericContext
result = GenericContext[str](value="Any string")
print(result)
#> GenericContext(value='Any string')
```

Now, if you try to pass a dictionary, it will raise an exception

```python
try:
    result = GenericContext[str](value={"x": "foo", "y": 5.0})
except ValueError as e:
    print(e)
#> 1 validation error for GenericContext[str]
#> value
#>   Input should be a valid string [type=string_type, input_value={'x': 'foo', 'y': 5.0}, input_type=dict]
#>     For further information visit https://errors.pydantic.dev/2.11/v/string_type
```

Once again, we pydantic validation to cut down on boilerplate while also providing type safety:

```python
print(GenericContext[str].model_validate(100))
#> GenericContext[str](value='100')
```

### Common Context Types

`ccflow` provides a number of other commonly used contexts for standardization. Below are some examples

```python
from datetime import date
from cubist_core.flow import (
    DateContext,
    DateRangeContext,
    UniverseContext,
    UniverseDateContext,
    VersionedDateContext,
)
print(DateContext(date=date.today()))
print(DateRangeContext(start_date=date(2022, 1, 1), end_date=date(2023, 2, 2)))
print(VersionedDateContext(date=date.today(), entry_time_cutoff=datetime.utcnow()))
print(UniverseContext(universe="US"))
print(UniverseDateContext(universe="US", date=date.today()))
```

Since date-based contexts are particularly popular, `ccflow` also provides additional pydantic validation to make it as easy as possible to construct them from basic types like strings and tuples. This comes in handy when running workflows from the command line.

```python
from ccflow import DateContext, DateRangeContext
print(DateContext.model_validate("2025-01-01"))
#> DateContext(date=datetime.date(2025, 1, 1))
print(DateContext.model_validate("0d"))
#> DateContext(date=datetime.date(2025, 4, 4))
print(DateRangeContext.model_validate(("-7d", "0d")))
#> DateRangeContext(start_date=datetime.date(2025, 3, 28), end_date=datetime.date(2025, 4, 4))
```

### Custom Contexts

The author of a workflow will frequently want to define a context which is specific to that workflow, or to a collection of related workflows. It is very simple to define your own custom context:

```python
from ccflow import ContextBase
from datetime import datetime

class LocationTimestampContext(ContextBase):
    latitude: float
    longitude: float
    timestamp: datetime

print(LocationTimestampContext(latitude=40.7128, longitude=-74.0060, timestamp=datetime.now()))
#> LocationTimestampContext(latitude=40.7128, longitude=-74.006, timestamp=datetime.datetime(2025, 4, 5, 12, 47, 43, 303252))
```

### Properties

Contexts by default are declared as `frozen` in Pydantic; they are immutable, and ideally they should be hashable as well. This is so that more advanced tools (i.e. such as caching) can use contexts as dictionary keys.

## Callable Models

With the context and result types defined, we are now ready to define a workflow step/task.
Following the design objectives, and the exposition of how best to use configuration objects, we define a `CallableModel` abstract base class, which is essentially a configurable object where one must implement the `__call__` method as a function of the context. It has a few other features which we will explore below.
Note that since every CallableModel is a BaseModel, this class is configurable using all the configuration logic described in the Configuration Tutorial. In particular any workflow step can be named and added to the registry for later reference.

First, we illustrate a trivial example of defining a `CallableModel` that implements the "(FizzBuzz)[https://en.wikipedia.org/wiki/Fizz_buzz]" programming problem. operating on a `GenericContext` and returning a `GenericResult`

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext

class FizzBuzzModel(CallableModel):
    fizz: str = "Fizz"
    buzz: str = "Buzz"

    @Flow.call
    def __call__(self, context: GenericContext[int]) -> GenericResult[list[int|str]]:
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

One of the first things that jumps out about the implementation above, is the use of the decorator `@Flow.call` (which is required by `CallableModel`).
We will discuss this more later, but the idea is to hide any advanced functionality of the framework regarding evaluation of the steps behind this decorator.
This is what lets us control type validation, logging, caching and other advanced evaluation methods.
The decorator itself is also a configurable object, but by default, it only provides type checking on the input and output types.

### Type Checking

In the above example, we could simply pass the value `15` to the `__call__` method as the `context` argument: the decorator invokes pydantic type checking which validates the input and converts it to the appropriate type (`GenericContext[int]` in this case).
Similarly, we are able to return the result list directly, and the decorator will validate this as well, ensuring that the output conforms to the expected type (`GenericResult[list[int|str]]`).
Thus, the `@Flow.call` decorator combines with the pydantic type checking to reduce boilerplate for end users while preserving runtime type safety.

To further illustrate this point, if we try to pass an invalid input to the model, it will result in an error:

```python
model = FizzBuzzModel()
try:    
    model("not an integer")
except ValueError as e:
    print(e)
#> 1 validation error for GenericContext[int]
#> value
#>   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='not an integer', input_type=str]
#>     For further information visit https://errors.pydantic.dev/2.11/v/int_parsing
```

In some cases, leveraging type hints for type checking may be limiting (especially if some underlying logic is not entirely type safe), and so we provide other ways to specify the context and result types.
In particular, there is a `context_type` and `result_type` property on every `CallableModel` that will return the types used for validation. By default, it will infer these types from the type signature of the `__call__` method, but they can also be overridden.
For example:

```python
model = FizzBuzzModel()
print(model.context_type)
print(model.result_type)
#> <class 'ccflow.context.GenericContext[int]'>
#> <class 'ccflow.result.generic.GenericResult[list[Union[int, str]]]'>
```

Suppose that either the context type or the result type depends on the configuration of the object. To make this type-safe from a static type checking point of view, one option would be to use Generics, but this option is overly complicated for many users (and still doesn't allow one to do everything due to python's lack of "higher kinded types").
A simpler method is to re-implement `context_type` and/or `result_type`.

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext
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
        # do something with the context
        return context.value
```

We can now configure this model to take an integer and return a string. Because of the runtime type checking, this conversion will be done automatically for us:

```python
model = DynamicTypedModel(input_type=int, output_type=str)
print(model(5))
#> GenericResult[str](value='5')
```

If no type information is provided on the model at all, an error will be thrown:

```python
from ccflow import CallableModel, Flow

class NoTypeModel(CallableModel):
    @Flow.call
    def __call__(self, context):
        return {"foo": 5}

model = NoTypeModel()
try:
    model()
except TypeError as e:
    print(e)
#> Must either define a type annotation for context on __call__ or implement 'context_type'
```

### Flow Decorator

Behind the `@Flow.call` decorator is where all the magic lives, including advanced features which are not yet well tested or have not yet been implemented. However, we walk through some of the more basic functionality to give an idea of how it can be used.

The behavior of the `@Flow.call` decorator can be controlled in two ways:

- by using the FlowOptionsOverride context (which allows you to scope changes to specific models, specific model types or all models)
- by passing arguments to it when defining the CallableModel to customize model-specific behavior

An example of the former is to change the log level for all model evaluations:

```python
import logging

from cubist_core.flow import FlowOptionsOverride

logger = logging.getLogger()
logger.setLevel(logging.WARN)

model = FizzBuzzModel()
with FlowOptionsOverride(options={"log_level": logging.WARN}):
    print(model(15))

#[FizzBuzzModel]: Start evaluation of __call__ on GenericContext[int](value=15).
#[FizzBuzzModel]: FizzBuzzModel(meta=MetaData(name=''), fizz='Fizz', buzz='Buzz')
#[FizzBuzzModel]: End evaluation of __call__ on GenericContext[int](value=15) (time elapsed: 0:00:00.000035).
#> GenericResult[list[Union[int, str]]](value=[1, 2, 'Fizz', 4, 'Buzz', 'Fizz', 7, 8, 'Fizz', 'Buzz', 11, 'Fizz', 13, 14, 'FizzBuzz'])
```

An example of the latter (model-specific options) is to disable validation of the result type on a particular model

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext

class NoValidationModel(CallableModel):
    @Flow.call(validate_result=False)
    def __call__(self, context: GenericContext[str]) -> GenericResult[float]:
        return "foo"

model = NoValidationModel()
print(model("foo"))
#> foo
```

To see a list of all the available options, you can look at the schema definition of the FlowOptions class:

```python
from ccflow import FlowOptions
print(FlowOptions.model_fields)
#> {'log_level': FieldInfo(annotation=int, required=False, default=10, description="If no 'evaluator' is set, will use a LoggingEvaluator with this log level"),
#> 'verbose': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use verbose logging'),
#> 'validate_result': FieldInfo(annotation=bool, required=False, default=True, description="Whether to validate the result to the model's result_type before returning"),
#> 'volatile': FieldInfo(annotation=bool, required=False, default=False, description='Whether this function is volatile (i.e. always returns a different value), and hence should always be excluded from caching'),
#> 'cacheable': FieldInfo(annotation=bool, required=False, default=False, description='Whether the model results should be cached if possible. This is False by default so that caching is opt-in'),
#> 'evaluator': FieldInfo(annotation=Union[Annotated[EvaluatorBase, InstanceOf], NoneType], required=False, default=None, description='A hook to set a custom evaluator')}
```

### Metadata

For convenience (and consistency), the `CallableModel` base class comes with a `meta` attribute that stores metadata about the step as represented by the `MetaData` class.
Usage of this is optional, and the `name` and `description` fields are mostly placeholders. At the moment

- The `name` field is used by the logging evaluator to identify the model in the logs (and will be automatically set when models are loaded into the `ModelRegistry` from `hydra` configs).
- The `description` is helpful when building a catalog of callable models from the `ModelRegistry`

As the number and size of workflows grows, it is useful to set these. especially for users that are trying to understand large workflows that other people have configured.

The `MetaData` class also has a field called `options` which is another way that `FlowOptions` can be set (as an alternative to the `FlowOptionsOverride` context manager, or setting them directly in the `@Flow.call` decorator).
The advantage of using the `meta` attribute for this is that the `FlowOptions` can be specified in the configuration file.

## Evaluators

One of the fields on `FlowOptions` above was the evaluator. This is a hook that allows for customization of how the steps in the workflow are evaluated.
At a high level, an evaluator is a special type of `CallableModel` that operates on a combination of the `__call__` function and the context, and returns a new function to be evaluated.
In such a way, additional functionality can be implemented, including

- Logging (i.e. `LoggingEvaluator`), which is the default
- Local caching (i.e. `MemoryCacheEvaluator`)
- Graph evaluator (i.e. `GraphEvaluator`)
- Dependency tracking (not open sourced yet)
- Disk based caching (not open sourced yet)
- Distributed evaluation (not open sourced yet)
- User defined evaluators

### In-memory Caching

Since we use standard python code to schedule the tasks within the workflow and pass data between tasks, diamond dependencies will cause the same task to be called more than once.
This may have performance implications as well as other unwanted side-effects if the tasks are not idempotent. Thus, `ccflow` provides an in-memory caching evaluator (`MemoryCacheEvaluator`) that will make sure that each task is evaluated only once for each context.

As caching is not always suitable for all models (if they are not referentially transparent), caching is opt-in by default by setting `FlowOptions.cacheable=True`.
Since it is also possible to set the default for all models to be cacheable via `FlowOptionsOverride`, there is also the ability to opt-out of caching for a specific model by setting `FlowOptions.volatile=True` in the `@Flow.call` decorator.

To illustrate, we start with a model that is neither `cacheable` nor `volatile`, and run it the standard way:

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext

class FibonacciModel(CallableModel):
    salt: int = 0

    @Flow.call
    def __call__(self, context: GenericContext[int]) -> GenericResult[int]:
        print(f"Calling model with {context}")
        if context.value <= 1:
            return context.value
        else:
            return self(context.value - 1).value + self(context.value - 2).value

model = FibonacciModel()
print(model(4))
#> Calling model with GenericContext[int](value=4)
#> Calling model with GenericContext[int](value=3)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=0)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=3)
```

Now we run it with `FlowOptions.cacheable=True` and the `MemoryCacheEvaluator`:

```python
from ccflow.evaluators import MemoryCacheEvaluator
from ccflow import FlowOptionsOverride

evaluator = MemoryCacheEvaluator()
with FlowOptionsOverride(options={"cacheable":True, "evaluator": evaluator}):
    print(model(4))
#> Calling model with GenericContext[int](value=4)
#> Calling model with GenericContext[int](value=3)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=3)
```

If we call it again with the same evaluator, results will still be pulled from the cache. This holds even if we construct a new model object with the same attributes:

```python
model = FibonacciModel()
with FlowOptionsOverride(options={"cacheable":True, "evaluator": evaluator}):
    print(model(2))
#> GenericResult[int](value=1)
```

Since the model attributes are part of the cache key, changing them will cause the cache to be invalidated:

```python
model = FibonacciModel(salt=1) 
with FlowOptionsOverride(options={"cacheable":True, "evaluator": evaluator}):
    print(model(2))
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=1)
```

### Graph Evaluation

Dependencies between tasks/steps in a workflow is one of the defining characteristics of a workflow orchestration framework. Earlier we covered how dependencies defined (via composition) between configuration objects.
However, since a `CallableModel` is a `BaseModel`, the same principle applies. Until now, we have been evaluating the `__call__` functions of models in the same order that python would as part of standard execution,
and we used caching to avoid redundant evaluations.

In other cases, it may be desirable to inspect the dependencies between tasks and evaluate them in a more optimal order. However, to do this, we need to describe an addition feature of the `CallableModel` class.
We eschew fancy inspection tricks to discover dependencies automatically, and instead rely on the user to define them explicitly.

We illustrate by subclassing the `FibonacciModel` from above to create a `FibonacciDepsModel` with dependencies defined:

```python
class FibonacciDepsModel(FibonacciModel):
    @Flow.deps
    def __deps__(self, context: GenericContext[int]):
      if context.value <= 1:
        return []
      return [(self, [GenericContext[int](value=context.value - 2), GenericContext[int](value=context.value - 1)])]
```

The type of the return value is `ccflow.GraphDepList`, which is `List[Tuple[CallableModelType, List[ContextType]]]`. In other words, for each `CallableModel` the evaluation depends on, the user lists which contexts it depends on for that model.
Note that the `__deps__` method is not called directly by users, but rather it is used internally by evaluator implementations. Since `__deps__` are explicitly specified by the creator of the model, they can return dependencies on models whose outputs do not directly feed into the current model (i.e. perhaps a model that writes a file to a shared location).
Similarly, the `__deps__` method can explicitly leave out actual dependencies if they can always be evaluated alongside the current model.

With the dependencies specified, we can now use this information to evaluate the workflow in a more optimal order. For example, the `GraphEvaluator` will topologically sort the dependency graph and evaluate the calls in that order.
However, because the later calls will still execute the standard code in the body of the `__call__` method, we need to use the `MemoryCacheEvaluator` to pull these from the cache.

```python
from ccflow.evaluators import GraphEvaluator, MemoryCacheEvaluator, MultiEvaluator
model = FibonacciDepsModel() 
evaluator = MultiEvaluator(evaluators=[GraphEvaluator(), MemoryCacheEvaluator()])
with FlowOptionsOverride(options={"cacheable":True, "evaluator": evaluator}):
    print(model(4))
#> Calling model with GenericContext[int](value=0)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=3)
#> Calling model with GenericContext[int](value=4)
#> GenericResult[int](value=3)
```

This also opens up possibilities for distributed evaluation.

### User Defined Evaluators

Most importantly, it is straightforward to implement your own evaluators, as no single library would be able to provide evaluators to cover all use cases.
Custom evaluators can be used to

- run models on other computing platforms (i.e. `dask`, `ray`, `spark`, etc.)
- to implement custom caching strategies (i.e. `redis`, `s3`, etc.)
- to implement custom logging strategies
- to implement custom evaluation strategies (i.e. `batching`)

An evaluator is basically another form of callable model, with a few caveats

- It doesn't use the `@Flow.call` decorator
- It uses the `ModelEvaluationContext` as the context type

The `ModelEvaluationContext` has fields for the model, the context, the function to evaluate (i.e. `__call__`), and the `FlowOptions`.
It too, has a `__call__` method that will evaluate the function on the model with the provided context (but ignoring any options).

Below we illustrate how to write a really simple evaluator that just prints the options and delegates to the `ModelEvaluationContext` to get the normal result.

```python
from ccflow import EvaluatorBase, ModelEvaluationContext, ResultType

class MyEvaluator(EvaluatorBase):

    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        print("Custom evaluator with options:", context.options)
        return context()

evaluator = MyEvaluator()
model = FibonacciModel()
with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
    print(model(0))

#> Custom evaluator with options: {'cacheable': True, 'type_': 'ccflow.callable.FlowOptions'}
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=0)
```
