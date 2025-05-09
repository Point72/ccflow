- [Workflows with `ccflow`](#workflows-with-ccflow)
  - [Result Type](#result-type)
    - [`GenericResult`](#genericresult)
    - [Common Result Types](#common-result-types)
    - [Custom Results](#custom-results)
  - [Context](#context)
    - [`NullContext`](#nullcontext)
    - [`GenericContext`](#genericcontext)
    - [Common Context Types](#common-context-types)
    - [Custom Contexts](#custom-contexts)
    - [Properties](#properties)
  - [`CallableModel`](#callablemodel)

# Workflows with `ccflow`

As described in the [Workflow Design Goals](Design-Goals#workflow-design-goals), `ccflow` wishes to make it easy to define (via configuration) the tasks/steps that make up a workflow. In the [Using Configuration](Configuration#using-configuration) section and in some of the examples on that page, we described how to bind functionality to the configuration by adding `__call__` methods to the models.

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
These include `PandasResult`, `NDArrayResult`, `ArrowResult`, `XArrayResult` and `NarwhalsFrameResult` (for generic compatibility across dataframe libraries using the excellent [Narwhals](https://narwhals-dev.github.io/narwhals/) library).
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

## `CallableModel`

With the context and result types defined, we are now ready to define a workflow step/task.
Following the design objectives, and the exposition of how best to use configuration objects, we define a `CallableModel` abstract base class, which is essentially a configurable object where one must implement the `__call__` method as a function of the context. It has a few other features which we will explore below.
Note that since every CallableModel is a BaseModel, this class is configurable using all the configuration logic described in the previous section. In particular any workflow step can be named and added to the registry for later reference.

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

In the above example, we could simply pass the value `15` to the `__call__` method as the `context` argument: the decorator invokes pydantic type checking which validates the input and converts it to the appropriate type (`GenericContext[int]` in this case).
Similarly, we are able to return the result list directly, and the decorator will validate this as well, ensuring that the output conforms to the expected type (`GenericResult[list[int|str]]`).
Thus, the `@Flow.call` decorator combines with the pydantic type checking to reduce boilerplate for end users while preserving runtime type safety.

To further illustrate this point, if we try to pass an invalid input to the model, it will result in an error:

```python
try:
    model("not an integer")
except ValueError as e:
    print(e)
#> 1 validation error for GenericContext[int]
#> value
#>   Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='not an integer', input_type=str]
#>     For further information visit https://errors.pydantic.dev/2.11/v/int_parsing
```
