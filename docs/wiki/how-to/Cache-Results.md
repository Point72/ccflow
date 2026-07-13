# Cache Results

Because `ccflow` schedules tasks with ordinary Python, a diamond-shaped dependency graph will call the same task more than once. This guide shows how to cache results to avoid redundant work, how to evaluate an explicit dependency graph, and how to write your own evaluator. It builds on [Defining Workflows](Defining-Workflows); the full evaluator catalog is in [Built-in Models](Built-in-Models#evaluators).

The examples use this model, which prints on each call so you can see when it actually runs:

```python
from ccflow import CallableModel, Flow, GenericResult, GenericContext, FlowOptionsOverride
from ccflow.evaluators import MemoryCacheEvaluator

class FibonacciModel(CallableModel):
    salt: int = 0

    @Flow.call
    def __call__(self, context: GenericContext[int]) -> GenericResult[int]:
        print(f"Calling model with {context}")
        if context.value <= 1:
            return context.value
        return self(context.value - 1).value + self(context.value - 2).value
```

## Enable in-memory caching

Caching is opt-in. Set `cacheable=True` and supply a `MemoryCacheEvaluator`, scoped with `FlowOptionsOverride`, so each `(model, context)` runs once:

```python
model = FibonacciModel()
evaluator = MemoryCacheEvaluator()
with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
    print(model(4))
#> Calling model with GenericContext[int](value=4)
#> Calling model with GenericContext[int](value=3)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=3)
```

The redundant calls that plain evaluation would make are gone. Reusing the same evaluator keeps serving from the cache, even for a freshly constructed model with the same fields:

```python
model = FibonacciModel()
with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
    print(model(2))
#> GenericResult[int](value=1)
```

To keep a model out of the cache even when caching is on globally, mark it `volatile=True` in its `@Flow.call`.

## Understand cache invalidation

A model's fields are part of its cache key, so changing them invalidates the cache automatically:

```python
model = FibonacciModel(salt=1)
with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
    print(model(2))
#> Calling model with GenericContext[int](value=2)
#> ...
```

If a model's behavior depends on code *outside* the class body (a module-level helper or shared class), list those in `__ccflow_tokenizer_deps__` so the cache key changes when they change:

```python
def helper(x):
    return x + 1

class SharedLogic:
    def transform(self, x):
        return x * 2

class MyModel(CallableModel):
    __ccflow_tokenizer_deps__ = [helper, SharedLogic]
```

Wrapping a model in *transparent* evaluators (logging, timing) does not change its cache identity. For the full key-derivation rules, see [Built-in Models](Built-in-Models#how-cache-keys-are-built).

## Evaluate a dependency graph

To evaluate steps in an optimal order rather than Python's call order, declare dependencies explicitly with `@Flow.deps` and use the `GraphEvaluator` (together with the cache, since graph nodes still run their `__call__` bodies):

```python
class FibonacciDepsModel(FibonacciModel):
    @Flow.deps
    def __deps__(self, context: GenericContext[int]):
        if context.value <= 1:
            return []
        return [(self, [GenericContext[int](value=context.value - 2), GenericContext[int](value=context.value - 1)])]
```

`__deps__` returns a `GraphDepList` — for each model an evaluation depends on, the list of contexts it needs. Then combine the evaluators:

```python
from ccflow.evaluators import GraphEvaluator, MultiEvaluator

model = FibonacciDepsModel()
evaluator = MultiEvaluator(evaluators=[GraphEvaluator(), MemoryCacheEvaluator()])
with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
    print(model(4))
#> Calling model with GenericContext[int](value=0)
#> Calling model with GenericContext[int](value=1)
#> Calling model with GenericContext[int](value=2)
#> Calling model with GenericContext[int](value=3)
#> Calling model with GenericContext[int](value=4)
#> GenericResult[int](value=3)
```

Note the topological order (0, 1, 2, 3, 4), and that each node runs once. This is also the foundation for distributed evaluation.

## Write a custom evaluator

No library can provide every execution strategy, so evaluators are extensible. An evaluator is a model that takes a `ModelEvaluationContext` (which carries the model, context, function, and options) and returns a result. Override `is_transparent` to return `True` if it does not change the result (so caching ignores it):

```python
from ccflow import EvaluatorBase, ModelEvaluationContext, ResultType

class MyEvaluator(EvaluatorBase):
    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        print("Custom evaluator with options:", context.options)
        return context()

with FlowOptionsOverride(options={"cacheable": True, "evaluator": MyEvaluator()}):
    print(FibonacciModel()(0))
#> Custom evaluator with options: {'cacheable': True, 'type_': 'ccflow.callable.FlowOptions'}
#> Calling model with GenericContext[int](value=0)
#> GenericResult[int](value=0)
```

Custom evaluators can run models on other platforms (Dask, Ray, Spark), implement other caching backends (Redis, S3), or add batching and custom logging.

## See also

- [Retry on Failure](Retry-on-Failure) — the retry evaluator and `RetryModel`.
- [Built-in Models](Built-in-Models#evaluators) — the full evaluator catalog and cache-key rules.
- [Defining Workflows](Defining-Workflows) — where `FlowOptions` and evaluators are introduced.
