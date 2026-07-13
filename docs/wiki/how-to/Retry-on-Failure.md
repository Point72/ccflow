# Retry on Failure

To make a flaky step resilient — a network fetch, a rate-limited API — retry it on failure. `ccflow` offers two approaches: a `RetryEvaluator` that wraps whatever runs in its scope, and a `RetryModel` that bakes a retry policy into the graph. This guide covers both. They share the same `RetryPolicy` mechanics; the evaluator catalog is in [Built-in Models](Built-in-Models#evaluators).

## Retry with an evaluator

Configure a `RetryEvaluator` and apply it through `FlowOptionsOverride`. It supports exponential backoff, jitter, and stop conditions:

```python
from ccflow import FlowOptionsOverride
from ccflow.evaluators import RetryEvaluator

retry = RetryEvaluator(
    max_attempts=5,
    wait_initial=0.5,       # first retry waits ~0.5s
    wait_multiplier=2.0,    # then 1s, 2s, 4s, ...
    wait_max=30.0,          # cap any single wait at 30s
    wait_jitter=0.25,       # add up to 0.25s of random jitter
    max_delay=120.0,        # stop once cumulative waiting would exceed 2 minutes
    retry_exceptions=[TimeoutError, ConnectionError],
)

with FlowOptionsOverride(options={"evaluator": retry}):
    result = my_model(my_context)
```

`max_delay` caps the *cumulative* sleep time between retries (not total runtime): once the next backoff would push accumulated waiting over the budget, no further retries happen. Use `retry_exceptions` / `no_retry_exceptions` to choose which exceptions are retried.

> [!NOTE]
> `retry_exceptions` and `no_retry_exceptions` accept either a bare exception class (as above) or an importable path string (e.g. `"httpx.ConnectError"`, the form for YAML/Hydra configs). Paths are imported eagerly, so a third-party exception requires that package installed.

The evaluator is transparent (a successful result is identical to a direct call, so caching and dependency graphs are unaffected) and holds no per-call state, so one instance is safe to share across threads and combine with parallel evaluators. Place it *inside* a parallel evaluator to retry each task independently, or *outside* to retry the whole dispatch as a unit.

## Choose which models are retried

An evaluator wraps *every* model in its scope, including dependencies. To narrow that, from coarse to fine:

**Scope the override to model types or instances.**

```python
with FlowOptionsOverride(options={"evaluator": retry}, model_types=(FetchFromApi,)):
    result = pipeline(context)          # only FetchFromApi instances are retried
```

Use `models=(instance,)` to target a single instance.

**Override per call.**

```python
result = my_model(context, _options={"evaluator": retry})
```

**Pin it on the model.** Set `model.meta.options` so a model always carries its retry policy.

## Bake a policy into the graph with `RetryModel`

When you want the retry policy to be part of the workflow itself — declared statically in config, visible in serialization and the dependency graph — wrap the model in a `RetryModel` instead. It becomes a first-class node and preserves the wrapped model's `context_type` / `result_type`:

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

Use `RetryEvaluator` for a cross-cutting "how to run" policy applied at runtime; use `RetryModel` when the policy belongs to one specific model as part of the graph.

## See also

- [Cache Results](Cache-Results) — combine retries with caching and graph evaluation.
- [Built-in Models](Built-in-Models#evaluators) — the full evaluator catalog.
