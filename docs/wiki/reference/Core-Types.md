# Core Types

Reference for the base classes and options at the center of `ccflow`. For conceptual orientation see [Core Concepts](Core-Concepts); for usage see the [Tutorials](Tutorials).

## `BaseModel`

The base class for all configuration in `ccflow`. Inherits from [pydantic](https://docs.pydantic.dev/latest/)'s `BaseModel` and adds:

- serialization of the model's own type as `type_`, aliased to `_target_`, so a serialized config reconstructs as the correct subclass and is instantiable by Hydra;
- rejection of unknown/misnamed fields (extra inputs forbidden) by default;
- compatibility with Hydra instantiation and the `ModelRegistry`.

All pydantic model features (validation, coercion, `model_dump`, `model_validate`, JSON schema) apply.

## `ModelRegistry`

A named, mutable collection of `BaseModel` instances that behaves like a `collections.abc.Mapping`.

- `add(name, model, overwrite=False)` — register a model; raises if `name` exists unless `overwrite=True`; validates that `model` is a `BaseModel`.
- `__getitem__` / `get(name, default=...)` — retrieve by name; supports path syntax (`"a/b"`) across nested registries.
- `models` — a mapping of the top-level entries only.
- iteration / `len` — over all combined keys, including those in sub-registries.
- `clear()` — remove all entries (and reset registrations/dependencies of previously registered models).
- `load_config(dict)` — load a nested dictionary of configs, interpreting nested dicts (without `_target_`) as sub-registries and resolving string references against the root registry.
- `load_config_from_path(path, config_key=...)` — load configuration from Hydra files; `config_key` selects the subtree to register.

Registries may contain other registries, forming a hierarchy.

### `ModelRegistry.root()`

Returns the singleton root registry. Name references in configs (e.g. `"data/source1"`) are resolved against this root, which is how `ccflow` performs dependency injection over shared instances.

Models expose their placement in the registry:

- `get_registrations()` — list of `(registry, name)` tuples.
- `get_registered_names()` — list of path strings.
- `get_registry_dependencies()` — the registered models this model depends on, found recursively.

## `ContextBase`

Base class for contexts — the runtime parameters that templatize a workflow. A `ContextBase` is a `BaseModel` and also a `ResultBase` (so a context can be produced as a step's result). Contexts are **frozen** (immutable) and hashable by default, so they can serve as cache keys. See [Contexts and Results](Contexts-and-Results) for the built-in context types.

## `ResultBase`

Base class for the results returned by workflow steps. A `ResultBase` is a `BaseModel`, giving uniform validation, serialization, and support for delayed/cached evaluation. See [Contexts and Results](Contexts-and-Results) for the built-in result types.

## `CallableModel`

A `BaseModel` that is called with a context and returns a result. Subclasses implement `__call__`, decorated with `@Flow.call`.

- `context_type` — the type used to validate the incoming context; inferred from the `__call__` annotation by default, overridable as a property.
- `result_type` — the type used to validate the returned result; inferred by default, overridable as a property.
- `meta` — a `MetaData` instance (see below).

Raises `TypeError` on call if no context type can be determined (neither an annotation nor a `context_type` property is provided).

## The `Flow` decorators

- `@Flow.call` — required on a `CallableModel.__call__`. Applies input/output type validation and is the seam through which the framework injects logging, caching, and evaluation. Accepts options such as `validate_result=False`. `@Flow.call(auto_context=True)` lets `__call__` declare context fields as keyword-only parameters instead of one context object.
- `@Flow.deps` — decorates a `__deps__` method that returns a `GraphDepList` (`List[Tuple[CallableModelType, List[ContextType]]]`), declaring explicit dependencies for graph evaluation.
- `@Flow.model` — turns a plain Python function into a `CallableModel`. See [Flow Model](Flow-Model).
- `@Flow.context_transform` — defines reusable contextual rewrites for `@Flow.model`. See [Flow Model](Flow-Model).

## `MetaData`

Metadata carried on every `CallableModel` via its `meta` attribute:

- `name` — used by the logging evaluator to identify the model; set automatically when models are loaded from Hydra configs.
- `description` — free text, useful for cataloging models in a registry.
- `options` — a `FlowOptions` payload, allowing flow options to be specified in a config file and travel with the model instance.

## Flow Options

`FlowOptions` controls the behavior injected by `@Flow.call`. It can be set as decorator arguments, via `FlowOptionsOverride`, on `meta.options`, or per call with `_options`.

| Field             | Type                    | Default | Description                                                                                      |
| :---------------- | :---------------------- | :------ | :----------------------------------------------------------------------------------------------- |
| `log_level`       | `int`                   | `10`    | If no `evaluator` is set, uses a `LoggingEvaluator` at this level.                               |
| `verbose`         | `bool`                  | `True`  | Whether to use verbose logging.                                                                  |
| `validate_result` | `bool`                  | `True`  | Whether to validate the result against the model's `result_type` before returning.               |
| `volatile`        | `bool`                  | `False` | Whether the function is volatile (always returns a new value) and must be excluded from caching. |
| `cacheable`       | `bool`                  | `False` | Whether results may be cached; caching is opt-in.                                                |
| `evaluator`       | `EvaluatorBase \| None` | `None`  | A hook to set a custom evaluator.                                                                |

### `FlowOptionsOverride`

A context manager that applies `FlowOptions` within its scope. Accepts `options=...` plus optional `model_types=(...)` or `models=(...)` to restrict the override to specific model classes or instances.

## Evaluator types

- `EvaluatorBase` — base class for evaluators. An evaluator uses `ModelEvaluationContext` as its context type and does not use `@Flow.call`. Override `is_transparent(context) -> bool` to return `True` for evaluators that do not change the result (logging, timing), so cache-key computation skips them.
- `ModelEvaluationContext` — carries the model, context, function to evaluate (`__call__` or `__deps__`), and `FlowOptions`; calling it evaluates the function on the model with the context (ignoring options).

See [Built-in Models](Built-in-Models#evaluators) for the concrete evaluators.

## Publisher types

- `BasePublisher` (`ccflow.publishers`) — base interface for components that send data somewhere; a common interface lets publishers be swapped through configuration.
- `PublisherModel` — a `CallableModel` that runs a `model` and hands its result to a `publisher` in one step.

See [Built-in Models](Built-in-Models#publishers) for the concrete publishers.
