# Core Concepts

`ccflow` (Composable Configuration Flow) is really two frameworks that share a foundation:

- a **configuration framework** for defining hierarchical, strongly typed configuration and the relationships between pieces of it, and
- a **workflow framework** for associating code with that configuration, so that a configuration graph becomes a runnable graph of tasks.

The two halves fit together because everything in `ccflow` is a *model*. A workflow step is just a configuration object that also knows how to run. This page introduces the vocabulary you will meet throughout the rest of the documentation and explains how the pieces relate. It is a conceptual map, not an API listing — for the exact classes and their fields, see the [Reference](Reference).

## Models are configuration

The central abstraction is the **model**, `ccflow.BaseModel`. The naming follows [pydantic](https://docs.pydantic.dev/latest/), whose `BaseModel` it extends: think of a model as a *configurable class* — a set of typed, validated attributes. Because it is a pydantic model, you get schema validation, coercion, and serialization for free.

`ccflow` extends pydantic's base class in a few deliberate ways: it serializes the model's own type alongside its fields (so a serialized config can be reconstructed as the right subclass), aliases that type field to `_target_` for [`hydra`](Configuration-and-Hydra) compatibility, and rejects unknown or misnamed fields so configuration mistakes surface immediately rather than silently.

Using classes rather than raw dictionaries for configuration is a design choice, not an accident. A schema catches typos and type errors *when the configuration loads* instead of when it is used, lets you validate combinations of parameters, and makes configuration self-documenting. [Design Goals](Design-Goals) discusses this reasoning in depth.

## The registry binds configuration together

A **`ModelRegistry`** is a named collection of models — conceptually a catalog. Registries can contain other registries, forming a hierarchy, and a single **root** registry sits at the top as a singleton.

The registry is what turns a pile of independent config objects into a connected graph. Instead of passing configuration around by hand, a model can refer to another model *by its name in the root registry*, and `ccflow` resolves that reference to the actual instance. This is dependency injection: a high-level config depends on lower-level configs by name, changes propagate through the shared instance, and any part of the hierarchy can be swapped without disturbing the rest.

This name-based linking is subtly but importantly different from copying configuration around. Two configs that both reference `data/prices` point at the *same object*, so a change in one place is seen everywhere — a graph, not a tree of duplicated values.

## Contexts parameterize workflows

A **context** (`ccflow.ContextBase`) is the set of parameters that *templatize* a workflow at runtime. The classic example is a date: the same pipeline runs for many dates, so date is a context rather than fixed configuration.

Separating context from configuration lets a workflow author expose only the few parameters that vary between runs (date, region, symbol, universe) while keeping everything else fixed in configuration. It also keeps configuration immutable during a run — contexts, not configs, carry the per-run variation, which is why contexts are frozen and hashable (so they can serve as cache keys). A context is itself a kind of result, so one step can compute the context another step consumes.

## Results carry data between steps

A **result** (`ccflow.ResultBase`) is the typed output of a workflow step. Insisting that every step return a `ResultBase` (again, a pydantic model) gives the framework a uniform thing to validate, serialize, and operate on generically — and it is what makes features like delayed and cached evaluation possible. Results range from the catch-all `GenericResult` to typed wrappers around common data structures.

## Callable models are workflow steps

A **`CallableModel`** is a `BaseModel` that can be *called*: given a context, it returns a result. It is the point where configuration meets code. You implement its `__call__` method and decorate it with `@Flow.call`; the decorator is where the framework injects its machinery.

Because a `CallableModel` is still a `BaseModel`, a workflow step is configurable, serializable, registrable, and composable exactly like any other configuration. Steps depend on other steps through ordinary object composition — Python itself passes data between them and determines evaluation order — so wiring a workflow needs no new language or scheduler. `@Flow.model` offers an alternative, function-first way to define the same kind of model from a plain Python signature.

## The Flow decorator hides the machinery

`@Flow.call` (and its sibling `@Flow.model`) is where "the magic lives". By default it performs runtime type checking of the context and result, cutting boilerplate while preserving type safety. But it is also the seam through which the framework layers on logging, caching, and alternative evaluation strategies — without the model's author writing any of that. The behavior is controlled by **`FlowOptions`**, which can be set on the decorator, on the model's metadata, through a scoped override context manager, or per call.

## Evaluators control *how* steps run

An **evaluator** decides how a callable model is actually executed. The default logs each evaluation; others cache results in memory, evaluate an explicit dependency graph in topological order, retry on failure, or (in extensions) distribute work across a cluster. Evaluators are themselves models, so the execution strategy is just more configuration. This separation — *what* the workflow computes versus *how* it is run — is deliberate: you can add caching, retries, or distribution to an existing workflow by changing an evaluator, not by rewriting steps.

## Publishers send data out

A **publisher** is a model that knows how to take data and send it somewhere — a file, an object store, a database, an email, a report. Giving all publishers a common interface means one can be substituted for another purely through configuration, so switching from writing a local file to emailing a report is a config change rather than a code change.

## How it fits together

A complete `ccflow` application is a registry of models. Some are pure configuration; some are callable steps wired together by composition and registry references; contexts parameterize them; evaluators decide how they run; publishers move their outputs. Because every one of these is a `BaseModel`, the whole thing can be defined interactively in Python for research, or loaded from files with [`hydra`](Configuration-and-Hydra) for production — the same objects either way.

From here:

- [Design Goals](Design-Goals) — the requirements and reasoning behind these abstractions.
- [Configuration and Hydra](Configuration-and-Hydra) — why file-based composition and a CLI matter, and how config groups build on the registry.
- [Tutorials](Tutorials) — put the concepts into practice, starting with [First Steps](First-Steps).
- [Reference](Reference) — the exact contexts, results, models, publishers, and evaluators that ship with `ccflow`.
