# Configuration and Hydra

`ccflow` can be used entirely from Python, but it is designed to work hand-in-hand with [Hydra](https://hydra.cc/). This page explains *why* file-based configuration and a command line matter, what Hydra brings to the table, and how the `ModelRegistry` complements it. If you want the steps rather than the reasoning, see the [Configuring Models](Configuring-Models) tutorial, the [Composing an ETL Application](Composing-an-ETL-Application) tutorial, and the [Run Workflows from the CLI](Run-Workflows-from-the-CLI) guide.

## Two ways of working, one set of objects

Real applications accumulate configuration. A data pipeline has sources, credentials, transforms, output locations, schedules, calendars, retry policies, and execution settings — and each of those has its own parameters. `ccflow` targets two quite different ways of interacting with all that configuration:

- **Static and versioned.** In production you want the entire configuration pinned down in files, reviewed, version-controlled, and reproducible. You run it from the command line and occasionally override a value.
- **Dynamic and interactive.** In research you want to construct, modify, and re-run configurations from a notebook or script, without editing files or restarting anything.

The important design goal is that these are *the same objects* seen two ways. A model you build interactively in Python is the same model Hydra instantiates from a YAML file. This is what lets a workflow move from a research notebook into production without being rewritten. The interactive path is served by the [`ModelRegistry`](Core-Concepts); the file-and-CLI path is served by Hydra.

## Why not just dictionaries and `argparse`?

It is tempting to reach for plain dictionaries, environment variables, or `argparse` for configuration. These break down as an application grows:

- **No schema.** A misspelled key (writing `weight` where the schema expects `weights`) or a wrong type (the string `"1234"` where an integer is expected) fails silently or far from its cause. `ccflow`'s pydantic models catch these when the configuration loads. See [Validate and Coerce Configuration](Configure-Complex-Values).
- **No composition.** A flat namespace does not let you assemble configuration from reusable pieces, spread it across files, or reuse the same fragment in several places.
- **No reuse of *choices*.** The hard part of large configuration is not the individual values, it is expressing "use the S3 cache here, the SQLite cache there, and swap them without touching anything else."

Hydra exists to solve the composition and command-line problems; pydantic (via `ccflow.BaseModel`) solves the schema problem. `ccflow` uses both.

## What Hydra provides

[Hydra](https://hydra.cc/), built on [OmegaConf](https://omegaconf.readthedocs.io/), is a framework for **composing hierarchical configuration from multiple files** and **overriding it from the command line**. A few of its ideas do most of the work.

### Composition from files

Configuration can be split across many small YAML files in a directory tree and recombined. A top-level file declares a `defaults` list that pulls in the others:

```yaml
defaults:
  - _self_
  - cache: memory
  - execution: local
```

Each entry brings in another file, and `_self_` marks where the current file's own content merges relative to those pieces. This keeps each concern in its own small, readable, independently reviewable file instead of one sprawling document.

### Config groups: the key idea

A **config group** is a directory of interchangeable options for one slice of the application. If you have a folder `cache/` containing `memory.yaml`, `redis.yaml`, and `s3.yaml`, then `cache` is a config group and each file is an option. You select one in the defaults list (`- cache: memory`) or from the command line (`cache=redis`).

This is the mechanism that makes an application genuinely *composable*. Each subsystem — the cache, the execution policy, the set of credentials, the output target — becomes a named, swappable slot. Changing which implementation an application uses is a matter of naming a different option, not editing code or wiring. An application assembled this way is really a small matrix of choices; a concrete run is one selection through that matrix.

Config groups also compose *across packages*. By adding a search path, an application can pull config groups out of an installed library:

```yaml
hydra:
  searchpath:
    - pkg://mytoolkit.config
```

This is what lets a reusable toolkit ship a library of config groups (calendars, credential shapes, cache backends, output writers) that any downstream application can select from — a connector package can contribute `cache=redis` simply by being installed. The [Composing an ETL Application](Composing-an-ETL-Application) tutorial builds an application in exactly this style.

### Packages and where config lands

The `@package` directive controls *where* a file's content is placed in the final configuration tree. A file that begins with `# @package _global_` merges its content at the root rather than under its group name, which is useful for base configurations that define top-level keys. Packages let a config group option write to whatever part of the tree it needs to.

### Overrides and interpolation

Once composed, any node can be overridden from the command line without touching a file — append a new key (`+context.date=2025-01-01`), change an existing one (`++cache.ttl=60`), or select a different group option (`cache=redis`). This is the fast, safe way to run one-off variations of a pinned configuration.

Within the configuration, OmegaConf **interpolation** lets one node reference another (`${extract.output.name}`), so a value is written once and reused. Resolvers extend this — `${oc.select:key,default}` resolves a key if it exists and falls back otherwise, which is a natural way to make a configuration *dispatch* between alternatives (for example, "run the backfill wrapper if one is selected, otherwise run the task directly").

### Instantiation into objects

Finally, Hydra turns configuration into objects. A mapping with a `_target_` key names a class to construct. Because `ccflow.BaseModel` serializes its type to `_target_`, Hydra-composed configuration instantiates `ccflow` models directly — the composed YAML *is* the object graph.

## How the registry complements Hydra

Hydra is excellent at composing files and instantiating objects, but on its own it stops at instantiation. `ccflow`'s `ModelRegistry` adds two things Hydra does not.

**Shared instances instead of copies.** OmegaConf interpolation (`${...}`) copies configuration: two nodes that interpolate the same source end up configured identically but as *separate instances*. The registry instead lets a model reference another *by name*, resolving to the **same object**. When configuration forms a dependency graph — one data source feeding many signals feeding one portfolio — you usually want the shared-instance semantics so a change at the source propagates everywhere. This name-based dependency injection is a `ccflow` addition on top of Hydra. [Core Concepts](Core-Concepts) and the [Configuring Models](Configuring-Models) tutorial show it in action.

**The same power without files.** The registry provides the interactive half of the story. `ModelRegistry.load_config(...)` loads a dictionary of configs (resolving name references as it goes), and `load_config_from_path(...)` loads the same Hydra files a CLI would use — so a researcher can pull the production configuration into a notebook, inspect it, tweak an object, and re-run, all as live Python objects.

## The payoff

Put together, the configuration story gives you:

- **One application, many configurations** — a versioned base plus command-line overrides for day-to-day variation.
- **Swappable subsystems** — change caches, executors, outputs, or credentials by selecting a different config group option, including options contributed by installed packages.
- **CLI-based dispatch** — choose *which* callable to run, and with what context, from the command line, so one entry point serves an entire catalog of workflows.
- **Reproducibility and testability** — configuration is data: reviewable in a pull request, diffable, and testable independently of the logic it drives.
- **A research-to-production path** — the same objects, whether built in a notebook or composed from files.

This composition style is not just a convenience for large projects; it is the foundation for building a reusable workflow toolkit that others assemble applications from. The [Composing an ETL Application](Composing-an-ETL-Application) tutorial walks through building exactly such a thing.

## Where to go next

- [Configuring Models](Configuring-Models) — build and register configurations interactively, then load them from files.
- [Composing an ETL Application](Composing-an-ETL-Application) — assemble a config-group-driven, CLI-dispatched application.
- [Run Workflows from the CLI](Run-Workflows-from-the-CLI) — the mechanics of running and overriding configured workflows.
- Hydra's own [documentation](https://hydra.cc/docs/intro/) — the full override grammar, config groups, and compose API.
