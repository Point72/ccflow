# Composing an ETL Application

In [Building an ETL Pipeline](Building-an-ETL-Pipeline) you wrote a whole pipeline in one YAML file. That is fine for one pipeline, but real applications need to *swap* pieces — a different source here, a different output there, a durable cache in production and a no-op cache in tests — and to run *many* workflows from one command. This tutorial grows the pipeline into a reusable, config-group-driven application with command-line dispatch.

By the end you will have built the skeleton of a reusable ETL *toolkit*: a package of models plus a library of config groups that downstream applications assemble and run through a single entry point. The reasoning behind this style is in [Configuration and Hydra](Configuration-and-Hydra); here we build it.

## From one file to config groups

Recall the single-file config from the last tutorial. The first move is to give each stage its own file inside a **config group** — a directory of interchangeable options for one slice of the app.

```
config/
  base.yaml
  extract/
    rest.yaml
  transform/
    links.yaml
  load/
    db.yaml
```

`base.yaml` composes the pieces with a `defaults` list:

```yaml
# config/base.yaml
defaults:
  - extract: rest
  - transform: links
  - load: db
```

Each stage lives in its own file — an *option* within its group:

```yaml
# config/extract/rest.yaml
_target_: ccflow.PublisherModel
model:
  _target_: ccflow.examples.etl.models.RestModel
publisher:
  _target_: ccflow.publishers.GenericFilePublisher
  name: raw
  suffix: .html
field: value
```

```yaml
# config/transform/links.yaml
_target_: ccflow.PublisherModel
model:
  _target_: ccflow.examples.etl.models.LinksModel
  file: ${extract.publisher.name}${extract.publisher.suffix}
publisher:
  _target_: ccflow.publishers.GenericFilePublisher
  name: extracted
  suffix: .csv
field: value
```

```yaml
# config/load/db.yaml
_target_: ccflow.examples.etl.models.DBModel
file: ${transform.publisher.name}${transform.publisher.suffix}
db_file: etl.db
table: links
```

This composes to exactly the same configuration as before — but now each concern is a small, independently reviewable file.

## Swapping an option

The payoff appears when a group has more than one option. Add a second way to extract — reading from a local file instead of over HTTP:

```yaml
# config/extract/file.yaml
_target_: ccflow.PublisherModel
model:
  _target_: ccflow.examples.etl.models.LocalReadModel   # your own model
  path: ./raw.html
publisher:
  _target_: ccflow.publishers.GenericFilePublisher
  name: raw
  suffix: .html
field: value
```

Now the extract subsystem is swappable by name, at the command line, with nothing else changing:

```bash
python -m ccflow.examples.etl +callable=extract              # uses extract/rest (the default)
python -m ccflow.examples.etl +callable=extract extract=file # swaps in extract/file
```

That is the essence of config-group composition: an application is a small matrix of choices, and a run is one path through it.

## Dispatching which callable to run

You have already used `+callable=extract` to choose *which* step to run. That works because the shared entry point runs whatever the top-level `callable` key names. Making `callable` a first-class, overridable key is what turns one entry point into a dispatcher over a whole catalog of workflows.

A common pattern is to compute `callable` from an optional selection so the same app can run a task directly *or* wrapped in something else. `ccflow` uses OmegaConf's `oc.select` resolver for this:

```yaml
# config/base.yaml
defaults:
  - _self_
  - extract: rest
  - transform: links
  - load: db

# The concrete task the app runs by default:
task: ${load}

# Run an optional wrapper if one is selected, otherwise the task itself:
callable: ${oc.select:wrapper,/task}
```

`${oc.select:wrapper,/task}` resolves to `wrapper` if a `wrapper` key has been selected, and otherwise falls back to the registered `/task`. Selecting a wrapper is then just another config-group choice (`+wrapper=/wrappers/retry`, say), and the same command runs either shape. Leading-slash names like `/task` refer to models by their path in the root registry.

## Sharing an execution policy

An application usually wants the *same* execution behavior everywhere — graph evaluation, memory caching, and logging — regardless of which workflow runs. Set it once with a shared `FlowOptions` block that the entry point applies:

```yaml
# config/base.yaml (continued)
cli:
  model:
    _target_: ccflow.FlowOptions
    evaluator:
      _target_: ccflow.evaluators.MultiEvaluator
      evaluators:
        - _target_: ccflow.evaluators.GraphEvaluator
        - _target_: ccflow.evaluators.MemoryCacheEvaluator
        - _target_: ccflow.evaluators.LoggingEvaluator
    cacheable: true
```

Now every run of this application evaluates its dependency graph, caches repeated sub-results, and logs — because the execution strategy is configuration, not code. See [Cache Results](Cache-Results) for how the graph and cache evaluators cooperate.

## Packaging groups for reuse

Everything so far lived beside one application. To build a *toolkit* — reusable ETL building blocks that many applications assemble — move the models and their config groups into an installable package.

Give the package a config directory of domain-neutral, swappable groups:

```
mytoolkit/
  __init__.py            # exports your CallableModels, publishers, etc.
  cli.py                 # the shared entry point (below)
  config/
    base.yaml            # sensible defaults, marked global
    cache/
      noop.yaml
    execution/
      default.yaml
    credentials/
      default.yaml
    calendars/
      default.yaml
```

Mark the package base so its keys land at the root of any config that includes it:

```yaml
# mytoolkit/config/base.yaml
# @package _global_

defaults:
  - _self_
  - cache: noop
  - execution: default
  - credentials: default
```

The `# @package _global_` directive places this file's content at the top level rather than nested under `base`, which is what you want for a shared base that defines top-level keys. Each group (`cache`, `execution`, ...) ships a safe default option, and installed connector packages can contribute more options to the same groups — a package that provides `cache=redis` becomes selectable simply by being installed.

A downstream application then keeps only *its* configuration and pulls the toolkit's groups in via a search path:

```yaml
# app/config/pipeline.yaml
defaults:
  - _self_
  - /cache: noop          # a group provided by the toolkit
  - /execution: default

hydra:
  searchpath:
    - pkg://mytoolkit.config

model:
  _target_: myapp.MyPipeline

task: ${model}
callable: ${oc.select:wrapper,/task}
```

`pkg://mytoolkit.config` tells Hydra to look inside the installed package for config groups, so the application composes its own file together with the toolkit's shared groups. Switching a subsystem — say, from the no-op cache to a durable one contributed by a connector package — is then a one-word change (`cache=redis`) with no code edits.

## A shared entry point

Finally, give the toolkit one command that every application uses. Build it on the same helpers from [Building an ETL Pipeline](Building-an-ETL-Pipeline):

```python
# mytoolkit/cli.py
import hydra
from ccflow.utils.hydra import cfg_run, cfg_explain_cli


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    return cfg_run(cfg)


def explain():
    cfg_explain_cli(config_path="config", config_name="base", hydra_main=main)
```

Expose them as console scripts so they install as real commands:

```toml
# pyproject.toml
[project.scripts]
cc-run = "mytoolkit.cli:main"
cc-run-explain = "mytoolkit.cli:explain"
```

Because these are ordinary Hydra apps, any downstream application can point the same command at its own config directory:

```bash
# Run an application's pipeline through the shared entry point:
cc-run --config-dir ./app/config --config-name pipeline +context=[]

# Swap a subsystem by naming a different group option:
cc-run --config-dir ./app/config --config-name pipeline cache=redis +context=[]

# Inspect the composed configuration without running it:
cc-run-explain --config-dir ./app/config --config-name pipeline
```

One command, one place to look, and a matrix of config-group choices behind it.

## What you built

- A config directory where each subsystem is a **config group** of swappable options.
- Command-line **dispatch** of which callable runs, via a `callable` key resolved with `oc.select`.
- A shared **execution policy** applied to every run through a `FlowOptions` block.
- A reusable **toolkit**: models plus config groups shipped in a package, pulled into applications with `pkg://` search paths, and driven by one console entry point.

This is the shape of a production `ccflow` application: a versioned base, swappable subsystems, and a single dispatchable command — with the same objects still available interactively for research.

## Next steps

- [Building a Configurable Calculator](Building-a-Configurable-Calculator) — the capstone tutorial: bring in the functional `@Flow.model` API and drive a registry of functions entirely from the CLI.
- [Configuration and Hydra](Configuration-and-Hydra) — the reasoning behind config groups, packages, and dispatch.
- [Run Workflows from the CLI](Run-Workflows-from-the-CLI) — a focused guide to running, overriding, and explaining.
- [Cache Results](Cache-Results) and [Retry on Failure](Retry-on-Failure) — reliability and performance for the workflows you dispatch.
