# Run Workflows from the CLI

Once a workflow is configured, you run it from the command line through a small [Hydra](Configuration-and-Hydra) entry point. This guide covers running a configured callable, choosing which one to run, applying overrides, and inspecting the composed configuration. It assumes you have a config directory and an entry point as built in [Building an ETL Pipeline](Building-an-ETL-Pipeline) and [Composing an ETL Application](Composing-an-ETL-Application).

## Set up the entry point

An entry point wires Hydra to `ccflow`'s runner and explainer:

```python
import hydra
from ccflow.utils.hydra import cfg_run, cfg_explain_cli


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    return cfg_run(cfg)

def explain():
    cfg_explain_cli(config_path="config", config_name="base", hydra_main=main)
```

`cfg_run` executes the callable named by the configuration's top-level `callable` key. `cfg_explain_cli` opens a UI over the composed configuration without running anything.

## Run a callable

Select which callable to run with `+callable`, and pass a context with `+context`:

```bash
# Run the 'extract' callable with an empty context:
python -m myapp +callable=extract +context=[]

# Run it with a populated context:
python -m myapp +callable=extract +context=["http://lobste.rs"]
```

`+key=value` *adds* a key that is not already in the config; use it for `callable` and `context`, which are supplied at run time.

## Override configuration values

Use Hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) to change any node of the composed configuration:

```bash
# Change a nested value (++ forces an existing key):
python -m myapp +callable=extract +context=[] ++extract.publisher.name=lobsters

# Point a stage at different inputs:
python -m myapp +callable=transform +context=[] ++transform.model.file=lobsters.html
```

- `++key=value` sets an existing key (force-override).
- `+key=value` appends a new key.
- `key=value` selects a [config group](Composing-an-ETL-Application#swapping-an-option) option.

## Swap a config-group option

If a subsystem is a config group, choose a different option by name — no `+`/`++`:

```bash
python -m myapp +callable=extract +context=[] extract=file    # use extract/file instead of the default
```

## Point at your own config directory

Any Hydra app accepts a config directory and root config name on the command line, so a shared entry point can run different applications:

```bash
python -m myapp --config-dir ./config --config-name pipeline +context=[]
```

## Inspect the composed configuration

Before (or instead of) running, use the explain entry point to see exactly what Hydra composed — invaluable for confirming overrides landed where you expect:

```bash
python -m myapp.explain
python -m myapp.explain ++extract.publisher.name=test
```

<img src="https://github.com/point72/ccflow/raw/main/docs/img/wiki/etl/explain2.png?raw=true" width="400">

## See also

- [Composing an ETL Application](Composing-an-ETL-Application) — build a config-group-driven app around this entry point.
- [Configuration and Hydra](Configuration-and-Hydra) — the composition and override model in depth.
- Hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) — the full command-line syntax.
