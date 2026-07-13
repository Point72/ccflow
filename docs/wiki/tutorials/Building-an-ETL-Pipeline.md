# Building an ETL Pipeline

Now let's put [Defining Workflows](Defining-Workflows) to work and build an end-to-end [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) pipeline. You will write three callable models — extract, transform, and load — and drive them from a single [Hydra](Configuration-and-Hydra) configuration file with a command-line entry point.

The goal is to construct a set of callable models into which we pass contexts and get back results, and to define the pipeline via a static configuration file. This is a toy example, but it exercises every core idea end to end.

> [!NOTE]
> The full source is in-tree at [`ccflow/examples/etl`](https://github.com/Point72/ccflow/tree/main/ccflow/examples/etl), and can be run directly with `python -m ccflow.examples.etl`.

The pipeline does three things:

- **Extract** a website's HTML and save it.
- **Transform** that HTML into a CSV of link names and URLs.
- **Load** that CSV into a queryable SQLite database.

## A context for the pipeline

Our first step takes a context, so we can point the pipeline at different sites at runtime. Contexts are ideal for the small handful of parameters that vary between runs.

```python
from ccflow import ContextBase
from pydantic import Field

class SiteContext(ContextBase):
    site: str = Field(default="https://news.ycombinator.com")
```

Once the CLI is wired up, we will be able to select a context at runtime:

```bash
etl-cli +context=[]                      # default: hacker news
etl-cli +context=["http://lobste.rs"]    # a different site
```

## Extract

The extract step queries a site over HTTP and returns its HTML:

```python
from typing import Optional
from httpx import Client
from ccflow import CallableModel, Flow, GenericResult


class RestModel(CallableModel):
    @Flow.call
    def __call__(self, context: Optional[SiteContext] = None) -> GenericResult[str]:
        context = context or SiteContext()
        resp = Client().get(context.site, follow_redirects=True)
        return GenericResult[str](value=resp.text)
```

The key elements:

- It inherits from `CallableModel`, so it runs given a context and returns a result from a `@Flow.call`-decorated `__call__`.
- It takes a `SiteContext`.
- It returns a `GenericResult[str]`. (You could define a custom result type for stronger typing.)

Before adding the other steps, let's get this one running from a CLI.

## Wiring up a CLI

`ccflow` provides helpers that connect [Hydra](Configuration-and-Hydra) to the callable-models framework:

```python
import hydra
from ccflow.utils.hydra import cfg_run, cfg_explain_cli


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    cfg_run(cfg)

def explain():
    cfg_explain_cli(config_path="config", config_name="base", hydra_main=main)
```

`cfg_run` takes a Hydra configuration hierarchy and executes its top-level `callable`. `cfg_explain_cli` launches a UI for browsing the composed configuration. Both are runnable directly from the shipped example: `python -m ccflow.examples.etl` and `python -m ccflow.examples.etl.explain`.

## Configuring the extract step

Hydra is driven by YAML. Here is a configuration for the extract step:

```yaml
# ccflow/examples/etl/config/base.yaml
extract:
  _target_: ccflow.PublisherModel
  model:
    _target_: ccflow.examples.etl.models.RestModel
  publisher:
    _target_: ccflow.publishers.GenericFilePublisher
    name: raw
    suffix: .html
  field: value
```

The `extract` key is a `PublisherModel` — a `CallableModel` that runs the given `model` and hands its result to the given `publisher`. Here `RestModel` produces the HTML and `GenericFilePublisher` writes it to `raw.html`. (We could have written the file directly, but this shows how to reuse the same publisher anywhere.)

Run it:

```bash
python -m ccflow.examples.etl +callable=extract +context=[]
```

This calls `extract`, which calls `RestModel` and feeds the result to `GenericFilePublisher`, producing `raw.html`. Point it at a different site:

```bash
python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"]
```

Hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) lets you tweak any node — for example, change the output file name:

```bash
python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"] ++extract.publisher.name=lobsters
```

## Transform

The transform step reads an HTML file, extracts its links, and produces CSV:

```python
from csv import DictWriter
from io import StringIO
from bs4 import BeautifulSoup
from ccflow import CallableModel, Flow, GenericResult, NullContext

class LinksModel(CallableModel):
    file: str

    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult[str]:
        with open(self.file, "r") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        links = [{"name": a.text, "url": href} for a in soup.find_all("a", href=True) if (href := a["href"]).startswith("http")]

        io = StringIO()
        writer = DictWriter(io, fieldnames=["name", "url"])
        writer.writeheader()
        writer.writerows(links)
        return GenericResult[str](value=io.getvalue())
```

## Load

The load step reads a CSV file and loads it into SQLite:

```python
import sqlite3
from csv import DictReader
from pydantic import Field
from ccflow import CallableModel, Flow, GenericResult, NullContext


class DBModel(CallableModel):
    file: str
    db_file: str = Field(default="etl.db")
    table: str = Field(default="links")

    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult[str]:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (name TEXT, url TEXT)")
        with open(self.file, "r") as f:
            reader = DictReader(f)
            for row in reader:
                cursor.execute(f"INSERT INTO {self.table} (name, url) VALUES (?, ?)", (row["name"], row["url"]))
        conn.commit()
        return GenericResult[str](value="Data loaded into database")
```

## The full pipeline in one config

Register all three steps in the same YAML file:

```yaml
extract:
  _target_: ccflow.PublisherModel
  model:
    _target_: ccflow.examples.etl.models.RestModel
  publisher:
    _target_: ccflow.publishers.GenericFilePublisher
    name: raw
    suffix: .html
  field: value

transform:
  _target_: ccflow.PublisherModel
  model:
    _target_: ccflow.examples.etl.models.LinksModel
    file: ${extract.publisher.name}${extract.publisher.suffix}
  publisher:
    _target_: ccflow.publishers.GenericFilePublisher
    name: extracted
    suffix: .csv
  field: value

load:
  _target_: ccflow.examples.etl.models.DBModel
  file: ${transform.publisher.name}${transform.publisher.suffix}
  db_file: etl.db
  table: links
```

Notice the `transform` step references the extract step's output with Hydra/OmegaConf [interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation): `${extract.publisher.name}${extract.publisher.suffix}` resolves to `raw.html`. The `load` step needs no publisher — it writes the database directly.

## Running the pipeline

Run each step the same way, with overrides as needed:

```bash
# Transform
python -m ccflow.examples.etl +callable=transform +context=[]

# Transform with overrides: read lobsters.html, write lobsters.csv
python -m ccflow.examples.etl +callable=transform +context=[] ++transform.model.file=lobsters.html ++transform.publisher.name=lobsters

# Load
python -m ccflow.examples.etl +callable=load +context=[]

# Load with overrides: read lobsters.csv into an in-memory database
python -m ccflow.examples.etl +callable=load +context=[] ++load.file=lobsters.csv ++load.db_file=":memory:"
```

## Visualizing the configuration

Because Hydra loads all this into a single registry, you can inspect the resolved configuration with the explain entry point:

```bash
python -m ccflow.examples.etl.explain
```

<img src="https://github.com/point72/ccflow/raw/main/docs/img/wiki/etl/explain1.png?raw=true" width="400">

Combine it with overrides to confirm everything resolves as intended:

```bash
python -m ccflow.examples.etl.explain ++extract.publisher.name=test
```

<img src="https://github.com/point72/ccflow/raw/main/docs/img/wiki/etl/explain2.png?raw=true" width="400">

## What you learned

- Each ETL stage is a plain `CallableModel`, wired together through configuration.
- `PublisherModel` pairs a model with a publisher to compute-and-write in one step.
- A single YAML file, with interpolation between stages, defines the whole pipeline.
- One CLI runs any stage, and the explain UI shows exactly what was composed.

## Next steps

You wrote the whole pipeline in one file. Real applications split configuration into swappable pieces and dispatch between many workflows from the command line. That is the subject of the next tutorial.

- [Composing an ETL Application](Composing-an-ETL-Application) — break this into config groups and build a reusable, CLI-driven application.
- [Configuration and Hydra](Configuration-and-Hydra) — why this composition style is valuable.
- [Run Workflows from the CLI](Run-Workflows-from-the-CLI) — a focused reference for running and overriding.
