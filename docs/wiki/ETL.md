- [Building an ETL System](#building-an-etl-system)
  - [Background](#background)
  - [Models](#models)
    - [Context](#context)
    - [Extract](#extract)
  - [CLI](#cli)
  - [Configuration](#configuration)
  - [Remaining Models](#remaining-models)
    - [Transform](#transform)
    - [Load](#load)
  - [Full Configuration](#full-configuration)
  - [Running](#running)
  - [Visualizing](#visualizing)
  - [Appendix - Multiple Files](#appendix---multiple-files)

# Building an ETL System

Let's take what we learned in [Workflows](Workflows) and build an end-to-end [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) system with `ccflow`.

The goal is to construct a set of [Callable Models](Workflows#callable-models) into which we can pass [Contexts](Workflows#context) and get back [Results](Workflows#result-type), and to define our workflows via static configuration files using [hydra](https://hydra.cc).

In order to generically operate on the data that communicated between steps of the workflow, `ccflow` defines a base class for the results that a step returns.
These are also child classes of `BaseModel` so one gets all the type validation and serialization features of pydantic models for free. Insisting on a base class for all result types also allows
the framework to perform additional magic (such as delayed evaluation).

Some result types are provided with `ccflow`, but it is straightforward to define your own.

## Background

Let's start with a simple set of tasks:

- Extract a website's `html` content and save it
- Given a saved `.html` file, transform it into a `csv` file of link names and corresponding URLs
- Given a saved `.csv` file of names and URLs, load it into a queryable sqlite database

This is a toy example, but we will use it to put the concepts in [Workflows](Workflows) to practice.
Additionally, we will grow our understanding of [hydra](https://hydra.cc) with a concrete example.

> [!NOTE]
> Source code is available in-source, in [ccflow/examples/etl](https://github.com/Point72/ccflow/tree/main/ccflow/examples/etl).

## Models

We'll define a single [Context](Workflows#context) as the argument to our first model.
This is not strictly necessary, but we'll use it as an example of a context.
In general, contexts are very useful for storing global reference data, like the current date being processed in a backfill pipeline.

### Context

```python
from ccflow import ContextBase
from pydantic import Field

class SiteContext(ContextBase):
    site: str = Field(default="https://en.wikipedia.org/wiki/Main_Page")
```

This class has a single attribute, `site`.
When we finish writing our ETL CLI, we will be able to pass in different contexts at runtime:

```bash
etl-cli +context=[]  # use default wikipedia
etl-cli +context=["http://lobste.rs"]  # query lobste.rs instead
```

### Extract

For the `extract` stage of our ETL pipeline, let's write a basic model to query a website over REST and return the `HTML` content:

```python
from typing import Optional
from httpx import Client
from ccflow import CallableModel, Flow, GenericResult, NullContext


class RestModel(CallableModel):
    @Flow.call
    def __call__(self, context: Optional[SiteContext] = None) -> GenericResult[str]:
        context = context or SiteContext()
        resp = Client().get(context.site, follow_redirects=True)
        return GenericResult[str](value=resp.text)
```

There are a few key elements here:

- **CallableModel**: Our class inherits from `CallableModel`, which means it is expected to execute given a context and return a result in a `@Flow.call` decorated `__call__` method
- **SiteContext**\*: Our class will take a specific flavor of context, defined above
- **GenericResult[str]**: Our class returns a `GenericResult`, which, as the name suggests, is just a generic container of a `value` (in this case, a `str`). We could've made this a custom result type as well, allowing greater type safety around our pipeline.

When we execute our `RestModel`, it will take in a `SiteContext` instance, make an `HTTP` request to the site, and return the string `html` result.

Before we move on to our other models, let's get this stage working with a CLI.

## CLI

`ccflow` provides some helper functions for linking together [hydra](https://hydra.cc) with our Callable models framework.

```python
import hydra
from ccflow.utils.hydra import cfg_run, cfg_explain_cli


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    cfg_run(cfg)

def explain():
    cfg_explain_cli(config_path="config", config_name="base", hydra_main=main)
```

The first function here takes a `hydra` configuration hierarchy, and tries to execute the top level `callable` attribute.
When we write the configuration files for our `RestModel`, we'll see how this links together.

The second function here launches a helpful UI to browse the `hydra` configuration hierarchy we've constructed.

> ![NOTE]
> These can be run directly from `ccflow`:
> `python -m ccflow.examples.etl` and `python -m ccflow.examples.etl.explain`

## Configuration

`hydra` is driven by `yaml` files, so let's write one of these for our ETL examples.

**ccflow/examples/etl/config/base.yaml**.

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
```

Our config here defines a key `extract` as a `PublisherModel`, which is itself a `CallableModel` that will execute the given `model` subkey and generate results using the corresponding `publisher` subkey.

This isn't strictly necessary for our example, as we could've just written the file ourselves from the `RestModel`, but it's a good example of leveraging the same callable models in multiple places (in this case, a `GenericFilePublisher`).

We can now execute this with the CLI we created above:

```bash
python -m ccflow.examples.etl +callable=extract +context=[]
```

This will load our configuration and call `/extract` (a `PublisherModel`).
`PublisherModel` will itself call `RestModel`, and then feed the results to `GenericFilePublisher`.
The end result should be the extracted result as a file called `raw.html` containing the `HTML` content (as configured on the \`GenericFilePublisher).

We can run the same CLI with a different context to extract a different website:

```bash
    python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"]
```

We can even leverage hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) to make tweaks at any layer of our configuration hierarchy:

```bash
# Change the GenericFilePublisher's output file name to lobsters, generating lobsters.html instead of raw.html
python -m ccflow.examples.etl +callable=extract +context=["http://lobste.rs"] ++extract.publisher.name=lobsters
```

Let's implement the remaining steps of our pipeline as their own models

## Remaining Models

### Transform

Our `LinksModel` will read an `HTML` file, extract all the links therein, and generate a `csv` string.

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

        # Use beautifulsoup to convert links into csv of name, url
        soup = BeautifulSoup(html, "html.parser")
        links = [{"name": a.text, "url": href} for a in soup.find_all("a", href=True) if (href := a["href"]).startswith("http")]

        io = StringIO()
        writer = DictWriter(io, fieldnames=["name", "url"])
        writer.writeheader()
        writer.writerows(links)
        output = io.getvalue()
        return GenericResult[str](value=output)
```

### Load

Our `DBModel` will read a `csv` file and load it into a `SQLite` database.

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

## Full Configuration

Let's register all of our models in the same configuration yaml:

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

Our `transform` step will also use a `PublisherModel` to generate a `csv` file.
It will uses's `hydra`'s native support for [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) interpolation to reference fields in other configuration blocks, e.g. `${extract.publisher.name}`.

Our `load` does not require a publisher as it will optionally produce a `.db` file directly.

> [!TIP]
> `hydra` provides a lot of utilities for breaking this config across multiple files and folders,
> providing defaults and overrides, etc. We will see an example of this below.

## Running

We've seen how to run our previous step with `+callable=extract`.
Our subsequent steps can be run similarly:

```bash

# Transform step:
python -m ccflow.examples.etl +callable=transform +context=[]

# Transform step with overrides, read input lobsters.html and generate lobsters.csv
python -m ccflow.examples.etl +callable=transform +context=[] ++transform.model.file=lobsters.html ++transform.publisher.name=lobsters

# Load step:
python -m ccflow.examples.etl +callable=load +context=[]

# Load step with overrides, read input lobsters.csv and generate in-memory sqlite db
python -m ccflow.examples.etl +callable=load +context=[] ++load.file=lobsters.csv ++load.db_file=":memory:"
```

Similarly, we can tweak configurations as well

## Visualizing

`hydra` is reading `yaml` files and loading up the `ccflow` models into a single registry, which we can visualize in `JSON` form using the other CLI we wrote above:

```bash
python -m ccflow.examples.etl.explain
```

This can be super helpful to see what fields are set to what values.

<img src="https://github.com/point72/ccflow/raw/main/docs/img/wiki/etl/explain1.png?raw=true" width="400">

We can also combine it with `hydra` overrides to ensure we're configuring everything correctly!

```bash
python -m ccflow.examples.etl.explain ++extract.publisher.name=test
```

<img src="https://github.com/point72/ccflow/raw/main/docs/img/wiki/etl/explain2.png?raw=true" width="400">

## Appendix - Multiple Files

In `hydra`, its conveniet to have things broken up across multiple files.
In our example above, we can do this as follows:

**etl/config/base.yaml**

```yaml
defaults:
    - extract: rest
    - transform: links
    - load: db
```

**etl/config/extract/rest.yaml**

```yaml
_target_: ccflow.PublisherModel
model:
  _target_: ccflow.examples.etl.models.RestModel
publisher:
  _target_: ccflow.publishers.GenericFilePublisher
  name: raw
  suffix: .html
field: value
```

**etl/config/transform/links.yaml**

```yaml
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

**etl/config/load/db.yaml**

```yaml
_target_: ccflow.examples.etl.models.DBModel
file: ${transform.publisher.name}${transform.publisher.suffix}
db_file: etl.db
table: links
```

In this organizational scheme, its easy to leveage `hydra` to provide a wide array of separate but interoperable functionality.
`hydra` has [much more documentation on this topic](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/).
