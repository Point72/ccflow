# Bind Logic to Configs

A configuration object becomes useful when code lives close to the configuration it needs. This guide shows how to attach business logic to configuration classes so that a config is also a runnable unit — a reader, a publisher, or a stage in a data pipeline.

The pattern is to keep logic *close* to the configuration it uses, rather than reaching into the registry for values throughout your codebase (which recreates the problems of global variables — see [Core Concepts](Core-Concepts)). For full workflow steps with contexts, results, and evaluators, use a [`CallableModel`](Defining-Workflows); the plain-method pattern here is handy for simpler cases.

## Add a `__call__` method

To make a config runnable, give it a `__call__` method. Following the [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle), keep one purpose per class:

```python
import pandas as pd
from pathlib import Path
from ccflow import BaseModel

class MyParquetConfig(BaseModel):
    file: Path
    description: str = ""

    def __call__(self):
        """Read the file as a pandas data frame."""
        return pd.read_parquet(self.file)

df = MyParquetConfig(file="data.parquet")()
```

There are no restrictions on what lives inside — validation conforms the *fields*, but the logic is ordinary Python. Branch on configuration as needed:

```python
import pyarrow.parquet as pq
from typing import Literal

class MyParquetConfig(BaseModel):
    file: Path
    library: Literal["pandas", "pyarrow"] = "pandas"

    def read_pandas(self):
        return pd.read_parquet(self.file)

    def read_arrow(self):
        return pq.read_table(self.file)

    def __call__(self):
        return self.read_pandas() if self.library == "pandas" else self.read_arrow()
```

## Write a custom publisher

To send data somewhere (files, an object store, email, a report), implement the publisher interface `ccflow.publishers.BasePublisher`. A common interface means publishers can be swapped purely through configuration. Here is one that renders a list of strings as HTML in a notebook, using a [Jinja template](Configure-Complex-Values#configure-a-jinja-template-eg-a-sql-query):

```python
from typing import List
from IPython.display import display, HTML
from ccflow import JinjaTemplate, BasePublisher

class MyPublisher(BasePublisher):
    data: List[str] = None
    html_template: JinjaTemplate

    def __call__(self):
        display(HTML(self.get_name()))
        display(HTML(self.html_template.template.render(data="<BR>".join(self.data))))

p = MyPublisher(
    name="<b>My {{desc}} publisher:</b>",
    html_template="""<p style="color:blue;">{{data}}</p>""",
)
p.name_params = dict(desc="test")
p.data = ["Blue text.", "More blue text."]
p()
```

Because `data` is a typed field, invalid data is rejected before publishing:

```python
try:
    p.data = {}
except ValueError as v:
    print(v)
```

`ccflow` ships publishers for common cases — see [Built-in Models](Built-in-Models#publishers).

## Build a data pipeline

To wire several stages together, give each stage a config with a `__call__`, and connect them by registry reference. The models below read, augment, and summarize data with [Polars](https://docs.pola.rs/) (requires `polars`):

```python
from typing import List, Optional
from pathlib import Path
import polars as pl
from ccflow import BaseModel, ModelRegistry
from ccflow.exttypes.polars import PolarsExpression


class PolarsCallable(BaseModel):
    """Base class for models that return a polars LazyFrame."""
    def __call__(self) -> pl.LazyFrame: ...

class ParquetDataReader(PolarsCallable):
    file: Path
    n_rows: Optional[int] = None

    def __call__(self):
        return pl.scan_parquet(self.file, n_rows=self.n_rows)

class FeatureAdder(PolarsCallable):
    data_input: PolarsCallable
    group_col: str
    average_cols: List[str]

    def __call__(self):
        df = self.data_input()
        agg = [pl.col(c).mean().alias(f"{c}_mean") for c in self.average_cols]
        avg = df.group_by(pl.col(self.group_col)).agg(agg)
        joined = df.join(avg, on=self.group_col)
        resid = [(pl.col(c) - pl.col(f"{c}_mean")).abs().alias(f"{c}_resid") for c in self.average_cols]
        return joined.with_columns(resid)

class TopKFinder(PolarsCallable):
    data_input: PolarsCallable
    by: str
    k: int = 1

    def __call__(self):
        return self.data_input().top_k(k=self.k, by=self.by)

class ColumnSelector(PolarsCallable):
    data_input: PolarsCallable
    exprs: List[PolarsExpression]

    def __call__(self):
        return self.data_input().select(self.exprs)
```

Register configurations, letting several downstream models share one upstream stage:

```python
root = ModelRegistry.root().clear()
root.add("Raw Data", ParquetDataReader(file="example.parquet"))
root.add("Augmented Data", FeatureAdder(data_input="Raw Data", group_col="State", average_cols=["Sales", "Profit"]))
root.add("TopK Profit residuals", TopKFinder(data_input="Augmented Data", by="Profit_resid", k=3))
root.add("Mean residuals", ColumnSelector(data_input="Augmented Data", exprs=[pl.col("Sales_resid").mean(), pl.col("Profit_resid").mean()]))
```

Now load any model from the registry and call it — you get back a `polars.LazyFrame` (you built the Polars graph via `ccflow`), which you materialize with `collect()`:

```python
print(root["TopK Profit residuals"]().collect())
```

Because both downstream models reference the same `"Augmented Data"` instance, changing it once (for example, `root["Augmented Data"].group_col = "Region"`) flows through to every result that depends on it.

## See also

- [Defining Workflows](Defining-Workflows) — the full `CallableModel` pattern with contexts, results, and evaluators.
- [Configure Complex Values](Configure-Complex-Values) — richer field types for your configs.
- [Built-in Models](Built-in-Models) — publishers and models that ship with `ccflow`.
