# Configure Complex Values

Configuration is rarely just strings and numbers. This guide collects recipes for putting richer values into `ccflow` configs — custom coercion, templates, expressions, arrays, arbitrary objects, and references to Python objects by path. Each section is independent; jump to the one you need.

All examples assume:

```python
from ccflow import BaseModel
from typing import List
```

## Coerce a non-dict value into a model

To let a config accept a convenient shorthand (e.g. a plain string) instead of a full dictionary, add a wrap validator that reshapes the input:

```python
from pydantic import model_validator

class NameConfig(BaseModel):
    first_name: str
    last_name: str

    @model_validator(mode="wrap")
    @classmethod
    def coerce_from_string(cls, data, handler):
        if isinstance(data, str):
            names = data.split(" ")
            if len(names) == 2:
                data = dict(first_name=names[0], last_name=names[1])
        return handler(data)

print(NameConfig.model_validate("John Doe"))
#> NameConfig(first_name='John', last_name='Doe')
```

Now anywhere `NameConfig` is expected, a string is accepted and coerced:

```python
class MyConfig(BaseModel):
    name: NameConfig

print(MyConfig(name="John Doe"))
#> MyConfig(name=NameConfig(first_name='John', last_name='Doe'))
```

For the full range of validators, see the pydantic docs on [validators](https://docs.pydantic.dev/latest/concepts/validators/).

## Configure a Jinja template (e.g. a SQL query)

To parameterize a template safely from configuration, use the `JinjaTemplate` extension type. It converts a string into a renderable [Jinja](https://jinja.palletsprojects.com/) template:

```python
from ccflow import JinjaTemplate

class MyTemplateConfig(BaseModel):
    greeting: JinjaTemplate
    user: str
    place: str

config = MyTemplateConfig(
    greeting="Hello {{user|upper}}, welcome to {{place}}!",
    user="friend",
    place="the tutorial",
)
print(config.greeting.template.render(config.dict()))
#> Hello FRIEND, welcome to the tutorial!
```

This is the safe way to build SQL queries from parameters rather than string-formatting them:

```python
from datetime import date

class MyQueryTemplate(BaseModel):
    query: JinjaTemplate
    columns: List[str]
    where: str = "Test"
    query_date: date = date(2024, 1, 1)
    filters: List[str] = []

query = """select {{columns|join(",\n\t")}}
from MyDatabase
where WhereCol = '{{where}}'
    and Date = '{{query_date}}'
    {% for filter in filters %}and {{filter}} {% endfor %}
"""

config = MyQueryTemplate(query=query, columns=["Col1", "Col2 as MyOtherCol", "SomeID"])
config.query_date = date(2022, 1, 1)
config.filters = ["SomeID IS NOT NULL"]
print(config.query.template.render(config.dict()))
```

## Configure a Polars expression

To define data transformations as configuration, use the `PolarsExpression` extension type, which parses a string into a [Polars](https://docs.pola.rs/) expression (requires `polars`):

```python
from ccflow.exttypes.polars import PolarsExpression
import polars as pl

class PolarsConfig(BaseModel):
    columns: List[PolarsExpression]
    where: PolarsExpression

config = {
    "columns": ["pl.col('Col1')", "pl.col('Col2').alias('MyOtherCol')"],
    "where": "pl.col('WhereCol')=='Test'",
}
config_model = PolarsConfig.model_validate(config)
assert isinstance(config_model.columns[0], pl.Expr)
assert isinstance(config_model.where, pl.Expr)
```

## Configure a NumPy array

To hold array data as a real `numpy.ndarray` (with the input conformed automatically), use `NDArray`:

```python
from ccflow import NDArray
import numpy as np

class MyNumpyConfig(BaseModel):
    my_array: NDArray[np.float64]
    my_list: List[float]

config = MyNumpyConfig(my_array=[1, 2, 3], my_list=[1, 2, 3])
assert isinstance(config.my_array, np.ndarray)
assert isinstance(config.my_list, list)
```

## Allow an arbitrary (non-built-in) type

To store objects that are not standard types, opt in with `arbitrary_types_allowed` and let Hydra construct the value. Pydantic still validates that the result is an instance of the declared type:

```python
from hydra.utils import instantiate

class MyCustomType:
    pass

class MyConfigWithCustomType(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    custom: MyCustomType

config = {
    "_target_": "__main__.MyConfigWithCustomType",
    "custom": {"_target_": "__main__.MyCustomType"},
}
config_model = instantiate(config)
assert isinstance(config_model.custom, MyCustomType)
```

## Reference a Python object by import path

To point at something already defined in code — a function, constant, enum, or class — use `PyObjectPath`:

```python
from ccflow import PyObjectPath

class MyConfigWithPaths(BaseModel):
    builtin_func: PyObjectPath = PyObjectPath("builtins.len")
    separator: PyObjectPath = PyObjectPath("ccflow.REGISTRY_SEPARATOR")

config = MyConfigWithPaths()
assert config.builtin_func.object([1, 2, 3]) == 3
print(config.separator.object)
#> /
```

## Reuse registry aliases and Python defaults

`ccflow.compose` offers small helpers for interacting with the registry and Python-backed defaults from configuration:

- `model_alias(model_name)` — resolve a model instance by string name from the active registry.
- `update_from_template(base, update=None, target_class=None)` — build a new instance or dict by shallow-copying `base` and applying updates. `base` may be a registry alias (pass the alias string, or use `_target_: ccflow.compose.model_alias`) or a dict; the shallow copy preserves nested `BaseModel` identity.
- `from_python(py_object_path, indexer=None)` — resolve any Python object by import path, optionally indexing into it with a list of keys.

## See also

- [Configuring Models](Configuring-Models) — the tutorial these recipes build on.
- [Bind Logic to Configs](Bind-Logic-to-Configs) — attach behavior to configuration classes.
