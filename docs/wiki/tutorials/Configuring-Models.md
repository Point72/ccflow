# Configuring Models

In [First Steps](First-Steps) you loaded a couple of configs into a registry. This tutorial slows down and builds the configuration framework up properly: typed schemas, hierarchy, serialization, the registry, and dependency injection. By the end you will be able to define your own configuration objects and wire them together, both interactively and from files.

Follow along in a Python session. Every block runs as shown. We will lean on [pydantic](https://docs.pydantic.dev/latest/) throughout — `ccflow.BaseModel` is a pydantic model with a few `ccflow` conveniences added.

Start with these imports:

```python
from ccflow import BaseModel, ModelRegistry
from datetime import date
from pathlib import Path
from pprint import pprint
```

## Your first config

Pydantic calls its classes "Models", so we do too — think of a model as a *configurable class*. Define one by subclassing `BaseModel`:

```python
class MyFileConfig(BaseModel):
    file: Path
    asof: date = date(2024, 1, 1)
    description: str = ""
```

That is a schema. Create an instance and watch pydantic conform the inputs to the declared types:

```python
c = MyFileConfig(file="sample.txt")
print(c)
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 1, 1), description='')
```

Configs are mutable by default, so you can adjust them after construction:

```python
c.description = "Sample description"
print(c)
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 1, 1), description='Sample description')
```

You can also build one from a dictionary — useful when configuration comes from a file:

```python
config = {"file": "sample.txt", "asof": "2024-02-02"}
print(MyFileConfig.model_validate(config))
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 2, 2), description='')
```

Notice the string path became a `PosixPath` and the string date became a `datetime.date`.

The point of a schema is that mistakes are caught early. A value that cannot be conformed raises immediately:

```python
try:
    c.asof = "foo"
except ValueError as e:
    print(e)
#> 1 validation error for MyFileConfig
#  asof
#    Input should be a valid date or datetime, input is too short ...
```

`ccflow.BaseModel` goes one step further than plain pydantic and rejects unknown or misnamed fields, so a typo cannot silently create a stray attribute:

```python
try:
    MyFileConfig(file="sample.txt", AsOf=date(2024, 1, 1))
except ValueError as e:
    print(e)
#> 1 validation error for MyFileConfig
#  AsOf
#    Extra inputs are not permitted ...
```

## Hierarchical configs

Configuration is naturally nested, and models compose. Define a config whose fields are themselves configs:

```python
class MyTransformConfig(BaseModel):
    x: MyFileConfig
    y: MyFileConfig = None
    param: float = 0.
```

You can build it by composition:

```python
x = MyFileConfig(file="source1.csv")
y = MyFileConfig(file="source2.csv")
transform = MyTransformConfig(x=x, y=y, param=1.)
print(transform.x.file)
#> source1.csv
```

Because this is ordinary object composition, editing the local `x` edits the object inside `transform`:

```python
x.description = "First Source"
print(transform.x.description)
#> First Source
```

Keep that behavior in mind — it becomes important once you register configs.

Pydantic will also coerce nested dictionaries into the declared types, so you can pass raw data all the way down:

```python
print(MyTransformConfig(x={"file": "source1.csv", "asof": "2024-02-02"}))
#> MyTransformConfig(x=MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 2, 2), description=''), y=None, param=0.0)
```

## Serializing configs

Pydantic serializes models richly:

```python
pprint(transform.model_dump(mode="json"))
```

`ccflow` adds one thing to the default: it records each model's `type_` alongside its fields. That lets you reconstruct the correct subclass from a serialized config:

```python
transform2 = BaseModel.model_validate(transform.model_dump(mode="json"))
assert isinstance(transform2, MyTransformConfig)
```

For [Hydra](Configuration-and-Hydra) compatibility, that type field is also aliased to `_target_`:

```python
pprint(transform.model_dump(mode="json", by_alias=True))
#> {'param': 1.0,
#   '_target_': '__main__.MyTransformConfig',
#   'x': {..., '_target_': '__main__.MyFileConfig'},
#   'y': {..., '_target_': '__main__.MyFileConfig'}}
```

## Inheritance and templatization

Because configs are classes, you can share fields through inheritance:

```python
class DateRangeMixin(BaseModel):
    start_date: date
    end_date: date

class RegionMixin(BaseModel):
    region: str

class MyConfig(DateRangeMixin, RegionMixin):
    parameter: int

print(MyConfig(parameter=4, region="US", start_date=date(2022, 1, 1), end_date=date(2023, 1, 1)))
#> MyConfig(region='US', start_date=datetime.date(2022, 1, 1), end_date=datetime.date(2023, 1, 1), parameter=4)
```

For generic/templated configs, see pydantic's docs on [Generic Models](https://docs.pydantic.dev/latest/concepts/models/#generic-models).

## Registering configurations

A `ModelRegistry` is a named collection of configs — a catalog. Create one and add configs to it:

```python
registry = ModelRegistry(name="My Raw Data")
registry.add("source1", MyFileConfig(file="source1.csv", description="First"))
registry.add("source2", MyFileConfig(file="source2.csv", description="Second"))

print(list(registry))
#> ['source1', 'source2']
```

The registry validates what goes in — only models are allowed:

```python
try:
    registry.add("bad_data", {"foo": 5, "bar": 6})
except TypeError as e:
    print(e)
#> model must be a child class of <class 'ccflow.base.BaseModel'>, not '<class 'dict'>'.
```

Look configs up with `__getitem__` or `get`, and note that adding an existing name needs `overwrite=True`:

```python
assert registry["source1"] is registry.get("source1", default=None)

try:
    registry.add("source1", registry["source1"])
except ValueError as e:
    print(e)
#> Cannot add 'source1' to 'My Raw Data' as it already exists!
```

Registries can contain other registries, and a single **root** registry is available as a singleton to tie everything together:

```python
root = ModelRegistry.root()
assert root is ModelRegistry.root()  # singleton
root.add("data", registry, overwrite=True)

print(list(root))
#> ['data', 'data/source1', 'data/source2']
```

From the root you can reach any config three equivalent ways:

```python
root["data"]["source1"]          # dictionary syntax
root["data/source1"]             # path syntax
root.get("data").get("source1")  # getter syntax
```

## Dependencies and dependency injection

Here is where the registry earns its keep. A config can depend on another *by its name in the root registry*, and `ccflow` resolves the name to the actual instance:

```python
root = ModelRegistry.root()
root.add("data", registry, overwrite=True)

new_config = MyTransformConfig(x="data/source1")
print(new_config.x.file)
#> source1.csv
```

Because the reference resolves to the shared instance, editing the source edits it here too:

```python
root["data"]["source1"].description = "Test"
assert new_config.x.description == "Test"
```

This is dependency injection: `new_config` did not need to know how `source1` was built, only its name. Register the composite and keep wiring by simple assignment:

```python
root.add("transform", new_config, overwrite=True)
root["transform"].y = "data/source2"
assert root["transform"].y.file == Path("source2.csv")
```

Configs can report where they are registered and what they depend on:

```python
print(new_config.get_registered_names())
#> ['/transform']
print(new_config.get_registry_dependencies())
#> [['/data/source1'], ['/data/source2']]
```

> This shared-instance behavior is a deliberate difference from Hydra's `${...}` interpolation, which *copies* configuration. See [Configuration and Hydra](Configuration-and-Hydra) for why that matters.

## Loading configs from data and files

You rarely add configs one call at a time. `load_config` takes a nested dictionary and loads the whole thing — interpreting nested dictionaries as sub-registries and resolving string references:

```python
all_configs = {
    "data": {
        "source1": {"_target_": "__main__.MyFileConfig", "file": "source1.csv", "description": "First"},
        "source2": {"_target_": "__main__.MyFileConfig", "file": "source2.csv", "description": "Second"},
    },
    "transform": {
        "_target_": "__main__.MyTransformConfig",
        "param": 1.0,
        "x": "data/source1",
        "y": "data/source2",
    },
}
root = ModelRegistry.root().clear()
root.load_config(all_configs)
print(list(root))
#> ['data', 'data/source1', 'data/source2', 'transform']
```

The `_target_` keys are what let `ccflow` (and Hydra) know which class to build. In fact, a config dict with `_target_` is exactly what `hydra.utils.instantiate` consumes:

```python
from hydra.utils import instantiate
config = {"_target_": "__main__.MyTransformConfig", "param": 1.0,
          "x": {"file": "source1.csv", "description": "First"}, "y": {"file": "source2.csv"}}
print(instantiate(config).param)
#> 1.0
```

To load configuration straight from Hydra files, use `load_config_from_path`, pointing `config_key` at the part of the file that holds the registry:

```python
import ccflow.examples
root = ModelRegistry.root().clear()
absolute_path = Path(ccflow.examples.__file__).parent / "config/conf.yaml"
root.load_config_from_path(path=absolute_path, config_key="registry")
```

Hydra's own [documentation](https://hydra.cc/docs/intro/) covers how to author those files; the [Composing an ETL Application](Composing-an-ETL-Application) tutorial builds a full file-based application step by step.

## What you learned

- `BaseModel` gives you typed, validated, self-documenting configuration.
- Configs compose into hierarchies and serialize round-trip via `_target_`.
- The `ModelRegistry` catalogs configs; the root registry links them by name.
- Name references are dependency injection over *shared instances*.
- `load_config` and `load_config_from_path` load whole trees from data or Hydra files.

## Next steps

- [Defining Workflows](Defining-Workflows) — make these configuration objects runnable.
- [Configure Complex Values](Configure-Complex-Values) — custom validation, Jinja/SQL templates, expressions, arrays, and arbitrary types in configs.
- [Bind Logic to Configs](Bind-Logic-to-Configs) — attach business logic to configuration classes.
