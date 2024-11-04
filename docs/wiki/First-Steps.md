- [Quick Start](#quick-start)
- [Configuration with `ccflow`](#configuration-with-ccflow)
  - [Basic Config with BaseModel](#basic-config-with-basemodel)
  - [Hierarchical Configs](#hierarchical-configs)
  - [Serializing Configs](#serializing-configs)
  - [Config Inheritance and Templatization](#config-inheritance-and-templatization)
- [Registering Configurations](#registering-configurations)
  - [The Model Registry](#the-model-registry)
  - [Dependencies and Dependency Injection](#dependencies-and-dependency-injection)
  - [Integration with `hydra`](#integration-with-hydra)
- [Advanced Config Examples](#advanced-config-examples)
  - [Custom Validation and Coercion](#custom-validation-and-coercion)
  - [Jinja Templates and SQL Queries](#jinja-templates-and-sql-queries)
  - [Polars Expressions](#polars-expressions)
  - [Numpy Arrays](#numpy-arrays)
  - [Arbitrary Types](#arbitrary-types)
  - [Loading Objects by Path](#loading-objects-by-path)
- [Using Configuration](#using-configuration)
  - [Adding a `__call__` method](#adding-a-__call__-method)
  - [Example: Reading a file](#example-reading-a-file)
  - [Example: Custom publisher](#example-custom-publisher)
  - [Example: Simple data pipeline in polars](#example-simple-data-pipeline-in-polars)
  - [Example: Gaussian process regression in sklearn](#example-gaussian-process-regression-in-sklearn)

## Quick Start

This short example shows some of the key features of the configuration framework in `ccflow`:

```python
from ccflow import BaseModel, ModelRegistry

# Define config objects
class MyFileConfig(BaseModel):
    file: str
    description: str = ""

class MyTransformConfig(BaseModel):
    x: MyFileConfig
    y: MyFileConfig = None
    param: float = 0.


# Define json configs
configs = {
    "data": {
        "source1": {
            "_target_": "__main__.MyFileConfig",
            "file": "source1.csv",
            "description": "First",
        },
        "source2": {
            "_target_": "__main__.MyFileConfig",
            "file": "source2.csv",
            "description": "Second",
        },
        "source3": {
            "_target_": "__main__.MyFileConfig",
            "file": "source3.csv",
            "description": "Third",
        },
    },
    "transform": {
        "_target_": "__main__.MyTransformConfig",
        "x": "data/source1",
        "y": "data/source2",
    },
}

# Register configs
root = ModelRegistry.root().clear()
root.load_config(configs)

# List the keys in the registry
print(list(root))
#> ['data', 'data/source1', 'data/source2', 'data/source3', 'transform']

# Access configs from the registry
print(root["transform"])
#> MyTransformConfig(
#    x=MyFileConfig(file='source1.csv', description='First'),
#    y=MyFileConfig(file='source2.csv', description='Second'),
#    param=0)

# Assign config objects by name
root["transform"].x = "data/source3"
print(root["transform"].x)
#> MyFileConfig(file='source3.csv', description='Third')

# Propagate low-level changes to the top
root["data/source3"].file = "source3_amended.csv"
# See that it changes in the "transform" definition
print(root["transform"].x.file)
#> source3_amended.csv
```

## Configuration with `ccflow`

Let's dive deeper into some of the ideas above, focusing on how we leverage features of `pydantic` for the purposes of configuration.

```python
from ccflow import BaseModel, ModelRegistry
from datetime import date
from pathlib import Path
from pprint import pprint
```

### Basic Config with BaseModel

Let's get started with some very simple examples. Pydantic calls their classes "Models", and so we use the same terminology; think of a "Model" as a "Configurable" class.

The `BaseModel` is our base class for all configuration. We have subclassed Pydantic's `BaseModel` to change some of the default configuration options, and to make the objects play nicer with Hydra and the rest of our framework. However, as they are still Pydantic models, everything you can do with Pydantic's [Models](https://pydantic-docs.helpmanual.io/usage/models/) can be done with these.

We begin with a dummy example, but one which illustrates how new config schemas and values can be easily defined and manipulated on-the-fly.

```python
class MyFileConfig(BaseModel):
    file: Path
    asof: date = date(2024,1,1)    
    description: str = ""   
```

This is not very exciting yet, basically just the definition of a config schema, but it already illustrates how Pydantic will conform input data to the right types (i.e. Path, date and str in this case).

```python
c = MyFileConfig(file="sample.txt")
print(c)
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 1, 1), description='')
```

Note that the config object is mutable by default (though they can be frozen too). This makes it easy to change configs, especially once they get nested

```python
c.description = "Sample description"
print(c)
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 1, 1), description='Sample description')
```

Pydantic allows for objects to be created directly from dictionaries

```python
config = {"file": "sample.txt", "asof": "2024-02-02"}
print(MyFileConfig.model_validate(config))
#> MyFileConfig(file=PosixPath('sample.txt'), asof=datetime.date(2024, 2, 2), description='')
```

Note that Pydantic automatically converted the string path to a `PosixPath` and the string date to a `datetime.date`.

Pydantic also provides a [JSON schema ](https://json-schema.org/) in standardized format that can users understand the parameters on the config object, though this only works on models that only contain json-compatible types (even though Pydantic supports arbitrary types as we will see later). For example:

```python
pprint(MyFileConfig.schema())
#> {'additionalProperties': False,
#   'properties': {'asof': {'default': '2024-01-01',
#                           'format': 'date',
#                           'title': 'Asof',
#                           'type': 'string'},
#                  'description': {'default': '',
#                                  'title': 'Description',
#                                  'type': 'string'},
#                  'file': {'format': 'path', 'title': 'File', 'type': 'string'}},
#   'required': ['file'],
#   'title': 'MyFileConfig',
#   'type': 'object'}
```

Pydantic's type validation will catch cases that are incompatible with our schema definition. In fact, Pydantic can be used to place even greater constraints on the values themselves (i.e. asof date must not be in the future)

```python
try:
    c.asof = "foo"
except ValueError as e:
    print(e)
#> 1 validation error for MyFileConfig
#  asof
#    Input should be a valid date or datetime, input is too short [type=date_from_datetime_parsing, input_value='foo', input_type=str]
#      For further information visit https://errors.pydantic.dev/2.9/v/date_from_datetime_parsing
```

Furthermore, in `ccflow.BaseModel`, we have enabled the option by default to raise exceptions when field names are mis-specified (or extra fields are provided) to catch potential configuration mistakes.

```python
try:
    MyFileConfig(file="sample.txt", AsOf=date(2024,1,1))
except ValueError as e:
    print(e)

#> 1 validation error for MyFileConfig
#  AsOf
#    Extra inputs are not permitted [type=extra_forbidden, input_value=datetime.date(2024, 1, 1), input_type=date]
#      For further information visit https://errors.pydantic.dev/2.9/v/extra_forbidden
```

### Hierarchical Configs

Hierarchical configs are also easy to work with. Below, we create a new config which consists of two file configs, and we can easily modify nested attributes using standard python syntax.

```python
class MyTransformConfig(BaseModel):
    x: MyFileConfig
    y: MyFileConfig = None
    param: float = 0.
```

The traditional construction of a nested model by object composition would look something
like this

```python
x = MyFileConfig(file="source1.csv")
y = MyFileConfig(file="source2.csv")
transform = MyTransformConfig(x=x, y=y, param=1.)
print(transform)
#> MyTransformConfig(x=MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 1, 1), description=''), y=MyFileConfig(file=PosixPath('source2.csv'), asof=datetime.date(2024, 1, 1), description=''), param=1.0)
```

Note that because of object composition, changing an attribute on the local variable `x` will change it on the object in `transform`:

```python
x.description="First Source"
print(transform.x.description)
#> 'First Source'
```

This becomes important later when we register the config.

Pydantic also provides the ability to coerce dictionaries recursively into structured types, so long as the types have been declared on the schema. For example, it will automatically create the `MyFileConfig` instance if we just pass a dictionary to `MyTransformConfig`:

```python
print(MyTransformConfig(x={"file": "source1.csv", "asof": "2024-02-02"}))
#> MyTransformConfig(x=MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 2, 2), description=''), y=None, param=0.0)
```

### Serializing Configs

Pydantic provides a rich set of tools for serializing models, for example

```python
pprint(transform.model_dump(mode="json"))
#> {'param': 1.0,
#   'type_': '__main__.MyTransformConfig',
#   'x': {'asof': '2024-01-01',
#         'description': 'First Source',
#         'file': 'source1.csv',
#         'type_': '__main__.MyFileConfig'},
#   'y': {'asof': '2024-01-01',
#         'description': '',
#         'file': 'source2.csv',
#         'type_': '__main__.MyFileConfig'}}
```

Note that the custom `BaseModel` in `ccflow` serializes the `type_` of the model in addition to the fields (which does not happen by default in Pydantic). This allows for generic reconstruction of the right config class (or subclass) from the serialized version:

```python
transform2 = BaseModel.model_validate(transform.model_dump(mode="json"))
assert isinstance(transform2, MyTransformConfig)
```

For compatibility with `hydra` (further on in this tutorial), we also alias the `type_` field to `_target_`, i.e.

```python
pprint(transform.model_dump(mode="json", by_alias=True))
#> {'param': 1.0,
#   '_target_': '__main__.MyTransformConfig',
#   'x': {'asof': '2024-01-01',
#         'description': 'First Source',
#         'file': 'source1.csv',
#         '_target_': '__main__.MyFileConfig'},
#   'y': {'asof': '2024-01-01',
#         'description': '',
#         'file': 'source2.csv',
#         '_target_': '__main__.MyFileConfig'}}
```

### Config Inheritance and Templatization

In addition to the composition of configs as illustrated above, there may be cases when inheritance and templatization is required. Pydantic supports both of these out of the box.

Below we provide an example of multiple inheritance of model objects, which further illustrates the power of having schema classes for configuration over raw dictionaries.

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

For examples of templatization, refer to the section of the Pydantic documentation on [Generic Models](https://pydantic-docs.helpmanual.io/usage/models/#generic-models).

## Registering Configurations

Next we explain the power of being able to register configurations.

```python
from ccflow import ModelRegistry
```

### The Model Registry

`ccflow` provides a `ModelRegistry` class which represents a collection of `ccflow.BaseModel` instances (configs). Later we will see how config files can be mapped to a registry, but for now we illustrate how it can be used interactively.

```python
registry = ModelRegistry(name="My Raw Data")
registry.add(
    "source1",
    MyFileConfig(
        file="source1.csv", description="First"
    ),
)
registry.add(
    "source2",
    MyFileConfig(
        file="source2.csv", description="Second"
    ),
)
print(registry)
#> ModelRegistry(name='My Raw Data')

print(list(registry))
#> ['source1', 'source2']

pprint(registry.models)
#> mappingproxy({'source1': MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 1, 1), description='First'),
#                'source2': MyFileConfig(file=PosixPath('source2.csv'), asof=datetime.date(2024, 1, 1), description='Second')})
```

At this point, a `ModelRegistry` just looks and behaves like a dictionary. However, a bit of extra functionality has been built in, such as validation of items that go into the registry to make sure they are config classes, i.e.

```python
try:
    registry.add("bad_data", {"foo": 5, "bar": 6})
except TypeError as e:
    print(e)
#> model must be a child class of <class 'ccflow.base.BaseModel'>, not '<class 'dict'>'.
```

This may seem like an unnecessary restriction, but enforcing that all registry elements are using BaseModel means that we can deliver more powerful functionality over time by extending the BaseModel implementation.

From any registry, you can access configs directly using `__getitem__` or `get` (which allows for a default value)

```python
assert "source1" in registry
assert registry["source1"] is registry.get("source1", default=None)
```

Note that by default, if you try to add a configuration that already exists, it will raise an error, unless you pass `overwrite=True`:

```python
try:
    registry.add("source1", registry["source1"])
except ValueError as e:
    print(e)
#> Cannot add 'source1' to 'My Raw Data' as it already exists!

assert registry.add("source1", registry["source1"], overwrite=True)
```

As the amount of configuration grows, there is a desire to organize these objects in a hierarchy, and so, the registry class can contain other registries (since they are configuration objects themselves).

Furthermore, instead of passing various registries around in the code, it is sometimes helpful to have a single registry that is a singleton at the "root" of all these registries. `ccflow` provides this:

```python
root = ModelRegistry.root()
assert root is ModelRegistry.root()  # It is a singleton.
root.add("data", registry, overwrite=True)
print(root)
#> RootModelRegistry()
print(root.models)
#> {'data': ModelRegistry(name='My Raw Data')}
```

From the root registry, there are three different ways to get underlying configs, using dictionary syntax, file path or getter syntax:

```python
root["data"]["source1"]  # Dictionary syntax
root["data/source1"]  # File path syntax
root.get("data").get("source1") # Getter syntax
```

Note that the same object can be registered under multiple different names. If one thinks of a registry as a "catalog" of data or configurations, it makes sense that the same item could be indexed in different ways.

The registry is a `collections.abc.Mapping` over all the registered models and on any models that belong to sub-registries. In other words, it will return all possible combined keys, i.e.:

```python
print(list(root))
#> ['data', 'data/source1', 'data/source2']
print(len(root))
#> 3
```

If you only want to access the top level of the registry, use `registry.models`, which in this case returns a single item (the "data" sub-registry):

```python
print(list(root.models))
#> ['data']
```

To clear all the entries from a registry, one can execute:

```python
root.clear()
print(root.models)
#> {}
```

### Dependencies and Dependency Injection

We wish to allow configuration objects to depend on each other in such a way that the linkage is dynamic. In the `ccflow` framework, this is done through object composition (and mutability of configs). However, to make things easier, we allow for configs to be referenced by their name in the **root** registry! For example, we can create a new config object like so:

```python
root = ModelRegistry.root()
root.add("data", registry, overwrite=True)

new_config = MyTransformConfig(x="data/source1")
print(new_config)
#> MyTransformConfig(x=MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 1, 1), description='First'), y=None, param=0.0)
```

Just like before, if we now change the values on this config model in the registry, they will change in this newly created object as well:

```python
root["data"]["source1"].description = "Test"
assert new_config.x.description == "Test"
```

Note, however, that if we replace the config object in the registry with an entirely new config object, the dependency will still reference the old object. This is why you need to pass `overwrite=True` when adding an object to the registry with a name that already exists.

With these dependencies set up, we can now register this object as well

```python
root.add("transform", new_config, overwrite=True)
print(list(root))
#> ['data', 'data/source1', 'data/source2', 'transform']
```

Even once registered, linkages between objects can be added through simple assignment

```python
root["transform"].y = "/data/source2"
assert root["transform"].y.file == Path("source2.csv")
```

The config objects in `ccflow` can tell you where they are registered (which may be in more than one place), either as a tuple of (registry, name), or as a path by which the object could be accessed. i.e. For the composite configuration `"data config"`, registered in the root registry:

```python
print(new_config.get_registrations())
#> [(RootModelRegistry(), 'transform')]
print(new_config.get_registered_names())
#> ['/transform']
```

Below is an example of that for the file config, as accessed from the transform config:

```python
print(root["transform"].y.get_registrations())
#> [(ModelRegistry(name='My Raw Data'), 'source2')]
print(root["transform"].y.get_registered_names())
#> ['/data/source2']
```

The config objects can also tell you their dependencies on other registered config objects. It will look recursively through the the entire nested configuration structure to find other models that are in the registry (even if some intermediate levels are not registered):

```python
print(new_config.get_registry_dependencies())
#> [['/data/source1'], ['/data/source2']]
```

When you clear all the models out of the root registry, the registrations and dependencies on previously registered objects are also reset:

```python
root.clear()
print(new_config.get_registrations())
#> []
print(new_config.get_registry_dependencies())
#> []
```

### Integration with `hydra`

As shown above, Pydantic provides tools to serialize and deserialize model configurations. The `BaseModel` class in `ccflow` is also compatible with [Hydra](https://hydra.cc/docs/intro/). While hydra integration is optional, it provides powerful tools for working with configuration files and the command line that we can leverage directly.

```python
from hydra.utils import instantiate
config = {
    '_target_': '__main__.MyTransformConfig',
    'param': 1.0,
    'x': {'description': 'First',
          'file': 'source1.csv'},
    'y': {'file': 'source2.csv'}
}
print(instantiate(config))
#> MyTransformConfig(x=MyFileConfig(file=PosixPath('source1.csv'), asof=datetime.date(2024, 1, 1), description='First'), y=MyFileConfig(file=PosixPath('source2.csv'), asof=datetime.date(2024, 1, 1), description=''), param=1.0)
```

Note that this is the same as calling `BaseModel.model_validate(config)`.

Furthermore, the `ModelRegistry` class has a convenience method that will take a dictionary of configs, and add them all to the registry. It is even clever enough to interpret nested dictionaries (with no `_target_`) as nested registries, and to resolve dependencies specified as strings (by looking up the string path in the **root** registry). Below is a complete example:

```python
all_configs = {
    "data": {
        "source1": {
            "_target_": "__main__.MyFileConfig",
            "file": "source1.csv",
            "description": "First",
        },
        "source2": {
            "_target_": "__main__.MyFileConfig",
            "file": "source2.csv",
            "description": "Second",
        },
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

In the above, note that the "x" and "y" variables are passed as strings referencing other objects in the registry. When using hydra, something similar could be accomplished by using [variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) in OmegaConf, but there is one key difference. In hydra/omegaconf, variable interpolation makes a copy of the yaml config, so `transform.x` would point to an object that is configured the same way as `data/source1`, but it would not be pointing to the same *instance*. Thus, if changes were made to `data/source1`, they would not be reflected in `transform.x`.

With the above behavior in place, especially the ability to load dictionaries of configs into the root `ModelRegistry`, the next step is to be able to load the configurations from files. Fortunately, this piece has already been solved by hydra. In particular, it allows for configurations to be split across multiple sub-directories and files, re-used in multiple places, and recombined as needed. It also provides advanced command line tools to configure your application.

While hydra is primarily concerned with loading file-based configurations into command-line applications, their [Compose API](https://hydra.cc/docs/advanced/compose_api/) provides a way to load the configs interactively in a notebook, as a special dictionary type. However, to save people the work of first loading config files into config dictionaries, and then adding those into the registry, we've provided a function to do this directly, shown below. Note that these config files are loading models which are defined in `ccflow` and `ccflow.examples`.

```python
import ccflow.examples
root = ModelRegistry.root().clear()
absolute_path = Path(ccflow.examples.__file__).parent / "config/conf.yaml"
root.load_config_from_path(path=absolute_path, config_key="registry")
```

The "config_key" argument in the function call above points to the subset of the hydra configs to load into the registry, as there may be parts of the config which you do not which to load into the registry (such as configuration of hydra itself, or potentially other global configuration variables that are only meant to exist in the file layer).

It is out-of-scope for this tutorial to cover the various ways in which hydra can be used to generate configs, but please check out their [documentation](https://hydra.cc/docs/intro/) for more information.

## Advanced Config Examples

In this section we cover some more advanced config examples. It can be skipped on the first read if desired.

### Custom Validation and Coercion

Pydantic provides a lot of functionality for custom validation. We don't cover all of it in the tutorial, but encourage people to read the section of the Pydantic docs on [validators](https://pydantic-docs.helpmanual.io/usage/validators/).

One key feature is the ability to coerce non-dictionary objects into Pydantic models, which is very powerful from a configuration perspective. We use this trick in several places in `ccflow` to improve usability, and mention it here in case others find it useful. We illustrate this below:

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

NameConfig.model_validate("John Doe")
#> NameConfig(first_name='John', last_name='Doe')
```

Now, when a string name is passed to a config where `NameConfig` is expected, Pydantic will coerce the data to the correct type.

```python
class MyConfig(BaseModel):
    name: NameConfig

print(MyConfig(name="John Doe"))
#> MyConfig(name=NameConfig(first_name='John', last_name='Doe'))
```

### Jinja Templates and SQL Queries

Another aspect of configuration that we haven't touched on so far is the need to specify a template document, and then to fill in the data. This occurs commonly when building database queries from parameters or when plugging data into an email template or HTML report. One common solution is to leverage python's string formatting capabilities, but this provides a minimal amount of validation to guard against accidents in the template definition, or malicious users (i.e. SQL injection attacks). The standard solution to this problem is to leverage [Jinja templates](https://jinja.palletsprojects.com/), which are extremely powerful (as they enable some amount of scripting inside the template itself).

`ccflow` has defined a Pydantic extension type that corresponds to Jinja templates, so that they can be used in configuration objects. We illustrate this below (note that unused template arguments are ignored).

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
print(config.greeting)
#> Hello {{user|upper}}, welcome to {{place}}!
print(config.greeting.template.render(config.dict()))
#> Hello FRIEND, welcome to the tutorial!
config.place = "a different configuration"
print(config.greeting.template.render(config.dict()))
#> Hello FRIEND, welcome to a different configuration!
config.greeting = "Dear {{user}}, templates can also be easily switched"
print(config.greeting.template.render(config.dict()))
#> Dear friend, templates can also be easily switched
```

While the above example may be useful for a templatized email or report, we provide a more complex and realistic example that illustrates how to easily configure a SQL query:

```python
from datetime import date
from typing import List

class MyQueryTemplate(BaseModel):
    query: JinjaTemplate
    columns: List[str]
    where: str = "Test"
    query_date: date = date(2024, 1, 1)
    filters: List[str] = []

query = """select  {{columns|join(",\n\t")}}
from MyDatabase
where WhereCol = '{{where}}'
    and Date = '{{query_date}}'
    {% for filter in filters %}and {{filter}} {% endfor %}
"""

config = MyQueryTemplate(
    query=query,
    columns=["Col1", "Col2 as MyOthercol", "SomeID"],
)
print(config.query.template.render(config.dict()))
#> select  Col1,
#  	   Col2 as MyOtherCol,
# 	   SomeID
#  from MyDatabase
#  where WhereCol = 'Test'
#      and Date = '2024-01-01'
```

Now it's easy to reconfigure the query by, i.e. changing the date and adding filters:

```python
config.query_date = date(2022, 1, 1)
config.filters = ["SomeID IS NOT NULL", "Col1 in 'blerg'"]
print(config.query.template.render(config.dict()))
#> select  Col1,
#      Col2 as MyOtherCol,
#      SomeID
#  from MyDatabase
#  where WhereCol = 'Test'
#      and Date = '2022-01-01'
#      and SomeID IS NOT NULL and Col1 in 'blerg' 
```

### Polars Expressions

Instead of configuring SQL queries as shown above, libraries like [Polars](https://docs.pola.rs/) have popularized the use of expressions for defining data transformation. `ccflow` also provides an extension type for polars expressions, so that they can be easily converted from strings (i.e. from a config file):

```python
from ccflow.exttypes.polars import PolarsExpression
import polars as pl

class PolarsConfig(BaseModel):
    columns: List[PolarsExpression]
    where: PolarsExpression

config = {"columns": ["pl.col('Col1')", "pl.col('Col2').alias('MyOtherCol')", "pl.col('SomeID')"], "where": "pl.col('WhereCol')=='Test'" }
config_model = PolarsConfig.model_validate(config)
assert isinstance(config_model.columns[0], pl.Expr)
assert isinstance(config_model.where, pl.Expr)
```

### Numpy Arrays

Sometimes it is more convenient to work with numpy array objects instead of python lists. `ccflow` provides tools to do this easily, as shown in the following example (which conforms the input data to the declared types automatically).

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

### Arbitrary Types

Often the need arises for configuration to create objects that are not built-in types (str, int, float, etc). Pydantic supports a number of additional types (see the [documentation](https://pydantic-docs.helpmanual.io/usage/types/) for a full list), but can also handle completely arbitrary types. We can also use hydra to instantiate these arbitrary types from the configs as well, and Pydantic will validate that the created object is an instance of the desired type. Furthermore, we specify some extension types  in `ccflow` that have additional validation and functionality (where `JinjaTemplate`, `PolarsExpression`, and `NDArray` described above are examples).

First, to illustrate how custom types work, we define our own custom object type, and then a configuration object (i.e. `BaseModel`) that contains that type. To prevent accidental inclusion of custom types, Pydantic must be explicitly told to include them using the `model_config` option.

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

### Loading Objects by Path

Sometimes one needs to refer to an object that is already defined in the codebase by its import name. For example, in some cases, it can be easier easier to construct a config object in python, such as when lots of custom classes, enum types, or lambda functions are involved. For this case, `ccflow` also has a solution that is able to refer to any python object by path, as illustrated below:

```python
from ccflow import PyObjectPath

class MyConfigWithPaths(BaseModel):
    builtin_func: PyObjectPath = PyObjectPath("builtins.len")
    separator: PyObjectPath = PyObjectPath("ccflow.REGISTRY_SEPARATOR")

config = MyConfigWithPaths()
assert config.builtin_func.object([1, 2, 3]) == 3
print("Separator: ", config.separator.object)
#> Separator:  /
```

## Using Configuration

With the ability to define arbitrarily complex configuration structures following the examples above, now arises the question of how best to use all that config information.

One option is simply to access the relevant config information from the root registry whenever it is needed in the code. This method is **fragile** and **strongly discouraged**. It is equivalent to using global variables throughout the code, which has the following problems

- Causes very tight coupling between parts of the code, and adds dependencies everywhere on, i.e. the naming and structure of the configuration, which may need to change frequently to stay organized
- Since every configuration option is available to every piece of code, it makes it difficult to reason about which code depends on which parts of configuration. One of the original design goals was to make it very easy to remove un-needed configuration classes (i.e. for unused/unsuccessful experimentation)
- Because configurations are mutable (by design), using them everywhere makes it harder to reason about the state of the system (nothing is "pure" or "const correct" any more). It would be better to separate the parts of the code that depend on configuration (and are subject to change) from those that do not (i.e. analytics, pure/idempotent functions, etc)

### Adding a `__call__` method

One solution to this problem is to write code that lives as "close" to the relevant configuration as possible, such that the scope is limited to those configuration parameters it needs and the dependencies are more clear. In this way, the registry is not used at all for run-time access, except perhaps as an initial entry-point into the logic. Furthermore, this code can serve as the bridge between the configuration graph (i.e. `ccflow`) and any other computational graphs which will be doing the heavy lifting. For example, the configurations can be used to define tensor graphs (tensorflow, pytorch), event processing graphs (csp, kafka streams), task graphs (ray, dask), etc, which are then executed separately.

One easy way to bind together user-defined business logic with the configuration classes is simply to add a method on the class. Following the [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle), each of these classes should ideally have only one purpose, and hence following the python convention, we can name the primary method that commplishes this `__call__` so that the BaseModel becomes callable. The following examples illustrate this idea.

### Example: Reading a file

```python
import pandas as pd

class MyParquetConfig(BaseModel):
    file: Path
    description: str = ""

    def __call__(self):
        """Read the file as a pandas data frame"""        
        return pd.read_parquet(self.file)
```

```python
config = MyParquetConfig(
    file=Path(ccflow.examples.__file__).parent / "example.parquet", description="Example Data"
)
df = config()
df.head()
```

Thus, using all the machinery in the previous section, we can define configs that are either pure data containers, or that correspond to some piece of arbitrary user-defined functionality, i.e. a "step" in a workflow. While we do leverage Pydantic for conforming data, there are **no restrictions** on the kind of code that could live inside the `__call__` method (or strictly speaking, how many methods there are or what those methods are called). Here is a slightly more complex example for illustration:

```python
import pyarrow.parquet as pq
from typing import Literal

class MyParquetConfig(BaseModel):
    """This is an example of a config class."""

    file: Path
    description: str = ""
    library: Literal["pandas", "pyarrow"] = "pandas"

    def read_pandas(self):
        return pd.read_parquet(self.file)

    def read_arrow(self):
        return pq.read_table(self.file)

    def __call__(self):
        """Read the file as a pandas data frame or arrow table"""
        if self.library == "pandas":
            return self.read_pandas()
        else:
            return self.read_arrow()
```

```python
config = MyParquetConfig(file="ccflow/examples/example.parquet", library="pandas")
df = config()
print(type(df))
#> <class 'pandas.core.frame.DataFrame'>

config = MyParquetConfig(file="ccflow/examples/example.parquet", library="pyarrow")
df = config()
print(type(df))
#> <class 'pyarrow.lib.Table'>
```

### Example: Custom publisher

A common use case in any research/production framework is to send data from the current process to some other location. In extract-transform-load (ETL) processes, the target location is usually files, an object store or a database. However, automation of research reports containing tables, charts and HTML is another common use case; those may be written to files, sent to an experiment tracking framework or simply emailed to a set of recipients.

In `ccflow`, we used the principles above to define a very simple interface for "publishers" (`ccflow.publishers.BasePublisher`), which are configurable components that know how to take data and send it somewhere else. The reason for having a `BasePublisher` class is so that publishers can easily be substituted for one another (and validated) as part of the configuration of a larger workflow. If publishers were all implemented a little bit differently, then it would be difficult to switch from, i.e. writing files to sending an email purely based on configuration.

While `ccflow` provides some implementations out of the box for common use cases (files, email)), custom implementations of the interface are straightforward and encouraged. Below we will create a custom publisher that uses IPython's "display" function to display a list of strings as html. We use Jinja templating as described in a previous section to define the html template.

```python
from IPython.display import display, HTML
from ccflow import JinjaTemplate, BasePublisher
from typing import List

class MyPublisher(BasePublisher):
    data: List[str] = None
    html_template: JinjaTemplate

    def __call__(self):
        display(HTML(self.get_name()))
        display(HTML(self.html_template.template.render(data="<BR>".join(self.data))))

# Create the publisher (i.e. via static configuration)
p = MyPublisher(
    name="<b>My {{desc}} publisher:</b>",
    html_template="""<p style="color:blue;">{{data}}</p>""",
)

# Set the data that we want to publish (i.e. at runtime)
p.name_params = dict(desc="test")
p.data = ["Blue text.", "More blue text."]
p()
```

Even though "data" is a standard attribute on BasePublisher, implementations can override it to define more specific data types that the publisher allows, and Pydantic provides the validation. For example, passing a dictionary to "data" in the example above results in an error (before the call to publish is even made):

```python
p = MyPublisher(
    name="<b>My {{desc}} publisher:</b>",
    html_template="""<p style="color:blue;">{{data}}</p>""",
)
try:
    p.data = {}
except ValueError as v:
    print(v)
```

### Example: Simple data pipeline in polars

In the example below, we use the dataframe library [Polars](https://docs.pola.rs/) along with `ccflow` to configure a very simple data pipeline. For this example to run `polars` must be installed.

First we define some simple models to read, transform and summarize data

```python
from ccflow import BaseModel, ModelRegistry
from ccflow.exttypes.polars import PolarsExpression
from pathlib import Path
from typing import List, Optional

import polars as pl


class PolarsCallable(BaseModel):
    """A base class for models that return a polars LazyFrame"""
    def __call__(self) -> pl.LazyFrame:
        ...
    
class ParquetDataReader(PolarsCallable):
    """A model to read data from a parquet file"""
    file: Path
    n_rows: Optional[int] = None
    row_index_name: Optional[str] = None
    # Could expose additional options from the `pl.scan_parquet` interface here
    
    def __call__(self):
        return pl.scan_parquet(self.file, n_rows=self.n_rows, row_index_name=self.row_index_name)
    
class FeatureAdder(PolarsCallable):
    """This model adds extra columns (features) to the original dataset. 
    The extra columns are averages over 'average_cols' grouped by 'group_col', and
    the absolute difference between the original values and the grouped averages.
    """
    data_input: PolarsCallable
    group_col: str
    average_cols: List[str]
    
    def __call__(self):
        df = self.data_input()
        agg_cols = [pl.col(col).mean().alias(f"{col}_mean") for col in self.average_cols]
        avg_df = df.group_by(pl.col(self.group_col)).agg(agg_cols)
        joined_df = df.join(avg_df, on=self.group_col)
        residual_columns = [(pl.col(col) - pl.col(f"{col}_mean")).abs().alias(f"{col}_resid") for col in self.average_cols]
        return joined_df.with_columns(residual_columns)
    
    
class TopKFinder(PolarsCallable):
    """This model finds the records corresponding to the max   """
    data_input: PolarsCallable    
    by: str
    k: int = 1
    reverse: bool = False
    
    def __call__(self):
        df = self.data_input()
        return df.top_k(k=self.k, by=self.by, reverse=self.reverse)

class ColumnSelector(PolarsCallable):
    """This model generically selects column expressions from a source"""
    data_input: PolarsCallable
    exprs: List[PolarsExpression]
    
    def __call__(self):
        return self.data_input().select(self.exprs)
```

With these models, we now register certain configurations in the registry

```python
root = ModelRegistry.root().clear()
root.add("Raw Data", DataReader(file=Path(ccflow.examples.__file__).parent / "example.parquet"))
root.add("Augmented Data", FeatureAdder(data_input="Raw Data", group_col="State", average_cols=["Sales", "Profit"]))
root.add("TopK Profit residuals", TopKFinder(data_input="Augmented Data", by="Profit_resid", k=3))
root.add("TopK Sales residuals", TopKFinder(data_input="Augmented Data", by="Sales_resid", k=5))
root.add("Mean residuals", ColumnSelector(data_input="Augmented Data", exprs=[pl.col("Sales_resid").mean(), pl.col("Profit_resid").mean()]))
```

Note that we have registered several different configurations all referencing the same "Augmented Data" component. Two of these are slightly different configurations of `TopKFinder`, but the other is a totally different summary of the data.

To use these configurations, we can now load a residuals model from the registry and call it. This returns a `polars.LazyFrame`, i.e. we are using `ccflow` to construct the polars graph. We can then call `collect()` on the result to materialize a data frame:

```python
print(root["TopK Profit residuals"]().collect())
print(root["Mean residuals"]().collect())
```

Suppose instead we wanted to group by "Region" instead of "State", this can be done by injecting the change directly on "Augmented Data". Note that this change will impact multiple downstream results, as they all depend on the same "Augmented Data" config:

```python
root["Augmented Data"].group_col = "Region"
print(root["TopK Profit residuals"]().collect())
print(root["Mean residuals"]().collect())
```

At the moment, "Mean residuals" is taking the mean of the residuals from all the groups output from "Augmented Data". What if we wanted to take the mean of only the top sales residuals? We can reconfigure this easily by pointing the input of "Mean residuals" to "TopK Sales residuals" rather than the original "Augmented Data"

```python
root["Mean residuals"].data_input = "TopK Sales residuals"
print(root["Mean residuals"]().collect())
```

Suppose that we now need to go back and change the way the data is being read. Often, this might require pointing to a different file or data source, but sometimes it could just be modifying the additional arguments to the additional read call. For example, perhaps we want to change the name of the row index column. We want to be able to do this without changing any code, and especially not by exposing this parameter to any downstream logic (as most data loading interfaces have many different parameters that might need to be configured). Even though this parameter is more than one level down in the call tree, we can inject a new value for it using `ccflow`, and this will still work even if additional processing steps are added in the middle of the workflow:

```python
root["Raw Data"].row_index_name = "My Index Row"
assert "My Index Row" in root["TopK Profit residuals"]().collect_schema().names()
```

### Example: Gaussian process regression in sklearn

Gaussian Processes (GP) are a generic supervised learning method designed to solve non-linear regression and probabilistic classification problems. The GaussianProcessRegressor in `sklearn` implements Gaussian processes (GP) for regression purposes. For this, the prior of the GP needs to be specified. The prior mean is assumed to be constant and zero (for normalize_y=False) or the training data’s mean (for normalize_y=True). The prior’s covariance is specified by passing a kernel object.

In the example below, we see how `ccflow` can be used to configure a GP Regression (including the kernel object). Even though the kernel object is not a Pydantic type, we can still configure it in this framework.

Note that `sklearn` must be installed to run this example. For the meaning of the parameters, refer to the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor); we follow their example for usage of the GP Regressor.

```python
from ccflow import BaseModel
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from typing import Callable, Union
from hydra.utils import instantiate

class GPRegressionModel(BaseModel):
    """Wrapping of sklearn's GaussianProcessRegressor for configuration"""

    # Here we tell Pydantic to allow the "Kernel" type, which is not a standard Pydantic type
    model_config = {"arbitrary_types_allowed": True}

    kernel: Kernel
    alpha: float = 1e-10
    optimizer: Union[str, Callable] = "fmin_l_bfgs_b"
    n_restarts_optimizer: int = 0
    normalize_y: bool = False
    random_state: int = None

    def __call__(self):
        """Build the GP Regressor object"""
        # Rather than passing in each attribute individually, we can use the dict representation of the config class (leaving out the "type_" attribute),
        # as we named everything consistently with the sklearn parameter names
        return GaussianProcessRegressor(**self.model_dump(exclude={"type_"}))

# Define the config as a dictionary (potentially to live in a config file)
gpr_config = {
    "_target_": "__main__.GPRegressionModel",
    "kernel": {
        "_target_": "sklearn.gaussian_process.kernels.Sum",
        "k1": {
            "_target_": "sklearn.gaussian_process.kernels.DotProduct",
            "sigma_0": 1.0,
        },
        "k2": {
            "_target_": "sklearn.gaussian_process.kernels.WhiteKernel",
            "noise_level": 1.0,
        },
    },
    "optimizer": "fmin_l_bfgs_b",
    "random_state": 0,
}

# Load the config in the root registry
root = ModelRegistry.root().clear()
root.load_config({"gpr": gpr_config})

# Use the config to train the configured model
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
gpr = root["gpr"]().fit(X, y)
print(gpr.score(X, y))

# We can now change the config interactively to experiment with different kernels,
# without needing to go back to config dictionaries or files
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

root["gpr"].kernel = RBF() + WhiteKernel()
gpr = root["gpr"]().fit(X, y)
print(gpr.score(X, y))
```
