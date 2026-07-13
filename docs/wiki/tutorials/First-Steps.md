# First Steps

This is the shortest possible tour of `ccflow`. In a few minutes you will define two configuration objects, load them into a registry from plain data, look them up, and watch the registry keep linked objects in sync. Type it into a Python session and follow along — everything here runs as shown.

You only need `ccflow` installed (see [Installation](Installation)). Everything below happens in Python; no files or command line yet.

Paste the whole block into a Python session:

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

## What you just did

- Defined two configuration schemas as `BaseModel` subclasses, so their fields are typed and validated.
- Loaded a nested dictionary of configs into the root `ModelRegistry` with `load_config`, which turned nested dictionaries into a hierarchy and resolved the string `"data/source1"` into a real reference.
- Looked configs up by path (`root["transform"]`).
- Rewired a dependency by name (`root["transform"].x = "data/source3"`) and saw that editing a low-level object (`root["data/source3"].file`) propagated up through the shared instance.

That last point is the heart of `ccflow`: configuration is a graph of shared, strongly typed objects, not a tree of copied values.

## Next steps

- [Configuring Models](Configuring-Models) — build these ideas up properly: hierarchical configs, validation, serialization, and dependency injection.
- [Defining Workflows](Defining-Workflows) — make configuration objects *runnable*.
- [Core Concepts](Core-Concepts) — the vocabulary and the reasoning behind the design.
