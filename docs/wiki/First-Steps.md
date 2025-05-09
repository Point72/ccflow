# First Steps

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
