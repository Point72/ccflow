"""Sample python objects for testing from_python and identity preservation."""

from typing import Dict

from ccflow import BaseModel

# Module-level objects
SHARED_CFG: Dict[str, int] = {"x": 1, "y": 2}
OTHER_CFG: Dict[str, int] = {"x": 10, "y": 20}
"""Dict samples; identity for dicts is not guaranteed by Pydantic."""


class SharedHolder(BaseModel):
    name: str
    cfg: Dict[str, int]


class SharedModel(BaseModel):
    val: int = 0


# Module-level instance to be resolved via from_python
SHARED_MODEL = SharedModel(val=42)


class Consumer(BaseModel):
    shared: SharedModel
    tag: str = ""
