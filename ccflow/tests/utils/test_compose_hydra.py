import os

import pytest

from ccflow import BaseModel, ModelRegistry


def _config_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "config", name)


@pytest.fixture
def registry():
    ModelRegistry.root().clear()
    yield ModelRegistry.root()
    ModelRegistry.root().clear()


def test_hydra_conf_registry_reference_identity():
    # Config supplies registry names for nested BaseModel arguments; identity should be preserved
    from ccflow.tests.data.python_object_samples import Consumer, SharedHolder, SharedModel

    path = _config_path("conf_from_python.yaml")
    cfg = ModelRegistry.root().create_config_from_path(path=path)
    ModelRegistry.root().load_config(cfg, overwrite=True)

    shared = ModelRegistry.root()["shared_model"]
    consumer = ModelRegistry.root()["consumer"]
    consumer_updated = ModelRegistry.root()["consumer_updated"]

    assert isinstance(shared, SharedModel)
    assert isinstance(consumer, Consumer)
    # Identity: consumer.shared should be the same instance as registry shared_model
    assert consumer.shared is shared
    # model_copy_update preserves shared identity and applies field updates
    assert consumer_updated.shared is shared
    assert consumer_updated.tag == "consumer2"

    # Also check dict-returning from_python works and holder is constructed
    holder = ModelRegistry.root()["holder"]
    assert isinstance(holder, SharedHolder)
    assert isinstance(holder.cfg, dict)


# Removed: dict identity is not guaranteed by Pydantic; identity preservation is tested for BaseModel fields.


def test_model_copy_update_shared_identity():
    # Ensure shared sub-fields remain identical objects when alias-update is used
    from hydra.utils import instantiate

    ModelRegistry.root().clear()

    class Shared(BaseModel):
        val: int = 1

    class A(BaseModel):
        s: Shared
        x: int = 0

    # Register a base object and a shared object by name
    shared = Shared(val=5)
    base = A(s=shared, x=10)
    ModelRegistry.root().add("shared", shared, overwrite=True)
    ModelRegistry.root().add("base", base, overwrite=True)

    # Compose a config that uses model_copy_update to update only a primitive field
    cfg = {
        "updated": {
            "_target_": "ccflow.compose.model_copy_update",
            "model_name": "base",
            "update": {"x": 99},
        }
    }

    # Hydra instantiate calls the function, which uses model_copy(update=...)
    obj = instantiate(cfg["updated"], _convert_="all")
    assert isinstance(obj, A)
    assert obj.x == 99
    # Ensure the shared sub-field refers to the same object as in the registry
    assert obj.s is shared

    # Additional: Using model_copy_update without changing shared should preserve identity
    obj2 = instantiate(cfg["updated"], _convert_="all")
    assert obj2.s is shared
