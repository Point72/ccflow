from pydantic import Field

from ccflow import BaseModel, ModelRegistry
from ccflow.compose import model_copy_update


class DB(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="default_db")


class Parent(BaseModel):
    name: str = Field(default="default_name")
    version: str = Field(default="0.0")
    enabled: bool = Field(default=False)
    child: DB


def setup_module(_):
    ModelRegistry.root().clear()
    r = ModelRegistry.root()
    r.add("db_default", DB())
    r.add("db_other", DB(host="override.local", port=6543, name="other_db"))
    r.add("parent", Parent(child=DB(name="child")))


def teardown_module(_):
    ModelRegistry.root().clear()


def test_model_copy_update_returns_model_from_registry_and_updates():
    # alias + update helper
    m2 = model_copy_update("db_default", update={"host": "h", "port": 100})
    assert isinstance(m2, DB)
    assert m2.host == "h"
    assert m2.port == 100


def test_model_copy_update_equivalent():
    m = model_copy_update("db_default", update={"name": "x"})
    assert isinstance(m, DB)
    assert m.name == "x"


def test_model_copy_update_preserves_shared_identity_on_update():
    # Identity of nested fields preserved when using shallow dict update
    class Shared(BaseModel):
        val: int = 1

    class A(BaseModel):
        s: Shared
        x: int = 0

    shared = Shared(val=5)
    base = A(s=shared, x=10)
    ModelRegistry.root().add("baseA", base, overwrite=True)
    updated = model_copy_update("baseA", update={"x": 11})
    assert isinstance(updated, A)
    assert updated.x == 11
    assert updated.s is shared


def test_model_alias_resolve_by_name():
    base = ModelRegistry.root()["db_default"]
    out = BaseModel.model_validate("db_default")
    assert out is base


def test_model_copy_update_with_no_changes_returns_diff_object():
    base = ModelRegistry.root()["db_default"]
    out = model_copy_update("db_default")
    assert out is not base


def test_model_copy_update_applies_multiple_updates():
    out = model_copy_update("db_default", update={"host": "u.local", "port": 9999, "name": "u"})
    assert out.host == "u.local"
    assert out.port == 9999
    assert out.name == "u"


def test_model_copy_update_does_not_affect_original():
    base = ModelRegistry.root()["db_default"]
    out = model_copy_update("db_default", update={"name": "changed"})
    assert base.name != out.name


def test_model_copy_update_handles_empty_update():
    out = model_copy_update("db_default", update={})
    assert isinstance(out, DB)


def test_model_copy_update_hydra_like_call():
    # Simulate Hydra instantiate of function target using args + update
    from hydra.utils import instantiate

    cfg = {
        "_target_": "ccflow.compose.model_copy_update",
        "model_name": "db_other",
        "update": {"port": 7654},
    }
    out = instantiate(cfg, _convert_="all")
    assert out.port == 7654
    assert out.host == "override.local"


def test_from_python_hydra_like():
    from ccflow.compose import from_python

    obj = from_python("ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG")
    assert isinstance(obj, dict)


def test_model_copy_update_preserves_type_and_fields():
    base = ModelRegistry.root()["db_default"]
    out = model_copy_update("db_default", update={"name": "new"})
    assert isinstance(out, DB)
    assert out.host == base.host


def test_model_copy_update_multiple_fields():
    out = model_copy_update("db_default", update={"name": "m", "host": "m.local", "port": 1111})
    assert out.name == "m"
    assert out.host == "m.local"
    assert out.port == 1111


def test_model_copy_update_multiple_calls_independent_instances():
    base = ModelRegistry.root()["db_default"]
    a = model_copy_update("db_default", update={"name": "a"})
    b = model_copy_update("db_default", update={"name": "b"})
    assert a.name == "a"
    assert b.name == "b"
    assert base.name != a.name and base.name != b.name
