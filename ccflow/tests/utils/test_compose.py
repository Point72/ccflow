from pydantic import Field

from ccflow import BaseModel, ModelRegistry
from ccflow.compose import from_python, update_from_base


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


def test_update_from_base_returns_model_and_updates():
    # alias + update helper
    base = BaseModel.model_validate("db_default")
    m2 = update_from_base(base=base, update={"host": "h", "port": 100})
    assert isinstance(m2, DB)
    assert m2.host == "h"
    assert m2.port == 100


def test_update_from_base_equivalent():
    base = BaseModel.model_validate("db_default")
    m = update_from_base(base=base, update={"name": "x"})
    assert isinstance(m, DB)
    assert m.name == "x"


def test_update_from_base_preserves_shared_identity_on_update():
    # Identity of nested fields preserved when using shallow dict update
    class Shared(BaseModel):
        val: int = 1

    class A(BaseModel):
        s: Shared
        x: int = 0

    shared = Shared(val=5)
    base = A(s=shared, x=10)
    ModelRegistry.root().add("baseA", base, overwrite=True)
    updated = update_from_base(base=BaseModel.model_validate("baseA"), update={"x": 11})
    assert isinstance(updated, A)
    assert updated.x == 11
    assert updated.s is shared


def test_model_alias_resolve_by_name():
    base = ModelRegistry.root()["db_default"]
    out = BaseModel.model_validate("db_default")
    assert out is base


def test_update_from_base_with_no_changes_returns_diff_object():
    base = ModelRegistry.root()["db_default"]
    out = update_from_base(base=base)
    assert out is not base


def test_update_from_base_applies_multiple_updates():
    out = update_from_base(base=BaseModel.model_validate("db_default"), update={"host": "u.local", "port": 9999, "name": "u"})
    assert out.host == "u.local"
    assert out.port == 9999
    assert out.name == "u"


def test_update_from_base_does_not_affect_original():
    base = ModelRegistry.root()["db_default"]
    out = update_from_base(base=base, update={"name": "changed"})
    assert base.name != out.name


def test_update_from_base_handles_empty_update():
    out = update_from_base(base=BaseModel.model_validate("db_default"), update={})
    assert isinstance(out, DB)


def test_update_from_base_hydra_like_call_with_model_alias_base():
    # Simulate Hydra instantiate of function target using args + update
    from hydra.utils import instantiate

    cfg = {
        "_target_": "ccflow.compose.update_from_base",
        "base": {"_target_": "ccflow.compose.model_alias", "model_name": "db_other"},
        "update": {"port": 7654},
    }
    out = instantiate(cfg, _convert_="all")
    assert out.port == 7654
    assert out.host == "override.local"


def test_from_python_hydra_like():
    obj = from_python("ccflow.tests.data.python_object_samples.SHARED_CFG")
    assert isinstance(obj, dict)
    assert obj == {"x": 1, "y": 2}


def test_update_from_base_preserves_type_and_fields():
    base = ModelRegistry.root()["db_default"]
    out = update_from_base(base=base, update={"name": "new"})
    assert isinstance(out, DB)
    assert out.host == base.host


def test_update_from_base_multiple_fields():
    out = update_from_base(base=BaseModel.model_validate("db_default"), update={"name": "m", "host": "m.local", "port": 1111})
    assert out.name == "m"
    assert out.host == "m.local"
    assert out.port == 1111


def test_update_from_base_multiple_calls_independent_instances():
    base = ModelRegistry.root()["db_default"]
    a = update_from_base(base=base, update={"name": "a"})
    b = update_from_base(base=base, update={"name": "b"})
    assert a.name == "a"
    assert b.name == "b"
    assert base.name != a.name and base.name != b.name


def test_from_python_with_indexer_resolves_nested_dict():
    obj = from_python(
        "ccflow.tests.data.python_object_samples.NESTED_CFG",
        indexer=["db"],
    )
    assert isinstance(obj, dict)
    assert obj == {"host": "seed.local", "port": 7000, "name": "seed"}


def test_update_from_base_hydra_chain_with_from_python_indexer_target_class():
    from hydra.utils import instantiate

    cfg = {
        "_target_": "ccflow.compose.update_from_base",
        "base": {
            "_target_": "ccflow.compose.from_python",
            "py_object_path": "ccflow.tests.data.python_object_samples.NESTED_CFG",
            "indexer": ["db"],
        },
        "update": {"port": 7654},
        "target_class": "ccflow.tests.utils.test_compose.DB",
    }
    out = instantiate(cfg, _convert_="all")
    assert isinstance(out, DB)
    assert out.host == "seed.local"
    assert out.port == 7654
    assert out.name == "seed"


def test_update_from_base_with_base_as_string_alias_preserves_identity():
    base_parent = ModelRegistry.root()["parent"]
    updated = update_from_base(base="parent", update={"version": "1.0"})
    assert isinstance(updated, Parent)
    assert updated.version == "1.0"
    # child identity preserved via shallow copy
    assert updated.child is base_parent.child


def test_update_from_base_with_dict_base_returns_updated_dict_without_target():
    src = {"a": 1}
    out = update_from_base(base=src, update={"b": 2})
    assert isinstance(out, dict)
    assert out == {"a": 1, "b": 2}
    assert out is not src


def test_update_from_base_with_dict_base_and_type_target_class():
    out = update_from_base(base={"host": "x", "port": 1, "name": "n"}, target_class=DB, update={"port": 2})
    assert isinstance(out, DB)
    assert out.host == "x"
    assert out.port == 2
    assert out.name == "n"


def test_update_from_base_with_none_base_and_type_target_class():
    out = update_from_base(base=None, target_class=DB, update={"host": "h", "port": 9, "name": "n"})
    assert isinstance(out, DB)
    assert out.host == "h" and out.port == 9 and out.name == "n"


def test_update_from_base_with_none_base_without_target_returns_dict():
    out = update_from_base(base=None, update={"x": 1})
    assert isinstance(out, dict)
    assert out == {"x": 1}


def test_update_from_base_with_empty_update_on_dict_and_none():
    out1 = update_from_base(base={"a": 1}, update={})
    assert out1 == {"a": 1}
    out2 = update_from_base(base=None, update={})
    assert out2 == {}


def test_update_from_base_invalid_base_type_raises():
    import pytest

    with pytest.raises(TypeError):
        update_from_base(base=[1, 2, 3], update={"x": 1})


def test_update_from_base_with_base_model_and_explicit_target_class():
    class DB2(BaseModel):
        host: str
        port: int
        name: str

    base_db = ModelRegistry.root()["db_default"]
    out = update_from_base(base=base_db, target_class=DB2, update={"name": "changed"})
    assert isinstance(out, DB2)
    assert out.host == base_db.host
    assert out.port == base_db.port
    assert out.name == "changed"
