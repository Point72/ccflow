from pathlib import Path

import pytest
from pydantic import Field

from ccflow import BaseModel, ModelRegistry
from ccflow.compose import model_alias_update, model_compose


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


def test_model_alias_update_returns_model_from_registry_and_updates():
    # alias + update helper
    m2 = model_alias_update("db_default", host="h", port=100)
    assert isinstance(m2, DB)
    assert m2.host == "h"
    assert m2.port == 100


def test_model_alias_update_equivalent():
    m = model_alias_update("db_default", name="x")
    assert isinstance(m, DB)
    assert m.name == "x"


def test_model_compose_updates_without_path():
    base = ModelRegistry.root()["db_default"]
    out = model_compose("db_default", name="z")
    assert out.name == "z"
    assert out.host == base.host


def test_model_compose_with_path_and_keys():
    # Use existing test data as python source
    path = "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG"
    out = model_compose("db_default", path=path, keys="database")
    assert out.host == "localhost"
    assert out.port == 5432
    assert out.name == "test_db"


def test_model_compose_merge_semantics():
    path = "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG"
    # explicit_wins keeps explicit value for host, fills port from mapping
    out = model_compose("db_default", path=path, keys="database", merge="explicit_wins", host="exp")
    assert out.host == "exp"
    assert out.port == 5432

    # resolved_wins overrides explicit value for host
    out2 = model_compose("db_default", path=path, keys="database", merge="resolved_wins", host="exp")
    assert out2.host == "localhost"

    # raise_on_conflict errors on conflicting key
    with pytest.raises(ValueError):
        model_compose("db_default", path=path, keys="database", merge="raise_on_conflict", host="exp")


def test_model_compose_filter_extras():
    # mapping contains extra key not present on DB
    path = "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG"
    out = model_compose("db_default", path=path, keys="database", filter_extras=True)
    assert isinstance(out, DB)


def test_model_compose_allowed_prefixes_enforced():
    path = "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG"
    with pytest.raises(ValueError):
        model_compose("db_default", path=path, allowed_prefixes=["some.other.module"])


def test_model_compose_with_instance_and_update_dict():
    inst = ModelRegistry.root()["db_default"]
    out = model_compose(inst, update={"name": "via_update"})
    assert out.name == "via_update"


def test_hydra_target_usage_like():
    # Simulate Hydra instantiate by calling function target
    target = model_compose
    cfg = dict(model="db_other", path=None, update={"port": 7654})
    out = target(**cfg)
    assert out.port == 7654
    assert out.host == "override.local"
