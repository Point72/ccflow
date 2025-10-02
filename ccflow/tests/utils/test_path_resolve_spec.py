from dataclasses import dataclass
from typing import Mapping

import pytest
from pydantic import BaseModel as PydanticBaseModel

from ccflow.utils import PathResolveSpec

NESTED_PATH = "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG"


def test_extract_from_values_top_level():
    values = {
        "ccflow_path": NESTED_PATH,
        "ccflow_keys": "database",
        "ccflow_merge": "explicit_wins",
        "ccflow_filter_extras": True,
        "foo": 1,
    }

    spec, mutated = PathResolveSpec.extract_from_values(values)
    assert spec is not None
    assert str(spec.path).endswith("path_key_resolver_samples.NESTED_CONFIG")
    assert spec.keys == "database"
    assert spec.merge == "explicit_wins"
    assert spec.filter_extras is True
    # ccflow_* keys are popped from the mutated dict
    assert "ccflow_path" not in mutated
    assert mutated["foo"] == 1


def test_extract_from_values_namespaced_prefixed():
    values = {
        "ccflow": {
            "ccflow_path": NESTED_PATH,
            "ccflow_keys": "database",
            "ccflow_merge": "resolved_wins",
            "ccflow_filter_extras": False,
        },
        "bar": 2,
    }

    spec, mutated = PathResolveSpec.extract_from_values(values)
    assert spec is not None
    assert spec.filter_extras is False
    assert mutated["bar"] == 2
    assert "ccflow" not in mutated


def test_resolve_traverse_as_mapping_allowed_prefix():
    spec = PathResolveSpec.model_validate({"path": NESTED_PATH, "keys": "database"})
    obj = spec.resolve_object(allowed_prefixes=["ccflow.tests.data.path_key_resolver_samples"])
    node = spec.traverse(obj)
    mapping = spec.as_mapping(node)
    assert isinstance(mapping, Mapping)
    assert mapping["host"] == "localhost"
    assert mapping["port"] == 5432


def test_resolve_object_disallowed_prefix_raises():
    spec = PathResolveSpec.model_validate({"path": NESTED_PATH})
    with pytest.raises(ValueError):
        spec.resolve_object(allowed_prefixes=["some.other.module"])


def test_filter_extras_map_true_and_false():
    spec = PathResolveSpec.model_validate({"path": NESTED_PATH, "keys": "database", "filter_extras": True})
    mapping = {"host": "localhost", "port": 5432, "extra": 1}
    fields = {"host": None, "port": None}  # Only allow these keys
    filtered = spec.filter_extras_map(mapping, fields)
    assert filtered == {"host": "localhost", "port": 5432}

    spec2 = PathResolveSpec.model_validate({"path": NESTED_PATH, "keys": "database", "filter_extras": False})
    filtered2 = spec2.filter_extras_map(mapping, fields)
    assert filtered2 == mapping


def test_merge_into_resolved_wins_and_explicit_wins():
    mapping = {"host": "localhost", "port": 5432}
    values = {"host": "explicit"}

    spec_resolved = PathResolveSpec.model_validate({"path": NESTED_PATH, "merge": "resolved_wins"})
    out_resolved = spec_resolved.merge_into(values, mapping)
    assert out_resolved["host"] == "localhost"
    assert out_resolved["port"] == 5432

    spec_explicit = PathResolveSpec.model_validate({"path": NESTED_PATH, "merge": "explicit_wins"})
    out_explicit = spec_explicit.merge_into(values, mapping)
    assert out_explicit["host"] == "explicit"
    assert out_explicit["port"] == 5432


def test_merge_into_raise_on_conflict():
    mapping = {"host": "localhost", "port": 5432}
    values_conflict = {"host": "explicit"}
    spec_conflict = PathResolveSpec.model_validate({"path": NESTED_PATH, "merge": "raise_on_conflict"})
    with pytest.raises(ValueError):
        spec_conflict.merge_into(values_conflict, mapping)

    values_no_conflict = {"name": "x"}
    out = spec_conflict.merge_into(values_no_conflict, mapping)
    assert out["port"] == 5432
    assert out["host"] == "localhost"


def test_as_mapping_with_pydantic_and_dataclass():
    class PModel(PydanticBaseModel):
        a: int
        b: str

    @dataclass
    class DModel:
        a: int
        b: str

    pm = PModel(a=1, b="x")
    dm = DModel(a=2, b="y")
    m1 = PathResolveSpec.as_mapping(pm)
    m2 = PathResolveSpec.as_mapping(dm)
    assert m1 == {"a": 1, "b": "x"}
    assert m2 == {"a": 2, "b": "y"}


def test_debug_meta():
    spec = PathResolveSpec.model_validate({"path": NESTED_PATH, "keys": ["database"], "merge": "explicit_wins", "filter_extras": True})
    meta = spec.debug_meta()
    assert meta["path"].endswith("path_key_resolver_samples.NESTED_CONFIG")
    assert meta["keys"] == ["database"]
    assert meta["merge"] == "explicit_wins"
    assert meta["filter_extras"] is True
