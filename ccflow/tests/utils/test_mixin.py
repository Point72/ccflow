from typing import ClassVar, List

import pytest
from pydantic import ConfigDict, Field, ValidationError

from ccflow import BaseModel
from ccflow.utils import PathKeyResolverMixin


# Top-level models for Hydra integration tests
class MixinSimpleModel(BaseModel, PathKeyResolverMixin):
    name: str = Field(default="default_name")
    version: str = Field(default="0.0")
    enabled: bool = Field(default=False)


class MixinDatabaseConfigModel(BaseModel, PathKeyResolverMixin):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="default_db")


class PlainDatabaseConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="default_db")


class MixinParentModel(BaseModel, PathKeyResolverMixin):
    name: str = Field(default="default_name")
    version: str = Field(default="0.0")
    enabled: bool = Field(default=False)
    child: MixinDatabaseConfigModel
    plain_child: PlainDatabaseConfig


# Local complex objects for import-path tests
class _InnerBM(BaseModel):
    val: int


LOCAL_COMPLEX_WITH_BM = {"sub": _InnerBM(val=5)}


class _PM(BaseModel):
    a: int
    b: str


LOCAL_PM = {"pm": _PM(a=1, b="x")}


class TestPathKeyResolverMixin:
    """Test cases for PathKeyResolverMixin."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create a simple test model that uses the mixin
        class SimpleTestModel(BaseModel, PathKeyResolverMixin):
            name: str = Field(default="default_name")
            version: str = Field(default="0.0")
            enabled: bool = Field(default=False)

        # Create a model with additional fields for merge testing
        class MergeTestModel(BaseModel, PathKeyResolverMixin):
            name: str = Field(default="default_name")
            version: str = Field(default="0.0")
            enabled: bool = Field(default=False)
            extra_field: str = Field(default="extra_value")

        self.SimpleTestModel = SimpleTestModel
        self.MergeTestModel = MergeTestModel

    def test_dict_without_path_key_is_unchanged(self):
        """Test that dictionaries without 'path' key are processed normally."""
        normal_config = {"name": "normal_model", "version": "1.5", "enabled": False}

        model = self.SimpleTestModel(**normal_config)

        assert model.name == "normal_model"
        assert model.version == "1.5"
        assert model.enabled is False

    def test_basic_path_resolution(self):
        """Test basic PyObjectPath resolution without key parameter."""
        # Test data is in the test_data module
        path_config = {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG"}

        model = self.SimpleTestModel(**path_config)

        # Values should be resolved from SIMPLE_CONFIG
        assert model.name == "test_model"
        assert model.version == "1.0"
        assert model.enabled is True

    def test_path_with_key_resolution(self):
        """Test PyObjectPath resolution with key parameter."""

        # Create a test model that matches the database config structure
        class DatabaseConfigModel(BaseModel, PathKeyResolverMixin):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="default_db")

        path_config = {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG", "ccflow_keys": "database"}

        model = DatabaseConfigModel(**path_config)

        # Values should be resolved from NESTED_CONFIG["database"]
        assert model.host == "localhost"
        assert model.port == 5432
        assert model.name == "test_db"

    def test_value_merging_precedence(self):
        """Test that resolved values take precedence over explicitly provided values."""
        path_config = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "name": "overridden_name",  # Will be overridden by resolved name
            "version": "3.0",  # Will be overridden by resolved version
            # enabled should come from the resolved config
        }

        model = self.SimpleTestModel(**path_config)

        # Resolved values should override explicitly provided values
        assert model.name == "test_model"  # From SIMPLE_CONFIG, not "overridden_name"
        assert model.version == "1.0"  # From SIMPLE_CONFIG, not "3.0"
        # Value from resolved config should be used
        assert model.enabled is True

    def test_partial_resolution_merging(self):
        """Test merging when resolved config only provides some fields."""

        # Create a model with more fields than what's in MIXED_TYPES_CONFIG
        class ExtendedModel(BaseModel, PathKeyResolverMixin):
            # Allow extra keys from the resolved dict to be ignored for this test
            model_config = ConfigDict(extra="ignore")
            string_val: str = Field(default="default_string")
            int_val: int = Field(default=0)
            bool_val: bool = Field(default=False)
            missing_field: str = Field(default="default_missing")
            extra_provided: str = Field(default="default_extra")

        path_config = {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.MIXED_TYPES_CONFIG", "extra_provided": "explicitly_set"}

        model = ExtendedModel(**path_config)

        # Values from resolved config
        assert model.string_val == "hello"
        assert model.int_val == 42
        assert model.bool_val is True
        # Field not in resolved config should use default
        assert model.missing_field == "default_missing"
        # Explicitly provided field
        assert model.extra_provided == "explicitly_set"

    def test_invalid_path_handling(self):
        """Test error handling for invalid PyObjectPath."""
        path_config = {"ccflow_path": "non_existent.module.INVALID_CONFIG"}

        with pytest.raises(Exception):  # Should raise some kind of validation error
            self.SimpleTestModel(**path_config)

    def test_invalid_key_handling(self):
        """Test error handling for invalid key in resolved object."""
        path_config = {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG", "ccflow_keys": "non_existent_key"}

        with pytest.raises(KeyError):
            self.SimpleTestModel(**path_config)

    def test_path_resolves_to_non_dict(self):
        """Test error handling when path resolves to non-dict object."""
        # Add a non-dict object to our test data for this test
        path_config = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.MIXED_TYPES_CONFIG",
            "ccflow_keys": "string_val",  # This resolves to a string, not a dict
        }

        # The mixin will try to call values.update() with a string, which should cause a validation error
        with pytest.raises((AttributeError, ValueError, TypeError)):
            self.SimpleTestModel(**path_config)

    def test_empty_path_handling(self):
        """Test that empty path parameter is handled correctly."""
        path_config = {"ccflow_path": ""}

        with pytest.raises(Exception):
            self.SimpleTestModel(**path_config)

    def test_nested_models_mixed_mixins(self):
        """Parent model (mixin) containing child mixin model and non-mixin model."""

        class DatabaseConfigModel(BaseModel, PathKeyResolverMixin):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="default_db")

        class PlainDatabaseConfig(BaseModel):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="default_db")

        class ParentModel(BaseModel, PathKeyResolverMixin):
            name: str = Field(default="default_name")
            version: str = Field(default="0.0")
            enabled: bool = Field(default=False)
            child: DatabaseConfigModel
            plain_child: PlainDatabaseConfig

        parent_config = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "child": {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG", "ccflow_keys": "database"},
            "plain_child": {"host": "plain.example.com", "port": 3306, "name": "plain_db"},
        }

        model = ParentModel(**parent_config)

        # Parent resolved from SIMPLE_CONFIG
        assert model.name == "test_model"
        assert model.version == "1.0"
        assert model.enabled is True

        # Child resolved via nested path/key
        assert model.child.host == "localhost"
        assert model.child.port == 5432
        assert model.child.name == "test_db"

        # Plain child uses provided explicit values only
        assert model.plain_child.host == "plain.example.com"
        assert model.plain_child.port == 3306
        assert model.plain_child.name == "plain_db"

    def test_nested_models_grandchild_mixins(self):
        """Parent (mixin) -> child (mixin) with nested mixin fields resolved independently."""

        class DatabaseConfigModel(BaseModel, PathKeyResolverMixin):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="default_db")

        class FeaturesModel(BaseModel, PathKeyResolverMixin):
            feature_a: bool = Field(default=False)
            feature_b: bool = Field(default=False)

        class ChildWithFeatures(BaseModel, PathKeyResolverMixin):
            db: DatabaseConfigModel
            features: FeaturesModel

        class ParentModel(BaseModel, PathKeyResolverMixin):
            name: str = Field(default="default_name")
            version: str = Field(default="0.0")
            enabled: bool = Field(default=False)
            child: ChildWithFeatures

        parent_config = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "child": {
                "db": {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG", "ccflow_keys": "database"},
                "features": {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG", "ccflow_keys": "features"},
            },
        }

        model = ParentModel(**parent_config)

        # Parent resolved from SIMPLE_CONFIG
        assert model.name == "test_model"
        assert model.version == "1.0"
        assert model.enabled is True

        # Child nested models resolved correctly
        assert model.child.db.host == "localhost"
        assert model.child.db.port == 5432
        assert model.child.db.name == "test_db"
        assert model.child.features.feature_a is True
        assert model.child.features.feature_b is False

    def test_plain_child_with_path_raises(self):
        """Non-mixin nested models should not accept 'path'/'key' and must raise."""

        class PlainDatabaseConfig(BaseModel):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="default_db")

        class ParentModel(BaseModel, PathKeyResolverMixin):
            name: str = Field(default="default_name")
            version: str = Field(default="0.0")
            enabled: bool = Field(default=False)
            plain_child: PlainDatabaseConfig

        parent_config = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "plain_child": {"ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG", "ccflow_keys": "database"},
        }
        with pytest.raises(ValidationError):
            ParentModel(**parent_config)

    def test_ccflow_merge_explicit_wins(self):
        class Model(BaseModel, PathKeyResolverMixin):
            a: str = Field(default="x")
            b: int = Field(default=0)

        cfg = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "ccflow_merge": "explicit_wins",
            "a": "explicit",
        }
        m = Model(**cfg)
        # 'a' stays explicit, 'b' filled from resolved if present, else default
        assert m.a == "explicit"
        assert m.b == 0  # 'b' not in SIMPLE_CONFIG, so default used

    def test_ccflow_keys_list_traversal(self):
        class EnvDB(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

        cfg = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.COMPLEX_CONFIG",
            "ccflow_keys": ["environments", "dev", "database"],
        }
        m = EnvDB(**cfg)
        assert m.host == "dev.example.com"
        assert m.port == 5432

    def test_ccflow_keys_top_level_int(self):
        class Server(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

        cfg = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SERVERS",
            "ccflow_keys": 1,  # select second server by index
        }
        s = Server(**cfg)
        assert s.host == "server2.local"
        assert s.port == 2222

    def test_ccflow_namespaced_dict_prefixed_keys(self):
        class DB(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

        cfg = {
            "ccflow": {
                "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
                "ccflow_keys": "database",
                "ccflow_merge": "explicit_wins",
                "ccflow_filter_extras": True,
            }
        }
        m = DB(**cfg)
        assert m.host == "localhost"
        assert m.port == 5432

    def test_ccflow_merge_raise_on_conflict(self):
        class Model(BaseModel, PathKeyResolverMixin):
            name: str
            version: str
            enabled: bool

        cfg_ok = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "ccflow_merge": "raise_on_conflict",
            "name": "test_model",  # same as resolved
        }
        m = Model(**cfg_ok)
        assert m.name == "test_model"
        assert m.version == "1.0"
        assert m.enabled is True

        cfg_conflict = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
            "ccflow_merge": "raise_on_conflict",
            "version": "9.9",  # conflicts with resolved "1.0"
        }
        with pytest.raises(ValueError):
            Model(**cfg_conflict)

    def test_ccflow_allowed_prefixes_enforced(self):
        class StrictDB(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

            # Restrict to our test data module
            ccflow_allowed_prefixes: ClassVar[List[str]] = ["ccflow.tests.data.path_key_resolver_samples"]

        # Allowed path
        ok = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
            "ccflow_keys": "database",
        }
        m = StrictDB(**ok)
        assert m.host == "localhost"
        assert m.port == 5432

        # Disallowed path: valid import path but outside allowed prefix
        bad = {
            "ccflow_path": "ccflow.tests.utils.test_mixin.MixinSimpleModel",
            "ccflow_keys": [],
        }
        with pytest.raises(ValueError) as ei:
            StrictDB(**bad)
        assert "not allowed by allowed prefixes" in str(ei.value)

    def test_ccflow_debug_metadata(self):
        class DB(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

        cfg = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
            "ccflow_keys": "database",
            "ccflow_merge": "explicit_wins",
        }
        m = DB(**cfg)
        assert hasattr(m, "__ccflow_source__")
        meta = m.__ccflow_source__
        assert meta["path"].endswith("path_key_resolver_samples.NESTED_CONFIG")
        assert meta["keys"] == ["database"]
        assert meta["merge"] == "explicit_wins"

    def test_mapping_with_pydantic_instance_value(self):
        class Outer(BaseModel, PathKeyResolverMixin):
            sub: _InnerBM

        cfg = {
            "ccflow_path": "ccflow.tests.utils.test_mixin.LOCAL_COMPLEX_WITH_BM",
        }
        o = Outer(**cfg)
        assert isinstance(o.sub, _InnerBM)
        assert o.sub.val == 5

    def test_traverse_to_pydantic_model_node(self):
        class Target(BaseModel, PathKeyResolverMixin):
            a: int
            b: str

        cfg = {
            "ccflow_path": "ccflow.tests.utils.test_mixin.LOCAL_PM",
            "ccflow_keys": "pm",
        }
        t = Target(**cfg)
        assert t.a == 1
        assert t.b == "x"

    def test_private_attr_is_per_instance(self):
        class DB(BaseModel, PathKeyResolverMixin):
            host: str
            port: int

        cfg1 = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
            "ccflow_keys": "database",
        }
        cfg2 = {
            "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.OTHER_NESTED_CONFIG",
            "ccflow_keys": "database_alt",
        }

        m1 = DB(**cfg1)
        m2 = DB(**cfg2)

        assert hasattr(m1, "__ccflow_source__") and hasattr(m2, "__ccflow_source__")
        # Metadata differs per instance
        assert m1.__ccflow_source__["keys"] == ["database"]
        assert m2.__ccflow_source__["keys"] == ["database_alt"]

        # Mutating one should not affect the other
        m1.__ccflow_source__["mut"] = 1
        assert "mut" not in m2.__ccflow_source__
