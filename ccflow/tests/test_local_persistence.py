"""Tests for local class registration in ccflow.

The local persistence module allows classes defined inside functions (with '<locals>'
in their __qualname__) to work with PyObjectPath serialization by registering them
on ccflow.local_persistence with unique names.

Key behaviors tested:
1. Local classes get __ccflow_import_path__ set at definition time
2. Module-level classes are NOT registered (they're already importable)
3. Cross-process cloudpickle works via _sync_to_module
4. UUID-based naming provides uniqueness
"""

import re
import subprocess
import sys

import pytest
import ray

import ccflow.local_persistence as local_persistence
from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, NullContext
from ccflow.local_persistence import create_ccflow_model


class ModuleLevelModel(BaseModel):
    value: int


class ModuleLevelContext(ContextBase):
    value: int


class ModuleLevelCallable(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult:
        return GenericResult(value="ok")


# =============================================================================
# Tests for _register function
# =============================================================================


def test_base_module_available_after_import():
    """Test that ccflow.local_persistence module is available after importing ccflow."""
    assert local_persistence.LOCAL_ARTIFACTS_MODULE_NAME in sys.modules


def test_register_preserves_module_qualname_and_sets_import_path():
    """Test that _register sets __ccflow_import_path__ without changing __module__ or __qualname__."""

    def build():
        class Foo:
            pass

        return Foo

    Foo = build()
    original_module = Foo.__module__
    original_qualname = Foo.__qualname__

    local_persistence._register(Foo)

    # __module__ and __qualname__ should NOT change (preserves cloudpickle)
    assert Foo.__module__ == original_module, "__module__ should not change"
    assert Foo.__qualname__ == original_qualname, "__qualname__ should not change"
    assert "<locals>" in Foo.__qualname__, "__qualname__ should contain '<locals>'"

    # __ccflow_import_path__ should be set and point to the registered class
    import_path = Foo.__ccflow_import_path__
    registered_name = import_path.split(".")[-1]
    module = sys.modules[local_persistence.LOCAL_ARTIFACTS_MODULE_NAME]
    assert hasattr(module, registered_name), "Class should be registered in module"
    assert getattr(module, registered_name) is Foo, "Registered class should be the same object"
    assert import_path.startswith("ccflow.local_persistence._Local_"), "Import path should have expected prefix"


def test_register_handles_class_name_starting_with_digit():
    """Test that _register handles class names starting with a digit by prefixing with underscore."""
    # Create a class with a name starting with a digit
    cls = type("3DModel", (), {})
    local_persistence._register(cls)

    import_path = cls.__ccflow_import_path__
    registered_name = import_path.split(".")[-1]

    # The registered name should start with _Local__ (underscore added for digit)
    assert registered_name.startswith("_Local__"), "Registered name should start with _Local__"
    assert "_3DModel_" in registered_name, "Registered name should contain _3DModel_"

    # Should be registered on ccflow.local_persistence
    module = sys.modules[local_persistence.LOCAL_ARTIFACTS_MODULE_NAME]
    assert getattr(module, registered_name) is cls, "Class should be registered on module"


def test_sync_to_module_registers_class_not_yet_on_module():
    """Test that _sync_to_module registers a class that has __ccflow_import_path__ but isn't on the module yet.

    This happens in cross-process unpickle scenarios where cloudpickle recreates the class
    with __ccflow_import_path__ set, but the class isn't yet on ccflow.local_persistence.
    """
    # Simulate a class that has __ccflow_import_path__ but isn't registered on ccflow.local_persistence
    # (like what happens after cross-process cloudpickle unpickle)
    cls = type("SimulatedUnpickled", (), {})
    unique_name = "_Local_SimulatedUnpickled_test123abc"
    cls.__ccflow_import_path__ = f"{local_persistence.LOCAL_ARTIFACTS_MODULE_NAME}.{unique_name}"

    # Verify class is NOT on ccflow.local_persistence yet
    module = sys.modules[local_persistence.LOCAL_ARTIFACTS_MODULE_NAME]
    assert getattr(module, unique_name, None) is None, "Class should NOT be on module before sync"

    # Call _sync_to_module
    local_persistence._sync_to_module(cls)

    # Verify class IS now on ccflow.local_persistence
    assert getattr(module, unique_name, None) is cls, "Class should be on module after sync"


# =============================================================================
# Tests for local class registration via BaseModel
# =============================================================================


class TestLocalPersistencePreservesCloudpickle:
    """Tests verifying that local persistence preserves cloudpickle behavior."""

    def test_qualname_has_locals_for_function_defined_class(self):
        """Verify that __qualname__ contains '<locals>' for classes defined in functions."""

        def create_class():
            class Inner(BaseModel):
                x: int

            return Inner

        cls = create_class()
        assert "<locals>" in cls.__qualname__
        assert cls.__module__ != local_persistence.LOCAL_ARTIFACTS_MODULE_NAME

    def test_module_not_changed_to_local_artifacts(self):
        """Verify that __module__ is NOT changed to ccflow.local_persistence."""

        def create_class():
            class Inner(ContextBase):
                value: str

            return Inner

        cls = create_class()
        # __module__ should be this test module, not ccflow.local_persistence
        assert cls.__module__ == "ccflow.tests.test_local_persistence"
        assert cls.__module__ != local_persistence.LOCAL_ARTIFACTS_MODULE_NAME

    def test_ccflow_import_path_is_set(self):
        """Verify that __ccflow_import_path__ is set for local classes."""

        def create_class():
            class Inner(BaseModel):
                y: float

            return Inner

        cls = create_class()
        assert hasattr(cls, "__ccflow_import_path__")
        assert cls.__ccflow_import_path__.startswith(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME + ".")

    def test_class_registered_in_base_module(self):
        """Verify that the class is registered in ccflow.local_persistence under import path."""

        def create_class():
            class Inner(BaseModel):
                z: bool

            return Inner

        cls = create_class()
        import_path = cls.__ccflow_import_path__
        registered_name = import_path.split(".")[-1]

        artifacts_module = sys.modules[local_persistence.LOCAL_ARTIFACTS_MODULE_NAME]
        assert hasattr(artifacts_module, registered_name)
        assert getattr(artifacts_module, registered_name) is cls


class TestPyObjectPathWithImportPath:
    """Tests for PyObjectPath integration with __ccflow_import_path__."""

    def test_type_property_uses_import_path(self):
        """Verify that the type_ property returns a path using __ccflow_import_path__."""

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        instance = cls(value=123)
        type_path = str(instance.type_)

        # type_ should use the __ccflow_import_path__, not module.qualname
        assert type_path == cls.__ccflow_import_path__
        assert type_path.startswith(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME)

    def test_type_path_can_be_imported(self):
        """Verify that the type_ path can be used to import the class."""
        import importlib

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        instance = cls(value=456)
        type_path = str(instance.type_)

        # Should be able to import using the path
        parts = type_path.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        imported_cls = getattr(module, parts[1])
        assert imported_cls is cls

    def test_type_property_for_context_base(self):
        """Verify type_ works for ContextBase subclasses."""

        def create_class():
            class LocalContext(ContextBase):
                name: str

            return LocalContext

        cls = create_class()
        instance = cls(name="test")
        type_path = str(instance.type_)

        assert type_path == cls.__ccflow_import_path__
        assert instance.type_.object is cls

    def test_json_serialization_includes_target(self):
        """Verify JSON serialization includes _target_ using __ccflow_import_path__."""

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        instance = cls(value=789)
        data = instance.model_dump(mode="python")

        assert "type_" in data or "_target_" in data
        # The computed field should use __ccflow_import_path__
        type_value = data.get("type_") or data.get("_target_")
        assert str(type_value).startswith(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME)


class TestCloudpickleSameProcess:
    """Tests for same-process cloudpickle behavior."""

    def test_cloudpickle_class_roundtrip_same_process(self):
        """Verify cloudpickle can serialize and deserialize local classes in same process."""
        from ray.cloudpickle import dumps, loads

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        restored_cls = loads(dumps(cls))

        # Should be the same object (cloudpickle recognizes it's in the same process)
        assert restored_cls is cls

    def test_cloudpickle_instance_roundtrip_same_process(self):
        """Verify cloudpickle can serialize and deserialize instances in same process."""
        from ray.cloudpickle import dumps, loads

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        instance = cls(value=42)
        restored = loads(dumps(instance))

        assert restored.value == 42
        assert type(restored) is cls

    def test_cloudpickle_preserves_type_path(self):
        """Verify type_ works after cloudpickle roundtrip in same process."""
        from ray.cloudpickle import dumps, loads

        def create_class():
            class LocalModel(BaseModel):
                value: int

            return LocalModel

        cls = create_class()
        instance = cls(value=100)
        original_type_path = str(instance.type_)

        restored = loads(dumps(instance))
        restored_type_path = str(restored.type_)

        assert restored_type_path == original_type_path


# =============================================================================
# Cross-process cloudpickle tests
# =============================================================================


@pytest.fixture
def pickle_file(tmp_path):
    """Provide a temporary pickle file path with automatic cleanup."""
    pkl_path = tmp_path / "test.pkl"
    yield str(pkl_path)


class TestCloudpickleCrossProcess:
    """Tests for cross-process cloudpickle behavior (subprocess tests).

    These tests verify that ccflow classes (BaseModel, ContextBase, CallableModel)
    with local or __main__ scope can be pickled in one process and unpickled in another,
    with .type_ (PyObjectPath) working correctly after unpickle.
    """

    @pytest.mark.parametrize(
        "create_code,load_code",
        [
            pytest.param(
                # Local-scope BaseModel
                """
from ray.cloudpickle import dump
from ccflow import BaseModel

def create_local():
    class LocalModel(BaseModel):
        value: int
    return LocalModel

LocalModel = create_local()
assert "<locals>" in LocalModel.__qualname__
assert hasattr(LocalModel, "__ccflow_import_path__")

instance = LocalModel(value=42)
_ = instance.type_

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.value == 42
assert obj.type_.object is type(obj)
print("SUCCESS")
""",
                id="local_basemodel",
            ),
            pytest.param(
                # Local-scope ContextBase
                """
from ray.cloudpickle import dump
from ccflow import ContextBase

def create_local():
    class LocalContext(ContextBase):
        name: str
        value: int
    return LocalContext

LocalContext = create_local()
assert "<locals>" in LocalContext.__qualname__

instance = LocalContext(name="test", value=42)
_ = instance.type_

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.name == "test"
assert obj.value == 42
assert obj.type_.object is type(obj)
print("SUCCESS")
""",
                id="local_context",
            ),
            pytest.param(
                # Local-scope CallableModel (also tests callable execution)
                """
from ray.cloudpickle import dump
from ccflow import CallableModel, ContextBase, GenericResult, Flow

def create_local():
    class LocalContext(ContextBase):
        x: int

    class LocalCallable(CallableModel):
        multiplier: int = 2

        @Flow.call
        def __call__(self, context: LocalContext) -> GenericResult:
            return GenericResult(value=context.x * self.multiplier)

    return LocalContext, LocalCallable

LocalContext, LocalCallable = create_local()
model = LocalCallable(multiplier=3)
ctx = LocalContext(x=10)
result = model(ctx)
assert result.value == 30

with open("{pkl_path}", "wb") as f:
    dump((model, ctx), f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    model, ctx = load(f)

result = model(ctx)
assert result.value == 30
assert model.type_.object is type(model)
assert ctx.type_.object is type(ctx)
print("SUCCESS")
""",
                id="local_callable",
            ),
            pytest.param(
                # __main__ module class (not inside a function)
                # cloudpickle recreates but doesn't add to sys.modules["__main__"]
                """
from ray.cloudpickle import dump
from ccflow import ContextBase

class MainContext(ContextBase):
    value: int

assert MainContext.__module__ == "__main__"
assert hasattr(MainContext, "__ccflow_import_path__")

instance = MainContext(value=42)
_ = instance.type_

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.value == 42
assert obj.type_.object is type(obj)
print("SUCCESS")
""",
                id="main_module",
            ),
        ],
    )
    def test_cross_process_cloudpickle(self, pickle_file, create_code, load_code):
        """Test that ccflow classes work with cloudpickle across processes."""
        pkl_path = pickle_file

        create_result = subprocess.run(
            [sys.executable, "-c", create_code.format(pkl_path=pkl_path)],
            capture_output=True,
            text=True,
        )
        assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"
        assert "SUCCESS" in create_result.stdout

        load_result = subprocess.run(
            [sys.executable, "-c", load_code.format(pkl_path=pkl_path)],
            capture_output=True,
            text=True,
        )
        assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"
        assert "SUCCESS" in load_result.stdout


# =============================================================================
# Module-level classes should not be affected
# =============================================================================


class TestModuleLevelClassesUnaffected:
    """Tests verifying that module-level classes are not affected by local persistence."""

    def test_module_level_class_no_import_path(self):
        """Verify module-level classes don't get __ccflow_import_path__."""
        assert not hasattr(ModuleLevelModel, "__ccflow_import_path__")
        assert not hasattr(ModuleLevelContext, "__ccflow_import_path__")
        assert not hasattr(ModuleLevelCallable, "__ccflow_import_path__")

    def test_module_level_class_type_path_uses_qualname(self):
        """Verify module-level classes use standard module.qualname for type_."""
        instance = ModuleLevelModel(value=1)
        type_path = str(instance.type_)

        # Should use standard path, not ccflow.local_persistence._Local_...
        assert type_path == "ccflow.tests.test_local_persistence.ModuleLevelModel"
        assert "_Local_" not in type_path

    def test_module_level_standard_pickle_works(self):
        """Verify standard pickle works for module-level classes."""
        from pickle import dumps, loads

        instance = ModuleLevelModel(value=42)
        restored = loads(dumps(instance))
        assert restored.value == 42
        assert type(restored) is ModuleLevelModel


# =============================================================================
# Ray task tests
# =============================================================================


class TestRayTaskWithLocalClasses:
    """Tests for Ray task execution with locally-defined classes."""

    def test_local_callable_model_ray_task(self):
        """Test that locally-defined CallableModels can be sent to Ray tasks."""

        def create_local_callable():
            class LocalContext(ContextBase):
                x: int

            class LocalCallable(CallableModel):
                multiplier: int = 2

                @Flow.call
                def __call__(self, context: LocalContext) -> GenericResult:
                    return GenericResult(value=context.x * self.multiplier)

            return LocalContext, LocalCallable

        LocalContext, LocalCallable = create_local_callable()

        # Verify <locals> is in qualname (ensures cloudpickle serializes definition)
        assert "<locals>" in LocalCallable.__qualname__
        assert "<locals>" in LocalContext.__qualname__

        # Verify __ccflow_import_path__ is set
        assert hasattr(LocalCallable, "__ccflow_import_path__")
        assert hasattr(LocalContext, "__ccflow_import_path__")

        @ray.remote
        def run_callable(model, context):
            result = model(context)
            # Verify type_ works inside the Ray task (cross-process PyObjectPath)
            _ = model.type_
            _ = context.type_
            return result.value

        model = LocalCallable(multiplier=3)
        context = LocalContext(x=10)

        with ray.init(num_cpus=1):
            result = ray.get(run_callable.remote(model, context))

        assert result == 30

    def test_local_context_ray_task(self):
        """Test that locally-defined ContextBase can be sent to Ray tasks."""

        def create_local_context():
            class LocalContext(ContextBase):
                name: str
                value: int

            return LocalContext

        LocalContext = create_local_context()
        assert "<locals>" in LocalContext.__qualname__

        @ray.remote
        def process_context(ctx):
            # Access fields and type_ inside Ray task
            _ = ctx.type_
            return f"{ctx.name}:{ctx.value}"

        context = LocalContext(name="test", value=42)

        with ray.init(num_cpus=1):
            result = ray.get(process_context.remote(context))

        assert result == "test:42"

    def test_local_base_model_ray_task(self):
        """Test that locally-defined BaseModel can be sent to Ray tasks."""

        def create_local_model():
            class LocalModel(BaseModel):
                data: str

            return LocalModel

        LocalModel = create_local_model()
        assert "<locals>" in LocalModel.__qualname__

        @ray.remote
        def process_model(m):
            # Access type_ inside Ray task
            type_path = str(m.type_)
            return f"{m.data}|{type_path}"

        model = LocalModel(data="hello")

        with ray.init(num_cpus=1):
            result = ray.get(process_model.remote(model))

        assert result.startswith("hello|ccflow.local_persistence._Local_")


# =============================================================================
# UUID uniqueness tests
# =============================================================================


class TestUUIDUniqueness:
    """Tests verifying UUID-based naming provides uniqueness."""

    def test_multiple_local_classes_same_name_get_unique_paths(self):
        """Test that multiple local classes with same name get unique import paths."""

        def create_model_a():
            class SameName(BaseModel):
                value: int

            return SameName

        def create_model_b():
            class SameName(BaseModel):
                value: str

            return SameName

        ModelA = create_model_a()
        ModelB = create_model_b()

        # Both have same class name but different import paths
        assert ModelA.__name__ == ModelB.__name__ == "SameName"
        assert ModelA.__ccflow_import_path__ != ModelB.__ccflow_import_path__

        # Both should be accessible
        instance_a = ModelA(value=42)
        instance_b = ModelB(value="hello")
        assert instance_a.type_.object is ModelA
        assert instance_b.type_.object is ModelB

    def test_uuid_format_is_valid(self):
        """Test that the UUID portion of names is valid hex."""

        def create_class():
            class TestModel(BaseModel):
                x: int

            return TestModel

        Model = create_class()
        import_path = Model.__ccflow_import_path__

        # Extract UUID portion
        match = re.search(r"_Local_TestModel_([a-f0-9]+)$", import_path)
        assert match is not None, f"Import path doesn't match expected format: {import_path}"

        uuid_part = match.group(1)
        assert len(uuid_part) == 12, f"UUID should be 12 hex chars, got {len(uuid_part)}"
        assert all(c in "0123456789abcdef" for c in uuid_part)


# =============================================================================
# Nested class and inheritance tests
# =============================================================================


class OuterClass:
    """Module-level outer class for testing nested class importability."""

    class NestedModel(BaseModel):
        """A BaseModel nested inside a module-level class."""

        value: int


class TestNestedClasses:
    """Tests for classes nested inside other classes."""

    def test_nested_class_inside_module_level_class_not_registered(self):
        """Verify that a nested class inside a module-level class is NOT registered.

        Classes nested inside module-level classes (like OuterClass.NestedModel)
        have qualnames like 'OuterClass.NestedModel' without '<locals>' and ARE
        importable via the standard module.qualname path.
        """
        # The qualname has a '.' indicating nested class, but no '<locals>'
        assert "." in OuterClass.NestedModel.__qualname__
        assert OuterClass.NestedModel.__qualname__ == "OuterClass.NestedModel"
        assert "<locals>" not in OuterClass.NestedModel.__qualname__

        # Should NOT have __ccflow_import_path__
        assert "__ccflow_import_path__" not in OuterClass.NestedModel.__dict__

        # type_ should use standard path
        instance = OuterClass.NestedModel(value=42)
        type_path = str(instance.type_)
        assert type_path == "ccflow.tests.test_local_persistence.OuterClass.NestedModel"
        assert "_Local_" not in type_path
        assert instance.type_.object is OuterClass.NestedModel

    def test_nested_class_inside_function_is_registered(self):
        """Verify that a class nested inside a function-defined class IS registered."""

        def create_outer():
            class Outer:
                class Inner(BaseModel):
                    value: int

            return Outer

        Outer = create_outer()
        # The inner class has <locals> in its qualname (from the function)
        assert "<locals>" in Outer.Inner.__qualname__

        # Should be registered and have __ccflow_import_path__
        assert hasattr(Outer.Inner, "__ccflow_import_path__")


class TestInheritanceDoesNotPropagateImportPath:
    """Tests verifying that __ccflow_import_path__ is not inherited by subclasses."""

    def test_subclass_of_local_class_gets_own_registration(self):
        """Verify that subclassing a local class doesn't inherit __ccflow_import_path__."""

        def create_base():
            class LocalBase(BaseModel):
                value: int

            return LocalBase

        def create_derived(base_cls):
            class LocalDerived(base_cls):
                extra: str = "default"

            return LocalDerived

        Base = create_base()
        Derived = create_derived(Base)

        # Both should have __ccflow_import_path__ in their own __dict__
        assert "__ccflow_import_path__" in Base.__dict__
        assert "__ccflow_import_path__" in Derived.__dict__

        # They should have DIFFERENT import paths
        assert Base.__ccflow_import_path__ != Derived.__ccflow_import_path__

        # Both should be importable
        base_instance = Base(value=1)
        derived_instance = Derived(value=2, extra="test")

        assert base_instance.type_.object is Base
        assert derived_instance.type_.object is Derived

    def test_subclass_of_module_level_class_is_registered(self):
        """Verify that subclassing a module-level class inside a function creates a local class."""

        def create_subclass():
            class LocalSubclass(ModuleLevelModel):
                extra: str = "default"

            return LocalSubclass

        Subclass = create_subclass()

        # The subclass is local (defined in function), so needs registration
        assert "<locals>" in Subclass.__qualname__
        assert "__ccflow_import_path__" in Subclass.__dict__

        # But the parent should NOT have __ccflow_import_path__
        assert "__ccflow_import_path__" not in ModuleLevelModel.__dict__


# =============================================================================
# Generic types tests
# =============================================================================


class TestGenericTypes:
    """Tests for generic types and PyObjectPath."""

    def test_unparameterized_generic_type_path(self):
        """Test that unparameterized generic types work with type_."""
        from typing import Generic, TypeVar

        T = TypeVar("T")

        def create_generic():
            class GenericModel(BaseModel, Generic[T]):
                data: T  # Will be Any when unparameterized

            return GenericModel

        GenericModel = create_generic()

        # Create an unparameterized instance
        instance = GenericModel(data=42)

        # type_ should work
        type_path = str(instance.type_)
        assert "_Local_" in type_path
        assert "GenericModel" in type_path
        assert instance.type_.object is GenericModel

    def test_generic_base_class_is_registered(self):
        """Test that the unparameterized generic class is correctly registered.

        Note: Parameterized generics (e.g., GenericModel[int]) create new classes
        that lose the '<locals>' marker in their qualname due to Python/pydantic's
        generic handling. Use unparameterized generics for local class scenarios,
        or define concrete subclasses.
        """
        from typing import Generic, TypeVar

        T = TypeVar("T")

        def create_generic():
            class GenericModel(BaseModel, Generic[T]):
                data: T

            return GenericModel

        GenericModel = create_generic()

        # The unparameterized class should be registered
        assert "<locals>" in GenericModel.__qualname__
        assert hasattr(GenericModel, "__ccflow_import_path__")
        assert GenericModel.__ccflow_import_path__.startswith(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME)


# =============================================================================
# Import string tests
# =============================================================================


class TestImportString:
    """Tests for the import_string function."""

    def test_import_string_handles_nested_class_path(self):
        """Verify our import_string handles nested class paths that pydantic's ImportString cannot."""
        from pydantic import ImportString, TypeAdapter

        from ccflow.exttypes.pyobjectpath import import_string

        nested_path = "ccflow.tests.test_local_persistence.OuterClass.NestedModel"

        # Pydantic's ImportString fails on nested class paths
        pydantic_adapter = TypeAdapter(ImportString)
        with pytest.raises(Exception, match="No module named"):
            pydantic_adapter.validate_python(nested_path)

        # Our import_string handles it correctly
        result = import_string(nested_path)
        assert result is OuterClass.NestedModel

    def test_import_string_still_works_for_simple_paths(self):
        """Verify import_string still works for simple module.ClassName paths."""
        from ccflow.exttypes.pyobjectpath import import_string

        # Simple class path
        result = import_string("ccflow.tests.test_local_persistence.ModuleLevelModel")
        assert result is ModuleLevelModel

        # Built-in module
        result = import_string("os.path.join")
        import os.path

        assert result is os.path.join


# =============================================================================
# Registration strategy tests
# =============================================================================


class TestRegistrationStrategy:
    """Tests verifying the registration strategy for different class types."""

    def test_module_level_classes_not_registered(self):
        """Module-level classes should NOT have __ccflow_import_path__ set."""
        # ModuleLevelModel is defined at module level in this file
        assert "__ccflow_import_path__" not in ModuleLevelModel.__dict__
        assert "<locals>" not in ModuleLevelModel.__qualname__

        # Nested classes at module level also shouldn't need registration
        assert "__ccflow_import_path__" not in OuterClass.NestedModel.__dict__

    def test_local_class_registered_immediately(self):
        """Local classes (with <locals> in qualname) should be registered during definition."""
        from unittest import mock

        # Must patch where it's used (base.py), not where it's defined (local_persistence)
        with mock.patch("ccflow.base._register") as mock_do_reg:

            def create_local():
                class LocalModel(BaseModel):
                    value: int

                return LocalModel

            LocalModel = create_local()

            # _register SHOULD be called immediately for local classes
            mock_do_reg.assert_called_once()
            # Verify it has <locals> in qualname
            assert "<locals>" in LocalModel.__qualname__


# =============================================================================
# Tests for create_ccflow_model wrapper
# =============================================================================


class TestCreateCcflowModelWrapper:
    """Tests for the create_ccflow_model wrapper function."""

    def test_create_ccflow_model_basic(self):
        """Test basic create_ccflow_model usage with ContextBase."""
        DynamicContext = create_ccflow_model(
            "DynamicContext",
            __base__=ContextBase,
            name=(str, ...),
            value=(int, 0),
        )

        # Should be a valid ContextBase subclass
        assert issubclass(DynamicContext, ContextBase)

        # Should be registered
        assert hasattr(DynamicContext, "__ccflow_import_path__")
        assert DynamicContext.__ccflow_import_path__.startswith(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME)

        # Should be usable
        ctx = DynamicContext(name="test", value=42)
        assert ctx.name == "test"
        assert ctx.value == 42

    def test_create_ccflow_model_with_base_model(self):
        """Test create_ccflow_model with ccflow BaseModel as base."""
        DynamicModel = create_ccflow_model(
            "DynamicModel",
            __base__=BaseModel,
            data=(str, "default"),
            count=(int, 0),
        )

        assert issubclass(DynamicModel, BaseModel)
        assert hasattr(DynamicModel, "__ccflow_import_path__")

        instance = DynamicModel(data="hello", count=5)
        assert instance.data == "hello"
        assert instance.count == 5

    def test_create_ccflow_model_type_property_works(self):
        """Test that type_ property works for dynamically created models."""
        DynamicContext = create_ccflow_model(
            "DynamicContext",
            __base__=ContextBase,
            x=(int, ...),
        )

        ctx = DynamicContext(x=10)
        type_path = str(ctx.type_)

        # type_ should use __ccflow_import_path__
        assert type_path == DynamicContext.__ccflow_import_path__
        assert "_Local_" in type_path
        assert ctx.type_.object is DynamicContext

    def test_create_ccflow_model_can_be_imported(self):
        """Test that dynamically created models can be imported via their path."""
        import importlib

        DynamicModel = create_ccflow_model(
            "ImportableModel",
            __base__=BaseModel,
            value=(int, 0),
        )

        import_path = DynamicModel.__ccflow_import_path__
        parts = import_path.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        imported_cls = getattr(module, parts[1])

        assert imported_cls is DynamicModel

    def test_create_ccflow_model_with_docstring(self):
        """Test create_ccflow_model with custom docstring."""

        DynamicModel = create_ccflow_model(
            "DocumentedModel",
            __base__=BaseModel,
            __doc__="A dynamically created model for testing.",
            value=(int, 0),
        )

        assert DynamicModel.__doc__ == "A dynamically created model for testing."

    def test_create_ccflow_model_with_complex_fields(self):
        """Test create_ccflow_model with various field types."""
        from typing import List, Optional

        from pydantic import Field

        DynamicModel = create_ccflow_model(
            "ComplexModel",
            __base__=BaseModel,
            name=(str, ...),
            tags=(List[str], Field(default_factory=list)),
            description=(Optional[str], None),
            count=(int, 0),
        )

        instance = DynamicModel(name="test")
        assert instance.name == "test"
        assert instance.tags == []
        assert instance.description is None
        assert instance.count == 0

        instance2 = DynamicModel(name="test2", tags=["a", "b"], description="desc", count=5)
        assert instance2.tags == ["a", "b"]
        assert instance2.description == "desc"
        assert instance2.count == 5

    def test_create_ccflow_model_multiple_unique_names(self):
        """Test that multiple models with same name get unique registration paths."""

        Model1 = create_ccflow_model("SameName", __base__=BaseModel, value=(int, 0))
        Model2 = create_ccflow_model("SameName", __base__=BaseModel, value=(str, ""))

        # Both should be registered with different paths
        assert Model1.__ccflow_import_path__ != Model2.__ccflow_import_path__

        # Both should have the same __name__
        assert Model1.__name__ == Model2.__name__ == "SameName"

        # Both should be accessible via their own paths
        assert Model1(value=42).type_.object is Model1
        assert Model2(value="test").type_.object is Model2

    def test_create_ccflow_model_inherits_from_context_base(self):
        """Test that models inheriting from ContextBase have frozen config."""
        DynamicContext = create_ccflow_model(
            "FrozenContext",
            __base__=ContextBase,
            value=(int, 0),
        )

        ctx = DynamicContext(value=42)

        # ContextBase subclasses should be frozen
        with pytest.raises(Exception):  # ValidationError for frozen model
            ctx.value = 100


class TestCreateCcflowModelCloudpickleSameProcess:
    """Tests for cloudpickle with dynamically created models in the same process."""

    def test_cloudpickle_instance_roundtrip(self):
        """Test cloudpickle roundtrip for instances of dynamically created models."""
        from ray.cloudpickle import dumps, loads

        DynamicModel = create_ccflow_model(
            "PickleTestModel",
            __base__=BaseModel,
            value=(int, 0),
        )

        instance = DynamicModel(value=123)
        restored = loads(dumps(instance))

        assert restored.value == 123
        assert type(restored) is DynamicModel

    def test_cloudpickle_class_roundtrip(self):
        """Test cloudpickle roundtrip for dynamically created model classes."""
        from ray.cloudpickle import dumps, loads

        DynamicModel = create_ccflow_model(
            "ClassPickleModel",
            __base__=BaseModel,
            name=(str, ""),
        )

        restored_cls = loads(dumps(DynamicModel))
        assert restored_cls is DynamicModel

    def test_cloudpickle_preserves_type_path(self):
        """Test that type_ path is preserved after cloudpickle roundtrip."""
        from ray.cloudpickle import dumps, loads

        DynamicModel = create_ccflow_model(
            "TypePathModel",
            __base__=BaseModel,
            value=(int, 0),
        )

        instance = DynamicModel(value=42)
        original_path = str(instance.type_)

        restored = loads(dumps(instance))
        restored_path = str(restored.type_)

        assert restored_path == original_path


class TestCreateCcflowModelCloudpickleCrossProcess:
    """Tests for cross-process cloudpickle with dynamically created models (via create_ccflow_model)."""

    @pytest.mark.parametrize(
        "create_code,load_code",
        [
            pytest.param(
                # Context only
                """
from ray.cloudpickle import dump
from ccflow import ContextBase
from ccflow.local_persistence import create_ccflow_model

DynamicContext = create_ccflow_model(
    "CrossProcessContext",
    __base__=ContextBase,
    name=(str, ...),
    value=(int, 0),
)

assert hasattr(DynamicContext, "__ccflow_import_path__")

ctx = DynamicContext(name="test", value=42)
assert "_Local_" in str(ctx.type_)

with open("{pkl_path}", "wb") as f:
    dump(ctx, f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    ctx = load(f)

assert ctx.name == "test"
assert ctx.value == 42
assert "_Local_" in str(ctx.type_)
assert ctx.type_.object is type(ctx)
print("SUCCESS")
""",
                id="context_only",
            ),
            pytest.param(
                # Dynamic context with CallableModel
                """
from ray.cloudpickle import dump
from ccflow import CallableModel, ContextBase, GenericResult, Flow
from ccflow.local_persistence import create_ccflow_model

DynamicContext = create_ccflow_model(
    "CallableModelContext",
    __base__=ContextBase,
    x=(int, ...),
    multiplier=(int, 2),
)

def create_callable():
    class DynamicCallable(CallableModel):
        @Flow.call
        def __call__(self, context: DynamicContext) -> GenericResult:
            return GenericResult(value=context.x * context.multiplier)
    return DynamicCallable

DynamicCallable = create_callable()
model = DynamicCallable()
ctx = DynamicContext(x=10, multiplier=3)
result = model(ctx)
assert result.value == 30

with open("{pkl_path}", "wb") as f:
    dump((model, ctx), f)
print("SUCCESS")
""",
                """
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    model, ctx = load(f)

result = model(ctx)
assert result.value == 30
assert ctx.type_.object is type(ctx)
assert model.type_.object is type(model)
print("SUCCESS")
""",
                id="with_callable",
            ),
        ],
    )
    def test_create_ccflow_model_cross_process(self, pickle_file, create_code, load_code):
        """Test that dynamically created models work across processes."""
        pkl_path = pickle_file

        create_result = subprocess.run(
            [sys.executable, "-c", create_code.format(pkl_path=pkl_path)],
            capture_output=True,
            text=True,
        )
        assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"
        assert "SUCCESS" in create_result.stdout

        load_result = subprocess.run(
            [sys.executable, "-c", load_code.format(pkl_path=pkl_path)],
            capture_output=True,
            text=True,
        )
        assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"
        assert "SUCCESS" in load_result.stdout


class TestCreateCcflowModelRayTask:
    """Tests for Ray task execution with dynamically created models."""

    def test_create_ccflow_model_ray_task(self):
        """Test that dynamically created models work in Ray tasks."""

        DynamicContext = create_ccflow_model(
            "RayTaskContext",
            __base__=ContextBase,
            name=(str, ...),
            value=(int, 0),
        )

        @ray.remote
        def process_context(ctx):
            # Access fields and type_ inside Ray task
            _ = ctx.type_
            return f"{ctx.name}:{ctx.value}"

        ctx = DynamicContext(name="ray_test", value=99)

        with ray.init(num_cpus=1):
            result = ray.get(process_context.remote(ctx))

        assert result == "ray_test:99"

    def test_create_ccflow_model_callable_model_ray_task(self):
        """Test CallableModel with dynamically created context in Ray tasks."""

        DynamicContext = create_ccflow_model(
            "RayCallableContext",
            __base__=ContextBase,
            x=(int, ...),
        )

        class RayCallable(CallableModel):
            factor: int = 2

            @Flow.call
            def __call__(self, context: DynamicContext) -> GenericResult:
                return GenericResult(value=context.x * self.factor)

        @ray.remote
        def run_callable(model, ctx):
            result = model(ctx)
            # Verify type_ works in Ray worker
            _ = model.type_
            _ = ctx.type_
            return result.value

        model = RayCallable(factor=5)
        ctx = DynamicContext(x=10)

        with ray.init(num_cpus=1):
            result = ray.get(run_callable.remote(model, ctx))

        assert result == 50


class TestCreateCcflowModelEdgeCases:
    """Tests for edge cases in create_ccflow_model wrapper."""

    def test_create_ccflow_model_no_fields(self):
        """Test create_ccflow_model with no custom fields."""

        EmptyModel = create_ccflow_model("EmptyModel", __base__=BaseModel)

        assert issubclass(EmptyModel, BaseModel)
        assert hasattr(EmptyModel, "__ccflow_import_path__")

        instance = EmptyModel()
        assert instance.type_.object is EmptyModel

    def test_create_ccflow_model_with_module_override(self):
        """Test create_ccflow_model with __module__ parameter."""

        CustomModuleModel = create_ccflow_model(
            "CustomModuleModel",
            __base__=BaseModel,
            __module__="custom.module.path",
            value=(int, 0),
        )

        # Module should be overridden
        assert CustomModuleModel.__module__ == "custom.module.path"

        # But should still be registered since it's not actually importable
        assert hasattr(CustomModuleModel, "__ccflow_import_path__")

    def test_create_ccflow_model_inheritance_from_custom_base(self):
        """Test create_ccflow_model inheriting from a custom ccflow class."""

        # First create a base class
        class CustomBase(ContextBase):
            base_field: str = "base"

        DerivedModel = create_ccflow_model(
            "DerivedModel",
            __base__=CustomBase,
            derived_field=(int, 0),
        )

        assert issubclass(DerivedModel, CustomBase)
        assert issubclass(DerivedModel, ContextBase)

        instance = DerivedModel(derived_field=42)
        assert instance.base_field == "base"
        assert instance.derived_field == 42

    def test_create_ccflow_model_special_characters_in_name(self):
        """Test create_ccflow_model handles special characters in name."""

        # Names with special characters should still work
        SpecialModel = create_ccflow_model(
            "Model-With-Dashes",
            __base__=BaseModel,
            value=(int, 0),
        )

        assert hasattr(SpecialModel, "__ccflow_import_path__")

        # The registered name should be sanitized
        import_path = SpecialModel.__ccflow_import_path__
        registered_name = import_path.split(".")[-1]
        # Should have sanitized the dashes
        assert "-" not in registered_name

    def test_create_ccflow_model_returns_correct_type(self):
        """Test that create_ccflow_model returns the model class, not an instance."""

        result = create_ccflow_model(
            "TypeCheckModel",
            __base__=BaseModel,
            value=(int, 0),
        )

        assert isinstance(result, type)
        assert issubclass(result, BaseModel)

    def test_create_ccflow_model_import_at_top_level(self):
        """Test that create_ccflow_model can be imported from ccflow."""
        from ccflow import create_ccflow_model as ccflow_create_model
        from ccflow.local_persistence import create_ccflow_model as lp_create_model

        # Both should be the same function
        assert ccflow_create_model is lp_create_model

        # And both should work
        Model = ccflow_create_model("TopLevelImportModel", __base__=BaseModel, value=(int, 0))
        assert hasattr(Model, "__ccflow_import_path__")
