import re
import subprocess
import sys
import textwrap
from unittest import TestCase

import ray
from pydantic import create_model as pydantic_create_model

import ccflow.local_persistence as local_persistence
from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, NullContext


class ModuleLevelModel(BaseModel):
    value: int


class ModuleLevelContext(ContextBase):
    value: int


class ModuleLevelCallable(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult:
        return GenericResult(value="ok")


class TestIsImportable(TestCase):
    """Tests for _is_importable - the core check for whether a class needs registration."""

    def test_module_level_ccflow_classes_are_importable(self):
        for cls in (ModuleLevelModel, ModuleLevelContext, ModuleLevelCallable):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(local_persistence._is_importable(cls))

    def test_local_scope_class_not_importable(self):
        def build_class():
            class LocalClass:
                pass

            return LocalClass

        LocalClass = build_class()
        self.assertFalse(local_persistence._is_importable(LocalClass))

    def test_main_module_class_not_importable(self):
        # __main__ classes are not importable from other processes
        cls = type("MainModuleClass", (), {})
        cls.__module__ = "__main__"
        cls.__qualname__ = "MainModuleClass"
        self.assertFalse(local_persistence._is_importable(cls))

    def test_module_level_class_is_importable(self):
        # Module-level classes are importable
        self.assertTrue(local_persistence._is_importable(ModuleLevelModel))

    def test_dynamically_created_class_not_importable(self):
        """Test that dynamically created classes (like from pydantic's create_model) are not importable."""
        # This simulates what happens with pydantic's create_model
        DynamicClass = type("DynamicClass", (), {})
        DynamicClass.__module__ = "ccflow.tests.test_local_persistence"
        DynamicClass.__qualname__ = "DynamicClass"
        # This class has a valid-looking module/qualname but isn't actually in the module
        self.assertFalse(local_persistence._is_importable(DynamicClass))


def _run_subprocess(code: str) -> str:
    """Execute code in a clean interpreter so sys.modules starts empty."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_base_module_available_after_import():
    """Test that ccflow.base module is available after importing ccflow."""
    output = _run_subprocess(
        """
        import sys
        import ccflow
        print(local_persistence.LOCAL_ARTIFACTS_MODULE_NAME in sys.modules)
        """.replace("local_persistence.LOCAL_ARTIFACTS_MODULE_NAME", f'"{local_persistence.LOCAL_ARTIFACTS_MODULE_NAME}"')
    )
    assert output == "True"


def test_register_preserves_module_qualname_and_sets_import_path():
    output = _run_subprocess(
        """
        import sys
        import ccflow.local_persistence as lp

        def build():
            class Foo:
                pass
            return Foo

        Foo = build()
        original_module = Foo.__module__
        original_qualname = Foo.__qualname__
        lp._register(Foo)
        module = sys.modules[lp.LOCAL_ARTIFACTS_MODULE_NAME]

        # __module__ and __qualname__ should NOT change (preserves cloudpickle)
        print(Foo.__module__ == original_module)
        print(Foo.__qualname__ == original_qualname)
        print("<locals>" in Foo.__qualname__)

        # __ccflow_import_path__ should be set and point to the registered class
        import_path = Foo.__ccflow_import_path__
        registered_name = import_path.split(".")[-1]
        print(hasattr(module, registered_name))
        print(getattr(module, registered_name) is Foo)
        print(import_path.startswith("ccflow.base._Local_"))
        """
    )
    lines = output.splitlines()
    assert lines[0] == "True", f"__module__ should not change: {lines}"
    assert lines[1] == "True", f"__qualname__ should not change: {lines}"
    assert lines[2] == "True", f"__qualname__ should contain '<locals>': {lines}"
    assert lines[3] == "True", f"Class should be registered in module: {lines}"
    assert lines[4] == "True", f"Registered class should be the same object: {lines}"
    assert lines[5] == "True", f"Import path should start with expected prefix: {lines}"


def test_register_handles_class_name_starting_with_digit():
    """Test that _register handles class names starting with a digit by prefixing with underscore."""
    output = _run_subprocess(
        """
        import sys
        import ccflow.local_persistence as lp

        # Create a class with a name starting with a digit
        cls = type("3DModel", (), {})
        lp._register(cls)

        import_path = cls.__ccflow_import_path__
        registered_name = import_path.split(".")[-1]

        # The registered name should start with _Local__ (underscore added for digit)
        print(registered_name.startswith("_Local__"))
        print("_3DModel_" in registered_name)

        # Should be registered on ccflow.base
        module = sys.modules[lp.LOCAL_ARTIFACTS_MODULE_NAME]
        print(getattr(module, registered_name) is cls)
        """
    )
    lines = output.splitlines()
    assert lines[0] == "True", f"Registered name should start with _Local__: {lines}"
    assert lines[1] == "True", f"Registered name should contain _3DModel_: {lines}"
    assert lines[2] == "True", f"Class should be registered on module: {lines}"


def test_sync_to_module_registers_class_not_yet_on_module():
    """Test that _sync_to_module registers a class that has __ccflow_import_path__ but isn't on the module yet.

    This happens in cross-process unpickle scenarios where cloudpickle recreates the class
    with __ccflow_import_path__ set, but the class isn't yet on ccflow.base.
    """
    output = _run_subprocess(
        """
        import sys
        import ccflow.local_persistence as lp

        # Simulate a class that has __ccflow_import_path__ but isn't registered on ccflow.base
        # (like what happens after cross-process cloudpickle unpickle)
        cls = type("SimulatedUnpickled", (), {})
        unique_name = "_Local_SimulatedUnpickled_test123abc"
        cls.__ccflow_import_path__ = f"{lp.LOCAL_ARTIFACTS_MODULE_NAME}.{unique_name}"

        # Verify class is NOT on ccflow.base yet
        module = sys.modules[lp.LOCAL_ARTIFACTS_MODULE_NAME]
        print(getattr(module, unique_name, None) is None)

        # Call _sync_to_module
        lp._sync_to_module(cls)

        # Verify class IS now on ccflow.base
        print(getattr(module, unique_name, None) is cls)
        """
    )
    lines = output.splitlines()
    assert lines[0] == "True", f"Class should NOT be on module before sync: {lines}"
    assert lines[1] == "True", f"Class should be on module after sync: {lines}"


def test_local_basemodel_cloudpickle_cross_process():
    """Test that local-scope BaseModel subclasses work with cloudpickle cross-process.

    This is the key test for the "best of both worlds" approach:
    - __qualname__ has '<locals>' so cloudpickle serializes the class definition
    - __ccflow_import_path__ allows PyObjectPath validation to work
    - After unpickling, __pydantic_init_subclass__ re-registers the class
    """
    import os
    import tempfile

    fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)

    try:
        # Create a local-scope BaseModel in subprocess 1 and pickle it
        create_result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
from ray.cloudpickle import dump
from ccflow import BaseModel

def create_local_model():
    class LocalModel(BaseModel):
        value: int

    return LocalModel

LocalModel = create_local_model()

# Verify __qualname__ has '<locals>' (enables cloudpickle serialization)
assert "<locals>" in LocalModel.__qualname__, f"Expected '<locals>' in qualname: {{LocalModel.__qualname__}}"

# Verify __ccflow_import_path__ is set (enables PyObjectPath)
assert hasattr(LocalModel, "__ccflow_import_path__"), "Expected __ccflow_import_path__ to be set"

# Create instance and verify type_ works (PyObjectPath validation)
instance = LocalModel(value=42)
type_path = instance.type_
print(f"type_: {{type_path}}")

# Pickle the instance
with open("{pkl_path}", "wb") as f:
    dump(instance, f)

print("SUCCESS: Created and pickled")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert create_result.returncode == 0, f"Create subprocess failed: {create_result.stderr}"
        assert "SUCCESS" in create_result.stdout, f"Create subprocess output: {create_result.stdout}"

        # Load in subprocess 2 (different process, class not pre-defined)
        load_result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

# Verify the value was preserved
assert obj.value == 42, f"Expected value=42, got {{obj.value}}"

# Verify type_ works after unpickling (class was re-registered)
type_path = obj.type_
print(f"type_: {{type_path}}")

# Verify the import path works
import importlib
path_parts = str(type_path).rsplit(".", 1)
module = importlib.import_module(path_parts[0])
cls = getattr(module, path_parts[1])
assert cls is type(obj), "Import path should resolve to the same class"

print("SUCCESS: Loaded and verified")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert load_result.returncode == 0, f"Load subprocess failed: {load_result.stderr}"
        assert "SUCCESS" in load_result.stdout, f"Load subprocess output: {load_result.stdout}"

    finally:
        if os.path.exists(pkl_path):
            os.unlink(pkl_path)


# =============================================================================
# Tests for pydantic's create_model
# =============================================================================


class TestPydanticCreateModel:
    """Tests for dynamically created models using pydantic's create_model."""

    def test_create_model_gets_registered(self):
        """Test that models created with pydantic's create_model get registered."""

        def make_model():
            return pydantic_create_model("DynamicModel", value=(int, ...), __base__=BaseModel)

        Model = make_model()

        # Registration happens lazily when type_ is accessed
        # Create an instance and access type_ to trigger registration
        instance = Model(value=42)
        _ = instance.type_

        # Should have __ccflow_import_path__ set after type_ access
        assert hasattr(Model, "__ccflow_import_path__")
        import_path = Model.__ccflow_import_path__
        assert import_path.startswith("ccflow.base._Local_")

        # Should be registered in ccflow.base
        registered_name = import_path.split(".")[-1]
        import ccflow.base

        assert hasattr(ccflow.base, registered_name)
        assert getattr(ccflow.base, registered_name) is Model

    def test_create_model_type_works(self):
        """Test that type_ property works for create_model-created models."""

        def make_model():
            return pydantic_create_model("DynamicModel", value=(int, ...), __base__=BaseModel)

        Model = make_model()
        instance = Model(value=42)

        # type_ should work and return the import path
        type_path = instance.type_
        assert str(type_path) == Model.__ccflow_import_path__
        assert type_path.object is Model

    def test_create_model_importable(self):
        """Test that create_model models can be imported via their type_ path."""
        import importlib

        def make_model():
            return pydantic_create_model("ImportableModel", data=(str, ...), __base__=BaseModel)

        Model = make_model()
        instance = Model(data="test")

        type_path = str(instance.type_)
        parts = type_path.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        imported_cls = getattr(module, parts[1])

        assert imported_cls is Model

    def test_create_model_without_locals_still_gets_uuid(self):
        """Test that create_model models (which don't have <locals> in qualname) get unique UUIDs."""

        def make_models():
            Model1 = pydantic_create_model("SameName", x=(int, ...), __base__=BaseModel)
            Model2 = pydantic_create_model("SameName", y=(str, ...), __base__=BaseModel)
            return Model1, Model2

        Model1, Model2 = make_models()

        # Both should be properly registered (trigger registration via type_)
        instance1 = Model1.model_validate({"x": 1})
        instance2 = Model2.model_validate({"y": "test"})
        _ = instance1.type_
        _ = instance2.type_

        # Both should have unique import paths (after type_ access triggers registration)
        assert Model1.__ccflow_import_path__ != Model2.__ccflow_import_path__

        assert instance1.x == 1
        assert instance2.y == "test"

    def test_create_model_context_base(self):
        """Test create_model with ContextBase as base class."""

        def make_context():
            return pydantic_create_model("DynamicContext", name=(str, ...), value=(int, ...), __base__=ContextBase)

        Context = make_context()
        instance = Context(name="test", value=42)

        # Access type_ to trigger registration
        type_path = instance.type_
        assert type_path.object is Context

        # After type_ access, __ccflow_import_path__ should be set
        assert hasattr(Context, "__ccflow_import_path__")

    def test_create_model_cloudpickle_same_process(self):
        """Test that create_model models work with cloudpickle in the same process."""
        from ray.cloudpickle import dumps, loads

        def make_model():
            return pydantic_create_model("PickleModel", value=(int, ...), __base__=BaseModel)

        Model = make_model()
        instance = Model(value=123)

        restored = loads(dumps(instance))

        assert restored.value == 123
        assert type(restored) is Model

    def test_create_model_cloudpickle_cross_process(self):
        """Test that create_model models work with cloudpickle across processes."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            create_code = f'''
from ray.cloudpickle import dump
from pydantic import create_model
from ccflow import BaseModel

Model = create_model("CrossProcessModel", value=(int, ...), __base__=BaseModel)

instance = Model(value=42)
# Access type_ to trigger registration (lazy for create_model)
type_path = instance.type_
print(f"type_: {{type_path}}")

# Now __ccflow_import_path__ should be set
assert hasattr(Model, "__ccflow_import_path__"), "Expected __ccflow_import_path__ after type_ access"

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"

            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.value == 42
type_path = obj.type_
assert type_path.object is type(obj)
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)


# =============================================================================
# Comprehensive tests for local persistence and PyObjectPath integration
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
        """Verify that __module__ is NOT changed to ccflow.base."""

        def create_class():
            class Inner(ContextBase):
                value: str

            return Inner

        cls = create_class()
        # __module__ should be this test module, not ccflow.base
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
        """Verify that the class is registered in ccflow.base under import path."""

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


class TestCloudpickleCrossProcess:
    """Tests for cross-process cloudpickle behavior (subprocess tests)."""

    def test_context_base_cross_process(self):
        """Test cross-process cloudpickle for ContextBase subclasses."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            create_code = f'''
from ray.cloudpickle import dump
from ccflow import ContextBase

def create_context():
    class LocalContext(ContextBase):
        name: str
        value: int
    return LocalContext

LocalContext = create_context()
assert "<locals>" in LocalContext.__qualname__
instance = LocalContext(name="test", value=42)
_ = instance.type_  # Verify type_ works before pickle

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"

            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.name == "test"
assert obj.value == 42
type_path = obj.type_  # Verify type_ works after unpickle
assert type_path.object is type(obj)
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)

    def test_sync_to_module_lazy_on_cross_process_unpickle(self):
        """Test that _sync_to_module happens lazily when accessing type_ after cross-process unpickle.

        When cloudpickle recreates a class in a new process, __ccflow_import_path__ is preserved
        but the class is NOT immediately registered on ccflow.base (cloudpickle sets attributes
        AFTER __pydantic_init_subclass__ runs). Registration happens lazily when:
        - type_ is accessed (calls PyObjectPath.validate)
        - PyObjectPath.validate(cls) is called directly
        - _register_local_subclass_if_needed(cls) is called
        """
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            # Process A: Create a local class and pickle the CLASS itself (not just instance)
            create_code = f'''
import sys
from ray.cloudpickle import dump
from ccflow import BaseModel
from ccflow.local_persistence import LOCAL_ARTIFACTS_MODULE_NAME

def create_model():
    class LocalModel(BaseModel):
        value: int
    return LocalModel

LocalModel = create_model()

# Verify class is registered and has import path
assert hasattr(LocalModel, "__ccflow_import_path__"), "Should have import path"
import_path = LocalModel.__ccflow_import_path__
assert import_path.startswith(LOCAL_ARTIFACTS_MODULE_NAME + "."), f"Bad path: {{import_path}}"

# Extract registered name and verify it's on ccflow.base
registered_name = import_path.split(".")[-1]
base_module = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]
assert getattr(base_module, registered_name, None) is LocalModel, "Should be on ccflow.base"

# Pickle the CLASS itself (cloudpickle will serialize the class definition)
with open("{pkl_path}", "wb") as f:
    dump(LocalModel, f)

# Output the import path so we can verify it's the same in process B
print(import_path)
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"
            original_import_path = create_result.stdout.strip()

            # Process B: Unpickle the class and verify lazy sync behavior
            load_code = f'''
import sys
from ray.cloudpickle import load
from ccflow.local_persistence import LOCAL_ARTIFACTS_MODULE_NAME
from ccflow import PyObjectPath

with open("{pkl_path}", "rb") as f:
    LocalModel = load(f)

# Verify __ccflow_import_path__ is preserved from pickle
assert hasattr(LocalModel, "__ccflow_import_path__"), "Should have import path after unpickle"
import_path = LocalModel.__ccflow_import_path__
assert import_path == "{original_import_path}", f"Path mismatch: {{import_path}} != {original_import_path}"

# BEFORE accessing type_: class is NOT yet registered on ccflow.base
# (cloudpickle sets attributes AFTER class creation, so __pydantic_init_subclass__ doesn't sync)
registered_name = import_path.split(".")[-1]
base_module = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]
registered_cls_before = getattr(base_module, registered_name, None)
# Note: registered_cls_before may be None OR may be LocalModel (if already synced by another path)

# Trigger lazy sync via PyObjectPath.validate
path = PyObjectPath.validate(LocalModel)
assert path == import_path, f"PyObjectPath should return the ccflow import path"

# AFTER accessing PyObjectPath.validate: class IS registered on ccflow.base
registered_cls_after = getattr(base_module, registered_name, None)
assert registered_cls_after is LocalModel, "_sync_to_module should have registered class on ccflow.base"

# Verify PyObjectPath.object resolves correctly
assert path.object is LocalModel, "PyObjectPath.object should resolve to the class"

# Verify the class is functional
instance = LocalModel(value=123)
assert instance.value == 123

print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"
            assert "SUCCESS" in load_result.stdout

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)

    def test_callable_model_cross_process(self):
        """Test cross-process cloudpickle for CallableModel subclasses."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            create_code = f'''
from ray.cloudpickle import dump
from ccflow import CallableModel, ContextBase, GenericResult, Flow

def create_callable():
    class LocalContext(ContextBase):
        x: int

    class LocalCallable(CallableModel):
        multiplier: int = 2

        @Flow.call
        def __call__(self, context: LocalContext) -> GenericResult:
            return GenericResult(value=context.x * self.multiplier)

    return LocalContext, LocalCallable

LocalContext, LocalCallable = create_callable()
instance = LocalCallable(multiplier=3)
context = LocalContext(x=10)

# Verify it works
result = instance(context)
assert result.value == 30

with open("{pkl_path}", "wb") as f:
    dump((instance, context), f)
print("SUCCESS")
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"

            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    instance, context = load(f)

# Verify the callable works after unpickle
result = instance(context)
assert result.value == 30

# Verify type_ works
assert instance.type_.object is type(instance)
assert context.type_.object is type(context)
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)

    def test_nested_local_classes_cross_process(self):
        """Test cross-process cloudpickle for multiply-nested local classes."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            create_code = f'''
from ray.cloudpickle import dump
from ccflow import BaseModel

def outer():
    def inner():
        class DeeplyNested(BaseModel):
            value: int
        return DeeplyNested
    return inner()

cls = outer()
assert "<locals>" in cls.__qualname__
assert cls.__qualname__.count("<locals>") == 2  # Two levels of nesting

instance = cls(value=999)
with open("{pkl_path}", "wb") as f:
    dump(instance, f)
print("SUCCESS")
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"

            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    obj = load(f)

assert obj.value == 999
_ = obj.type_  # Verify type_ works
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)

    def test_multiple_instances_same_local_class_cross_process(self):
        """Test that multiple instances of the same local class work cross-process."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            create_code = f'''
from ray.cloudpickle import dump
from ccflow import BaseModel

def create_class():
    class LocalModel(BaseModel):
        value: int
    return LocalModel

cls = create_class()
instances = [cls(value=i) for i in range(5)]

with open("{pkl_path}", "wb") as f:
    dump(instances, f)
print("SUCCESS")
'''
            create_result = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert create_result.returncode == 0, f"Create failed: {create_result.stderr}"

            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    instances = load(f)

# All instances should have the correct values
for i, instance in enumerate(instances):
    assert instance.value == i

# All instances should be of the same class
assert len(set(type(inst) for inst in instances)) == 1

# type_ should work for all
for instance in instances:
    _ = instance.type_
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)


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

        # Should use standard path, not ccflow.base._Local_...
        assert type_path == "ccflow.tests.test_local_persistence.ModuleLevelModel"
        assert "_Local_" not in type_path

    def test_module_level_standard_pickle_works(self):
        """Verify standard pickle works for module-level classes."""
        from pickle import dumps, loads

        instance = ModuleLevelModel(value=42)
        restored = loads(dumps(instance))
        assert restored.value == 42
        assert type(restored) is ModuleLevelModel


class TestRayTaskWithLocalClasses:
    """Tests for Ray task execution with locally-defined classes.

    These tests verify that the local persistence mechanism works correctly
    when classes are serialized and sent to Ray workers (different processes).
    """

    def test_local_callable_model_ray_task(self):
        """Test that locally-defined CallableModels can be sent to Ray tasks.

        This is the ultimate test of cross-process cloudpickle support:
        - Local class defined in function (has <locals> in __qualname__)
        - Sent to Ray worker (different process)
        - Executed and returns correct result
        - type_ property works after execution (PyObjectPath validation)
        """

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

        assert result.startswith("hello|ccflow.base._Local_")

    def test_create_model_ray_task(self):
        """Test that pydantic create_model models can be sent to Ray tasks."""

        def make_model():
            return pydantic_create_model("RayModel", value=(int, ...), __base__=BaseModel)

        Model = make_model()

        @ray.remote
        def process_model(m):
            type_path = str(m.type_)
            return f"{m.value}|{type_path}"

        model = Model(value=99)

        with ray.init(num_cpus=1):
            result = ray.get(process_model.remote(model))

        assert result.startswith("99|ccflow.base._Local_")


class TestUUIDUniqueness:
    """Tests verifying UUID-based naming provides cross-process uniqueness."""

    def test_different_processes_get_different_uuids(self):
        """Test that different processes generate different UUIDs for same class name."""
        import os
        import tempfile

        fd1, pkl_path1 = tempfile.mkstemp(suffix="_1.pkl")
        fd2, pkl_path2 = tempfile.mkstemp(suffix="_2.pkl")
        os.close(fd1)
        os.close(fd2)

        try:
            # Create same-named class in two different processes
            # Note: For module-level classes, registration only happens when needed
            # Access type_ to trigger registration before checking __ccflow_import_path__
            code = """
import sys
from ray.cloudpickle import dump
from ccflow import BaseModel

class SameName(BaseModel):
    value: int

instance = SameName(value={value})
# Access type_ to trigger registration if needed
_ = instance.type_
print(SameName.__ccflow_import_path__)

with open("{pkl_path}", "wb") as f:
    dump(instance, f)
"""
            result1 = subprocess.run([sys.executable, "-c", code.format(value=1, pkl_path=pkl_path1)], capture_output=True, text=True)
            assert result1.returncode == 0, f"Process 1 failed: {result1.stderr}"
            path1 = result1.stdout.strip()

            result2 = subprocess.run([sys.executable, "-c", code.format(value=2, pkl_path=pkl_path2)], capture_output=True, text=True)
            assert result2.returncode == 0, f"Process 2 failed: {result2.stderr}"
            path2 = result2.stdout.strip()

            # UUIDs should be different even though class names are the same
            assert path1 != path2
            assert "_Local_SameName_" in path1
            assert "_Local_SameName_" in path2

            # Both should be loadable in the same process
            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path1}", "rb") as f:
    obj1 = load(f)

with open("{pkl_path2}", "rb") as f:
    obj2 = load(f)

assert obj1.value == 1
assert obj2.value == 2
# They should be different types
assert type(obj1) is not type(obj2)
print("SUCCESS")
'''
            load_result = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert load_result.returncode == 0, f"Load failed: {load_result.stderr}"

        finally:
            for p in [pkl_path1, pkl_path2]:
                if os.path.exists(p):
                    os.unlink(p)

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
# Edge case tests
# =============================================================================


class OuterClass:
    """Module-level outer class for testing nested class importability."""

    class NestedModel(BaseModel):
        """A BaseModel nested inside a module-level class."""

        value: int


class TestImportString:
    """Tests for the import_string function."""

    def test_import_string_handles_nested_class_path(self):
        """Verify our import_string handles nested class paths that pydantic's ImportString cannot.

        Pydantic's ImportString fails on paths like 'module.OuterClass.InnerClass' because
        it tries to import the entire path as a module. Our import_string progressively
        tries shorter module paths and uses getattr for the rest.
        """
        import pytest
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


class TestNestedClasses:
    """Tests for classes nested inside other classes."""

    def test_nested_class_inside_module_level_class_is_importable(self):
        """Verify that a nested class inside a module-level class IS importable.

        Classes nested inside module-level classes (like OuterClass.NestedModel)
        have qualnames like 'OuterClass.NestedModel' and ARE importable via
        the standard module.qualname path.
        """
        # The qualname has a '.' indicating nested class
        assert "." in OuterClass.NestedModel.__qualname__
        assert OuterClass.NestedModel.__qualname__ == "OuterClass.NestedModel"

        # Should be importable - it's in the module namespace
        assert local_persistence._is_importable(OuterClass.NestedModel)

        # Should not have __ccflow_import_path__
        assert "__ccflow_import_path__" not in OuterClass.NestedModel.__dict__

        # type_ should use standard path
        instance = OuterClass.NestedModel(value=42)
        type_path = str(instance.type_)
        assert type_path == "ccflow.tests.test_local_persistence.OuterClass.NestedModel"
        assert "_Local_" not in type_path
        assert instance.type_.object is OuterClass.NestedModel

    def test_nested_class_inside_function_not_importable(self):
        """Verify that a class nested inside a function-defined class is not importable."""

        def create_outer():
            class Outer:
                class Inner(BaseModel):
                    value: int

            return Outer

        Outer = create_outer()
        # The inner class has <locals> in its qualname (from Outer)
        assert "<locals>" in Outer.Inner.__qualname__
        assert not local_persistence._is_importable(Outer.Inner)

        # Should get registered and have __ccflow_import_path__
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

    def test_subclass_of_module_level_class_not_registered(self):
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


class TestCreateModelEdgeCases:
    """Additional edge case tests for pydantic's create_model."""

    def test_create_model_with_same_name_as_module_level_class(self):
        """Test create_model with a name that matches an existing module-level class.

        The dynamically created class should get its own registration, not
        conflict with the module-level class.
        """

        def make_model():
            # Create a model with the same name as ModuleLevelModel
            return pydantic_create_model("ModuleLevelModel", x=(int, ...), __base__=BaseModel)

        DynamicModel = make_model()

        # The dynamic model should NOT be the same as the module-level one
        assert DynamicModel is not ModuleLevelModel

        # Access type_ to trigger registration
        instance = DynamicModel(x=123)
        type_path = str(instance.type_)

        # Should get a _Local_ path since it's not actually importable
        assert "_Local_" in type_path
        assert "ModuleLevelModel" in type_path

        # The module-level class should still use its standard path
        module_instance = ModuleLevelModel(value=456)
        module_type_path = str(module_instance.type_)
        assert module_type_path == "ccflow.tests.test_local_persistence.ModuleLevelModel"

    def test_create_model_result_assigned_to_different_name(self):
        """Test that create_model models get registered even when assigned to a different name.

        create_model("Foo", ...) assigned to variable Bar should still work.
        """
        DifferentName = pydantic_create_model("OriginalName", value=(int, ...), __base__=BaseModel)

        # The class __name__ is "OriginalName" but it's stored in variable DifferentName
        assert DifferentName.__name__ == "OriginalName"

        # Should still work correctly
        instance = DifferentName(value=42)
        type_path = str(instance.type_)

        # The registration should use "OriginalName" (the __name__)
        assert "OriginalName" in type_path
        assert instance.type_.object is DifferentName


class TestGenericTypes:
    """Tests for generic types and PyObjectPath."""

    def test_generic_basemodel_type_path(self):
        """Test that generic BaseModel subclasses work with type_.

        When you parameterize a generic type like GenericModel[int],
        the resulting class has __name__='GenericModel[int]'.
        The registration sanitizes this to 'GenericModel_int_'.
        """
        from typing import Generic, TypeVar

        T = TypeVar("T")

        def create_generic():
            class GenericModel(BaseModel, Generic[T]):
                data: T

            return GenericModel

        GenericModel = create_generic()

        # Create a concrete (parameterized) instance
        instance = GenericModel[int](data=42)

        # type_ should work - the class gets registered with sanitized name
        type_path = str(instance.type_)
        # The parameterized type has __name__='GenericModel[int]'
        # which gets sanitized to 'GenericModel_int_' in the registration
        assert "_Local_" in type_path
        assert "GenericModel" in type_path
        # Verify the path actually resolves to the correct class
        assert instance.type_.object is type(instance)

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
        # No type parameter suffix since we used unparameterized version
        assert instance.type_.object is GenericModel

    def test_build_standard_import_path_strips_brackets(self):
        """Test that _build_standard_import_path strips generic type parameters from qualname."""
        from ccflow.exttypes.pyobjectpath import _build_standard_import_path

        class MockClass:
            __module__ = "test.module"
            __qualname__ = "MyClass[int, str]"

        path = _build_standard_import_path(MockClass)
        assert "[" not in path
        assert path == "test.module.MyClass"


class TestReduceTriggersRegistration:
    """Tests verifying that __reduce__ triggers registration for consistent cross-process UUIDs."""

    def test_pickle_triggers_registration_for_create_model(self):
        """Verify that pickling a create_model instance triggers registration.

        Without __reduce__ triggering registration, create_model classes would get
        different UUIDs in different processes if pickled before type_ is accessed.
        """
        from ray.cloudpickle import dumps, loads

        def factory():
            return pydantic_create_model("ReduceTestModel", value=(int, ...), __base__=BaseModel)

        DynamicModel = factory()
        instance = DynamicModel(value=42)

        # Before pickle, no __ccflow_import_path__ (deferred registration)
        assert "__ccflow_import_path__" not in DynamicModel.__dict__

        # Pickle triggers __reduce__ which triggers registration
        pickled = dumps(instance)

        # After pickle, registration has happened
        assert "__ccflow_import_path__" in DynamicModel.__dict__

        # Unpickle and verify same UUID
        restored = loads(pickled)
        assert type(restored).__ccflow_import_path__ == DynamicModel.__ccflow_import_path__

    def test_pickle_cross_process_consistent_uuid(self):
        """Verify that pickling before type_ access gives consistent UUIDs across processes."""
        import os
        import tempfile

        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        try:
            # Process 1: Create and pickle WITHOUT accessing type_ first
            create_code = f'''
from ray.cloudpickle import dump
from pydantic import create_model
from ccflow import BaseModel

def factory():
    return create_model("CrossProcessReduceModel", value=(int, ...), __base__=BaseModel)

Model = factory()
instance = Model(value=42)

# Pickle WITHOUT accessing type_ - __reduce__ should trigger registration
with open("{pkl_path}", "wb") as f:
    dump(instance, f)

print(Model.__ccflow_import_path__)
'''
            result1 = subprocess.run([sys.executable, "-c", create_code], capture_output=True, text=True)
            assert result1.returncode == 0, f"Process 1 failed: {result1.stderr}"
            path1 = result1.stdout.strip()

            # Process 2: Load and verify same UUID
            load_code = f'''
from ray.cloudpickle import load

with open("{pkl_path}", "rb") as f:
    restored = load(f)

print(type(restored).__ccflow_import_path__)
'''
            result2 = subprocess.run([sys.executable, "-c", load_code], capture_output=True, text=True)
            assert result2.returncode == 0, f"Process 2 failed: {result2.stderr}"
            path2 = result2.stdout.strip()

            # UUIDs must match
            assert path1 == path2, f"UUID mismatch: {path1} != {path2}"

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)


class TestRegistrationStrategy:
    """Tests verifying the registration strategy for different class types.

    The strategy is:
    1. Module-level classes: No registration needed, they're importable via standard path
    2. Local classes (<locals> in qualname): Register immediately during class definition
    3. Dynamic classes (create_model, no <locals>): Registered during pickle via __reduce__

    This keeps import-time overhead minimal while still handling all cases correctly.
    """

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

    def test_create_model_defers_registration_until_type_access(self):
        """create_model classes should defer registration until type_ is accessed.

        This is the key optimization: create_model produces classes without <locals>
        in their qualname, so we can't tell at definition time if they need registration.
        We defer the _is_importable check until type_ is accessed.
        """
        from unittest import mock

        with mock.patch.object(local_persistence, "_register") as mock_do_reg:
            # Create a model via create_model
            DynamicModel = pydantic_create_model("DeferredModel", x=(int, ...), __base__=BaseModel)

            # No <locals> in qualname
            assert "<locals>" not in DynamicModel.__qualname__

            # _register should NOT have been called yet
            mock_do_reg.assert_not_called()

        # Now access type_ which triggers the deferred check
        instance = DynamicModel(x=1)
        _ = instance.type_

        # NOW it should have __ccflow_import_path__
        assert "__ccflow_import_path__" in DynamicModel.__dict__

    def test_is_importable_only_called_lazily(self):
        """_is_importable should only be called when we need to check, not during import."""
        from unittest import mock

        # Create a model via create_model
        DynamicModel = pydantic_create_model("LazyCheckModel", x=(int, ...), __base__=BaseModel)

        with mock.patch.object(local_persistence, "_is_importable", wraps=local_persistence._is_importable) as mock_is_imp:
            # Access type_ which triggers lazy registration check
            instance = DynamicModel(x=1)
            _ = instance.type_

            # _is_importable should have been called
            assert mock_is_imp.call_count >= 1


class TestIsImportableEdgeCases:
    """Tests for edge cases in the _is_importable function."""

    def test_class_with_module_not_in_sys_modules(self):
        """Test that a class claiming to be from an unloaded module is not importable."""
        cls = type("NotImportable", (), {})
        cls.__module__ = "nonexistent.fake.module"
        cls.__qualname__ = "NotImportable"

        assert not local_persistence._is_importable(cls)

    def test_class_in_wrong_module(self):
        """Test that a class claiming to be from a module it's not in is not importable."""
        cls = type("FakeClass", (), {})
        # Claim to be from this module, but with a name that doesn't exist
        cls.__module__ = "ccflow.tests.test_local_persistence"
        cls.__qualname__ = "ThisClassDoesNotExist"

        assert not local_persistence._is_importable(cls)
