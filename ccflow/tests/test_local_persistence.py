import subprocess
import sys
import textwrap
from collections import defaultdict
from itertools import count
from unittest import TestCase, mock

import ray

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


class TestNeedsRegistration(TestCase):
    def test_module_level_ccflow_classes_do_not_need_registration(self):
        for cls in (ModuleLevelModel, ModuleLevelContext, ModuleLevelCallable):
            with self.subTest(cls=cls.__name__):
                self.assertFalse(local_persistence._needs_registration(cls))

    def test_local_scope_class_needs_registration(self):
        def build_class():
            class LocalClass:
                pass

            return LocalClass

        LocalClass = build_class()
        self.assertTrue(local_persistence._needs_registration(LocalClass))

    def test_main_module_class_does_not_need_registration(self):
        cls = type("MainModuleClass", (), {})
        cls.__module__ = "__main__"
        cls.__qualname__ = "MainModuleClass"
        self.assertFalse(local_persistence._needs_registration(cls))

    def test_module_level_non_ccflow_class_does_not_need_registration(self):
        cls = type("ExternalClass", (), {})
        cls.__module__ = "ccflow.tests.test_local_persistence"
        cls.__qualname__ = "ExternalClass"
        self.assertFalse(local_persistence._needs_registration(cls))


class TestBuildUniqueName(TestCase):
    def test_build_unique_name_sanitizes_hint_and_increments_counter(self):
        with mock.patch.object(local_persistence, "_LOCAL_KIND_COUNTERS", defaultdict(lambda: count())):
            name = local_persistence._build_unique_name(
                kind_slug="callable_model",
                name_hint="module.path:MyCallable<One>",
            )
            self.assertTrue(name.startswith("callable_model__module_path_MyCallable_One_"))
            self.assertTrue(name.endswith("__0"))

            second = local_persistence._build_unique_name(
                kind_slug="callable_model",
                name_hint="module.path:MyCallable<One>",
            )
            self.assertTrue(second.endswith("__1"))

    def test_counters_are_namespaced_by_kind(self):
        with mock.patch.object(local_persistence, "_LOCAL_KIND_COUNTERS", defaultdict(lambda: count())):
            first_context = local_persistence._build_unique_name(kind_slug="context", name_hint="Context")
            first_callable = local_persistence._build_unique_name(kind_slug="callable_model", name_hint="Callable")
            second_context = local_persistence._build_unique_name(kind_slug="context", name_hint="Other")

        self.assertTrue(first_context.endswith("__0"))
        self.assertTrue(first_callable.endswith("__0"))
        self.assertTrue(second_context.endswith("__1"))

    def test_empty_hint_uses_fallback(self):
        with mock.patch.object(local_persistence, "_LOCAL_KIND_COUNTERS", defaultdict(lambda: count())):
            name = local_persistence._build_unique_name(kind_slug="model", name_hint="")
        self.assertEqual(name, "model__BaseModel__0")


def _run_subprocess(code: str) -> str:
    """Execute code in a clean interpreter so sys.modules starts empty."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_local_artifacts_module_is_lazy():
    output = _run_subprocess(
        """
        import sys
        import ccflow.local_persistence as lp

        print(lp.LOCAL_ARTIFACTS_MODULE_NAME in sys.modules)
        """
    )
    assert output == "False"


def test_local_artifacts_module_reload_preserves_dynamic_attrs():
    output = _run_subprocess(
        """
        import importlib
        import ccflow.local_persistence as lp

        def build_cls():
            class _Temp:
                pass
            return _Temp

        Temp = build_cls()
        lp._register_local_subclass(Temp, kind="demo")
        module = importlib.import_module(lp.LOCAL_ARTIFACTS_MODULE_NAME)

        # Extract the registered name from __ccflow_import_path__
        import_path = Temp.__ccflow_import_path__
        registered_name = import_path.split(".")[-1]

        before = getattr(module, registered_name) is Temp
        module = importlib.reload(module)
        after = getattr(module, registered_name) is Temp

        # __qualname__ should NOT have changed (preserves cloudpickle behavior)
        print(before, after, "<locals>" in Temp.__qualname__)
        """
    )
    assert output.split() == ["True", "True", "True"]


def test_register_local_subclass_preserves_module_qualname_and_sets_import_path():
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
        lp._register_local_subclass(Foo, kind="ModelThing")
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
        print(import_path.startswith("ccflow._local_artifacts.modelthing__"))
        """
    )
    lines = output.splitlines()
    assert lines[0] == "True", f"__module__ should not change: {lines}"
    assert lines[1] == "True", f"__qualname__ should not change: {lines}"
    assert lines[2] == "True", f"__qualname__ should contain '<locals>': {lines}"
    assert lines[3] == "True", f"Class should be registered in module: {lines}"
    assert lines[4] == "True", f"Registered class should be the same object: {lines}"
    assert lines[5] == "True", f"Import path should start with expected prefix: {lines}"


def test_local_basemodel_cloudpickle_cross_process():
    """Test that local-scope BaseModel subclasses work with cloudpickle cross-process.

    This is the key test for the "best of both worlds" approach:
    - __qualname__ has '<locals>' so cloudpickle serializes the class definition
    - __ccflow_import_path__ allows PyObjectPath validation to work
    - After unpickling, __pydantic_init_subclass__ re-registers the class
    """
    import os
    import tempfile

    pkl_path = tempfile.mktemp(suffix=".pkl")

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
        """Verify that __module__ is NOT changed to _local_artifacts."""

        def create_class():
            class Inner(ContextBase):
                value: str

            return Inner

        cls = create_class()
        # __module__ should be this test module, not _local_artifacts
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

    def test_class_registered_in_local_artifacts(self):
        """Verify that the class is registered in _local_artifacts under import path."""
        import sys

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

        pkl_path = tempfile.mktemp(suffix=".pkl")

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

    def test_callable_model_cross_process(self):
        """Test cross-process cloudpickle for CallableModel subclasses."""
        import os
        import tempfile

        pkl_path = tempfile.mktemp(suffix=".pkl")

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

        pkl_path = tempfile.mktemp(suffix=".pkl")

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

        pkl_path = tempfile.mktemp(suffix=".pkl")

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

        # Should use standard path, not _local_artifacts
        assert type_path == "ccflow.tests.test_local_persistence.ModuleLevelModel"
        assert local_persistence.LOCAL_ARTIFACTS_MODULE_NAME not in type_path

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

        assert result.startswith("hello|ccflow._local_artifacts.")
