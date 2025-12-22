import subprocess
import sys
import textwrap
from collections import defaultdict
from itertools import count
from unittest import TestCase, mock

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


def test_local_artifacts_module_reload_preserves_dynamic_attrs_and_qualname():
    output = _run_subprocess(
        """
        import importlib
        import ccflow.local_persistence as lp

        def build_cls():
            class _Temp:
                pass
            return _Temp

        Temp = build_cls()
        lp.register_local_subclass(Temp, kind="demo")
        module = importlib.import_module(lp.LOCAL_ARTIFACTS_MODULE_NAME)
        qual_before = Temp.__qualname__
        before = getattr(module, qual_before) is Temp
        module = importlib.reload(module)
        after = getattr(module, qual_before) is Temp
        print(before, after, qual_before == Temp.__qualname__)
        """
    )
    assert output.split() == ["True", "True", "True"]


def test_register_local_subclass_sets_module_qualname_and_origin():
    output = _run_subprocess(
        """
        import sys
        import ccflow.local_persistence as lp

        def build():
            class Foo:
                pass
            return Foo

        Foo = build()
        lp.register_local_subclass(Foo, kind="ModelThing")
        module = sys.modules[lp.LOCAL_ARTIFACTS_MODULE_NAME]
        print(Foo.__module__)
        print(Foo.__qualname__)
        print(hasattr(module, Foo.__qualname__))
        print(Foo.__ccflow_dynamic_origin__)
        """
    )
    lines = output.splitlines()
    assert lines[0] == "ccflow._local_artifacts"
    assert lines[2] == "True"
    assert lines[3] == "__main__.build.<locals>.Foo"
    assert lines[1].startswith("modelthing__")
    assert lines[1].endswith("__0")
