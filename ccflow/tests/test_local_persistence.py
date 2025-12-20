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
            with self.subTest(cls=cls):
                self.assertFalse(local_persistence._needs_registration(cls))

    def test_local_scope_class_needs_registration(self):
        def build_class():
            class LocalClass:
                pass

            return LocalClass

        LocalClass = build_class()
        self.assertTrue(local_persistence._needs_registration(LocalClass))

    def test_main_module_class_needs_registration(self):
        cls = type("MainModuleClass", (), {})
        cls.__module__ = "__main__"
        cls.__qualname__ = "MainModuleClass"
        self.assertTrue(local_persistence._needs_registration(cls))

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
