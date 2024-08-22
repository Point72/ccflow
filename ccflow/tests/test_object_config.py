import pickle
from unittest import TestCase

import pytest
from pydantic import ValidationError

from ccflow import BaseModel, LazyObjectConfig, ModelRegistry, ObjectConfig


class MyClass:
    def __init__(self, p="p", q=10.0):
        self.p = p
        self.q = q


class ContainerClass(BaseModel):
    config: ObjectConfig


class TestObjectConfig(TestCase):
    def tearDown(self):
        ModelRegistry.root().clear()

    def test_construction(self):
        config = ObjectConfig(
            object_type="ccflow.tests.test_object_config.MyClass",
            object_kwargs=dict(p="foo"),
        )
        self.assertIsInstance(config.object, MyClass)
        self.assertEqual(config.object.p, "foo")

        with pytest.raises((TypeError, ValidationError)):
            config.p = "bar"

        with pytest.raises(TypeError):
            config = ObjectConfig(
                object_type="ccflow.tests.test_object_config.MyClass",
                object_kwargs=dict(garbage="foo"),
            )

    def test_lazy_construction(self):
        config = LazyObjectConfig(
            object_type="ccflow.tests.test_object_config.MyClass",
            object_kwargs=dict(p="foo"),
        )
        self.assertIsInstance(config.object, MyClass)
        self.assertEqual(config.object.p, "foo")

        with pytest.raises((TypeError, ValidationError)):
            config.p = "bar"

        config = LazyObjectConfig(
            object_type="ccflow.tests.test_object_config.MyClass",
            object_kwargs=dict(garbage="foo"),
        )
        with pytest.raises(TypeError):
            config.object

    def test_validation(self):
        for Config in [ObjectConfig, LazyObjectConfig]:
            config = Config(
                object_type="ccflow.tests.test_object_config.MyClass",
                p="foo",
                q=5,
            )
            self.assertIsInstance(config.object, MyClass)

            # Check the result is as expected
            self.assertEqual(config.object.p, "foo")
            self.assertEqual(config.object.q, 5)

    def test_pickling(self):
        for Config in [ObjectConfig, LazyObjectConfig]:
            config = Config(
                object_type="ccflow.tests.test_object_config.MyClass",
                object_kwargs=dict(p="foo", q=5),
            )
            # Insert pickling step
            config = pickle.loads(pickle.dumps(config))

            self.assertIsInstance(config.object, MyClass)
            self.assertEqual(config.object.p, "foo")
            self.assertEqual(config.object.q, 5)

    def test_registration(self):
        """Test that validators on config objects don't interfere with BaseModel validators"""
        for Config in [ObjectConfig, LazyObjectConfig]:
            config = Config(
                object_type="ccflow.tests.test_object_config.MyClass",
                object_kwargs=dict(p="foo", q=5),
            )
            r = ModelRegistry.root()
            r.add("foo", config, overwrite=True)

            cc = ContainerClass(config="foo")
            self.assertEqual(cc.config, config)
