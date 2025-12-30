import pickle
from typing import Generic, TypeVar
from unittest import TestCase

from ccflow import PyObjectPath


class A:
    pass


T = TypeVar("T")


class B(Generic[T]):
    t: T


class TestPyObjectPath(TestCase):
    def test_basic(self):
        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.A")
        self.assertIsInstance(p, str)
        self.assertEqual(p.object, A)

        p = PyObjectPath("builtins.list")
        self.assertIsInstance(p, str)
        self.assertEqual(p.object, list)

    def test_validate(self):
        self.assertRaises(ValueError, PyObjectPath.validate, None)
        self.assertRaises(ValueError, PyObjectPath.validate, "foo")
        self.assertRaises(ValueError, PyObjectPath.validate, A())

        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.A")
        self.assertEqual(PyObjectPath.validate(p), p)
        self.assertEqual(PyObjectPath.validate(str(p)), p)
        self.assertEqual(PyObjectPath.validate(A), p)

        p = PyObjectPath("builtins.list")
        self.assertEqual(PyObjectPath.validate(p), p)
        self.assertEqual(PyObjectPath.validate(str(p)), p)
        self.assertEqual(PyObjectPath.validate(list), p)

    def test_generics(self):
        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B")
        self.assertEqual(PyObjectPath.validate(p), p)
        self.assertEqual(PyObjectPath.validate(str(p)), p)
        self.assertEqual(PyObjectPath.validate(B), p)

        p2 = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B[float]")
        self.assertRaises(ValueError, PyObjectPath.validate, p2)
        # Note that the type information gets stripped from the class, i.e. we compare with p, not p2
        self.assertEqual(PyObjectPath.validate(B[float]), p)
        # Re-creating the object from the path loses the type information at the moment
        self.assertEqual(PyObjectPath.validate(B[float]).object, B)

    def test_pickle(self):
        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.A")
        self.assertEqual(p, pickle.loads(pickle.dumps(p)))
        p = PyObjectPath.validate("ccflow.tests.exttypes.test_pyobjectpath.A")
        self.assertEqual(p, pickle.loads(pickle.dumps(p)))
        self.assertIsNotNone(p.object)
        self.assertEqual(p, pickle.loads(pickle.dumps(p)))
        self.assertEqual(p.object, pickle.loads(pickle.dumps(p.object)))

    def test_builtin_module_alias(self):
        """Test that objects with __module__ == '__builtin__' are handled correctly.

        In Python 2, built-in types had __module__ == '__builtin__', but in Python 3
        it's 'builtins'. Some C extensions or pickled objects may still report the
        old module name.
        """

        # Create a mock object that reports __builtin__ as its module
        class MockBuiltinObject:
            __module__ = "__builtin__"
            __qualname__ = "int"

        p = PyObjectPath.validate(MockBuiltinObject)
        self.assertEqual(p, "builtins.int")
        self.assertEqual(p.object, int)
