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
        # This case is special because pydantic 1 generic include the type information in __qualname__,
        # but this is not directly importable, so extra logic is needed to handle it.
        # self.assertEqual(B[float].__qualname__, "B[float]")
        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B")
        self.assertEqual(PyObjectPath.validate(p), p)
        self.assertEqual(PyObjectPath.validate(str(p)), p)
        self.assertEqual(PyObjectPath.validate(B), p)

        p2 = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B[float]")
        self.assertEqual(PyObjectPath.validate(p2), p2)
        # Note that the type information gets stripped from the class, i.e. we compare with p, not p2
        self.assertEqual(PyObjectPath.validate(B[float]), p)
        # Re-creating the object from the path loses the type information at the moment
        self.assertEqual(PyObjectPath.validate(B[float]).object, B)
