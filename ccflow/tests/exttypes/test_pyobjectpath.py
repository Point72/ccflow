from typing import Generic, TypeVar
from unittest import TestCase
from pydantic import TypeAdapter
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
        ta = TypeAdapter(PyObjectPath)
        self.assertRaises(ValueError, ta.validate_python, None)
        self.assertRaises(ValueError, ta.validate_python, "foo")
        self.assertRaises(ValueError, ta.validate_python, A())

        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.A")
        self.assertEqual(ta.validate_python(p), p)
        self.assertEqual(ta.validate_python(str(p)), p)
        self.assertEqual(ta.validate_python(A), p)

        p = PyObjectPath("builtins.list")
        self.assertEqual(ta.validate_python(p), p)
        self.assertEqual(ta.validate_python(str(p)), p)
        self.assertEqual(ta.validate_python(list), p)

    def test_generics(self):
        ta = TypeAdapter(PyObjectPath)
        # This case is special because pydantic 1 generic include the type information in __qualname__,
        # but this is not directly importable, so extra logic is needed to handle it.
        # self.assertEqual(B[float].__qualname__, "B[float]")
        p = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B")
        self.assertEqual(ta.validate_python(p), p)
        self.assertEqual(ta.validate_python(str(p)), p)
        self.assertEqual(ta.validate_python(B), p)

        p2 = PyObjectPath("ccflow.tests.exttypes.test_pyobjectpath.B[float]")
        self.assertEqual(ta.validate_python(p2), p2)
        # Note that the type information gets stripped from the class, i.e. we compare with p, not p2
        self.assertEqual(ta.validate_python(B[float]), p)
        # Re-creating the object from the path loses the type information at the moment
        self.assertEqual(ta.validate_python(B[float]).object, B)
