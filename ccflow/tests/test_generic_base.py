from unittest import TestCase

from ccflow.generic_base import GenericContext, GenericResult


class TestGenericResult(TestCase):
    def test_generic(self):
        v = {"a": 1, "b": [2, 3]}
        result = GenericResult(value=v)
        self.assertEqual(GenericResult.model_validate(v), result)
        self.assertIs(GenericResult.model_validate(result), result)

        v = {"value": 5}
        self.assertEqual(GenericResult.model_validate(v), GenericResult(value=5))
        self.assertEqual(GenericResult[int].model_validate(v), GenericResult[int](value=5))
        self.assertEqual(GenericResult[str].model_validate(v), GenericResult[str](value="5"))

        self.assertEqual(GenericResult.model_validate("foo"), GenericResult(value="foo"))
        self.assertEqual(GenericResult[str].model_validate(5), GenericResult[str](value="5"))

        result = GenericResult(value=5)
        # Note that this will work, even though GenericResult is not a subclass of GenericResult[str]
        self.assertEqual(GenericResult[str].model_validate(result), GenericResult[str](value="5"))


class TestGenericContext(TestCase):
    def test_generic_context(self):
        v = (1, [2, 3], {4, 5, 6})
        result = GenericContext(value=v)
        self.assertEqual(GenericContext.validate(v), result)

        v = {"value": 5}
        self.assertEqual(GenericContext.validate(v), GenericContext(value=5))
        self.assertEqual(GenericContext[int].validate(v), GenericContext[int](value=5))
        self.assertEqual(GenericContext[str].validate(v), GenericContext[str](value="5"))

        self.assertEqual(GenericContext.validate("foo"), GenericContext(value="foo"))
        self.assertEqual(GenericContext[str].validate(5), GenericContext[str](value="5"))

        result = GenericContext(value=5)
        # Note that this will work, even though GenericContext is not a subclass of GenericContext[str]
        self.assertEqual(GenericContext[str].validate(result), GenericContext[str](value="5"))

    def test_generics_conversion(self):
        v = (1, [2, 3], {4, 5, 6})
        self.assertEqual(GenericContext(value=GenericResult(value=v)), GenericContext(value=v))

        v = 5
        self.assertEqual(GenericContext[str](value=GenericResult(value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[str](value=GenericResult[str](value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[str](value=v)), GenericContext[int](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[int](value=v)), GenericContext[int](value=v))

        v = "5"
        self.assertEqual(GenericContext[str](value=GenericResult(value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[str](value=GenericResult[str](value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[str](value=v)), GenericContext[int](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[int](value=v)), GenericContext[int](value=v))

        self.assertEqual(GenericContext[str].validate(GenericResult(value=5)), GenericContext[str](value="5"))
