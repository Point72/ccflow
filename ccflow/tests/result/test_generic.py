from unittest import TestCase

from ccflow import GenericContext, GenericResult


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

    def test_generics_conversion(self):
        v = (1, [2, 3], {4, 5, 6})
        self.assertEqual(GenericResult(value=GenericContext(value=v)), GenericResult(value=v))

        v = 5
        self.assertEqual(GenericResult[str](value=GenericContext(value=v)), GenericResult[str](value=v))
        self.assertEqual(GenericResult[str](value=GenericContext[str](value=v)), GenericResult[str](value=v))
        self.assertEqual(GenericResult[int](value=GenericContext[str](value=v)), GenericResult[int](value=v))
        self.assertEqual(GenericResult[int](value=GenericContext[int](value=v)), GenericResult[int](value=v))

        v = "5"
        self.assertEqual(GenericResult[str](value=GenericContext(value=v)), GenericResult[str](value=v))
        self.assertEqual(GenericResult[str](value=GenericContext[str](value=v)), GenericResult[str](value=v))
        self.assertEqual(GenericResult[int](value=GenericContext[str](value=v)), GenericResult[int](value=v))
        self.assertEqual(GenericResult[int](value=GenericContext[int](value=v)), GenericResult[int](value=v))

        self.assertEqual(GenericResult[str].validate(GenericContext(value=5)), GenericResult[str](value="5"))
