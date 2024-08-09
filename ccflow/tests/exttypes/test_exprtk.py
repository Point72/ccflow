import unittest
from unittest import TestCase

from ccflow import BaseModel, ExprTkExpression


def _has_cexprtk() -> bool:
    try:
        import cexprtk  # noqa F401

        return True
    except ImportError:
        return False


class MyModel(BaseModel):
    expression: ExprTkExpression


class TestExprTkExpression(TestCase):
    @unittest.skipIf(_has_cexprtk(), "Requires cexprtk to not be installed")
    def test_no_cexprtk(self):
        self.assertRaisesRegex(ValueError, "Unable to import cexprtk. Please make sure you have it installed.", ExprTkExpression.validate, "1.0")

    @unittest.skipIf(not _has_cexprtk(), "Requires cexprtk to be installed")
    def test(self):
        import cexprtk

        symbol_table = cexprtk.Symbol_Table({"a": 1.0, "b": 2.0})

        # Constant
        e = ExprTkExpression.validate("1.0")
        self.assertAlmostEqual(1.0, e.expression(symbol_table)())

        # Valid
        e = ExprTkExpression.validate("1.0 + a * b")
        self.assertAlmostEqual(3.0, e.expression(symbol_table)())

        # Valid
        e = ExprTkExpression.validate("-a * b")
        self.assertAlmostEqual(-2.0, e.expression(symbol_table)())

        # Invalid
        self.assertRaisesRegex(ValueError, "Error parsing expression.*", ExprTkExpression.validate, "1a++")

        # Wrong types
        self.assertRaisesRegex(ValueError, ".*cannot be converted.*", ExprTkExpression.validate, None)

    def test_model(self):
        expression = "1.0 + a"
        if _has_cexprtk():
            import cexprtk

            m = MyModel(expression=expression)
            symbol_table = cexprtk.Symbol_Table({"a": 1.0})
            self.assertAlmostEqual(2.0, m.expression.expression(symbol_table)())
        else:
            self.assertRaisesRegex(ValueError, "Unable to import cexprtk. Please make sure you have it installed.", MyModel, expression=expression)
