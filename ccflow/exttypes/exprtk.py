from typing import Any

__all__ = ("ExprTkExpression",)


def _import_cexprtk():
    try:
        import cexprtk

        return cexprtk
    except ImportError:
        raise ValueError("Unable to import cexprtk. Please make sure you have it installed.")


class ExprTkExpression(str):
    """Wrapper around a string that represents an ExprTk expression."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field=None) -> Any:
        if isinstance(value, ExprTkExpression):
            return value

        if isinstance(value, str):
            cexprtk = _import_cexprtk()
            try:
                cexprtk.check_expression(value)
            except cexprtk.ParseException as e:
                raise ValueError(f"Error parsing expression {value}. {e}")

            return cls(value)

        raise ValueError(f"{value} cannot be converted into an ExprTkExpression.")

    def expression(self, symbol_table: Any) -> Any:
        """Make a cexprtk.Expression from a symbol table.

        Args:
            symbol_table: cexprtk.Symbol_Table

        Returns:
            An cexprtk.Expression.
        """
        cexprtk = _import_cexprtk()
        return cexprtk.Expression(str(self), symbol_table)
