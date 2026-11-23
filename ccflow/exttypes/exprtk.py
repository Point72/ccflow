import ctypes
import functools
import sys
import sysconfig
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import core_schema
from typing_extensions import Self

__all__ = ("ExprTkExpression",)

# Slot id of tp_clear in CPython's typeslots.h, and its offset in PyTypeObject on default (non free-threaded) builds.
_PY_TP_CLEAR = 51
_TP_CLEAR_OFFSET = 24 * ctypes.sizeof(ctypes.c_void_p)


def _disable_gc_clear(tp: type) -> None:
    # cexprtk's Symbol_Table.__dealloc__ writes through attributes that its tp_clear has already set to None, so a
    # Symbol_Table freed by the cyclic garbage collector corrupts the memory next to the None object. Without tp_clear,
    # the collector breaks such cycles through the other objects and the table is freed by reference counting.
    # This is what @cython.no_gc_clear would do; remove once cexprtk applies it to Symbol_Table.
    if sys.implementation.name != "cpython" or sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    get_slot = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_int)(("PyType_GetSlot", ctypes.pythonapi))
    slot = ctypes.c_void_p.from_address(id(tp) + _TP_CLEAR_OFFSET)
    if slot.value is not None and slot.value == get_slot(tp, _PY_TP_CLEAR):
        slot.value = None


@functools.cache
def _import_cexprtk():
    try:
        import cexprtk
    except ImportError:
        raise ValueError("Unable to import cexprtk. Please make sure you have it installed.")
    _disable_gc_clear(cexprtk.Symbol_Table)
    return cexprtk


class ExprTkExpression(str):
    """Wrapper around a string that represents an ExprTk expression."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: Any) -> Self:
        if isinstance(value, ExprTkExpression):
            return value

        if isinstance(value, str):
            cexprtk = _import_cexprtk()
            try:
                cexprtk.check_expression(value)
            except cexprtk.ParseException as e:
                raise ValueError(f"Error parsing expression {value}. {e}")

            return cls(value)

        raise ValueError("Value cannot be converted into an ExprTkExpression.")

    @classmethod
    def validate(cls, value: Any) -> Self:
        """Try to convert/validate an arbitrary value to a Frequency."""
        return _TYPE_ADAPTER.validate_python(value)

    def expression(self, symbol_table: Any) -> Any:
        """Make a cexprtk.Expression from a symbol table.

        Args:
            symbol_table: cexprtk.Symbol_Table

        Returns:
            An cexprtk.Expression.
        """
        cexprtk = _import_cexprtk()
        return cexprtk.Expression(str(self), symbol_table)


_TYPE_ADAPTER = TypeAdapter(ExprTkExpression)
