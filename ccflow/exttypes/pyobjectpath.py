"""This module contains extension types for pydantic."""

from functools import cached_property, lru_cache
from types import FunctionType, MethodType, ModuleType
from typing import Any, Type, get_origin

from pydantic import ImportString, TypeAdapter
from pydantic_core import core_schema
from typing_extensions import Self

_import_string_adapter = TypeAdapter(ImportString)


@lru_cache(maxsize=None)
def import_string(input_string: str):
    return _import_string_adapter.validate_python(input_string)


class PyObjectPath(str):
    """Similar to pydantic's ImportString (formerly PyObject in v1), this class represents the path to the object as a string.

    In pydantic v1, PyObject could not be serialized to json, whereas in v2, ImportString can.
    However, the round trip is not always consistent, i.e.
        >>> ta = TypeAdapter(ImportString)
        >>> ta.validate_json(ta.dump_json("math.pi"))
        3.141592653589793
        >>> ta = TypeAdapter(PyObjectPath)
        >>> ta.validate_json(ta.dump_json("math.pi"))
        'math.pi'

    Other differences are that ImportString can contain other arbitrary python values, whereas PyObjectPath is always a string
        >>> TypeAdapter(ImportString).validate_python(0)
        0
        >>> TypeAdapter(PyObjectPath).validate_python(0)
        raises
    """

    # TODO: It would be nice to make this also derive from Generic[T],
    #  where T could then by used for type checking in validate.
    #  However, this doesn't work: https://github.com/python/typing/issues/629

    @cached_property
    def object(self) -> Type:
        """Return the underlying object that the path corresponds to."""
        return import_string(str(self))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: Any):
        if isinstance(value, str):
            value = cls(value)
        else:  # Try to construct a string from the object that can then be used to import the object
            origin = get_origin(value)
            if origin:
                value = origin
            if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
                if value.__module__ == "__builtin__":
                    module = "builtins"
                else:
                    module = value.__module__
                qualname = value.__qualname__
                if "[" in qualname:
                    # This happens with Generic types in pydantic. We strip out the info for now.
                    # TODO: Find a way of capturing the underlying type info
                    qualname = qualname.split("[", 1)[0]
                if not module:
                    value = cls(qualname)
                else:
                    value = cls(module + "." + qualname)
            else:
                raise ValueError(f"ensure this value contains valid import path or importable object: unable to import path for {value}")
        try:
            value.object
        except ImportError as e:
            raise ValueError(f"ensure this value contains valid import path or importable object: {str(e)}")

        return value

    @classmethod
    @lru_cache(maxsize=None)
    def _validate_cached(cls, value: str) -> Self:
        return _TYPE_ADAPTER.validate_python(value)

    @classmethod
    def validate(cls, value) -> Self:
        """Try to convert/validate an arbitrary value to a PyObjectPath."""
        if isinstance(
            value, (str, type, FunctionType, ModuleType, MethodType)
        ):  # If the value is trivial, we cache it here to avoid the overhead of validation
            return cls._validate_cached(value)
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(PyObjectPath)
