"""This module contains extension types for pydantic."""

from functools import cached_property
from typing import Any, Type, get_origin

from pydantic import ImportString, TypeAdapter
from pydantic_core import core_schema

import_string = TypeAdapter(ImportString).validate_python


class PyObjectPath(str):
    """Similar to pydantic's PyObject, this class represents the path to the object as a string.

    This is useful because it can be serialized/deserialized (including to json), unlike Pydantic's PyObject,
    while also providing easy access to the underlying object.
    """

    # TODO: It would be nice to make this also derive from Generic[T],
    #  where T could then by used for type checking in validate.
    #  However, this doesn't work: https://github.com/python/typing/issues/629

    validate_always = True

    @cached_property
    def object(self) -> Type:
        """Return the underlying object that the path corresponds to."""
        return import_string(str(self))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, value: Any) -> Any:
        if isinstance(value, PyObjectPath):
            return value

        if isinstance(value, str):
            value = cls(value)
            try:
                value.object
            except ImportError as e:
                raise ValueError(f"ensure this value contains valid import path or importable object: {str(e)}")
        else:
            origin = get_origin(value)
            if origin is not None:
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
                if module is None:
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
