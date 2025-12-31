"""This module contains extension types for pydantic."""

import importlib
from functools import cached_property, lru_cache
from types import FunctionType, MethodType, ModuleType
from typing import Any, Type, get_origin

from pydantic import TypeAdapter
from pydantic_core import core_schema
from typing_extensions import Self


@lru_cache(maxsize=None)
def import_string(dotted_path: str) -> Any:
    """Import an object from a dotted path string.

    Handles nested class paths like 'module.OuterClass.InnerClass' by progressively
    trying shorter module paths and using getattr for the remaining parts.

    This is more flexible than pydantic's ImportString which can fail on nested classes.
    """
    if not dotted_path:
        raise ImportError("Empty path")

    parts = dotted_path.split(".")

    # Try progressively shorter module paths
    # e.g., for 'a.b.C.D', try 'a.b.C', then 'a.b', then 'a'
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            # Successfully imported module, now getattr for remaining parts
            for attr_name in parts[i:]:
                obj = getattr(obj, attr_name)
            return obj
        except ImportError:
            continue
        except AttributeError:
            # Module imported but attribute not found - keep trying shorter paths
            continue

    raise ImportError(f"No module named '{dotted_path}'")


def _build_standard_import_path(obj: Any) -> str:
    """Build 'module.qualname' path from an object with __module__ and __qualname__."""
    # Handle Python 2 -> 3 module name change for builtins
    if obj.__module__ == "__builtin__":
        module = "builtins"
    else:
        module = obj.__module__

    qualname = obj.__qualname__
    # Strip generic type parameters (e.g., "MyClass[int]" -> "MyClass")
    # This happens with Generic types in pydantic. Type info is lost but
    # at least the base class remains importable.
    # TODO: Find a way of capturing the underlying type info
    if "[" in qualname:
        qualname = qualname.split("[", 1)[0]
    return f"{module}.{qualname}" if module else qualname


class PyObjectPath(str):
    """A string representing an importable Python object path (e.g., "module.ClassName").

    Similar to pydantic's ImportString, but with consistent serialization behavior:
    - ImportString deserializes to the actual object
    - PyObjectPath deserializes back to the string path

    Example:
        >>> ta = TypeAdapter(ImportString)
        >>> ta.validate_json(ta.dump_json("math.pi"))
        3.141592653589793
        >>> ta = TypeAdapter(PyObjectPath)
        >>> ta.validate_json(ta.dump_json("math.pi"))
        'math.pi'

    PyObjectPath also only accepts importable objects, not arbitrary values:
        >>> TypeAdapter(ImportString).validate_python(0)
        0
        >>> TypeAdapter(PyObjectPath).validate_python(0)
        raises
    """

    # TODO: It would be nice to make this also derive from Generic[T],
    #  where T could then be used for type checking in validate.
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
        """Convert value (string path or object) to PyObjectPath, verifying it's importable."""
        if isinstance(value, str):
            path = cls(value)
        else:
            # Unwrap generic types (e.g., List[int] -> list)
            origin = get_origin(value)
            if origin:
                value = origin
            path = cls._path_from_object(value)

        # Verify the path is actually importable
        try:
            path.object
        except ImportError as e:
            raise ValueError(f"ensure this value contains valid import path or importable object: {str(e)}")

        return path

    @classmethod
    def _path_from_object(cls, value: Any) -> "PyObjectPath":
        """Build import path from an object.

        For ccflow BaseModel subclasses with __ccflow_import_path__ set (local classes),
        uses that path. Otherwise uses the standard module.qualname path.
        """
        if isinstance(value, type):
            # Use __ccflow_import_path__ if set (check __dict__ to avoid inheriting from parents).
            # Note: accessing .__dict__ is safe here because value is a type (class object),
            # and all class objects have __dict__. Only instances of __slots__ classes lack it.
            if "__ccflow_import_path__" in value.__dict__:
                return cls(value.__ccflow_import_path__)
            return cls(_build_standard_import_path(value))

        if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
            return cls(_build_standard_import_path(value))

        raise ValueError(f"ensure this value contains valid import path or importable object: unable to import path for {value}")

    @classmethod
    @lru_cache(maxsize=None)
    def _validate_cached(cls, value: str) -> Self:
        return _TYPE_ADAPTER.validate_python(value)

    @classmethod
    def validate(cls, value) -> Self:
        """Try to convert/validate an arbitrary value to a PyObjectPath.

        Uses caching for common value types to improve performance.
        """
        # Cache validation for common immutable types to avoid repeated work
        if isinstance(value, (str, type, FunctionType, ModuleType, MethodType)):
            return cls._validate_cached(value)
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(PyObjectPath)
