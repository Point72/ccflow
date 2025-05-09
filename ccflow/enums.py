"""We define our own enum type for use in ccflow.
The implementation of this enum depends on whether csp is found in the environment or not.
If it is found, it uses csp.Enum for seamless communication with the C++ layer of csp, otherwise it uses enum.Enum.
We define a new subclass whose parent is either of the above, so that testing whether something is an enum
(i.e. isinstance, or type hints) does not need to specify either of the implementations,
i.e. isinstance(x, ccflow.enums.Enum).

For consistency between python enum and csp enum, we need to
 - Make sure that "auto" enumeration is in the same namespace
 - Provide a convenience function for standard python enums that matches the DynamicEnum constructor API from csp

Notes:
    - csp.Enum allows for construction by string. For regular python enums (and csp as well), you can use __getitem__
    i.e. Currency['USD'].
"""

import inspect
from os import environ
from typing import Any, Callable, Dict, List, Union

from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

__all__ = ("auto", "Enum", "make_enum")


from enum import (
    Enum as BaseEnum,  # noqa: F401
    auto,
)

BaseEnum.auto = auto

_CSP_ENUM = False

if environ.get("CCFLOW_NO_CSP", "0") != "1":
    try:
        from csp import DynamicEnum, Enum as BaseEnum

        auto = BaseEnum.auto

        _CSP_ENUM = True

    except ImportError:
        try:
            from enum import auto

            from csp.impl.enum import DynamicEnum, Enum as BaseEnum

            _CSP_ENUM = True

        except ImportError:
            # if csp is not installed, rely on python Enum
            pass


class Enum(BaseEnum):
    """Hybrid enum type based on csp.Enum if csp is available, otherwise enum.Enum"""

    # NOTE: we overload the default numbering
    # behavior to start at 0 and match to csp
    def _generate_next_value_(name, start, count, last_values):
        return count

    @classmethod
    def validate(cls, v) -> "Enum":
        if isinstance(v, cls):
            return v
        elif isinstance(v, str):
            return cls[v]
        elif isinstance(v, int):
            return cls(v)
        raise ValueError(f"Cannot convert value to enum: {v}")

    @staticmethod
    def _serialize(value: "Enum") -> str:
        return value.name

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema.str_schema())
        field_schema.update(
            type="string",
            title=cls.__name__,
            description=cls.__doc__ or "An enumeration of {}".format(cls.__name__),
            enum=list(cls.__members__.keys()),
        )
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False, return_schema=core_schema.str_schema(), when_used="json"
            ),
        )


def make_enum(name: str, values: Union[Dict[str, float], List], start=0):
    """Create an enum type dynamically

    :param name: name of the class type
    :param values: either a dictionary of key : values or a list of enum names
    :param start: when providing a list of values, will start enumerating from start
    """

    module_name = inspect.currentframe().f_back.f_globals["__name__"]
    if _CSP_ENUM:
        e = DynamicEnum(name, values, start, module_name=module_name)
        # DynamicEnum is a BaseEnum (csp.Enum or csp.impl.enum.Enum) but it is not a ccflow.enums.Enum. To make
        # issubclass and isinstance return True when testing against ccflow.enums.Enum, we need to add it as a
        # base class.
        e.__bases__ = (Enum,) + e.__bases__
        return e
    if isinstance(values, list):
        values = {k: v + start for v, k in enumerate(values)}

    return Enum(name, values, module=module_name)
