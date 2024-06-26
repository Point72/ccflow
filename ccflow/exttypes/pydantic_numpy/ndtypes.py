"""Code from MIT-licensed open source library https://github.com/cheind/pydantic-numpy

MIT License

Copyright (c) 2022 Christoph Heindl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pydantic
from packaging import version
from pydantic import ValidationError
from typing import Any, Dict
from typing_extensions import get_args

from .ndarray import NDArray

__all__ = (
    "bool_",
    "float128",
    "float64",
    "float32",
    "float16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "complex256",
    "complex128",
    "complex64",
    "NDArrayFp128",
    "NDArrayFp64",
    "NDArrayFp32",
    "NDArrayFp16",
    "NDArrayInt64",
    "NDArrayInt32",
    "NDArrayInt16",
    "NDArrayInt8",
    "NDArrayUint64",
    "NDArrayUint32",
    "NDArrayUint16",
    "NDArrayUint8",
    "NDArrayComplex256",
    "NDArrayComplex128",
    "NDArrayComplex64",
    "NDArrayBool",
)


class _BaseDType:
    if version.parse(pydantic.__version__) < version.parse("2"):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update({"type": cls.__name__})

    else:

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            json_schema = handler(core_schema)
            json_schema.update({"type": cls.__name__})
            return json_schema

    @classmethod
    def __get_validators__(cls) -> Any:
        yield cls.validate

    @classmethod
    def validate(cls, val: Any, field) -> "_BaseDType":
        if field.sub_fields:
            msg = f"{cls.__name__} has no subfields"
            raise ValidationError(msg)
        if not isinstance(val, cls):
            return cls(val)
        return val

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation for pydantic v2"""
        from pydantic_core import core_schema

        def _validate(val):
            subtypes = get_args(source_type)
            if subtypes:
                msg = f"{cls.__name__} has no subfields"
                raise ValidationError(msg)
            if not isinstance(val, cls):
                return cls(val)
            return val

        return core_schema.no_info_plain_validator_function(_validate)


class bool_(np.bool_, _BaseDType):
    pass


class longdouble(np.longdouble, _BaseDType):
    pass


float128 = longdouble


class double(np.double, _BaseDType):
    pass


float64 = double


class single(np.single, _BaseDType):
    pass


float32 = single


class half(np.half, _BaseDType):
    pass


float16 = half


class int_(np.int_, _BaseDType):
    pass


int64 = int_


class intc(np.intc, _BaseDType):
    pass


int32 = intc


class short(np.short, _BaseDType):
    pass


int16 = short


class byte(np.byte, _BaseDType):
    pass


int8 = byte


class uint(np.uint, _BaseDType):
    pass


uint64 = uint


class uintc(np.uintc, _BaseDType):
    pass


uint32 = uintc


class ushort(np.ushort, _BaseDType):
    pass


uint16 = ushort


class ubyte(np.ubyte, _BaseDType):
    pass


uint8 = ubyte


class clongdouble(np.clongdouble, _BaseDType):
    pass


complex256 = clongdouble


class cdouble(np.cdouble, _BaseDType):
    pass


complex128 = cdouble


class csingle(np.csingle, _BaseDType):
    pass


complex64 = csingle

# NDArray typings

NDArrayFp128 = NDArray[float128]
NDArrayFp64 = NDArray[float64]
NDArrayFp32 = NDArray[float32]
NDArrayFp16 = NDArray[float16]

NDArrayInt64 = NDArray[int64]
NDArrayInt32 = NDArray[int32]
NDArrayInt16 = NDArray[int16]
NDArrayInt8 = NDArray[int8]

NDArrayUint64 = NDArray[uint64]
NDArrayUint32 = NDArray[uint32]
NDArrayUint16 = NDArray[uint16]
NDArrayUint8 = NDArray[uint8]

NDArrayComplex256 = NDArray[complex256]
NDArrayComplex128 = NDArray[complex128]
NDArrayComplex64 = NDArray[complex64]

NDArrayBool = NDArray[bool_]
