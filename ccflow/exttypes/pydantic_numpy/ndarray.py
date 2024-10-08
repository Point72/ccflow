"""Code adapted from MIT-licensed open source library https://github.com/cheind/pydantic-numpy

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

import sys
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.lib import NumpyVersion
from typing_extensions import get_args

T = TypeVar("T", bound=np.generic)

if sys.version_info < (3, 9) or NumpyVersion(np.__version__) < "1.22.0":
    nd_array_type = np.ndarray
else:
    nd_array_type = np.ndarray[Any, T]


class NDArray(Generic[T], nd_array_type):
    @classmethod
    def _serialize(cls, v, nxt):
        # Not as efficient as using orjson, but we need a list type to pass to pydantic,
        # and orjson produces us a string.
        if v is not None:
            v = v.tolist()
        return nxt(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        def _validate(v):
            subtypes = get_args(source_type)
            dtype = subtypes[0] if subtypes and subtypes[0] != Any else None
            try:
                if dtype is not None:
                    return np.asarray(v, dtype=dtype)
                return np.asarray(v)

            except TypeError:
                raise ValueError(f"Unable to convert {v} to an array.")

        return core_schema.no_info_before_validator_function(
            _validate,
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.list_schema(),
            ),
        )
