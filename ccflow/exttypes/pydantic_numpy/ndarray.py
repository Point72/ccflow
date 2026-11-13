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
from functools import lru_cache
from typing import Any, Generic, TypeVar, get_args

import numpy as np
from numpy.lib import NumpyVersion

T = TypeVar("T", bound=np.generic)

if sys.version_info < (3, 9) or NumpyVersion(np.__version__) < "1.22.0":
    nd_array_type = np.ndarray
else:
    nd_array_type = np.ndarray[Any, T]


class NDArray(Generic[T], nd_array_type):  # noqa: PYI059
    @classmethod
    def _serialize(cls, v, nxt):
        # Not as efficient as using orjson, but we need a list type to pass to pydantic,
        # and orjson produces us a string.
        if v is not None:
            v = v.tolist()
        return nxt(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic import TypeAdapter
        from pydantic_core import core_schema

        subtypes = get_args(source_type)
        dtype = subtypes[0] if subtypes and subtypes[0] != Any else None

        @lru_cache(maxsize=64)
        def _serialization_adapter(ndim):
            item_type = dtype
            for _ in range(ndim):
                item_type = list[item_type]
            return TypeAdapter(item_type)

        def _serialize(v, nxt, info):
            if not isinstance(v, np.ndarray) or v.dtype != object or not hasattr(dtype, "__get_pydantic_core_schema__"):
                return cls._serialize(v, nxt)
            # NumPy can coerce IntEnum items to numbers even in object arrays.
            if not isinstance(dtype, type) or any(not isinstance(item, dtype) for item in v.flat):
                return cls._serialize(v, nxt)
            options = {
                name: getattr(info, name)
                for name in ("mode", "include", "exclude", "by_alias", "exclude_unset", "exclude_defaults", "exclude_none", "round_trip")
            }
            # Newer Pydantic versions expose additional serialization controls.
            for name in ("context", "serialize_as_any", "exclude_computed_fields", "polymorphic_serialization"):
                if hasattr(info, name):
                    options[name] = getattr(info, name)
            return _serialization_adapter(v.ndim).dump_python(v.tolist(), **options)

        def _validate(v):
            try:
                if dtype:
                    return np.asarray(v, dtype=dtype)
                return np.asarray(v)

            except TypeError:
                raise ValueError("Unable to convert value to an array.")

        return core_schema.no_info_before_validator_function(
            _validate,
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                _serialize,
                info_arg=True,
                return_schema=core_schema.list_schema(),
            ),
        )
