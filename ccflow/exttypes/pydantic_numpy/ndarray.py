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
import sys
from abc import ABC, abstractmethod
from numpy.lib import NumpyVersion
from pathlib import Path
from pydantic import BaseModel, FilePath, validator
from typing import Any, Generic, Mapping, Optional, TypeVar
from typing_extensions import get_args

T = TypeVar("T", bound=np.generic)

if sys.version_info < (3, 9) or NumpyVersion(np.__version__) < "1.22.0":
    nd_array_type = np.ndarray
else:
    nd_array_type = np.ndarray[Any, T]


class NPFileDesc(BaseModel):
    path: FilePath
    key: Optional[str] = None

    @validator("path")
    def absolute(cls, value: Path) -> Path:
        return value.resolve().absolute()


class _CommonNDArray(ABC):
    @classmethod
    @abstractmethod
    def validate(cls, val: Any, field) -> nd_array_type: ...

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @staticmethod
    def _validate(val: Any, dtype=None) -> nd_array_type:
        if isinstance(val, Mapping):
            try:
                val = NPFileDesc(**val)
            except TypeError:
                raise ValueError("Unable to convert from mapping.")

        if isinstance(val, NPFileDesc):
            val: NPFileDesc

            if val.path.suffix.lower() not in [".npz", ".npy"]:
                raise ValueError("Expected npz or npy file.")

            if not val.path.is_file():
                raise ValueError(f"Path does not exist {val.path}")

            try:
                content = np.load(str(val.path))
            except FileNotFoundError:
                raise ValueError(f"Failed to load numpy data from file {val.path}")

            if val.path.suffix.lower() == ".npz":
                key = val.key or content.files[0]
                try:
                    data = content[key]
                except KeyError:
                    raise ValueError(f"Key {key} not found in npz.")
            else:
                data = content

        else:
            data = val

        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return np.asarray(data)

    @classmethod
    def _make_validator(cls, source_type):
        def _validate(v):
            subtypes = get_args(source_type)
            dtype = subtypes[0] if subtypes and subtypes[0] != Any else None
            return cls._validate(v, dtype)

        return _validate

    @classmethod
    def _serialize(cls, v, nxt):
        # Not as efficient as using orjson, but we need a list type to pass to pydantic,
        # and orjson produces us a string.
        if v is not None:
            v = v.tolist()
        return nxt(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Validation for pydantic v2"""
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            cls._make_validator(source_type),
            core_schema.any_schema(),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.list_schema(),
            ),
        )


class NDArray(Generic[T], nd_array_type, _CommonNDArray):
    @classmethod
    def validate(cls, val: Any, field) -> nd_array_type:
        dtype_field = field.sub_fields[0] if field.sub_fields is not None else None
        dtype = None if dtype_field is None else dtype_field.type_
        return cls._validate(val, dtype)


class PotentialNDArray(Generic[T], nd_array_type, _CommonNDArray):
    """Like NDArray, but validation errors result in None."""

    @classmethod
    def validate(cls, val: Any, field) -> Optional[nd_array_type]:
        try:
            dtype_field = field.sub_fields[0] if field.sub_fields is not None else None
            return cls._validate(val, dtype_field.type_ if dtype_field else None)
        except ValueError:
            return None

    @classmethod
    def _make_validator(cls, source_type):
        def _validate(v):
            subtypes = get_args(source_type)
            dtype = subtypes[0] if subtypes and subtypes[0] != Any else None
            try:
                return cls._validate(v, dtype)
            except ValueError:
                return None

        return _validate
