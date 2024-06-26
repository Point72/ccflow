# ruff: noqa: F811

# BSD 3-Clause License

# Copyright (c) 2014, Anaconda, Inc. and contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import binascii
import cloudpickle
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import os
import pathlib
import pickle
import types
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Optional


################
# dask/core.py #
################
class literal:
    """A small serializable object to wrap literal values without copying"""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "literal<type=%s>" % type(self.data).__name__

    def __reduce__(self):
        return (literal, (self.data,))

    def __call__(self):
        return self.data


#################
# dask/utils.py #
#################
class Dispatch:
    """Simple single dispatch."""

    def __init__(self, name=None):
        self._lookup = {}
        self._lazy = {}
        if name:
            self.__name__ = name

    def register(self, type, func=None):
        """Register dispatch of `func` on arguments of type `type`"""

        def wrapper(func):
            if isinstance(type, tuple):
                for t in type:
                    self.register(t, func)
            else:
                self._lookup[type] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def register_lazy(self, toplevel, func=None):
        """
        Register a registration function which will be called if the
        *toplevel* module (e.g. 'pandas') is ever loaded.
        """

        def wrapper(func):
            self._lazy[toplevel] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls):
        """Return the function implementation for the given ``cls``"""
        lk = self._lookup
        for cls2 in cls.__mro__:
            # Is a lazy registration function present?
            toplevel, _, _ = cls2.__module__.partition(".")
            try:
                register = self._lazy[toplevel]
            except KeyError:
                pass
            else:
                register()
                self._lazy.pop(toplevel, None)
                return self.dispatch(cls)  # recurse
            try:
                impl = lk[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache lookup
                    lk[cls] = impl
                return impl
        raise TypeError(f"No dispatch for {cls}")

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of argument.
        """
        meth = self.dispatch(type(arg))
        return meth(arg, *args, **kwargs)

    @property
    def __doc__(self):
        try:
            func = self.dispatch(object)
            return func.__doc__
        except TypeError:
            return "Single Dispatch for %s" % self.__name__


###################
# dask/hashing.py #
###################
hashers = []  # In decreasing performance order


# Timings on a largish array:
# - CityHash is 2x faster than MurmurHash
# - xxHash is slightly slower than CityHash
# - MurmurHash is 8x faster than SHA1
# - SHA1 is significantly faster than all other hashlib algorithms

try:
    import cityhash  # `python -m pip install cityhash`
except ImportError:
    pass
else:
    # CityHash disabled unless the reference leak in
    # https://github.com/escherba/python-cityhash/pull/16
    # is fixed.
    if cityhash.__version__ >= "0.2.2":

        def _hash_cityhash(buf):
            """
            Produce a 16-bytes hash of *buf* using CityHash.
            """
            h = cityhash.CityHash128(buf)
            return h.to_bytes(16, "little")

        hashers.append(_hash_cityhash)

try:
    import xxhash  # `python -m pip install xxhash`
except ImportError:
    pass
else:

    def _hash_xxhash(buf):
        """
        Produce a 8-bytes hash of *buf* using xxHash.
        """
        return xxhash.xxh64(buf).digest()

    hashers.append(_hash_xxhash)

try:
    import mmh3  # `python -m pip install mmh3`
except ImportError:
    pass
else:

    def _hash_murmurhash(buf):
        """
        Produce a 16-bytes hash of *buf* using MurmurHash.
        """
        return mmh3.hash_bytes(buf)

    hashers.append(_hash_murmurhash)


def _hash_sha1(buf):
    """
    Produce a 20-bytes hash of *buf* using SHA1.
    """
    return hashlib.sha1(buf).digest()


hashers.append(_hash_sha1)


def hash_buffer(buf, hasher=None):
    """
    Hash a bytes-like (buffer-compatible) object.  This function returns
    a good quality hash but is not cryptographically secure.  The fastest
    available algorithm is selected.  A fixed-length bytes object is returned.
    """
    if hasher is not None:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            # Some hash libraries may have overly-strict type checking,
            # not accepting all buffers
            pass
    for hasher in hashers:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    raise TypeError(f"unsupported type for hashing: {type(buf)}")


def hash_buffer_hex(buf, hasher=None):
    """
    Same as hash_buffer, but returns its result in hex-encoded form.
    """
    h = hash_buffer(buf, hasher)
    s = binascii.b2a_hex(h)
    return s.decode()


################
# dask/base.py #
################
############
# Tokenize #
############
class TokenizationError(RuntimeError):
    pass


def tokenize(*args: object, ensure_deterministic: Optional[bool] = None, **kwargs: object) -> str:
    """Deterministic token

    >>> tokenize([1, 2, '3'])
    '06961e8de572e73c2e74b51348177918'

    >>> tokenize('Hello') == tokenize('Hello')
    True

    Parameters
    ----------
    args, kwargs:
        objects to tokenize
    ensure_deterministic: bool, optional
        If True, raise TokenizationError if the objects cannot be deterministically
        tokenized, e.g. two identical objects will return different tokens.
        Defaults to the `tokenize.ensure-deterministic` configuration parameter.
    """
    with _seen_ctx(reset=True), _ensure_deterministic_ctx(ensure_deterministic):
        token: object = _normalize_seq_func(args)
        if kwargs:
            token = token, _normalize_seq_func(sorted(kwargs.items()))

    # Pass `usedforsecurity=False` to support FIPS builds of Python
    return hashlib.md5(str(token).encode(), usedforsecurity=False).hexdigest()


# tokenize.ensure-deterministic flag, potentially overridden by tokenize()
_ensure_deterministic: ContextVar[bool] = ContextVar("_ensure_deterministic")

# Circular reference breaker used by _normalize_seq_func.
# This variable is recreated anew every time you call tokenize(). Note that this means
# that you could call tokenize() from inside tokenize() and they would be fully
# independent.
#
# It is a map of {id(obj): (<first seen incremental int>, obj)} which causes an object
# to be tokenized as ("__seen", <incremental>) the second time it's encountered while
# traversing collections. A strong reference to the object is stored in the context to
# prevent ids from being reused by different objects.
_seen: ContextVar[dict[int, tuple[int, object]]] = ContextVar("_seen")


@contextmanager
def _ensure_deterministic_ctx(ensure_deterministic: Optional[bool]) -> Iterator[bool]:
    try:
        ensure_deterministic = _ensure_deterministic.get()
        # There's a call of tokenize() higher up in the stack
        tok = None
    except LookupError:
        # Outermost tokenize(), or normalize_token() was called directly
        # if ensure_deterministic is None:
        #     ensure_deterministic = config.get("tokenize.ensure-deterministic")
        tok = _ensure_deterministic.set(ensure_deterministic)

    try:
        yield ensure_deterministic
    finally:
        if tok:
            tok.var.reset(tok)


def _maybe_raise_nondeterministic(msg: str) -> None:
    with _ensure_deterministic_ctx(None) as ensure_deterministic:
        if ensure_deterministic:
            raise TokenizationError(msg)


@contextmanager
def _seen_ctx(reset: bool) -> Iterator[dict[int, tuple[int, object]]]:
    if reset:
        # It is important to reset the token on tokenize() to avoid artifacts when
        # it is called recursively
        seen: dict[int, tuple[int, object]] = {}
        tok = _seen.set(seen)
    else:
        try:
            seen = _seen.get()
            tok = None
        except LookupError:
            # This is for debug only, for when normalize_token is called outside of
            # tokenize()
            seen = {}
            tok = _seen.set(seen)

    try:
        yield seen
    finally:
        if tok:
            tok.var.reset(tok)


normalize_token = Dispatch()


def identity(x):
    return x


normalize_token.register(
    (
        int,
        float,
        str,
        bytes,
        type(None),
        slice,
        complex,
        type(Ellipsis),
        decimal.Decimal,
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
        pathlib.PurePath,
    ),
    identity,
)


@normalize_token.register((types.MappingProxyType, dict))
def normalize_dict(d):
    return "dict", _normalize_seq_func(sorted(d.items(), key=lambda kv: str(kv[0])))


@normalize_token.register(OrderedDict)
def normalize_ordered_dict(d):
    return _normalize_seq_func((type(d), list(d.items())))


@normalize_token.register(set)
def normalize_set(s):
    # Note: in some Python version / OS combinations, set order changes every
    # time you recreate the set (even within the same interpreter).
    # In most other cases, set ordering is consistent within the same interpreter.
    return "set", _normalize_seq_func(sorted(s, key=str))


def _normalize_seq_func(seq: Iterable[object]) -> list[object]:
    with _seen_ctx(reset=False) as seen:
        out = []
        for item in seq:
            if isinstance(item, (str, bytes, int, float, bool, type(None))):
                # Basic data type. This is just for performance and compactness of the
                # output. It doesn't need to be a comprehensive list.
                pass
            elif id(item) in seen:
                # May or may not be a circular recursion. Maybe just a double reference.
                seen_when, _ = seen[id(item)]
                item = "__seen", seen_when
            else:
                seen[id(item)] = len(seen), item
                item = normalize_token(item)
            out.append(item)
        return out


@normalize_token.register((tuple, list))
def normalize_seq(seq):
    return type(seq).__name__, _normalize_seq_func(seq)


@normalize_token.register(literal)
def normalize_literal(lit):
    return "literal", normalize_token(lit())


# @normalize_token.register(Compose)
# def normalize_compose(func):
#     return _normalize_seq_func((func.first,) + func.funcs)


@normalize_token.register((partial,))
def normalize_partial(func):
    return _normalize_seq_func((func.func, func.args, func.keywords))


@normalize_token.register((types.MethodType, types.MethodWrapperType))
def normalize_bound_method(meth):
    return normalize_token(meth.__self__), meth.__name__


@normalize_token.register(types.BuiltinFunctionType)
def normalize_builtin_function_or_method(func):
    # Note: BuiltinMethodType is BuiltinFunctionType
    self = getattr(func, "__self__", None)
    if self is not None and not inspect.ismodule(self):
        return normalize_bound_method(func)
    else:
        return normalize_object(func)


@normalize_token.register(object)
def normalize_object(o):
    method = getattr(o, "__dask_tokenize__", None)
    if method is not None and not isinstance(o, type):
        return method()

    if type(o) is object:
        return _normalize_pure_object(o)

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _normalize_dataclass(o)

    try:
        return _normalize_pickle(o)
    except Exception:
        _maybe_raise_nondeterministic(
            f"Object {o!r} cannot be deterministically hashed. See "
            "https://docs.dask.org/en/latest/custom-collections.html#implementing-deterministic-hashing "
            "for more information."
        )
        return uuid.uuid4().hex


_seen_objects = set()


def _normalize_pure_object(o: object) -> tuple[str, int]:
    _maybe_raise_nondeterministic(
        "object() cannot be deterministically hashed. See "
        "https://docs.dask.org/en/latest/custom-collections.html#implementing-deterministic-hashing "
        "for more information."
    )
    # Idempotent, but not deterministic. Make sure that the id is not reused.
    _seen_objects.add(o)
    return "object", id(o)


def _normalize_pickle(o: object) -> tuple:
    buffers: list[pickle.PickleBuffer] = []
    pik: Optional[bytes]
    try:
        pik = pickle.dumps(o, protocol=5, buffer_callback=buffers.append)
        if b"__main__" in pik:
            pik = None
    except Exception:
        pik = None

    if pik is None:
        buffers.clear()
        pik = cloudpickle.dumps(o, protocol=5, buffer_callback=buffers.append)

    return hash_buffer_hex(pik), [hash_buffer_hex(buf) for buf in buffers]


def _normalize_dataclass(obj):
    fields = [(field.name, normalize_token(getattr(obj, field.name, None))) for field in dataclasses.fields(obj)]
    params = obj.__dataclass_params__
    params = [(attr, getattr(params, attr)) for attr in params.__slots__]

    return normalize_object(type(obj)), params, fields


@normalize_token.register_lazy("pandas")
def register_pandas():
    import pandas as pd

    @normalize_token.register(pd.Index)
    def normalize_index(ind):
        values = ind.array
        return [ind.name, normalize_token(values)]

    @normalize_token.register(pd.MultiIndex)
    def normalize_index(ind):
        codes = ind.codes
        return [ind.name] + [normalize_token(x) for x in ind.levels] + [normalize_token(x) for x in codes]

    @normalize_token.register(pd.Categorical)
    def normalize_categorical(cat):
        return [normalize_token(cat.codes), normalize_token(cat.dtype)]

    @normalize_token.register(pd.arrays.PeriodArray)
    @normalize_token.register(pd.arrays.DatetimeArray)
    @normalize_token.register(pd.arrays.TimedeltaArray)
    def normalize_period_array(arr):
        return [normalize_token(arr.asi8), normalize_token(arr.dtype)]

    @normalize_token.register(pd.arrays.IntervalArray)
    def normalize_interval_array(arr):
        return [
            normalize_token(arr.left),
            normalize_token(arr.right),
            normalize_token(arr.closed),
        ]

    @normalize_token.register(pd.Series)
    def normalize_series(s):
        return [
            s.name,
            s.dtype,
            normalize_token(s._values),
            normalize_token(s.index),
        ]

    @normalize_token.register(pd.DataFrame)
    def normalize_dataframe(df):
        mgr = df._mgr
        data = list(mgr.arrays) + [df.columns, df.index]
        return list(map(normalize_token, data))

    @normalize_token.register(pd.api.extensions.ExtensionArray)
    def normalize_extension_array(arr):
        import numpy as np

        return normalize_token(np.asarray(arr))

    # Dtypes
    @normalize_token.register(pd.api.types.CategoricalDtype)
    def normalize_categorical_dtype(dtype):
        return [normalize_token(dtype.categories), normalize_token(dtype.ordered)]

    @normalize_token.register(pd.api.extensions.ExtensionDtype)
    def normalize_period_dtype(dtype):
        return normalize_token(dtype.name)

    @normalize_token.register(type(pd.NA))
    def normalize_na(na):
        return pd.NA

    @normalize_token.register(pd.offsets.BaseOffset)
    def normalize_offset(offset):
        return offset.freqstr


@normalize_token.register_lazy("pyarrow")
def register_pyarrow():
    import pyarrow as pa

    @normalize_token.register(pa.DataType)
    def normalize_datatype(dt):
        return pickle.dumps(dt, protocol=4)


@normalize_token.register_lazy("numpy")
def register_numpy():
    import numpy as np

    @normalize_token.register(np.memmap)
    def normalize_mmap(mm):
        if hasattr(mm, "mode") and getattr(mm, "filename", None):
            if hasattr(mm.base, "ctypes"):
                offset = mm.ctypes._as_parameter_.value - mm.base.ctypes._as_parameter_.value
            else:
                offset = 0  # root memmap's have mmap object as base
            if hasattr(mm, "offset"):  # offset numpy used while opening, and not the offset to the beginning of file
                offset += mm.offset
            return (
                mm.filename,
                os.path.getmtime(mm.filename),
                mm.dtype,
                mm.shape,
                mm.strides,
                offset,
            )
        else:
            return normalize_object(mm)

    @normalize_token.register(np.ufunc)
    def normalize_ufunc(func):
        try:
            return _normalize_pickle(func)
        except Exception:
            _maybe_raise_nondeterministic(
                f"Cannot tokenize numpy ufunc {func!r}. Please use functions "
                "of the dask.array.ufunc module instead. See also "
                "https://docs.dask.org/en/latest/array-numpy-compatibility.html"
            )
            return uuid.uuid4().hex
