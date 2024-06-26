import numpy as np
import orjson
from typing import Any, Dict, List, Union

from .enums import Enum


def _remove_dict_enums(obj: Any) -> Dict:
    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, dict):
        return {_remove_dict_enums(k): _remove_dict_enums(v) for k, v in obj.items()}
    return obj


def orjson_dumps(v, default=None, *a, **kw) -> str:
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    # The default passed to orjson seems to be a partial function
    # with the json_encoders as the first argument. We try to perform the
    # conversion
    options = orjson.OPT_NON_STR_KEYS | orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY
    try:
        return orjson.dumps(
            v,
            default=default,
            option=options,
        ).decode()
    except orjson.JSONEncodeError:
        # if we fail, we try to remove the enums because
        # orjson serialization fails when csp enums are
        # used as dict keys
        return orjson.dumps(
            _remove_dict_enums(v),
            default=default,
            option=options,
        ).decode()


def make_ndarray_orjson_valid(arr: np.ndarray) -> Union[List[Any], np.ndarray]:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray instance, got {type(arr)}")
    # orjson supports these types:
    # https://github.com/ijl/orjson#numpy
    # Which types to check:
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.number
    is_number = np.issubdtype(arr.dtype, np.number)
    is_bool = np.issubdtype(arr.dtype, np.bool_)
    is_complex = np.issubdtype(arr.dtype, np.complexfloating)
    if not (is_bool or is_number) or is_complex:
        return arr.tolist()
    # Now, we have to make the numpy array c-contiguous. Why:
    # https://github.com/ijl/orjson/issues/100
    try:
        return np.ascontiguousarray(arr)
    except MemoryError:
        return arr.tolist()
