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

import json
from enum import IntEnum

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import PydanticSerializationError, core_schema

from ccflow import Enum, NDArray, float32


class MySettings(BaseModel):
    K: NDArray[float32]


def test_init_from_values():
    # Directly specify values
    cfg = MySettings(K=[1, 2])
    assert_allclose(cfg.K, [1.0, 2.0])
    assert cfg.K.dtype == np.float32
    assert cfg.model_dump_json()

    cfg = MySettings(K=np.eye(2))
    assert_allclose(cfg.K, [[1.0, 0], [0.0, 1.0]])
    assert cfg.K.dtype == np.float32


def test_unspecified_npdtype():
    # Not specifying a dtype will use numpy default dtype resolver

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

    cfg = MySettingsNoGeneric(K=[1, 2])
    assert_allclose(cfg.K, [1, 2])
    assert cfg.K.dtype == int


def test_json():
    import orjson

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

    cfg = MySettingsNoGeneric(K=[1, 2])
    jdata = orjson.loads(cfg.model_dump_json())

    assert "K" in jdata
    assert isinstance(jdata["K"], list)
    assert jdata["K"] == [1, 2]

    # Test round-trip
    cfg2 = MySettingsNoGeneric.model_validate_json(cfg.model_dump_json())
    np.testing.assert_array_equal(cfg.K, cfg2.K)


def test_optional_construction():
    class MySettingsOptional(BaseModel):
        K: NDArray[float32] | None = None

    cfg = MySettingsOptional()
    assert cfg.K is None

    cfg = MySettingsOptional(K=[1, 2])
    assert type(cfg.K) is np.ndarray
    assert cfg.K.dtype == np.float32


def test_subclass_basemodel():
    class MyModelField(BaseModel):
        K: NDArray[float32]

    class MyModel(BaseModel):
        L: dict[str, MyModelField]

    model_field = MyModelField(K=[1.0, 2.0])
    assert model_field.model_dump_json()

    model = MyModel(L={"a": MyModelField(K=[1.0, 2.0])})
    assert model.L["a"].K.dtype == np.dtype("float32")
    assert model.model_dump_json()


def test_default_value():
    class MyModelField(BaseModel):
        K: NDArray[float32] = np.array([1.0, 2.0])

    model_field = MyModelField()
    np.testing.assert_array_equal(model_field.K, np.array([1.0, 2.0]))


class MyDictArrayModel(BaseModel):
    x: NDArray[float32] | dict[int, NDArray[float32]]


def test_union_field():
    # Array
    model = MyDictArrayModel(x=[0, 1, 2])
    json = model.model_dump_json()
    model2 = MyDictArrayModel.model_validate_json(json)
    np.testing.assert_array_equal(model2.x, np.array([0.0, 1.0, 2.0]))

    # Dictionary
    model = MyDictArrayModel(x={0: [0, 1, 2]})
    json = model.model_dump_json()
    model2 = MyDictArrayModel.model_validate_json(json)
    assert 0 in model2.x
    assert len(model2.x) == 1
    np.testing.assert_array_equal(model2.x[0], np.array([0.0, 1.0, 2.0]))


class SerializedLabel:
    def __init__(self, value):
        self.value = value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        def serialize(value, info):
            prefix = (getattr(info, "context", None) or {}).get("prefix", "")
            return prefix + value.value

        return core_schema.is_instance_schema(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, info_arg=True, when_used="json"),
        )


@pytest.mark.parametrize(
    "shape,expected",
    [((2,), ["BUY", "SELL"]), ((1, 2), [["BUY", "SELL"]]), ((2, 1, 1), [[["BUY"]], [["SELL"]]])],
)
def test_custom_item_serializer(shape, expected):
    adapter = TypeAdapter(NDArray[SerializedLabel])
    array = adapter.validate_python(np.array([SerializedLabel("BUY"), SerializedLabel("SELL")], dtype=object).reshape(shape))
    original = array.copy()
    assert json.loads(adapter.dump_json(array)) == expected
    assert array.dtype == object
    np.testing.assert_array_equal(array, original)


def test_custom_item_serializer_retains_python_mode():
    value = SerializedLabel("BUY")
    adapter = TypeAdapter(NDArray[SerializedLabel])
    array = adapter.validate_python([value])
    assert adapter.dump_python(array)[0] is value
    assert adapter.dump_python(array, mode="json") == ["BUY"]


def test_custom_item_serializer_respects_index_filters():
    adapter = TypeAdapter(NDArray[SerializedLabel])
    array = adapter.validate_python([SerializedLabel("BUY"), SerializedLabel("SELL")])
    assert json.loads(adapter.dump_json(array, include={1})) == ["SELL"]
    assert json.loads(adapter.dump_json(array, exclude={0})) == ["SELL"]


@pytest.mark.parametrize("shape,expected", [((0,), []), ((2, 0), [[], []])])
def test_custom_item_serializer_empty_arrays(shape, expected):
    adapter = TypeAdapter(NDArray[SerializedLabel])
    assert json.loads(adapter.dump_json(np.empty(shape, dtype=object))) == expected


def test_custom_item_serializer_in_optional_field():
    class ArrayModel(BaseModel):
        values: NDArray[SerializedLabel] | None = None

    assert json.loads(ArrayModel().model_dump_json()) == {"values": None}
    assert json.loads(ArrayModel(values=[SerializedLabel("BUY")]).model_dump_json()) == {"values": ["BUY"]}


def test_custom_item_serializer_uses_context():
    import inspect

    if "context" not in inspect.signature(TypeAdapter.dump_json).parameters:
        pytest.skip("serialization context is unavailable in this Pydantic version")
    adapter = TypeAdapter(NDArray[SerializedLabel])
    array = adapter.validate_python([SerializedLabel("BUY")])
    assert json.loads(adapter.dump_json(array, context={"prefix": "trade:"})) == ["trade:BUY"]


def test_custom_item_serializer_preserves_model_options():
    class Payload(BaseModel):
        value: str = Field(alias="label")
        optional: str | None = None

    adapter = TypeAdapter(NDArray[Payload])
    array = adapter.validate_python([Payload(label="BUY"), Payload(label="SELL")])
    assert json.loads(adapter.dump_json(array, by_alias=True, exclude_none=True)) == [{"label": "BUY"}, {"label": "SELL"}]
    assert adapter.dump_python(array, include={1: {"value"}}) == [{"value": "SELL"}]


def test_custom_item_serializer_honors_ccflow_enum_names():
    class Side(Enum):
        BUY = Enum.auto()
        SELL = Enum.auto()

    adapter = TypeAdapter(NDArray[Side])
    array = adapter.validate_python(np.array([Side.BUY, Side.SELL], dtype=object))
    assert json.loads(adapter.dump_json(array)) == ["BUY", "SELL"]


def test_untyped_object_array_does_not_guess_custom_schema():
    with pytest.raises(PydanticSerializationError):
        TypeAdapter(NDArray).dump_json(np.array([SerializedLabel("BUY")], dtype=object))


def test_native_numpy_dtype_serialization_unchanged():
    adapter = TypeAdapter(NDArray[np.int64])
    array = adapter.validate_python([[1, 2], [3, 4]])
    assert json.loads(adapter.dump_json(array)) == [[1, 2], [3, 4]]


def test_numeric_array_does_not_apply_object_item_serializer():
    class NumericSide(IntEnum):
        BUY = 0
        SELL = 1

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return core_schema.is_instance_schema(
                cls,
                serialization=core_schema.plain_serializer_function_ser_schema(lambda value: value.name, when_used="json"),
            )

    adapter = TypeAdapter(NDArray[NumericSide])
    array = adapter.validate_python(np.array([NumericSide.BUY, NumericSide.SELL]))
    # NumPy has already coerced these values to integers, not enum instances.
    assert all(not isinstance(value, NumericSide) for value in array.flat)
    assert json.loads(adapter.dump_json(array)) == [0, 1]
