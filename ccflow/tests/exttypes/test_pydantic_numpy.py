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

from typing import Dict, Optional, Union

import numpy as np
from numpy.testing import assert_allclose
from pydantic import BaseModel

from ccflow import NDArray, float32


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
    assert jdata["K"] == list([1, 2])

    # Test round-trip
    cfg2 = MySettingsNoGeneric.model_validate_json(cfg.model_dump_json())
    np.testing.assert_array_equal(cfg.K, cfg2.K)


def test_optional_construction():
    class MySettingsOptional(BaseModel):
        K: Optional[NDArray[float32]] = None

    cfg = MySettingsOptional()
    assert cfg.K is None

    cfg = MySettingsOptional(K=[1, 2])
    assert type(cfg.K) is np.ndarray
    assert cfg.K.dtype == np.float32


def test_subclass_basemodel():
    class MyModelField(BaseModel):
        K: NDArray[float32]

    class MyModel(BaseModel):
        L: Dict[str, MyModelField]

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
    x: Union[NDArray[float32], Dict[int, NDArray[float32]]]


def test_union_field():
    # Array
    model = MyDictArrayModel(x=[0, 1, 2])
    json = model.json()
    model2 = MyDictArrayModel.parse_raw(json)
    np.testing.assert_array_equal(model2.x, np.array([0.0, 1.0, 2.0]))

    # Dictionary
    model = MyDictArrayModel(x={0: [0, 1, 2]})
    json = model.json()
    model2 = MyDictArrayModel.parse_raw(json)
    assert 0 in model2.x
    assert len(model2.x) == 1
    np.testing.assert_array_equal(model2.x[0], np.array([0.0, 1.0, 2.0]))
