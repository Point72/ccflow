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
import pytest
from numpy.testing import assert_allclose
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Dict, Optional

from ccflow import NDArray, PotentialNDArray, float32, orjson_dumps
from ccflow.exttypes.pydantic_numpy.ndarray import NPFileDesc


class MySettings(BaseModel):
    K: NDArray[float32]

    class Config:
        json_dumps = orjson_dumps


def test_init_from_values():
    # Directly specify values
    cfg = MySettings(K=[1, 2])
    assert_allclose(cfg.K, [1.0, 2.0])
    assert cfg.K.dtype == np.float32
    assert cfg.json()

    cfg = MySettings(K=np.eye(2))
    assert_allclose(cfg.K, [[1.0, 0], [0.0, 1.0]])
    assert cfg.K.dtype == np.float32


def test_load_from_npy_path(tmpdir):
    # Load from npy
    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npy"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_load_from_NPFileDesc(tmpdir):
    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K=NPFileDesc(path=Path(tmpdir) / "data.npy"))
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_load_field_from_npz(tmpdir):
    np.savez(Path(tmpdir) / "data.npz", values=np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npz", "key": "values"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_exceptional(tmpdir):
    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npz", "key": "values"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy", "key": "nosuchkey"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy"})

    with pytest.raises(ValidationError):
        MySettings(K="absc")


def test_unspecified_npdtype():
    # Not specifying a dtype will use numpy default dtype resolver

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

    cfg = MySettingsNoGeneric(K=[1, 2])
    assert_allclose(cfg.K, [1, 2])
    assert cfg.K.dtype == int


def test_json_encoders():
    import orjson

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

        class Config:
            json_dumps = orjson_dumps

    cfg = MySettingsNoGeneric(K=[1, 2])
    jdata = orjson.loads(cfg.json())

    assert "K" in jdata
    assert isinstance(jdata["K"], list)
    assert jdata["K"] == list([1, 2])


def test_optional_construction():
    class MySettingsOptional(BaseModel):
        K: Optional[NDArray[float32]] = None

    cfg = MySettingsOptional()
    assert cfg.K is None

    cfg = MySettingsOptional(K=[1, 2])
    assert type(cfg.K) is np.ndarray
    assert cfg.K.dtype == np.float32


def test_potential_array(tmpdir):
    class MySettingsPotential(BaseModel):
        K: PotentialNDArray[float32]

    np.savez(Path(tmpdir) / "data.npz", values=np.arange(5))

    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "data.npz", "key": "values"})
    assert cfg.K is not None
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])

    # Path not found
    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "nothere.npz", "key": "values"})
    assert cfg.K is None

    # Key not there
    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "data.npz", "key": "nothere"})
    assert cfg.K is None


def test_subclass_basemodel():
    class MyModelField(BaseModel):
        K: NDArray[float32]

        class Config:
            json_dumps = orjson_dumps

    class MyModel(BaseModel):
        L: Dict[str, MyModelField]

        class Config:
            json_dumps = orjson_dumps

    model_field = MyModelField(K=[1.0, 2.0])
    assert model_field.json()

    model = MyModel(L={"a": MyModelField(K=[1.0, 2.0])})
    assert model.L["a"].K.dtype == np.dtype("float32")
    assert model.json()


# We have added the tests below
def test_default_value():
    """Numpy type doesn't work with default values.
    The work-around is to use a default factory.
    See https://github.com/samuelcolvin/pydantic/issues/2923#issuecomment-885788163
    """
    from pydantic import Field

    class MyModelField(BaseModel):
        K: NDArray[float32] = Field(default_factory=lambda: np.array([1.0, 2.0]))

        class Config:
            json_dumps = orjson_dumps

    model_field = MyModelField()
    assert model_field.json()
