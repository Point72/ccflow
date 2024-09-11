import importlib
import json
from typing import Dict

import pytest
from pydantic import BaseModel, RootModel

from ccflow.enums import Enum, auto


class MyEnum(Enum):
    FIELD1 = auto()
    FIELD2 = auto()


class MyModel(BaseModel):
    enum: MyEnum
    enum_default: MyEnum = MyEnum.FIELD1


class MyDictModel(BaseModel):
    enum_dict: Dict[MyEnum, int] = None

    class Config:
        use_enum_values = True


def test_validation():
    assert MyModel(enum="FIELD2").enum == MyEnum.FIELD2
    assert MyModel(enum=0).enum == MyEnum.FIELD1
    assert MyModel(enum=MyEnum.FIELD1).enum == MyEnum.FIELD1
    with pytest.raises(ValueError):
        MyModel(enum=3.14)


def test_dict():
    assert dict(MyModel(enum=MyEnum.FIELD2)) == {"enum": MyEnum.FIELD2, "enum_default": MyEnum.FIELD1}
    assert MyModel(enum=MyEnum.FIELD2).model_dump(mode="python") == {"enum": MyEnum.FIELD2, "enum_default": MyEnum.FIELD1}
    assert MyModel(enum=MyEnum.FIELD2).model_dump(mode="json") == {"enum": "FIELD2", "enum_default": "FIELD1"}


def test_serialization():
    assert "enum" in MyModel.model_fields
    assert "enum_default" in MyModel.model_fields
    tm = MyModel(enum=MyEnum.FIELD2)
    assert json.loads(tm.model_dump_json()) == json.loads('{"enum": "FIELD2", "enum_default": "FIELD1"}')


class DictWrapper(RootModel[Dict[MyEnum, int]]):
    def __getitem__(self, item):
        return self.root[item]

    class Config:
        use_enum_values = True


class MyDictWrapperModel(BaseModel):
    enum_dict: DictWrapper

    class Config:
        use_enum_values = True


def test_enum_as_dict_key_json_serialization():
    dict_model = MyDictModel(enum_dict={MyEnum.FIELD1: 8, MyEnum.FIELD2: 19})
    assert dict_model.enum_dict[MyEnum.FIELD1] == 8
    assert dict_model.enum_dict[MyEnum.FIELD2] == 19

    assert json.loads(dict_model.model_dump_json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')

    dict_wrapper_model = MyDictWrapperModel(enum_dict=DictWrapper({MyEnum.FIELD1: 8, MyEnum.FIELD2: 19}))

    assert dict_wrapper_model.enum_dict[MyEnum.FIELD1] == 8
    assert dict_wrapper_model.enum_dict[MyEnum.FIELD2] == 19
    assert json.loads(dict_wrapper_model.model_dump_json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')


def test_json_schema_csp():
    if not importlib.util.find_spec("csp"):
        pytest.skip("Skipping test because csp not installed")
    assert MyModel.model_json_schema() == {
        "properties": {
            "enum": {"description": "An enumeration of MyEnum", "enum": ["FIELD1", "FIELD2"], "title": "MyEnum", "type": "string"},
            "enum_default": {
                "default": "FIELD1",
                "description": "An enumeration of MyEnum",
                "enum": ["FIELD1", "FIELD2"],
                "title": "MyEnum",
                "type": "string",
            },
        },
        "required": ["enum"],
        "title": "MyModel",
        "type": "object",
    }


def test_json_schema_no_csp():
    if importlib.util.find_spec("csp"):
        pytest.skip("Skipping test because csp installed")

    assert MyModel.model_json_schema() == {
        "properties": {
            "enum": {"description": "An enumeration.", "enum": ["FIELD1", "FIELD2"], "title": "MyEnum", "type": "string"},
            "enum_default": {
                "default": "FIELD1",
                "description": "An enumeration.",
                "enum": ["FIELD1", "FIELD2"],
                "title": "MyEnum",
                "type": "string",
            },
        },
        "required": ["enum"],
        "title": "MyModel",
        "type": "object",
    }
