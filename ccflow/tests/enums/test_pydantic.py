import importlib
import json
import pydantic
import pytest
from packaging import version
from pydantic import BaseModel
from typing import Dict

from ccflow.enums import Enum, auto
from ccflow.utils.pydantic1to2 import ValidationTypeError


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
    with pytest.raises(ValidationTypeError):
        MyModel(enum=3.14)


def test_dict():
    assert dict(MyModel(enum=MyEnum.FIELD2)) == {"enum": MyEnum.FIELD2, "enum_default": MyEnum.FIELD1}
    assert MyModel(enum=MyEnum.FIELD2).dict() == {"enum": MyEnum.FIELD2, "enum_default": MyEnum.FIELD1}
    if version.parse(pydantic.__version__) >= version.parse("2"):
        assert MyModel(enum=MyEnum.FIELD2).model_dump(mode="python") == {"enum": MyEnum.FIELD2, "enum_default": MyEnum.FIELD1}
        assert MyModel(enum=MyEnum.FIELD2).model_dump(mode="json") == {"enum": "FIELD2", "enum_default": "FIELD1"}


def test_serialization():
    assert "enum" in MyModel.__fields__
    assert "enum_default" in MyModel.__fields__
    tm = MyModel(enum=MyEnum.FIELD2)
    assert json.loads(tm.json()) == json.loads('{"enum": "FIELD2", "enum_default": "FIELD1"}')


if version.parse(pydantic.__version__) < version.parse("2"):

    class DictWrapper(BaseModel):
        __root__: Dict[MyEnum, int]

        def __getitem__(self, item):
            return self.__root__[item]

        class Config:
            use_enum_values = True

    class MyDictWrapperModel(BaseModel):
        enum_dict: DictWrapper

        class Config:
            use_enum_values = True

    # TODO: Enums as dict keys are not json serializeable
    def test_enum_as_dict_key_fails_json_serialization():
        dict_model = MyDictModel(enum_dict={MyEnum.FIELD1: 8, MyEnum.FIELD2: 19})
        assert dict_model.enum_dict[MyEnum.FIELD1] == 8
        assert dict_model.enum_dict[MyEnum.FIELD2] == 19

        with pytest.raises(TypeError):
            dict_model.json()

        dict_wrapper_model = MyDictWrapperModel(enum_dict=DictWrapper(__root__={MyEnum.FIELD1: 8, MyEnum.FIELD2: 19}))

        assert dict_wrapper_model.enum_dict[MyEnum.FIELD1] == 8
        assert dict_wrapper_model.enum_dict[MyEnum.FIELD2] == 19
        with pytest.raises(TypeError):
            dict_wrapper_model.json()

else:
    from pydantic import RootModel

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

        assert json.loads(dict_model.json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')

        dict_wrapper_model = MyDictWrapperModel(enum_dict=DictWrapper({MyEnum.FIELD1: 8, MyEnum.FIELD2: 19}))

        assert dict_wrapper_model.enum_dict[MyEnum.FIELD1] == 8
        assert dict_wrapper_model.enum_dict[MyEnum.FIELD2] == 19
        assert json.loads(dict_wrapper_model.json()) == json.loads('{"enum_dict":{"FIELD1":8,"FIELD2":19}}')


def test_json_schema_csp():
    if not importlib.util.find_spec("csp"):
        pytest.skip("Skipping test because csp not installed")
    if version.parse(pydantic.__version__) < version.parse("2"):
        assert MyModel.schema() == {
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
    else:
        assert MyModel.schema() == {
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

    if version.parse(pydantic.__version__) < version.parse("2"):
        assert MyModel.schema() == {
            "definitions": {"MyEnum": {"description": "An enumeration.", "enum": ["FIELD1", "FIELD2"], "title": "MyEnum", "type": "string"}},
            "properties": {"enum": {"$ref": "#/definitions/MyEnum"}, "enum_default": {"allOf": [{"$ref": "#/definitions/MyEnum"}], "default": 0}},
            "required": ["enum"],
            "title": "MyModel",
            "type": "object",
        }
    else:
        assert MyModel.schema() == {
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
