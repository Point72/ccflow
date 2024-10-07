import unittest
from typing import ClassVar, Dict, List, Optional, Type, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, ValidationError

from ccflow import BaseModel, NDArray, make_ndarray_orjson_valid
from ccflow.enums import Enum
from ccflow.exttypes.pydantic_numpy.ndtypes import bool_, complex64, float32, float64, int8, uint32


class ParentModel(BaseModel):
    field1: int


class ChildModel(ParentModel):
    field2: int


class NestedModel(BaseModel):
    a: ParentModel


class A(BaseModel):
    """Base class."""

    pass


class ArbitraryType:
    def __init__(self, x):
        self.x = x


class B(A):
    """B implements A and adds a json encoder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow z = MyClass, even though there is no validator

    x: ArbitraryType


class C(BaseModel):
    """C is composed of an A."""

    a: A


class MyEnum(Enum):
    FIRST = 1
    SECOND = 2


class D(BaseModel):
    value: MyEnum


class F(BaseModel):
    arr: NDArray


class G(BaseModel):
    foo: Optional[F] = None


class H_float64(BaseModel):
    arr: NDArray[float64]


class H_float32(BaseModel):
    arr: NDArray[float32]


class H_bool(BaseModel):
    arr: NDArray[bool_]


class H_complex64(BaseModel):
    arr: NDArray[complex64]


class H_uint32(BaseModel):
    arr: NDArray[uint32]


class H_int8(BaseModel):
    arr: NDArray[int8]


class TestBaseModelSerialization(unittest.TestCase):
    def _numpy_equality(self, val: BaseModel, other: BaseModel) -> bool:
        if val.__class__ == other.__class__ and len(val.__dict__) == len(other.__dict__):
            for k, v in val.__dict__.items():
                other_val = other.__dict__[k]
                if isinstance(v, np.ndarray):
                    np.testing.assert_array_equal(v, other.__dict__[k])
                else:
                    self.assertEqual(v, other_val)
        else:
            raise AssertionError

    def _check_serialization(self, model: BaseModel, equality_check=None):
        if equality_check is None:
            equality_check = self.assertEqual
        # Object serialization
        serialized = model.model_dump(mode="python")
        deserialized = type(model).model_validate(serialized)
        equality_check(model, deserialized)

        # JSON serialization
        serialized = model.model_dump_json()
        deserialized = type(model).model_validate_json(serialized)
        equality_check(model, deserialized)

    def test_make_ndarray_orjson_valid(self):
        try:
            make_ndarray_orjson_valid([9, 8])
        except TypeError:
            ...
        a = np.array([9, 8, 7, 12])
        self.assertTrue(a is make_ndarray_orjson_valid(a))
        b = a[::2]
        b_valid = make_ndarray_orjson_valid(b)
        # this is because b is not contiguous
        self.assertTrue(b is not b_valid)
        np.testing.assert_array_equal(b, b_valid)

        # complex values are not accepted by orjson currently
        complex_arr = np.array([0 + 3j, 7 + 2.1j], dtype=np.complex128)
        complex_arr_valid = make_ndarray_orjson_valid(complex_arr)
        self.assertTrue(not isinstance(complex_arr_valid, np.ndarray))
        self.assertTrue(complex_arr_valid == [0 + 3j, 7 + 2.1j])

    def test_serialization(self):
        self._check_serialization(ParentModel(field1=1))

    def test_serialization_subclass(self):
        self._check_serialization(ChildModel(field1=1, field2=2))

    def test_serialization_nested(self):
        self._check_serialization(NestedModel(a=ParentModel(field1=0)))

    def test_serialization_enum(self):
        self._check_serialization(D(value=MyEnum.FIRST))

    def test_serialization_nested_subclass(self):
        self._check_serialization(NestedModel(a=ChildModel(field1=0, field2=10)))

    def test_from_str_serialization(self):
        serialized = '{"_target_": "ccflow.tests.test_base_serialize.ChildModel", ' '"field1": 9, "field2": 4}'
        deserialized = BaseModel.model_validate_json(serialized)
        self.assertEqual(deserialized, ChildModel(field1=9, field2=4))

    def test_numpy_serialize(self):
        self._check_serialization(F(arr=np.array([9, 8])), self._numpy_equality)
        b = np.array([12, -11, 13, 14])
        b_skip = b[::2]
        self._check_serialization(F(arr=b_skip), self._numpy_equality)

        self._check_serialization(F(arr=[9, 0]), self._numpy_equality)
        self._check_serialization(G(), self._numpy_equality)
        for H in [H_float32, H_float64, H_uint32, H_bool]:
            self._check_serialization(H(arr=[1, 0]), self._numpy_equality)

        self._check_serialization(F(arr=["passes"]), self._numpy_equality)
        self.assertRaises(
            Exception,
            self._check_serialization,
            H_complex64(arr=[11, 12]),
            self._numpy_equality,
        )

        cut_off_array = H_int8(arr=[127, -128])
        np.testing.assert_array_equal(cut_off_array.arr, np.array([127, -128], dtype=np.int8))
        self._check_serialization(H_int8(arr=[127]))

    def test_base_model_config_inheritance(self):
        """Validate that pydantic model configs are inherited and defining configs in subclasses overrides only the
        configs set in the subclass."""

        class A(BaseModel):
            """No overriding of configs."""

            pass

        class B(BaseModel):
            """Override the config. Hopefully the config from our BaseModel doesn't get overridden."""

            model_config = ConfigDict(arbitrary_types_allowed=True)

        class C(PydanticBaseModel):
            """Override the config on a normal pydantic BaseModel (not our BaseModel)."""

            model_config = ConfigDict(arbitrary_types_allowed=True)

        self.assertRaises(ValidationError, A, extra_field1=1)

        # If configs are not inherited, B should allow extra fields.
        self.assertRaises(ValidationError, B, extra_field1=1)

        # C implements the normal pydantic BaseModel whichhould allow extra fields.
        _ = C(extra_field1=1)

    def test_serialize_as_any(self):
        # https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
        # https://github.com/pydantic/pydantic/issues/6423
        # This test could be removed once there is a different solution to the issue above
        from pydantic import SerializeAsAny
        from pydantic.types import constr

        class MyNestedModel(BaseModel):
            a1: A
            a2: Optional[Union[A, int]]
            a3: Dict[str, Optional[List[A]]]
            a4: ClassVar[A]
            a5: Type[A]
            a6: constr(min_length=1)

        target = {
            "a1": SerializeAsAny[A],
            "a2": Optional[Union[SerializeAsAny[A], int]],
            "a4": ClassVar[SerializeAsAny[A]],
            "a5": Type[A],
            "a6": constr(min_length=1),  # Uses Annotation
        }
        target["a3"] = dict[str, Optional[list[SerializeAsAny[A]]]]
        annotations = MyNestedModel.__annotations__
        self.assertEqual(str(annotations["a1"]), str(target["a1"]))
        self.assertEqual(str(annotations["a2"]), str(target["a2"]))
        self.assertEqual(str(annotations["a3"]), str(target["a3"]))
        self.assertEqual(str(annotations["a4"]), str(target["a4"]))
        self.assertEqual(str(annotations["a5"]), str(target["a5"]))
        self.assertEqual(str(annotations["a6"]), str(target["a6"]))
