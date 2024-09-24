from typing import ClassVar

from pydantic import model_validator, ConfigDict, PrivateAttr

from ccflow.base import ResultBase, make_lazy_result


class MyResult(ResultBase):
    total: ClassVar[int] = 0  # To track instantiations
    value: bool = False
    model_config = ConfigDict(extra='allow')
    _private: str = PrivateAttr(default='bar')

    @model_validator(mode="after")
    def _validate(self):
        # Track construction by incrementing the total each time the validation is called
        MyResult.total += 1


def test_make_lazy_result():
    assert MyResult.total == 0
    result = MyResult()
    assert MyResult.total == 1
    assert not result.value

    lazy_result = make_lazy_result(MyResult, lambda: MyResult(value=True))
    assert isinstance(lazy_result, MyResult)
    assert MyResult.total == 1  # Constructing the lazy result did not increment the total
    assert lazy_result.value
    assert MyResult.total == 2  # Accessing the value in the line above did increment the total
    assert lazy_result == MyResult(value=True)

    result = MyResult(value=True, extra_field='foo')
    assert "value" in result.__pydantic_fields_set__
    assert result.__pydantic_extra__['extra_field'] == 'foo'
    assert result.__pydantic_private__['_private'] == 'bar'

    lazy_result = make_lazy_result(MyResult, lambda: MyResult(value=True, extra_field='foo'))
    assert "value" in lazy_result.__pydantic_fields_set__
    assert lazy_result.__pydantic_extra__['extra_field'] == 'foo'
    assert lazy_result.__pydantic_private__['_private'] == 'bar'