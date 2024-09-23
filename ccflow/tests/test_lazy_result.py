from typing import ClassVar

from pydantic import model_validator

from ccflow.base import ResultBase, make_lazy_result


class MyResult(ResultBase):
    total: ClassVar[int] = 0  # To track instantiations
    value: bool = False

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
