from typing import Any, Set, TypeVar

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, create_model

__all__ = (
    "PydanticModelType",
    "PydanticDictOptions",
    "dict_to_model",
)

PydanticModelType = TypeVar("ModelType", bound=PydanticBaseModel)


class PydanticDictOptions(PydanticBaseModel):
    """See https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeldict"""

    model_config = ConfigDict(
        # Want to validate assignment so that if lists are assigned to include/exclude, they get validated
        validate_assignment=True
    )

    include: Set[str] = None
    exclude: Set[str] = set()
    by_alias: bool = False
    exclude_unset: bool = False
    exclude_defaults: bool = False
    exclude_none: bool = False


def dict_to_model(cls, v) -> PydanticBaseModel:
    """Validator to coerce dict to a pydantic base model without loss of data when no type specified.
    Without it, dict is coerced to PydanticBaseModel, losing all data.
    """
    if isinstance(v, dict):
        config = ConfigDict(arbitrary_types_allowed=True)

        fields = {f: (Any, None) for f in v}
        v = create_model("DynamicDictModel", **fields, __config__=config)(**v)
    return v
