from typing import Any, Iterable, Set, TypeVar, Union

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import create_model

__all__ = (
    "PydanticModelType",
    "PydanticDictOptions",
    "models_to_pandas",
    "dict_to_model",
)

PydanticModelType = TypeVar("ModelType", bound=PydanticBaseModel)


class PydanticDictOptions(PydanticBaseModel):
    """See https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeldict"""

    include: Set[str] = None
    exclude: Set[str] = set()
    by_alias: bool = False
    exclude_unset: bool = False
    exclude_defaults: bool = False
    exclude_none: bool = False

    class Config:
        # Want to validate assignment so that if lists are assigned to include/exclude, they get validated
        validate_assignment = True


_DEFAULT_OPTIONS = PydanticDictOptions()


def dict_to_model(cls, v) -> PydanticBaseModel:
    """Validator to coerce dict to a pydantic base model without loss of data when no type specified.
    Without it, dict is coerced to PydanticBaseModel, losing all data.
    """
    if isinstance(v, dict):

        class Config:
            arbitrary_types_allowed = True

        fields = {f: (Any, None) for f in v}
        v = create_model("DynamicDictModel", **fields, __config__=Config)(**v)
    return v


def models_to_pandas(
    models: Union[PydanticBaseModel, Iterable[PydanticBaseModel]], options: PydanticDictOptions = _DEFAULT_OPTIONS, **kwargs
) -> pd.DataFrame:
    """Converts a pydantic model or collection of models to a pandas DataFrame using pd.json_normalize.

    See https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html for more info.
    """
    if isinstance(models, PydanticBaseModel):
        models = [models]
    data = [model.model_dump(**options.model_dump(mode="python")) for model in models]
    return pd.json_normalize(data, **kwargs)


# TODO: The inverse function (pandas to list of models) is not yet implemented
#   See https://stackoverflow.com/questions/54776916/inverse-of-pandas-json-normalize
