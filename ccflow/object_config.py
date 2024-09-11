from typing import Any, Dict

from pydantic import PrivateAttr, model_validator
from pydantic.fields import Field

from .base import BaseModel
from .exttypes.pyobjectpath import PyObjectPath

__all__ = (
    "ObjectConfig",
    "LazyObjectConfig",
)


class ObjectConfig(BaseModel):  # TODO: Generic model version for type checking
    """Small class to help wrap an arbitrary python object as a BaseModel.

    This allows such objects to be registered by name in the registry,
    without having to define a custom pydantic wrapper for them.
    """

    object_type: PyObjectPath = Field(
        None,
        description="The type of the object this model wraps.",
    )
    object_kwargs: Dict[str, Any] = {}
    _object: Any = PrivateAttr(None)

    class Config(BaseModel.Config):
        ignored_types = (property,)
        extra = "allow"
        frozen = True  # Because we cache _object

    @model_validator(mode="wrap")
    def _kwarg_validator(cls, values, handler, info):
        if isinstance(values, dict):
            # Uplift extra fields into object_kwargs
            obj_kwargs = values.get("object_kwargs", {})
            for field in list(values):
                if field not in cls.model_fields:
                    obj_kwargs[field] = values.pop(field)
            values["object_kwargs"] = obj_kwargs
        return handler(values)

    def __getstate__(self):
        """Override pickling to ignore _object, so that configs can be pickled even if the underlying object cannot."""
        state_dict = self.__dict__.copy()
        state_dict.pop("_object", None)
        return {
            "__dict__": state_dict,
            "__pydantic_fields_set__": self.__pydantic_fields_set__,
            "__pydantic_extra__": self.__pydantic_extra__,
            "__pydantic_private__": self.__pydantic_private__,
        }

    def __setstate__(self, state):
        super().__setstate__(state)
        self._object = self.object_type.object(**self.object_kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Eagerly construct object. This way, if it fails to construct, it will be known immediately.
        self._object = self.object_type.object(**self.object_kwargs)

    @property
    def object(self):
        """Returns the pre-constructed object corresponding to the config."""
        return self._object


class LazyObjectConfig(ObjectConfig):
    """Like ObjectConfig, but the object is constructed lazily (on first access).

    One loses upfront validation that it's a valid config, but potentially gains performance benefits
    of not constructing unneeded objects.
    """

    def __init__(self, *args, **kwargs):
        super(ObjectConfig, self).__init__(*args, **kwargs)

    @property
    def object(self):
        """Returns the lazily-constructed object corresponding to the config."""
        if self._object is None:
            self._object = self.object_type.object(**self.object_kwargs)
        return self._object

    def __setstate__(self, state):
        super().__setstate__(state)
        self._object = None
