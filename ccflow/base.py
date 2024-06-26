"""This module defines the base model and registry for flow."""

import copy
import logging
import numpy as np
import orjson
import pandas as pd
import pathlib
import pydantic
from datetime import timedelta
from omegaconf import DictConfig
from packaging import version
from pydantic import BaseModel as PydanticBaseModel, PrivateAttr, ValidationError, root_validator, validator
from pydantic.fields import Field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar, get_args, get_origin

from .enums import Enum
from .exttypes.pandas import GenericPandasWrapper
from .exttypes.pyobjectpath import PyObjectPath
from .serialization import make_ndarray_orjson_valid, orjson_dumps

if version.parse(pydantic.__version__) < version.parse("2"):
    pass
else:
    from pydantic import TypeAdapter, model_validator


log = logging.getLogger(__name__)

__all__ = (
    "model_alias",
    "BaseModel",
    "ModelRegistry",
    "ModelType",
    "RegistryLookupContext",
    "RootModelRegistry",
    "REGISTRY_SEPARATOR",
    "ContextBase",
    "ContextType",
    "ResultBase",
    "ResultType",
)


REGISTRY_SEPARATOR = "/"


class RegistryKeyError(KeyError): ...


class _RegistryMixin:
    def get_registrations(self) -> List[Tuple["ModelRegistry", str]]:
        """Return the set of registrations that has happened for this model"""
        return self._registrations.copy()

    def get_registered_names(self) -> List[str]:
        """Return the set of names for this model in the root registry."""
        if self._registrations:
            full_names = []
            for registry, name in self._registrations:
                registry_names = registry.get_registered_names()
                for registry_name in registry_names:
                    full_names.append(REGISTRY_SEPARATOR.join([registry_name, name]))
            return full_names
        elif self is ModelRegistry.root():
            return [""]
        return []

    def get_registry_dependencies(self, types: Optional[Tuple["ModelType"]] = None) -> List[List[str]]:
        """Return the set of registered models that are contained by this model.
        It only returns names that are relative to the root registry.
        It types is specified, will only specify dependencies of the given model sub-types.
        """
        deps = []
        for _field_name, value in self:
            deps.extend(_get_registry_dependencies(value, types))
        return deps


if version.parse(pydantic.__version__) < version.parse("2"):

    class BaseModel(PydanticBaseModel, _RegistryMixin):
        """BaseModel is a base class for all pydantic models within the cubist flow framework.

        This gives us a way to add functionality to the framework, including
            - Type of object is part of serialization/deserialization
            - Registration by name, and coercion from string name
        """

        # We want to save the full path to the class type as part of the model when exporting
        # so that it's clear how to create it again.
        # Ideally, we would like to use _target_ as a field name, so that this lines up
        # with hydra. Unfortunately pydantic does not support "public" underscore fields
        # The suggested workaround is to use aliases
        # See https://github.com/samuelcolvin/pydantic/issues/288
        type_: PyObjectPath = Field(
            None,
            alias="_target_",
            repr=False,
            description="The (sub)type of BaseModel to be included in serialization "
            "to allow for faithful deserialization using hydra.utils.instantiate based on '_target_',"
            "which is the hydra convention.",
        )

        # We want to track under what names a model has been registered
        _registrations: List[Tuple["ModelRegistry", str]] = PrivateAttr(default_factory=list)

        class Config:
            validate_assignment = True
            # The flag below is needed to make validate_assignment work properly with aliases
            allow_population_by_field_name = True
            # Lots of bugs happen because of a mis-named field with a default value,
            # where the default behavior is just to drop the mis-named value. This prevents that
            extra = "forbid"
            json_loads = orjson.loads
            json_dumps = orjson_dumps
            json_encoders = {
                GenericPandasWrapper: lambda x: type(x).encode(x),
                np.ndarray: make_ndarray_orjson_valid,
                # encode is a classmethod
                pd.Index: lambda x: make_ndarray_orjson_valid(x.to_numpy()),
                Enum: lambda x: x.name,
                timedelta: lambda x: x.total_seconds(),
            }

        def __str__(self):
            # Because the standard string representation does not include class name
            return repr(self)

        def get_widget(
            self,
            json_kwargs: Optional[Dict[str, Any]] = None,
            widget_kwargs: Optional[Dict[str, Any]] = None,
        ):
            """Get an IPython widget to view the object."""
            from IPython.display import JSON

            kwargs = {"encoder": str}
            kwargs.update(json_kwargs or {})
            return JSON(self.json(**kwargs), **(widget_kwargs or {}))

        @root_validator(pre=True)
        def _type_validate(cls, values):
            values.pop("_target_", None)
            values["type_"] = PyObjectPath.validate(cls)
            return values

        @classmethod
        def validate(cls, v, field=None):
            """Validation will lookup a string in the root registry,
            and return the registered model if found."""
            if isinstance(v, str):
                return resolve_str(v)

            # If we already have an instance, run parent validation.
            if isinstance(v, cls):
                return super(BaseModel, cls).validate(v)

            # Look for type data on the object, because if it's a sub-class, need to instantiate it explicitly
            if isinstance(v, dict):
                if "_target_" in v:
                    type_ = v["_target_"]
                else:
                    type_ = v.get("type_")
            else:
                type_ = getattr(v, "type_", None)

            if type_ is not None:
                type_cls = PyObjectPath(type_).object
                if cls != type_cls:
                    v = type_cls.validate(v)
            return super(BaseModel, cls).validate(v)

        @classmethod
        def parse_obj(cls, obj):
            return cls.validate(obj)

        def __init_subclass__(cls, *args, **kwargs):
            if hasattr(cls, "Config") and hasattr(cls.Config, "json_encoders"):
                # Register the JSON encoder of subclasses to the BaseModel so that we can serialize from a reference to
                # the base class.
                BaseModel.__config__.json_encoders.update(cls.Config.json_encoders)
            super(BaseModel, cls).__init_subclass__(*args, **kwargs)

    class _ModelRegistryData(PydanticBaseModel):
        """A data structure representation of the model registry, without the associated functionality"""

        type_: PyObjectPath = Field(
            alias="_target_",
            repr=False,
        )
        name: str
        models: Dict[str, BaseModel]

else:
    import collections.abc
    import inspect
    import platform
    import typing
    import typing_extensions
    from pydantic import SerializeAsAny, computed_field, model_serializer, model_validator

    # Pydantic 2 has different handling of serialization.
    # This requires some workarounds at the moment until the feature is added to easily get a mode that
    # is compatible with Pydantic 1
    # This is done by adjusting annotations via a MetaClass for any annotation that includes a BaseModel,
    # such that the new annotation contains SerializeAsAny
    # https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
    # https://github.com/pydantic/pydantic/issues/6423
    # https://github.com/pydantic/pydantic-core/pull/740
    # See https://github.com/pydantic/pydantic/issues/6381 for inspiration on implementation
    from pydantic._internal._model_construction import ModelMetaclass

    # Required for py38 compatibility
    # In python 3.8, get_origin(List[float]) returns list, but you can't call list[float] to retrieve the annotation
    # Furthermore, Annotated is part of typing_Extensions and get_origin(Annotated[str, ...]) returns str rather than Annotated
    _IS_PY38 = version.parse(platform.python_version()) < version.parse("3.9")
    # For a more complete list, see https://github.com/alexmojaki/eval_type_backport/blob/main/eval_type_backport/eval_type_backport.py
    _PY38_ORIGIN_MAP = {
        tuple: typing.Tuple,
        list: typing.List,
        dict: typing.Dict,
        set: typing.Set,
        frozenset: typing.FrozenSet,
        collections.abc.Callable: typing.Callable,
        collections.abc.Iterable: typing.Iterable,
        collections.abc.Mapping: typing.Mapping,
        collections.abc.MutableMapping: typing.MutableMapping,
        collections.abc.Sequence: typing.Sequence,
    }

    def _adjust_annotations(annotation):
        origin = get_origin(annotation)
        if _IS_PY38:
            if isinstance(annotation, typing_extensions._AnnotatedAlias):
                return annotation
            else:
                origin = _PY38_ORIGIN_MAP.get(origin, origin)
        args = get_args(annotation)
        if inspect.isclass(annotation) and issubclass(annotation, PydanticBaseModel):
            return SerializeAsAny[annotation]
        elif origin and args:
            # Filter out typing.Type and generic types
            if origin is type or (inspect.isclass(origin) and issubclass(origin, Generic)):
                return annotation
            elif origin is ClassVar:  # ClassVar doesn't accept a tuple of length 1 in py39
                return ClassVar[_adjust_annotations(args[0])]
            else:
                try:
                    return origin[tuple(_adjust_annotations(arg) for arg in args)]
                except TypeError:
                    raise TypeError(f"Could not adjust annotations for {origin}")
        else:
            return annotation

    class _SerializeAsAnyMeta(ModelMetaclass):
        def __new__(self, name: str, bases: Tuple[type], namespaces: Dict[str, Any], **kwargs):
            annotations: dict = namespaces.get("__annotations__", {})

            for base in bases:
                for base_ in base.__mro__:
                    if base_ is PydanticBaseModel:
                        annotations.update(base_.__annotations__)

            for field, annotation in annotations.items():
                if not field.startswith("__"):
                    annotations[field] = _adjust_annotations(annotation)

            namespaces["__annotations__"] = annotations

            return super().__new__(self, name, bases, namespaces, **kwargs)

    class BaseModel(PydanticBaseModel, _RegistryMixin, metaclass=_SerializeAsAnyMeta):
        """BaseModel is a base class for all pydantic models within the cubist flow framework.

        This gives us a way to add functionality to the framework, including
            - Type of object is part of serialization/deserialization
            - Registration by name, and coercion from string name
        """

        @computed_field(
            alias="_target_",
            repr=False,
            description="The (sub)type of BaseModel to be included in serialization "
            "to allow for faithful deserialization using hydra.utils.instantiate based on '_target_',"
            "which is the hydra convention.",
        )
        @property
        def type_(self) -> PyObjectPath:
            return PyObjectPath.validate(type(self))

        # We want to track under what names a model has been registered
        _registrations: List[Tuple["ModelRegistry", str]] = PrivateAttr(default_factory=list)

        # Don't use ConfigDict/model_config here because many subclasses are still using ConfigDict
        class Config:
            validate_assignment = True
            populate_by_name = True
            coerce_numbers_to_str = True  # New in v2 for backwards compatibility with V1
            # Lots of bugs happen because of a mis-named field with a default value,
            # where the default behavior is just to drop the mis-named value. This prevents that
            extra = "forbid"
            ser_json_timedelta = "float"
            json_encoders = {
                GenericPandasWrapper: lambda x: type(x).encode(x),
                pd.Index: lambda x: make_ndarray_orjson_valid(x.to_numpy()),
            }

        def __str__(self):
            # Because the standard string representation does not include class name
            return repr(self)

        def __eq__(self, other: Any) -> bool:
            # Override the method from pydantic's base class so as not to include private attributes,
            # which was a change made in V2 (https://docs.pydantic.dev/latest/migration/)
            if isinstance(other, BaseModel):
                # When comparing instances of generic types for equality, as long as all field values are equal,
                # only require their generic origin types to be equal, rather than exact type equality.
                # This prevents headaches like MyGeneric(x=1) != MyGeneric[Any](x=1).
                self_type = self.__pydantic_generic_metadata__["origin"] or self.__class__
                other_type = other.__pydantic_generic_metadata__["origin"] or other.__class__
                return self_type == other_type and self.__dict__ == other.__dict__
            else:
                return NotImplemented  # delegate to the other item in the comparison

        def get_widget(
            self,
            json_kwargs: Optional[Dict[str, Any]] = None,
            widget_kwargs: Optional[Dict[str, Any]] = None,
        ):
            """Get an IPython widget to view the object."""
            from IPython.display import JSON

            kwargs = {"fallback": str, "mode": "json"}
            kwargs.update(json_kwargs or {})
            # Can't use self.model_dump_json or self.model_dump because they don't expose the fallback argument
            return JSON(self.__pydantic_serializer__.to_python(self, **kwargs), **(widget_kwargs or {}))

        @model_validator(mode="wrap")
        def _base_model_validator(cls, v, handler, info):
            if isinstance(v, str):
                try:
                    v = resolve_str(v)
                except RegistryKeyError as e:
                    # Need to throw a value error so that validation of Unions works properly.
                    raise ValueError from e
                return handler(v)

            # If we already have an instance, run parent validation.
            if isinstance(v, cls):
                return handler(v)

            # Look for type data on the object, because if it's a sub-class, need to instantiate it explicitly
            if isinstance(v, dict):
                type_ = None
                if "_target_" in v:
                    v = v.copy()
                    type_ = v.pop("_target_")
                if "type_" in v:
                    v = v.copy()
                    type_ = v.pop("type_")

                if type_ is not None:
                    type_cls = PyObjectPath(type_).object
                    if cls != type_cls:
                        return type_cls.model_validate(v)

            if isinstance(v, PydanticBaseModel):
                # Coerce from one BaseModel type to another (because it worked automatically in v1)
                v = v.model_dump(exclude={"type_"})

            return handler(v)

    class _ModelRegistryData(PydanticBaseModel):
        """A data structure representation of the model registry, without the associated functionality"""

        type_: PyObjectPath = Field(
            alias="_target_",
            repr=False,
        )
        name: str
        models: SerializeAsAny[Dict[str, BaseModel]]


def _get_registry_dependencies(value, types: Optional[Tuple[Type]]) -> List[List[str]]:
    deps = []
    if isinstance(value, BaseModel):
        if not types or isinstance(value, types):
            names = value.get_registered_names()
            if names:
                deps.append(names)
    if isinstance(value, PydanticBaseModel):
        for _field_name, v in value:
            deps.extend(_get_registry_dependencies(v, types))
    elif isinstance(value, dict):
        for k, v in value.items():
            deps.extend(_get_registry_dependencies(k, types))
            deps.extend(_get_registry_dependencies(v, types))
    elif isinstance(value, (list, tuple)):
        for v in value:
            deps.extend(_get_registry_dependencies(v, types))

    return deps


def _is_config_subregistry(value):
    """Test whether a config value is a subregistry, i.e. it is a dict which either
    contains a _target_ key, or recursively contains a dict that has a _target_ key.
    """
    if isinstance(value, (dict, DictConfig)):
        if "_target_" in value:
            return True
        else:
            for v in value.values():
                if _is_config_subregistry(v):
                    return True
    return False


def model_alias(model_name: str) -> BaseModel:
    """Function to alias a BaseModel by name in the root registry.

    Useful for configs in hydra where we want a config object to point directly to another config object, using
    _target_: ccflow.alias
    model_name: foo
    """
    return BaseModel.validate(model_name)


ModelType = TypeVar("ModelType", bound=BaseModel)


class ModelRegistry(BaseModel):
    """ModelRegistry represents a named collection of models.

    Because we want to be careful about how models are added and removed, the dict structure is not public.
    """

    name: str = Field(
        default="",
        description="The 'name' of the registry, purely for descriptive purposes",
    )
    _models: Dict[str, BaseModel] = PrivateAttr({})

    def __eq__(self, other: Any) -> bool:
        # Since our BaseModel ignored private attributes, the registry needs to explicitly compare them
        # Note that we want RootModelRegistry to compare as equal to a ModelRegistry, so we use isinstance.
        return isinstance(other, BaseModel) and self.name == other.name and self._models == other._models

    def __init__(self, *args, **kwargs):
        models = {}
        if "models" in kwargs:
            models = kwargs.pop("models")
            if not isinstance(models, dict):
                raise TypeError("models must be a dict")
        super(ModelRegistry, self).__init__(*args, **kwargs)
        for name, model in models.items():
            self.add(name, model)

    @validator("name")
    def _validate_name(cls, v):
        if not v:
            raise ValueError("name must be non-empty")
        return v

    if version.parse(pydantic.__version__) < version.parse("2"):

        def dict(self, **kwargs):
            """
            Custom implementation of dict method from parent to include a copy of the models dict
            """
            type_ = PyObjectPath.validate(ModelRegistry)
            data = _ModelRegistryData.construct(name=self.name, models=self.models.copy(), type_=type_)
            return data.dict(**kwargs)

        def json(self, **kwargs):
            """
            Custom implementation of json method from parent to include a copy of the models dict
            """
            type_ = PyObjectPath.validate(ModelRegistry)
            data = _ModelRegistryData.construct(name=self.name, models=self.models.copy(), type_=type_)
            return data.json(**kwargs)
    else:

        @model_serializer(mode="wrap")
        def _registry_serializer(self, handler):
            values = handler(self)
            values["models"] = handler(self._models)
            return values

    @property
    def _debug_name(self) -> str:
        """Returns the "full name" of the registry. Since registries can have multiple names"""
        registered_names = self.get_registered_names()
        return registered_names[-1] if registered_names else self.name

    @property
    def models(self) -> MappingProxyType:
        """Return an immutable pointer to the models dictionary."""
        return MappingProxyType(self._models)

    @classmethod
    def root(cls) -> "ModelRegistry":
        """Return a static instance of the root registry."""
        return _REGISTRY_ROOT

    def clear(self) -> "ModelRegistry":
        """Clear the registry (and remove any dependencies)."""
        names = list(self._models.keys())
        for name in names:
            self.remove(name)
        return self

    def remove(self, name: str) -> None:
        """Preferred API for removing model from the registry."""
        if name not in self._models:
            raise ValueError(f"Cannot remove '{name}' from '{self._debug_name}' as it does not exist there!")
        # Adjust registrations
        self._models[name]._registrations.remove((self, name))
        # Remove the model
        del self._models[name]
        log.debug("Removed '%s' from registry '%s'", name, self._debug_name)

    def add(self, name: str, model: ModelType, overwrite: bool = False) -> ModelType:
        """Preferred API for adding new models to registry."""
        if name in self._models and not overwrite:
            raise ValueError(f"Cannot add '{name}' to '{self._debug_name}' as it already exists!")
        if REGISTRY_SEPARATOR in name:
            raise ValueError(f"Cannot add '{name}' to '{self._debug_name}' because it contains '{REGISTRY_SEPARATOR}'")
        if not isinstance(model, BaseModel):
            raise TypeError(f"model must be a child class of {BaseModel}, not '{type(model)}'.")

        # Track dependencies
        if name in self._models and (self, name) in self._models[name]._registrations:
            # Remove the registered name from the model that's being replaced
            self._models[name]._registrations.remove((self, name))
        # Add the registered name to the new model
        model._registrations.append((self, name))

        self._models[name] = model
        log.debug("Added '%s' to registry '%s': %s", name, self._debug_name, model)
        return model

    def get(self, name: str, default=None):
        """Accessor for models by name with default value.

        See __getitem__ for how this is different from calling get on self.models.
        """
        try:
            return self.__getitem__(name)
        except KeyError:
            return default

    def __getitem__(self, item):
        """Accessor for models by name.

        Differs from accessing the models dict directly because it parses
        names from nexted registries containing "/",
        i.e. "foo/bar" is object "bar" from the sub-registry "foo" of the current registry
        and "/foo/bar" is object "bar" from the sub-registry "foo" of the root registry
        Note that "." and ".." are not allowed.
        """
        if REGISTRY_SEPARATOR in item:
            if "." in item:
                raise ValueError("Path references to registry objects do not support '.' or '..'")
            registry_name, name = item.split(REGISTRY_SEPARATOR, 1)
            if registry_name == "":
                registry = ModelRegistry.root()
            else:
                try:
                    registry = self._models[registry_name]
                except KeyError:
                    raise KeyError(
                        f"No sub-registry found by the name '{registry_name}' in registry '{self._debug_name}' " f"while looking up model '{item}'"
                    )
            return registry.__getitem__(name)
        else:
            if item in self._models:
                return self._models[item]
            else:
                raise KeyError(f"No registered model found by the name '{item}' in registry '{self._debug_name}'")

    def load_config(
        self,
        cfg: DictConfig,
        overwrite: bool = False,
        skip_exceptions: bool = False,
    ) -> "ModelRegistry":
        """Load from OmegaConf DictConfig that follows hydra conventions."""
        loader = _ModelRegistryLoader(overwrite=overwrite)
        return loader.load_config(cfg, self, skip_exceptions=skip_exceptions)

    def create_config_from_path(
        self,
        path: str,
        overrides: Optional[List[str]] = None,
        version_base: Optional[str] = None,
    ) -> DictConfig:
        """Create the config from the path.

        Args:
            path: The absolute path from which to load the config
            overrides: List of hydra-style override strings
            version_base: See https://hydra.cc/docs/upgrades/version_base/

        Returns:
            The instance of the model registry, with the configs loaded.
        """
        import hydra

        overrides = overrides or []
        path = pathlib.Path(path).absolute()  # Hydra requires absolute paths
        if not path.parent.exists():
            raise OSError(f"Path does not exist: {path.parent}")
        with hydra.initialize_config_dir(version_base=version_base, config_dir=str(path.parent)):
            cfg = hydra.compose(config_name=path.name, overrides=overrides)
        return cfg

    def load_config_from_path(
        self,
        path: str,
        config_key: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        overwrite: bool = False,
        version_base: Optional[str] = None,
    ) -> "ModelRegistry":
        """Create the config from the path, and then load that data into the registry.

        Args:
            path: The absolute path from which to load the config
            config_key: (optional) key from the config if only part of the config is getting loaded to the registry
            overrides: List of hydra-style override strings
            overwrite: Whether to over-write existing entries in the registry
            version_base: See https://hydra.cc/docs/upgrades/version_base/

        Returns:
            The instance of the model registry, with the configs loaded.
        """
        cfg = self.create_config_from_path(path=path, overrides=overrides, version_base=version_base)
        if config_key is not None:
            cfg = cfg[config_key]
        return self.load_config(cfg, overwrite=overwrite)


class RootModelRegistry(ModelRegistry):
    """
    Class to represent the singleton, i.e. "root" ModelRegistry,
    to make it easier to distinguish the repr from standard registries during errors, debugging, etc.
    """

    name: str = Field("", repr=False)

    @root_validator(pre=True, skip_on_failure=True)
    def _root_validate(cls, v):
        raise ValueError("You are not allowed to construct the RootModelRegistry directly. Use ModelRegistry.root().")

    @property
    def _debug_name(self) -> str:
        """Returns the "full name" of the registry. Since registries can have multiple names"""
        return "RootModelRegistry"


_REGISTRY_ROOT = RootModelRegistry.construct()


class _ModelRegistryLoader:
    def __init__(self, overwrite: bool):
        self._overwrite = overwrite

    def _make_subregistries(self, cfg, registries: List[ModelRegistry]) -> List[Tuple[List[ModelRegistry], str, DictConfig]]:
        registry = registries[-1]
        models_to_register = []
        for k, v in cfg.items():
            if not isinstance(v, (dict, DictConfig)):
                # Skip config "variables", i.e. strings, etc that could be re-used by reference across the
                # object configs
                continue
            elif "_target_" in v:
                models_to_register.append((registries, k, v))
            elif _is_config_subregistry(v):
                # Config value represents a sub-registry
                subregistry = ModelRegistry(name=k)
                registry.add(k, subregistry, overwrite=self._overwrite)
                models_to_register.extend(self._make_subregistries(v, registries + [subregistry]))
        return models_to_register

    def load_config(self, cfg: DictConfig, registry: ModelRegistry, skip_exceptions: bool = False) -> ModelRegistry:
        """Load from OmegaConf DictConfig that follows hydra conventions."""
        # Here we use hydra's 'instantiate' to instantiate models,
        # because it provides a standard way to resolve the class name
        # that's being constructed, through the "_target_" field.
        # This also allows for nested attributes on the model itself to
        # be constructed, even if they are not themselves of BaseModel type,
        # or if they are of a specific subclass of the parent.
        from hydra.errors import InstantiationException
        from hydra.utils import instantiate

        models_to_register = self._make_subregistries(cfg, [registry])
        while True:
            unresolved_models = []
            for registries, k, v in models_to_register:
                with RegistryLookupContext(registries=registries):
                    try:
                        model = instantiate(v, _convert_="all")
                    except InstantiationException as e:
                        if isinstance(e.__cause__, (RegistryKeyError, ValidationError)):
                            unresolved_models.append((registries, k, v))
                        elif not skip_exceptions:
                            raise e
                        continue

                if hasattr(model, "meta") and hasattr(model.meta, "name") and model.meta.name == "":
                    model.meta.name = k
                registries[-1].add(k, model, overwrite=self._overwrite)

            if not unresolved_models:
                break
            elif len(unresolved_models) == len(models_to_register):
                # Did not successfully register any more things, so stop
                break
            else:
                models_to_register = unresolved_models
                unresolved_models = []

        if not skip_exceptions and unresolved_models:
            # Raise the error from the first unresolved model by trying to instantiate it again
            _registries, _k, v = unresolved_models[0]
            model = instantiate(v, _convert_="all")
        return registry


class RegistryLookupContext:
    """This python context helps the model registry globally track the subregistry chain for resolving paths for
    validation of the BaseModel.

    Do not confuse the name with "Context" from callable.py.
    """

    _REGISTRIES = []

    def __init__(self, registries: List[ModelRegistry] = None):
        """Constructor.

        Args:
            registries: A list of registries to use as the search paths for string-based model references within
                this context.
        """
        self.registries = registries
        self._previous_registries = []

    @classmethod
    def registry_search_paths(cls) -> List[ModelRegistry]:
        """Return the active list of additional registry search paths."""
        return cls._REGISTRIES

    def __enter__(self):
        self._previous_registries = self._REGISTRIES
        RegistryLookupContext._REGISTRIES = self.registries

    def __exit__(self, exc_type, exc_value, exc_tb):
        RegistryLookupContext._REGISTRIES = self._previous_registries


def resolve_str(v: str) -> BaseModel:
    """Resolve a string value from the RootModelRegistry."""
    search_registries = RegistryLookupContext.registry_search_paths()
    idx = -1
    if not search_registries:
        search_registries = [ModelRegistry.root()]
    if v.startswith("/"):
        v = v.replace("/", "", 1)
        idx = 0
    elif v.startswith("./"):
        v = v.replace("./", "", 1)
    elif v.startswith("../"):
        while v.startswith("../"):
            search_registries = search_registries[:-1]
            if not search_registries:
                raise ValueError(f"Could not find enough parent registries for {v}")
            v = v.replace("../", "", 1)

    search_registry = search_registries[idx]
    try:
        return search_registry[v]
    except KeyError:
        raise RegistryKeyError(f"Could not find model {v} in {search_registry._debug_name}")


class ResultBase(BaseModel):
    """A Result is an object that holds the results from a callable model.

    It provides the equivalent of a strongly typed dictionary where the
    keys and schema are known upfront.

    All result types should derive from this base class.
    """

    # Note that as a result of allowing arbitrary types,
    # the standard pydantic serialization methods will not work
    # This is OK, because for results we want more control over
    # the serialization method, so we build our own serializers.
    class Config(BaseModel.Config):
        arbitrary_types_allowed = True
        if version.parse(pydantic.__version__) < version.parse("2"):
            # Result type might contain a lot of data, so we don't want to copy on validation
            copy_on_model_validation: str = "none"


class ContextBase(ResultBase):
    """A Context represents an immutable argument to a callable model.

    All contexts should derive from this base class.
    A context is also a type of result, as a CallableModel could be responsible for generating a context
    that is an input into another CallableModel.
    """

    class Config(ResultBase.Config):
        frozen = True
        arbitrary_types_allowed = False
        # This separator is used when parsing strings into contexts (i.e. from command line)
        separator = ","
        if version.parse(pydantic.__version__) < version.parse("2"):
            # Contexts should be immutable, so make sure we deep-copy the model on validation (i.e. during construction)
            copy_on_model_validation: str = "deep"

    if version.parse(pydantic.__version__) < version.parse("2"):

        @classmethod
        def validate(cls, v, field=None):
            # Try to validate as-is
            try:
                return super(ContextBase, cls).validate(v)
            except Exception as e:
                # If it fails, then perhaps it is a delimited string, tuple or list
                if isinstance(v, (str, tuple, list)):
                    if isinstance(v, str):
                        v = v.split(cls.Config.separator)
                    # Map these values to fields of the context (ignoring "type_", which is the first field)
                    fields = iter(cls.__fields__)
                    next(fields)
                    v = dict(zip(fields, v))
                    return super(ContextBase, cls).validate(v)
                raise e

    else:

        @model_validator(mode="wrap")
        def _context_validator(cls, v, handler, info):
            # Add deepcopy for v2 because it doesn't support copy_on_model_validation
            v = copy.deepcopy(v)

            if isinstance(v, (dict, cls)):
                return handler(v)

            BaseValidator = TypeAdapter(BaseModel)
            try:
                return handler(BaseValidator.validate_python(v))
            except Exception as e:
                if isinstance(v, (str, tuple, list)):
                    if isinstance(v, str):
                        v = v.split(cls.Config.separator)
                    v = dict(zip(cls.model_fields, v))
                    return handler(v)
                raise e


ContextType = TypeVar("ContextType", bound=ContextBase)
ResultType = TypeVar("ResultType", bound=ResultBase)
