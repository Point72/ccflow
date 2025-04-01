"""This module defines the base model and registry for flow."""

import collections.abc
import copy
import inspect
import logging
import os
import pathlib
import platform
import warnings
from types import GenericAlias, MappingProxyType
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from omegaconf import DictConfig
from packaging import version
from pydantic import (
    BaseModel as PydanticBaseModel,
    ConfigDict,
    PrivateAttr,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.fields import Field
from typing_extensions import Self

from .exttypes.pyobjectpath import PyObjectPath

log = logging.getLogger(__name__)

__all__ = (
    "model_alias",
    "BaseModel",
    "ModelRegistry",
    "ModelType",
    "RegistryLookupContext",
    "RootModelRegistry",
    "REGISTRY_SEPARATOR",
    "load_config",
    "ContextBase",
    "ContextType",
    "ResultBase",
    "ResultType",
    "make_lazy_result",
)


REGISTRY_SEPARATOR = "/"


class RegistryKeyError(KeyError):
    """Subclass for KeyError specific to Registry lookup errors."""

    ...


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

        Args:
            types: If specified, will only return dependencies of the given model sub-types.
        """
        deps = []
        for _field_name, value in self:
            deps.extend(_get_registry_dependencies(value, types))
        return deps


# Pydantic 2 has different handling of serialization.
# This requires some workarounds at the moment until the feature is added to easily get a mode that
# is compatible with Pydantic 1
# This is done by adjusting annotations via a MetaClass for any annotation that includes a BaseModel,
# such that the new annotation contains SerializeAsAny
# https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
# https://github.com/pydantic/pydantic/issues/6423
# https://github.com/pydantic/pydantic-core/pull/740
# See https://github.com/pydantic/pydantic/issues/6381 for inspiration on implementation
# NOTE: For this logic to be removed, require https://github.com/pydantic/pydantic-core/pull/1478
from pydantic._internal._model_construction import ModelMetaclass  # noqa: E402

_IS_PY39 = version.parse(platform.python_version()) < version.parse("3.10")


def _adjust_annotations(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if not _IS_PY39:
        from types import UnionType

        if origin is UnionType:
            origin = Union

    if isinstance(annotation, GenericAlias) or (inspect.isclass(annotation) and issubclass(annotation, PydanticBaseModel)):
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
        - Registration by name, and coercion from string name to allow for object re-use from the configs
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
        """The path to the object type"""
        return PyObjectPath.validate(type(self))

    # We want to track under what names a model has been registered
    _registrations: List[Tuple["ModelRegistry", str]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        # Note that validate_assignment only partially works: https://github.com/pydantic/pydantic/issues/7105
        validate_assignment=True,
        populate_by_name=True,
        coerce_numbers_to_str=True,  # New in v2 for backwards compatibility with V1
        # Lots of bugs happen because of a mis-named field with a default value,
        # where the default behavior is just to drop the mis-named value. This prevents that
        extra="forbid",
        ser_json_timedelta="float",
    )

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
        """Get an IPython widget to view the object.

        Args:
            json_kwargs: The kwargs to pass to the pydantic serializer
            widget_kwargs: The kwargs to pass to the JSON IPython widget
        """
        from IPython.display import JSON  # Heavy import, only import if used.

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
                raise ValueError(str(e)) from e
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

            if type_:
                if isinstance(type_, PyObjectPath):  # if we already have a PyObjectPath, we can use it directly and avoid expensive validation
                    type_cls = type_.object
                else:
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


def _is_config_model(value: Dict):
    """Test whether a config value is a model, i.e. it is a dict which contains a _target_ key."""
    if "_target_" in value:
        return True
    for key in value:
        # Catch potential misspellings (which will cause the config to be ignored and treated as a dict)
        if key.lower().strip("_") in ("target", "arget", "trget", "taget", "taret", "targt", "targe", "tagret"):
            warnings.warn(f"Found config value containing `{key}`, are you sure you didn't mean '_target_'?", SyntaxWarning)
    return False


def _is_config_subregistry(value):
    """Test whether a config value is a subregistry, i.e. it is a dict which either
    contains a _target_ key, or recursively contains a dict that has a _target_ key.
    """
    if isinstance(value, (dict, DictConfig)):
        if _is_config_model(value):
            return True
        else:
            for v in value.values():
                if _is_config_subregistry(v):
                    return True
    return False


def model_alias(model_name: str) -> BaseModel:
    """Function to alias a BaseModel by name in the root registry.

    Useful for configs in hydra where we want a config object to point directly to another config object.

    Args:
        model_name: The name of the underlying model to point to in the registry
    Example:
        _target_: ccflow.model_alias
        model_name: foo
    """
    return BaseModel.model_validate(model_name)


ModelType = TypeVar("ModelType", bound=BaseModel)


class ModelRegistry(BaseModel, collections.abc.Mapping):
    """ModelRegistry represents a named collection of models.

    Because we want to control how models are added and removed, the dict structure is not public.
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

    @field_validator("name")
    def _validate_name(cls, v):
        if not v:
            raise ValueError("name must be non-empty")
        return v

    @model_serializer(mode="wrap")
    def _registry_serializer(self, handler):
        values = handler(self)
        values["models"] = self._models
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
    def root(cls) -> Self:
        """Return a static instance of the root registry."""
        return _REGISTRY_ROOT

    def clear(self) -> Self:
        """Clear the registry (and remove any dependencies)."""
        names = list(self._models.keys())
        for name in names:
            self.remove(name)
        return self

    def remove(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: The name of the model to remove
        """
        if name not in self._models:
            raise ValueError(f"Cannot remove '{name}' from '{self._debug_name}' as it does not exist there!")
        # Adjust registrations
        self._models[name]._registrations.remove((self, name))
        # Remove the model
        del self._models[name]
        log.debug("Removed '%s' from registry '%s'", name, self._debug_name)

    def add(self, name: str, model: ModelType, overwrite: bool = False) -> ModelType:
        """Add a new model to the registry.

        Args:
            name: The name of the model to add
            model: The model to add
            overwrite: Whether to overwrite an existing model in the registry
        """
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

    def get(self, name: str, default=None) -> Optional[ModelType]:
        """Accessor for models by name with default value.

        Differs from calling self.models.get because it parses names from nested registries containing "/",
        i.e. "foo/bar" is object "bar" from the sub-registry "foo" of the current registry
        and "/foo/bar" is object "bar" from the sub-registry "foo" of the root registry
        Note that "." and ".." are not allowed.

        Args:
            name: The name of the model to get from the registry
            default: The default value to return if the model does not exist in the registry
        """
        try:
            return self.__getitem__(name)
        except KeyError:
            return default

    def __getitem__(self, item) -> ModelType:
        """Accessor for models by name.

        Differs from accessing the models dict directly because it parses names from nested registries containing "/",
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
                        f"No sub-registry found by the name '{registry_name}' in registry '{self._debug_name}' while looking up model '{item}'"
                    )
            return registry.__getitem__(name)
        else:
            if item in self._models:
                return self._models[item]
            else:
                raise KeyError(f"No registered model found by the name '{item}' in registry '{self._debug_name}'")

    def __iter__(self):
        for key, model in self._models.items():
            yield key
            if isinstance(model, ModelRegistry):
                for subkey in model:
                    yield REGISTRY_SEPARATOR.join((key, subkey))

    def __len__(self) -> int:
        count = len(self._models)
        for model in self._models.values():
            if isinstance(model, ModelRegistry):
                count += len(model)
        return count

    def load_config(
        self,
        cfg: DictConfig,
        overwrite: bool = False,
        skip_exceptions: bool = False,
    ) -> Self:
        """Load from OmegaConf DictConfig that follows hydra conventions.

        Args:
            cfg: An OmegaConf DictConfig
            overwrite: Whether to allow overwriting of names that already exist
            skip_exceptions: Whether to skip any exceptions that are thrown when validating and registering models
        """
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
        import hydra  # Heavy import, only import if used.

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
    ) -> Self:
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

    @model_validator(mode="before")
    def _root_validate(cls, v, info):
        raise ValueError("You are not allowed to construct the RootModelRegistry directly. Use ModelRegistry.root().")

    @property
    def _debug_name(self) -> str:
        """Returns the "full name" of the registry. Since registries can have multiple names"""
        return "RootModelRegistry"


_REGISTRY_ROOT = RootModelRegistry.model_construct()


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
            elif _is_config_model(v):
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
            registries, k, v = unresolved_models[0]
            with RegistryLookupContext(registries=registries):
                instantiate(v, _convert_="all")
            # The line above should have raised. If for whatever reason it didn't, we raise the error here.
            raise InstantiationException(f"Failed to instantiate {k} with config {v}, but cannot reproduce.")
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


def resolve_str(v: str) -> ModelType:
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
                raise ValueError(f"Could not resolve enough parent registries for '{v}'")
            v = v.replace("../", "", 1)

    search_registry = search_registries[idx]
    try:
        return search_registry[v]
    except KeyError:
        raise RegistryKeyError(f"Could not resolve model '{v}' in registry '{search_registry._debug_name}'")


def _find_parent_config_folder(config_dir: str = "config", config_name: str = "", *, basepath: str = ""):
    folder = pathlib.Path(basepath).resolve()
    exists = (folder / config_dir).exists() if not config_name else (folder / config_dir / f"{config_name}.yaml").exists()
    if not exists and (folder / config_dir / f"{config_name}.yml").exists():
        raise ValueError(
            f"Found config_name `{config_name}` with `.yml` suffix, which is not recognized by hydra. Please rename to `{config_name}.yaml`."
        )
    while not exists:
        folder = folder.parent
        if str(folder) == os.path.abspath(os.sep):
            raise FileNotFoundError(f"Could not find config folder: {config_dir} in folder {basepath}")
        exists = (folder / config_dir).exists() if not config_name else (folder / config_dir / f"{config_name}.yaml").exists()
        if not exists and (folder / config_dir / f"{config_name}.yml").exists():
            raise ValueError(
                f"Found config_name `{config_name}` with `.yml` suffix, which is not recognized by hydra. Please rename to `{config_name}.yaml`."
            )

    config_dir = (folder / config_dir).resolve()
    if not config_name:
        return folder.resolve(), config_dir, ""
    else:
        return folder.resolve(), config_dir, (folder / config_dir / f"{config_name}.yaml").resolve()


def load_config(
    root_config_dir: str,
    root_config_name: str,
    config_dir: str = "config",
    config_name: str = "",
    overrides: Optional[List[str]] = None,
    *,
    overwrite: bool = False,
    basepath: str = "",
) -> RootModelRegistry:
    """Helper function to load a hydra config into the root model registry.

    Hydra configs can be pulled from multiple places:
      1. A root configuration
      2. An optional user-provided config directory, and within that, an optional config name.

    Arguments:
        root_config_dir: The directory containing the root hydra config. This is typically the location of the configs in, i.e. "config"
         This is passed to hydra.initialize_config_dir to get the loading started.
        root_config_name: The config name within the base directory, i.e. "conf"
        config_dir: End user-provided additional directory to search for hydra configs.
        config_name: An optional config name to look for within the `config_dir`. This allows you to specify a particular config file to load.
        overrides: A list of hydra-style override strings to apply when loading the config.
        overwrite: Whether to overwrite existing entries in the registry when loading the config.
        basepath: The base path to start searching for the `config_dir`. This is useful when you want to load from an absolute (rather than relative) path.
    """
    # Heavy import, only import if used
    import os

    from hydra import compose, initialize_config_dir

    overrides = overrides or []
    with initialize_config_dir(config_dir=root_config_dir, version_base=None):
        if config_dir:
            hydra_folder, config_dir, _ = _find_parent_config_folder(config_dir=config_dir, config_name=config_name, basepath=basepath or os.getcwd())

            cfg = compose(config_name=root_config_name, overrides=[], return_hydra_config=True)
            searchpaths = cfg["hydra"]["searchpath"]
            searchpaths.extend([hydra_folder, config_dir])
            if config_name:
                config_group = pathlib.Path(config_dir).resolve().name
                overrides = [f"+{config_group}={config_name}", *overrides.copy(), f"hydra.searchpath=[{','.join(searchpaths)}]"]
            else:
                overrides = [*overrides.copy(), f"hydra.searchpath=[{','.join(searchpaths)}]"]

        cfg = compose(config_name=root_config_name, overrides=overrides)

    registry = ModelRegistry.root()
    registry.load_config(cfg, overwrite=overwrite)
    return registry


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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _onaccess_callback(*args, **kwargs):
        """Function to be called on every attribute access"""
        pass

    def __getattribute__(self, attr):
        """Call _onaccess_callback before allowing attribute access.

        Necessary to allow features like delayed evaluation.
        """
        if not attr.startswith("_lazy"):
            super().__getattribute__("_onaccess_callback")(self)
        return super().__getattribute__(attr)


class ContextBase(ResultBase):
    """A Context represents an immutable argument to a callable model.

    All contexts should derive from this base class.
    A context is also a type of result, as a CallableModel could be responsible for generating a context
    that is an input into another CallableModel.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=False,
        # This separator is used when parsing strings into contexts (i.e. from command line)
        separator=",",
    )

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
                    v = v.split(cls.model_config["separator"])
                if len(v) > len(cls.model_fields):  # Do not allow extra elements
                    raise e
                v = dict(zip(cls.model_fields, v))
                return handler(v)
            raise e


ContextType = TypeVar("ContextType", bound=ContextBase)
ResultType = TypeVar("ResultType", bound=ResultBase)


def _make_lazy_stub(res_type: ResultType, initializer_fn, *args, **kwargs) -> ResultType:
    """
    Makes a result stub without calling __init__, but instead initializer_fn on access

    Args:
        result_type: The template ResultType to use for the lazy result stub
        initializer_fn: A Callable that is called with the stub as first argument.
            Additional arguments are forwarded to initializer_fn.
    """
    if not issubclass(res_type, ResultBase):
        raise TypeError("Can only create delayed ResultBase-derived instances")
    new_instance = res_type.__new__(res_type)
    object.__setattr__(new_instance, "_lazy_is_delayed", True)

    def callback_to_inject(obj):
        # the callback is responsible for calling __init__ on the new instance
        # we will give it a clean instance without the callback
        object.__setattr__(obj, "_onaccess_callback", lambda *_, **__: None)
        initializer_fn(obj, *args, **kwargs)
        if hasattr(obj, "_lazy_is_delayed"):
            object.__delattr__(obj, "_lazy_is_delayed")
        # now __init__ must have been called

    object.__setattr__(new_instance, "_onaccess_callback", callback_to_inject)
    return new_instance


def make_lazy_result(res_type: ResultType, to_copy_fn: Callable[[], ResultType]) -> ResultType:
    """Creates a new ResultType based on the passed ResultType, but which will instantiate itself lazily on attribute access.

    Args:
        res_type: The original ResultType to model the lazy ResultType after
        to_copy_fn: The function which will be called to initialize instances of the return type when attributes are accessed.
            __dict__ will be copied from the result, as well as extra pydantic attributes if present.
            The copy mechanism is similar to that used in pydantic's model_construct.
    """

    def initializer(obj, validate=False, *args, **kwargs):
        new_obj = to_copy_fn()

        if hasattr(obj, "_lazy_validation_requested"):
            new_obj = res_type.model_validate(new_obj)
            object.__delattr__(obj, "_lazy_validation_requested")

        # now we copy the fields into obj : inspired by pydantic model_construct
        object.__setattr__(obj, "__dict__", new_obj.__dict__)
        if hasattr(new_obj, "__pydantic_fields_set__"):
            object.__setattr__(obj, "__pydantic_fields_set__", new_obj.__pydantic_fields_set__)
        if hasattr(new_obj, "__pydantic_extra__"):
            object.__setattr__(obj, "__pydantic_extra__", new_obj.__pydantic_extra__)
        if hasattr(new_obj, "__pydantic_private__"):
            object.__setattr__(obj, "__pydantic_private__", new_obj.__pydantic_private__)

    return _make_lazy_stub(res_type, initializer)
