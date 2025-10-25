from typing import ClassVar, List, Optional

from pydantic import PrivateAttr, model_validator

from .path_resolve_spec import PathResolveSpec

__all__ = ("PathKeyResolverMixin",)


class PathKeyResolverMixin:
    """Resolve configuration values from a python-import path with optional nested key traversal.

    Supported fields (either top-level or within a `ccflow` dict):
      - `ccflow_path`: import path string (coerced to PyObjectPath)
      - `ccflow_keys`: list[str|int] or dotted string for nested traversal (supports list indices)
      - `ccflow_merge`: one of {"resolved_wins", "explicit_wins", "raise_on_conflict"}
      - `ccflow_filter_extras`: bool; default True; drop resolved keys not present on the model

    Optional: subclasses may set `ccflow_allowed_prefixes` to restrict import paths.
      - `ccflow_allowed_prefixes` is a ClassVar[List[str]] of allowed import-path prefixes. If set, any
        `ccflow_path` not starting with one of these prefixes raises a ValueError during validation.
        This provides a lightweight allowlist to constrain dynamic imports in configs.
    """

    ccflow_allowed_prefixes: ClassVar[Optional[List[str]]] = None
    __ccflow_source__: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="wrap")
    def _resolve_and_attach(cls, v, handler):
        # Only operate on dict inputs; otherwise construct directly
        values = v
        spec: Optional[PathResolveSpec] = None
        if isinstance(values, dict):
            spec, values = PathResolveSpec.extract_from_values(values)

        # If we have a spec, resolve and merge before constructing the instance
        if spec is not None:
            obj = spec.resolve_object(allowed_prefixes=cls.ccflow_allowed_prefixes)
            node = spec.traverse(obj)
            mapping = spec.as_mapping(node)
            mapping = spec.filter_extras_map(mapping, getattr(cls, "model_fields", {}))
            values = spec.merge_into(values, mapping)

        # Construct instance
        result = handler(values)

        # Attach debug metadata if spec was provided
        if isinstance(result, object) and spec is not None:
            try:
                result.__ccflow_source__ = spec.debug_meta()
            except Exception:
                pass
        return result
