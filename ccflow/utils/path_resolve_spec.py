from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import AliasChoices, BaseModel as PydanticModel, Field

from ccflow import PyObjectPath


class PathResolveSpec(PydanticModel):
    """Specification for ccflow path/key resolution.

    - path: Import path or PyObjectPath to a resolvable object
    - keys: Nested traversal keys (list or dotted string)
    - merge: Merge strategy ('resolved_wins', 'explicit_wins', 'raise_on_conflict')
    - filter_extras: Whether to drop keys not present on the target model
    """

    # Accept both short names and ccflow_* variants via validation aliases
    path: PyObjectPath = Field(
        validation_alias=AliasChoices("path", "ccflow_path"),
        description="Python import path (coerced to PyObjectPath) to a mapping-like object or container for traversal.",
    )
    keys: Optional[Union[str, int, List[Union[str, int]]]] = Field(
        default=None,
        validation_alias=AliasChoices("keys", "ccflow_keys"),
        description="Nested traversal keys (str/int or list). Dotted strings are allowed (e.g., 'a.b.0.c').",
    )
    merge: Literal["resolved_wins", "explicit_wins", "raise_on_conflict"] = Field(
        default="resolved_wins",
        validation_alias=AliasChoices("merge", "ccflow_merge"),
        description="Merge strategy: 'resolved_wins' (resolved overrides explicit), 'explicit_wins' (explicit overrides resolved), or 'raise_on_conflict' (error if both define different values).",
    )
    filter_extras: bool = Field(
        default=True,
        validation_alias=AliasChoices("filter_extras", "ccflow_filter_extras"),
        description="Whether to drop keys from the resolved mapping that are not fields of the target model.",
    )

    @classmethod
    def extract_from_values(cls, values: Dict[str, Any]) -> Tuple[Optional["PathResolveSpec"], Dict[str, Any]]:
        """Extract spec from incoming values and pop ccflow hints. Returns (spec|None, mutated_values)."""
        if not isinstance(values, dict):
            return None, values

        spec_input: Dict[str, Any] = {}
        if "ccflow" in values and isinstance(values["ccflow"], dict):
            spec_input = values.pop("ccflow")
        else:
            if "ccflow_path" in values:
                spec_input["path"] = values.pop("ccflow_path")
            if "ccflow_keys" in values:
                spec_input["keys"] = values.pop("ccflow_keys")
            if "ccflow_merge" in values:
                spec_input["merge"] = values.pop("ccflow_merge")
            if "ccflow_filter_extras" in values:
                spec_input["filter_extras"] = values.pop("ccflow_filter_extras")

        if not spec_input:
            return None, values

        spec = cls.model_validate(spec_input)
        return spec, values

    @staticmethod
    def _normalize_keys(keys: Any) -> List[Any]:
        if keys is None:
            return []
        if isinstance(keys, int):
            return [keys]
        if isinstance(keys, str):
            parts = [p for p in keys.split(".") if p != ""]
            out: List[Any] = []
            for p in parts:
                if p.isdigit():
                    out.append(int(p))
                else:
                    out.append(p)
            return out
        if isinstance(keys, (list, tuple)):
            out: List[Any] = []
            for p in keys:
                if isinstance(p, str) and p.isdigit():
                    out.append(int(p))
                else:
                    out.append(p)
            return out
        return [keys]

    def resolve_object(self, *, allowed_prefixes: Optional[List[str]] = None) -> Any:
        if isinstance(self.path, str) and not self.path.strip():
            raise ValueError("ccflow_path cannot be empty")
        path_str = str(self.path)
        if allowed_prefixes and not any(path_str.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"ccflow_path '{path_str}' not allowed by allowed prefixes: {allowed_prefixes}")
        return self.path.object

    def traverse(self, obj: Any) -> Any:
        keys = self._normalize_keys(self.keys)
        cur = obj
        for k in keys:
            if isinstance(cur, dict):
                if k not in cur:
                    raise KeyError(f"Key '{k}' not found while traversing ccflow_keys={keys}. Available keys: {list(cur.keys())}")
                cur = cur[k]
            elif isinstance(cur, (list, tuple)):
                if not isinstance(k, int):
                    raise KeyError(f"Expected integer index while traversing a list, got key='{k}'")
                try:
                    cur = cur[k]
                except IndexError:
                    raise KeyError(f"List index out of range: {k} while traversing ccflow_keys={keys}")
            else:
                raise KeyError(f"Cannot traverse into type {type(cur).__name__} with key '{k}'")
        return cur

    @staticmethod
    def as_mapping(obj: Any) -> Mapping[str, Any]:
        from dataclasses import asdict, is_dataclass

        if isinstance(obj, dict):
            return obj
        if isinstance(obj, PydanticModel):
            return obj.model_dump()
        if is_dataclass(obj):
            return asdict(obj)
        raise ValueError("Resolved object is not a mapping and cannot be converted for ccflow_path resolution")

    def filter_extras_map(self, mapping: Mapping[str, Any], fields: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.filter_extras:
            return dict(mapping)
        allowed = fields.keys()
        return {k: v for k, v in mapping.items() if k in allowed}

    def merge_into(self, values: Dict[str, Any], mapping: Mapping[str, Any]) -> Dict[str, Any]:
        if self.merge not in ("resolved_wins", "explicit_wins", "raise_on_conflict"):
            raise ValueError("ccflow_merge must be one of {'resolved_wins','explicit_wins','raise_on_conflict'}")
        values = dict(values)
        if self.merge == "resolved_wins":
            values.update(mapping)
            return values
        elif self.merge == "explicit_wins":
            for k, v in mapping.items():
                if k not in values:
                    values[k] = v
            return values
        else:
            conflicts = [k for k, v in mapping.items() if k in values and values[k] != v]
            if conflicts:
                raise ValueError(f"ccflow_merge=raise_on_conflict detected conflicting keys: {conflicts}")
            for k, v in mapping.items():
                if k not in values:
                    values[k] = v
            return values

    def debug_meta(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "keys": self._normalize_keys(self.keys),
            "merge": self.merge,
            "filter_extras": self.filter_extras,
        }
