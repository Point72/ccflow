# PathKeyResolverMixin

The `PathKeyResolverMixin` enables models to populate their fields from a Python-importable object (e.g., a module-level dictionary), optionally traversing nested keys and controlling merge precedence.

Key Concepts

- `ccflow_path`: Import path string (or `PyObjectPath`) to an object that can be converted to a mapping.
- `ccflow_keys`: List of keys (strings and/or integer indexes) or a dotted string for nested traversal.
- `ccflow_merge`: Merge strategy; one of `resolved_wins` (default), `explicit_wins`, or `raise_on_conflict`.
- `ccflow_filter_extras`: Boolean; default `true`. If true, drops keys from the resolved mapping that are not model fields.
- Optional: subclasses may set `ccflow_allowed_prefixes: ClassVar[List[str]]` to restrict allowed import path prefixes.

Usage (Python)

```python
from pydantic import Field
from ccflow import BaseModel, PathKeyResolverMixin

class DBConfig(BaseModel, PathKeyResolverMixin):
    host: str
    port: int

cfg = {
    "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
    "ccflow_keys": "database",  # or ["database"]
}
db = DBConfig(**cfg)
assert db.host == "localhost"
assert db.port == 5432
```

Deep Traversal

```python
cfg = {
    "ccflow_path": "ccflow.tests.data.path_key_resolver_samples.COMPLEX_CONFIG",
    "ccflow_keys": ["environments", "dev", "database"],
}
```

Merge Strategies

- `resolved_wins`: values from the resolved mapping override explicitly provided values.
- `explicit_wins`: explicitly provided values override resolved ones; resolved values fill only missing fields.
- `raise_on_conflict`: raises `ValueError` if both resolved and explicit inputs define the same field with different values; otherwise fills missing fields.

```python
class Model(BaseModel, PathKeyResolverMixin):
    a: str = Field("x")
    b: int = Field(0)

m = Model(
    ccflow_path="ccflow.tests.data.path_key_resolver_samples.SIMPLE_CONFIG",
    ccflow_merge="explicit_wins",
    a="explicit",
)
assert m.a == "explicit"
```

Hydra Config Example

```yaml
mixin_db:
  _target_: mypkg.DBConfig
  ccflow_path: ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG
  ccflow_keys: database
```

Hydra Overrides

```text
mixin_db.ccflow_path=ccflow.tests.data.path_key_resolver_samples.OTHER_NESTED_CONFIG
mixin_db.ccflow_keys=database_alt
```

Error Handling

- Empty `ccflow_path` raises `ValueError`.
- Key traversal errors raise `KeyError` with the full traversal path and available keys at the failure point.
- Non-mapping objects raise a `ValueError` unless they are pydantic models or dataclasses (which are converted).

Namespaced Spec

- You can specify all options under a single `ccflow` dictionary. This makes configs cleaner and self-documenting.

```yaml
db:
  _target_: mypkg.DBConfig
  ccflow:
    path: ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG
    keys: database
    merge: resolved_wins
    filter_extras: true
```

Debug Metadata

- Instances using the mixin include `__ccflow_source__` as a private attribute with `{path, keys, merge, filter_extras}` for auditability.

Extras Filtering

- By default, keys not present on the model are dropped from the resolved mapping to avoid `extra_forbidden` errors.
- Set `ccflow_filter_extras: false` to allow all resolved keys (models may need `model_config = ConfigDict(extra="ignore")`).

Motivation

- Hydra-centric projects benefit from keeping most configuration in YAML, but some defaults are better expressed in Python (shared dicts, computed values, or tightly coupled settings).
- Referencing Python code allows:
  - Centralizing canonical configs used across modules, tests, and examples.
  - Environment- or context-specific defaults by selecting different objects via `ccflow_keys`.
  - Avoiding duplication across multiple YAML files; update once in code and reuse.
  - Type-checked evolution: the spec is validated, and tests can guard behavior.
- Merge strategies (`resolved_wins`, `explicit_wins`, `raise_on_conflict`) make precedence explicit, not implicit.
- Optional allowlist (`ccflow_allowed_prefixes`) keeps imports constrained to known-safe modules.

Example: Python-backed defaults with Hydra

- Define Python configs in a module like `ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG`.
- Select per environment in YAML with `ccflow_keys: database` or `ccflow_keys: database_alt`.
- Override dynamically at runtime:
  - `mixin_db.ccflow_path=ccflow.tests.data.path_key_resolver_samples.OTHER_NESTED_CONFIG`
  - `mixin_db.ccflow_keys=database_alt`

Allowlist Prefixes

- You can restrict dynamic import paths via a class-level allowlist.
- Define `ccflow_allowed_prefixes: ClassVar[List[str]]` on your model to constrain which modules may be referenced by `ccflow_path`.

```python
from typing import ClassVar, List
from ccflow import BaseModel, PathKeyResolverMixin

class StrictDB(BaseModel, PathKeyResolverMixin):
    host: str
    port: int

    # Only allow imports under our test data module
    ccflow_allowed_prefixes: ClassVar[List[str]] = ["ccflow.tests.data.path_key_resolver_samples"]

# Allowed
StrictDB(
    ccflow_path="ccflow.tests.data.path_key_resolver_samples.NESTED_CONFIG",
    ccflow_keys="database",
)

# Disallowed: valid import, but outside the allowed prefix
StrictDB(
    ccflow_path="ccflow.tests.utils.test_mixin.MixinSimpleModel",
    ccflow_keys=[],
)  # raises ValueError: not allowed by allowed prefixes
```

- The check uses `str(ccflow_path).startswith(prefix)` for each prefix.
- Leave `ccflow_allowed_prefixes` unset (None) to allow any import path.
