# Tokenization

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How Tokens Are Computed](#how-tokens-are-computed)
- [Behavior Hashing](#behavior-hashing)
- [Controlling Tokenization](#controlling-tokenization)
  - [Customizing Tokenization](#customizing-tokenization)
- [Cache Keys and MemoryCacheEvaluator](#cache-keys-and-memorycacheevaluator)
- [Limitations and Caveats](#limitations-and-caveats)
  - [Injecting External State into Tokens](#injecting-external-state-into-tokens)
- [Architecture](#architecture)

## Overview

Every ccflow `BaseModel` instance exposes a `model_token` property — a deterministic hex digest that uniquely identifies the model's **data** (field values) and optionally its **behavior** (source code of methods).

Tokens are used as cache keys by evaluators like `MemoryCacheEvaluator`, and can be used for change detection, deduplication, or audit trails.

```python
from ccflow import BaseModel

class MyModel(BaseModel):
    x: int = 1
    y: str = "hello"

m = MyModel()
print(m.model_token)  # e.g. "a1b2c3d4..."

# Same field values → same token
assert MyModel(x=1, y="hello").model_token == m.model_token

# Different field values → different token
assert MyModel(x=2).model_token != m.model_token
```

## Quick Start

**Tokens include both data and behavior by default:**

```python
from ccflow import BaseModel

class Config(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100

c1 = Config()
c2 = Config(learning_rate=0.02)
assert c1.model_token != c2.model_token
```

Methods defined on the class are automatically included in the token via bytecode hashing:

```python
class MyPipeline(BaseModel):
    x: int = 1

    def __call__(self, ctx):
        return self.x * 2  # changes to this body change the token
```

**Data-only tokens (excluding behavior):**

If you only want field data in the token (e.g. for pure config models with no meaningful methods):

```python
from ccflow.utils.tokenize import DefaultTokenizer

class PureConfig(BaseModel):
    __ccflow_tokenizer__ = DefaultTokenizer()  # data-only, no behavior hashing
    x: int = 1
```

## How Tokens Are Computed

A model token is a SHA-256 digest of the model's **canonical form**, which is a tuple of:

```
(fully_qualified_type_name, behavior_token_or_None, ((field1, normalized_value1), ...))
```

### Field Normalization

Each field value is recursively normalized to a deterministic canonical form:

| Type                                               | Canonical Form                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| Primitives (`int`, `str`, `bool`, `None`, `float`) | Identity                                                              |
| `datetime`, `date`, `time`                         | ISO format string                                                     |
| `enum.Enum`                                        | `("enum", type_path, value)`                                          |
| `list`, `tuple`, `set`, `dict`                     | Recursively normalized, sets/dicts sorted by `repr`                   |
| `numpy.ndarray`                                    | `("ndarray", dtype, shape, sha256_of_bytes)`                          |
| `pandas.DataFrame`                                 | `("dataframe", columns, dtypes, shape, sha256_of_hash_pandas_object)` |
| Frozen `BaseModel` child                           | `("__child__", child.model_token)` — Merkle tree shortcut             |
| Non-frozen `BaseModel` child                       | Recursively normalized inline                                         |
| Anything else                                      | cloudpickle fallback                                                  |

### Field Exclusion

Use pydantic's `Field(exclude=True)` to exclude fields from the token:

```python
from pydantic import Field

class MyModel(BaseModel):
    important: int = 42
    debug_info: str = Field(default="debug", exclude=True)  # not in token
```

### Token Caching

Frozen models (`frozen=True`, including `ContextBase`) cache their token automatically —
computed once and never invalidated, since the model cannot be mutated.

Mutable models recompute `model_token` on every access, guaranteeing the token always
reflects the current state.

To opt in to caching on a mutable model — for example, when it holds large input data
that is expensive to tokenize and will not be changed after construction — set
`cache_token=True`:

```python
class LargeInputModel(BaseModel):
    model_config = ConfigDict(cache_token=True)
    data: list  # large payload, expensive to tokenize
```

When `cache_token=True` on a mutable model, the cache is cleared whenever a field is
directly reassigned (via `validate_assignment`). However, mutating a **nested** child
in-place (e.g. `parent.child.x = 2`) will **not** invalidate the parent's cached token.
Only use `cache_token=True` when you know the model's content will not change after the
token is first accessed.

## Behavior Hashing

By default, `model_token` includes both field data **and** behavior (method bytecode). Two models with the same fields but different `__call__` implementations will have **different tokens**.

```python
class MyCallable(BaseModel):
    x: int = 1

    def __call__(self, ctx):
        return self.x + 1  # hashed into the token automatically
```

To disable behavior hashing for a class (data-only tokens), use a plain `DefaultTokenizer()`:

```python
class DataOnly(BaseModel):
    __ccflow_tokenizer__ = DefaultTokenizer()
    x: int = 1
```

### AST vs Bytecode

Two strategies are available for hashing function source code:

|                                   | `with_ast()`                                   | `with_bytecode()`                                      |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| **How it works**                  | Parses source → AST → `ast.unparse()` → SHA256 | Hashes `co_code` + `co_consts` → SHA256                |
| **Strips docstrings**             | ✅                                             | ✅                                                     |
| **Strips comments**               | ✅                                             | ✅ (comments aren't in bytecode)                       |
| **Immune to whitespace**          | ✅                                             | ✅                                                     |
| **Immune to variable renames**    | ❌ Different names → different hash            | ✅ Names in `co_varnames`, not `co_code`               |
| **Works without source**          | ❌ Falls back to bytecode                      | ✅ Always works                                        |
| **Stable across Python versions** | ✅ AST is stable                               | ⚠️ **`co_code` changes between Python minor versions** |
| **Works in REPL/Jupyter**         | ❌ `inspect.getsource()` often fails           | ✅ Always available                                    |
| **Performance**                   | Slower (parse + AST round-trip)                | ✅ Order of magnitude faster                           |

**Bytecode is the default** when behavior hashing is enabled via `compute_behavior_token()`. It is an order of magnitude faster than AST normalization. Use `with_ast()` if you need cross-version stability.

### Which Methods Are Hashed

When behavior hashing is enabled, **all methods defined directly on the class** (in `cls.__dict__`) are included. Inherited methods are NOT included — only methods the class itself defines.

This includes:

- Regular methods, `@classmethod`, `@staticmethod`
- Private methods (`_helper`, `__internal`)
- Pydantic validators (`@model_validator`, `@field_validator`)
- `__call__`, `__deps__`, any other dunder you define

This does **not** include:

- Methods inherited from parent classes
- Functions imported and called by your methods (no transitive dependency tracking)
- Methods added dynamically at runtime

### Adding Standalone Dependencies

If your class calls standalone functions that should affect the token, declare them:

```python
def my_transform(data):
    return data * 2

class MyPipeline(BaseModel):
    __ccflow_tokenizer_deps__ = [my_transform]
    x: int = 1

    def __call__(self, ctx):
        return my_transform(self.x)
```

## Controlling Tokenization

### Customizing Tokenization

There are three extension points, at different levels:

| Hook                            | Scope             | Use when                                                                                              |
| ------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------- |
| `__ccflow_tokenizer__` ClassVar | BaseModel class   | You want to change *how models are tokenized* (e.g. disable behavior hashing, use AST mode, use dask) |
| `normalize_token.register(T)`   | Any type (global) | You have a custom type that appears as a field value and needs a deterministic canonical form         |
| `__ccflow_tokenize__()` method  | Any instance      | Same as above, but defined on the class itself instead of registered globally                         |

The first is a high-level orchestration hook — it selects the tokenizer engine for a model class. The other two are leaf-value hooks that control how individual field values are canonicalized.

**`__ccflow_tokenizer__`** — select the tokenizer engine for a model class:

```python
class DataOnly(BaseModel):
    __ccflow_tokenizer__ = DefaultTokenizer()  # data-only, no behavior hashing
    x: int = 1

class WithAST(BaseModel):
    __ccflow_tokenizer__ = DefaultTokenizer.with_ast()  # AST normalization instead of bytecode
    x: int = 1
```

**`normalize_token.register()`** — register a global handler for a custom type:

```python
from ccflow.utils.tokenize import normalize_token

@normalize_token.register(MyDatabaseConnection)
def _(obj):
    return ("db", obj.host, obj.port, obj.database)
```

**`__ccflow_tokenize__()`** — define a canonical form on the class itself:

```python
class MySpecialType:
    def __init__(self, data):
        self.data = data

    def __ccflow_tokenize__(self):
        return ("MySpecialType", self.data.key)
```

If both a `normalize_token.register()` handler and `__ccflow_tokenize__()` exist for the same type, the singledispatch handler takes priority.

### Global Tokenizer Override

You can change the tokenizer for ALL `BaseModel` subclasses at runtime:

```python
from ccflow import BaseModel
from ccflow.utils.tokenize import DefaultTokenizer

# Switch to AST-based behavior hashing globally
BaseModel.__ccflow_tokenizer__ = DefaultTokenizer.with_ast()

# Or disable behavior hashing globally
BaseModel.__ccflow_tokenizer__ = DefaultTokenizer()
```

Subclasses that define their own `__ccflow_tokenizer__` are not affected.

### Building Custom Tokenizers

The tokenizer is composed from two pluggable components:

```python
from ccflow.utils.tokenize import (
    DefaultTokenizer,
    OwnMethodCollector,      # which functions to hash
    ASTSourceTokenizer,      # how to hash each function
    BytecodeSourceTokenizer,
)

# Full control over composition
tokenizer = DefaultTokenizer(
    collector=OwnMethodCollector(),
    source_tokenizer=BytecodeSourceTokenizer(),
)
```

Implement `FunctionCollector` or `SourceTokenizer` to create custom strategies.

## Cache Keys and MemoryCacheEvaluator

The `MemoryCacheEvaluator` uses `model_token` as the cache key:

```python
from ccflow import BaseModel, CallableModel, ContextBase, ResultBase
from ccflow.evaluators import MemoryCacheEvaluator

class MyContext(ContextBase):
    date: str = "2024-01-01"

class MyResult(ResultBase):
    value: float = 0.0

class MyCallable(CallableModel):
    multiplier: float = 1.0

    def __call__(self, ctx: MyContext) -> MyResult:
        return MyResult(value=float(ctx.date.replace("-", "")) * self.multiplier)

# The cache key is derived from model_token of the evaluation context
# Same context + same callable config → cache hit
cached = MemoryCacheEvaluator()
```

For `ModelEvaluationContext` (the wrapper that chains evaluators), the `model_token` implementation is smart about stripping "transparent" evaluator layers (like `LoggingEvaluator`) so that the cache key depends only on the actual computation.

## Limitations and Caveats

### Things That Will Produce Different Tokens When They Shouldn't

These produce **false cache misses** (safe but wasteful):

| Scenario                                               | Why                                                                | Mitigation                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------ |
| **Different Python minor version** (bytecode mode)     | `co_code` format changes between Python 3.11 → 3.12                | Use `with_ast()` for cross-version stability     |
| **Variable rename** (AST mode)                         | AST preserves variable names: `def f(x)` ≠ `def f(y)`              | Use `with_bytecode()` if renames are common      |
| **Pydantic injects `model_post_init`** into subclasses | Pydantic adds this to every `__dict__` even if you don't define it | Acceptable — consistent within a class hierarchy |

### Things That Will Produce the Same Token When They Shouldn't

These produce **false cache hits** (dangerous — stale results):

| Scenario                                                            | Why                                                                             | Mitigation                                                                                        |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Upstream code changes** (functions in other modules)              | No transitive dependency tracking — only methods on the class itself are hashed | Add critical dependencies to `__ccflow_tokenizer_deps__`                                          |
| **Python package version upgrades** (numpy, pandas, etc.)           | Package versions are not part of the hash                                       | Add a version field (see [Injecting External State](#injecting-external-state-into-tokens))       |
| **Data file changes on disk**                                       | File paths hash the same even if contents change                                | Add a file checksum field (see [Injecting External State](#injecting-external-state-into-tokens)) |
| **Environment variables / config changes**                          | External state not captured                                                     | Add env var fields (see [Injecting External State](#injecting-external-state-into-tokens))        |
| **Database schema or data changes**                                 | Only the query config is hashed, not the data                                   | Use time-based cache invalidation or include a data version field                                 |
| **Git branch / commit changes** (without code changes to the class) | No git integration                                                              | Add git hash as a field if needed                                                                 |

### Injecting External State into Tokens

Since `model_token` is computed from field values, you can include external state — package versions, file checksums, environment variables — by adding fields with `default_factory`. Use `repr=False` to keep them out of `__repr__` if desired:

**Package versions:**

```python
from pydantic import Field

class MyPipeline(BaseModel):
    x: int = 1
    pandas_version: str = Field(
        default_factory=lambda: __import__("pandas").__version__,
        repr=False,
    )
```

The version is captured once at construction. If pandas is upgraded and the model is re-created, the token changes automatically.

**Environment variables:**

```python
import os

class EnvAwareModel(BaseModel):
    x: int = 1
    deploy_env: str = Field(
        default_factory=lambda: os.environ.get("DEPLOY_ENV", "dev"),
        repr=False,
    )
```

**File checksums (using `model_validator`):**

When the extra data depends on another field (e.g. computing a checksum of a file path), use a `model_validator` instead of `default_factory`:

```python
import hashlib
from pydantic import Field, model_validator

class FileProcessor(BaseModel):
    input_path: str = "data.csv"
    input_checksum: str = Field(default="", repr=False)

    @model_validator(mode="after")
    def _compute_checksum(self):
        if not self.input_checksum:
            try:
                with open(self.input_path, "rb") as f:
                    self.input_checksum = hashlib.sha256(f.read()).hexdigest()
            except FileNotFoundError:
                self.input_checksum = "file_not_found"
        return self
```

Now the token changes whenever the file contents change, even if `input_path` stays the same. Users can also override `input_checksum` explicitly for testing.

> **Why not `@computed_field`?** Computed fields are evaluated lazily — every time the property is accessed. Since `model_token` reads all fields, using `@computed_field` would force evaluation on every token computation, which is wasteful for expensive operations (file I/O, subprocess calls). A regular field with `default_factory` or `model_validator` computes the value once at construction.

### Other Caveats

- **Large numpy arrays**: `tobytes()` copies the full array into memory for hashing. For very large arrays, this may be slow.
- **Polars / Arrow**: Work via cloudpickle fallback (no explicit optimized handlers).
- **Cycles**: Handled gracefully — a cycle produces `("__cycle__", type_path)` as a sentinel.
- **Unpicklable objects**: If cloudpickle cannot serialize an object and no `__ccflow_tokenize__()` method or `normalize_token` handler is registered, tokenization raises `TypeError`. Register a custom handler to support such types.

## Architecture

The tokenization system has two layers:

```
┌─────────────────────────────────────────┐
│  BaseModel API Layer (ccflow/base.py)   │
│  • model_token property                 │
│  • __ccflow_tokenizer__ ClassVar        │
│  • _model_token cache (PrivateAttr)     │
└──────────────┬──────────────────────────┘
               │ delegates to
┌──────────────▼──────────────────────────┐
│  Tokenizer Engine (utils/tokenize.py)   │
│                                         │
│  DefaultTokenizer                       │
│  ├── collector: FunctionCollector?       │
│  │   └── OwnMethodCollector             │
│  ├── source_tokenizer: SourceTokenizer? │
│  │   ├── ASTSourceTokenizer             │
│  │   └── BytecodeSourceTokenizer        │
│  └── normalize_token (singledispatch)   │
│      ├── int, str, float, ...           │
│      ├── numpy.ndarray                  │
│      ├── pandas.DataFrame               │
│      └── pydantic.BaseModel             │
└─────────────────────────────────────────┘
```

The engine (`utils/tokenize.py`) has **zero imports from ccflow** — it's a standalone leaf module that can be used independently.
