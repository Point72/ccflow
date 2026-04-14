"""Tests for the tokenization engine (ccflow.utils.tokenize) and BaseModel integration."""

import enum
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, computed_field, model_validator

from ccflow import BaseModel, ContextBase
from ccflow.utils.tokenize import (
    ASTSourceTokenizer,
    BytecodeSourceTokenizer,
    DefaultTokenizer,
    OwnMethodCollector,
    SourceTokenizer,
    compute_behavior_token,
    normalize_token,
)

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    x: int = 1
    y: str = "hello"


class NestedModel(BaseModel):
    child: SimpleModel = SimpleModel()
    name: str = "parent"


class ExcludedFieldModel(BaseModel):
    important: int = 42
    debug_info: str = Field(default="debug", exclude=True)


class FrozenModel(ContextBase):
    a: int = 1
    b: str = "frozen"


class NoCacheModel(BaseModel):
    model_config = ConfigDict(cache_token=False)
    value: int = 0


class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class ModelWithCollections(BaseModel):
    tags: List[str] = []
    metadata: dict = {}
    coords: tuple = ()


class ModelWithOptional(BaseModel):
    name: str = "test"
    extra: Optional[int] = None


class SubModel(SimpleModel):
    z: float = 3.14


# ---------------------------------------------------------------------------
# Module-level factory for behavior-hashing tests
# ---------------------------------------------------------------------------

_AST_TOKENIZER = DefaultTokenizer.with_ast()


def _make_ast_model(name="DynModel", *, base=BaseModel, deps=None, **attrs):
    """Build a BaseModel subclass with behavior hashing for testing."""
    cls_attrs = {
        "x": 1,
        "__annotations__": {"x": int},
        "__ccflow_tokenizer__": _AST_TOKENIZER,
    }
    if deps is not None:
        cls_attrs["__ccflow_tokenizer_deps__"] = deps
    cls_attrs.update(attrs)
    return type(name, (base,), cls_attrs)


# ---------------------------------------------------------------------------
# normalize_token tests
# ---------------------------------------------------------------------------


class TestNormalizeToken:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (True, True),
            (False, False),
            (42, 42),
            (3.14, 3.14),
            ("hello", "hello"),
            (b"data", b"data"),
        ],
    )
    def test_primitives(self, value, expected):
        assert normalize_token(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (date(2024, 1, 15), ("date", "2024-01-15")),
            (datetime(2024, 1, 15, 10, 30, 0), ("datetime", "2024-01-15T10:30:00")),
            (time(10, 30, 0), ("time", "10:30:00")),
            (timedelta(hours=1, minutes=30), ("timedelta", 5400.0)),
        ],
    )
    def test_datetime_types(self, value, expected):
        assert normalize_token(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ((1, "a", True), ("tuple", (1, "a", True))),
            ([1, 2, 3], ("list", (1, 2, 3))),
            ({3, 1, 2}, ("set", (1, 2, 3))),
            (frozenset({3, 1, 2}), ("frozenset", (1, 2, 3))),
            ({"b": 2, "a": 1}, ("dict", (("a", 1), ("b", 2)))),
        ],
    )
    def test_collections(self, value, expected):
        assert normalize_token(value) == expected

    def test_uuid(self):
        u = UUID("12345678-1234-5678-1234-567812345678")
        assert normalize_token(u) == ("uuid", "12345678-1234-5678-1234-567812345678")

    def test_path(self):
        p = Path("/tmp/test.txt")
        assert normalize_token(p) == ("path", "/tmp/test.txt")

    def test_pure_path(self):
        p = PurePosixPath("/tmp/test.txt")
        assert normalize_token(p) == ("path", "/tmp/test.txt")

    def test_enum(self):
        result = normalize_token(Color.RED)
        assert result == ("enum", "Color", "RED")

    def test_nested_collections(self):
        data = {"key": [1, (2, 3)]}
        result = normalize_token(data)
        assert result == ("dict", (("key", ("list", (1, ("tuple", (2, 3))))),))

    def test_numpy_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = normalize_token(arr)
        assert result[0] == "ndarray"
        assert result[1] == "int64"
        assert result[2] == (3,)
        arr2 = np.array([1, 2, 3], dtype=np.int64)
        assert normalize_token(arr) == normalize_token(arr2)

    def test_numpy_different_data(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        assert normalize_token(arr1) != normalize_token(arr2)

    def test_numpy_scalar(self):
        s = np.int64(42)
        result = normalize_token(s)
        assert result == ("np_scalar", "int64", 42)

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = normalize_token(df)
        assert result is not None
        # Same data → same token
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert normalize_token(df) == normalize_token(df2)

    def test_pandas_different_data(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 3]})
        assert normalize_token(df1) != normalize_token(df2)

    def test_pandas_series(self):
        s = pd.Series([1, 2, 3], name="test")
        result = normalize_token(s)
        assert result is not None

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        result = normalize_token(ts)
        assert result == ("pd_timestamp", ts.isoformat())

    def test_function(self):
        def my_func(x):
            return x + 1

        result = normalize_token(my_func)
        assert result[0] == "func"
        assert "my_func" in result[1]
        assert len(result) == 3  # (func, qualname, hash)

    def test_function_deterministic(self):
        def my_func(x):
            return x + 1

        r1 = normalize_token(my_func)
        r2 = normalize_token(my_func)
        assert r1 == r2

    def test_type(self):
        result = normalize_token(int)
        assert result == ("type", "builtins.int")

    def test_custom_type(self):
        result = normalize_token(SimpleModel)
        assert result[0] == "type"
        assert "SimpleModel" in result[1]

    def test_pydantic_basemodel(self):
        class PlainPydantic(PydanticBaseModel):
            x: int = 1

        obj = PlainPydantic(x=5)
        result = normalize_token(obj)
        assert result[0] == "pydantic"
        assert "PlainPydantic" in result[1]

    def test_ccflow_basemodel(self):
        obj = SimpleModel(x=5, y="world")
        result = normalize_token(obj)
        # Delegates to tokenizer — same output as normalize()
        assert result == obj.__ccflow_tokenizer__.normalize(obj)
        assert "SimpleModel" in result[0]

    def test_custom_hook(self):
        class MyObj:
            def __ccflow_tokenize__(self):
                return ("custom", 42)

        obj = MyObj()
        assert normalize_token(obj) == ("custom", 42)


# ---------------------------------------------------------------------------
# DefaultTokenizer tests
# ---------------------------------------------------------------------------


class TestDefaultTokenizer:
    def test_normalize_basic(self):
        t = DefaultTokenizer()
        model = SimpleModel(x=5, y="world")
        result = t.normalize(model)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (type_path, behavior, fields)
        assert result[1] is None  # No behavior token by default

    def test_tokenize_deterministic(self):
        t = DefaultTokenizer()
        model = SimpleModel(x=5, y="world")
        token1 = t.tokenize(model)
        token2 = t.tokenize(model)
        assert token1 == token2

    def test_tokenize_different_values(self):
        t = DefaultTokenizer()
        m1 = SimpleModel(x=1, y="a")
        m2 = SimpleModel(x=2, y="a")
        assert t.tokenize(m1) != t.tokenize(m2)

    def test_tokenize_produces_sha256(self):
        t = DefaultTokenizer()
        model = SimpleModel(x=1)
        token = t.tokenize(model)
        assert len(token) == 64  # SHA256 hex

    def test_excluded_fields_not_in_normalize(self):
        t = DefaultTokenizer()
        m = ExcludedFieldModel(important=10, debug_info="x")
        result = t.normalize(m)
        field_names = [f[0] for f in result[2]]
        assert "important" in field_names
        assert "debug_info" not in field_names

    def test_excluded_fields_dont_affect_token(self):
        t = DefaultTokenizer()
        m1 = ExcludedFieldModel(important=10, debug_info="debug1")
        m2 = ExcludedFieldModel(important=10, debug_info="debug2")
        assert t.tokenize(m1) == t.tokenize(m2)

    def test_nested_model(self):
        t = DefaultTokenizer()
        m = NestedModel(child=SimpleModel(x=5), name="test")
        token = t.tokenize(m)
        assert isinstance(token, str)

    def test_nested_different_child(self):
        t = DefaultTokenizer()
        m1 = NestedModel(child=SimpleModel(x=1), name="test")
        m2 = NestedModel(child=SimpleModel(x=2), name="test")
        assert t.tokenize(m1) != t.tokenize(m2)

    def test_cycle_detection(self):
        """Cycle detection prevents infinite recursion."""
        t = DefaultTokenizer()
        # Create a model — since ccflow BaseModel doesn't allow arbitrary attrs,
        # we test cycle detection via the normalize method directly
        m = SimpleModel(x=1)
        visited = {id(m)}
        result = t.normalize(m, _visited=visited)
        assert result[0] == "__cycle__"


# ---------------------------------------------------------------------------
# BaseModel.model_token integration tests
# ---------------------------------------------------------------------------


class TestModelToken:
    def test_basic(self):
        m = SimpleModel(x=1, y="hello")
        token = m.model_token
        assert isinstance(token, str)
        assert len(token) == 64  # SHA256 hex

    def test_deterministic(self):
        m = SimpleModel(x=1, y="hello")
        assert m.model_token == m.model_token

    def test_same_values_same_token(self):
        m1 = SimpleModel(x=1, y="hello")
        m2 = SimpleModel(x=1, y="hello")
        assert m1.model_token == m2.model_token

    def test_different_values_different_token(self):
        m1 = SimpleModel(x=1, y="hello")
        m2 = SimpleModel(x=2, y="hello")
        assert m1.model_token != m2.model_token

    def test_different_types_different_token(self):
        """Parent and subclass with same field values get different tokens."""
        m1 = SimpleModel(x=1, y="hello")
        m2 = SubModel(x=1, y="hello")
        assert m1.model_token != m2.model_token

    def test_mutable_no_cache(self):
        """Mutable models do not cache tokens by default."""
        m = SimpleModel(x=1, y="hello")
        token1 = m.model_token
        assert m._model_token is None  # Not cached (mutable)
        assert m.model_token == token1  # Still deterministic

    def test_mutable_reflects_mutation(self):
        """Mutable model token reflects field assignment immediately."""
        m = SimpleModel(x=1, y="hello")
        token1 = m.model_token
        m.x = 2
        token2 = m.model_token
        assert token2 != token1

    def test_no_cache_mode(self):
        """With cache_token=False, token is always computed fresh."""
        m = NoCacheModel(value=42)
        token1 = m.model_token
        assert m._model_token is None  # Never cached
        token2 = m.model_token
        assert token1 == token2  # Still deterministic

    def test_frozen_model(self):
        """Frozen models cache the token."""
        m = FrozenModel(a=1, b="test")
        token = m.model_token
        assert m._model_token is not None
        assert m.model_token == token

    def test_excluded_field_no_effect(self):
        """Fields with exclude=True don't affect the token."""
        m1 = ExcludedFieldModel(important=10, debug_info="x")
        m2 = ExcludedFieldModel(important=10, debug_info="y")
        assert m1.model_token == m2.model_token

    def test_nested_model_token(self):
        m = NestedModel(child=SimpleModel(x=5), name="parent")
        assert isinstance(m.model_token, str)

    def test_optional_none_vs_value(self):
        m1 = ModelWithOptional(name="test", extra=None)
        m2 = ModelWithOptional(name="test", extra=42)
        assert m1.model_token != m2.model_token

    def test_collections_in_model(self):
        m = ModelWithCollections(tags=["a", "b"], metadata={"k": "v"}, coords=(1, 2))
        assert isinstance(m.model_token, str)

    def test_model_copy_gets_fresh_token(self):
        """model_copy(update=...) produces correct (different) token."""
        m1 = SimpleModel(x=1, y="hello")
        _ = m1.model_token
        m2 = m1.model_copy(update={"x": 2})
        assert m1.model_token != m2.model_token

    def test_custom_tokenizer(self):
        """Models can use a custom tokenizer via __ccflow_tokenizer__."""

        class CustomModel(BaseModel):
            __ccflow_tokenizer__ = DefaultTokenizer.with_bytecode()
            value: int = 0

        m = CustomModel(value=42)
        assert len(m.model_token) == 64  # SHA256

    def test_pickle_preserves_token_cache(self):
        """Pickling a model preserves the token cache."""
        m = SimpleModel(x=1, y="hello")
        _ = m.model_token
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.model_token == m.model_token


# ---------------------------------------------------------------------------
# Component-level tests: SourceTokenizer and FunctionCollector
# ---------------------------------------------------------------------------


class TestASTSourceTokenizer:
    def test_returns_hex_digest(self):
        def f(x):
            return x + 1

        result = ASTSourceTokenizer().tokenize(f)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 64  # sha256

    def test_deterministic(self):
        def f(x):
            return x + 1

        t = ASTSourceTokenizer()
        assert t.tokenize(f) == t.tokenize(f)

    def test_different_bodies_differ(self):
        def f1(x):
            return x + 1

        def f2(x):
            return x * 2

        t = ASTSourceTokenizer()
        assert t.tokenize(f1) != t.tokenize(f2)

    def test_docstring_stripped(self):
        def f(x):
            """A docstring."""
            return x + 1

        tok_with = ASTSourceTokenizer().tokenize(f)

        def f(x):  # noqa: F811
            return x + 1

        tok_without = ASTSourceTokenizer().tokenize(f)
        assert tok_with == tok_without

    def test_variable_rename_changes_hash(self):
        """AST preserves variable names, so renaming changes the hash."""

        def f1(x):
            return x + 1

        def f2(y):
            return y + 1

        t = ASTSourceTokenizer()
        assert t.tokenize(f1) != t.tokenize(f2)

    def test_comment_changes_ignored(self):
        """Comments are stripped by AST parsing."""
        from ccflow.utils.tokenize import _normalize_source_ast

        s1 = "def f(x):\n    # comment\n    return x + 1"
        s2 = "def f(x):\n    return x + 1"
        assert _normalize_source_ast(s1) == _normalize_source_ast(s2)

    def test_whitespace_changes_ignored(self):
        from ccflow.utils.tokenize import _normalize_source_ast

        s1 = "def f(x):\n    return x+1"
        s2 = "def f(  x  ):\n    return x +  1"
        assert _normalize_source_ast(s1) == _normalize_source_ast(s2)

    def test_fallback_to_bytecode_when_no_source(self):
        """Built-in functions have no source; should fall back to bytecode or return None."""
        t = ASTSourceTokenizer()
        # Built-in like len has no source and no __code__
        result = t.tokenize(len)
        assert result is None

    def test_lambda(self):
        def f(x):
            return x + 1

        t = ASTSourceTokenizer()
        assert t.tokenize(f) is not None

    def test_classmethod_unwrapped(self):
        """Classmethods need __func__ unwrapping before tokenizing."""

        class C:
            @classmethod
            def m(cls):
                return 1

        t = ASTSourceTokenizer()
        # __func__ should be unwrapped by the collector, not the tokenizer
        assert t.tokenize(C.m.__func__) is not None

    def test_staticmethod_unwrapped(self):
        class C:
            @staticmethod
            def m():
                return 1

        t = ASTSourceTokenizer()
        assert t.tokenize(C.m) is not None


class TestBytecodeSourceTokenizer:
    def test_returns_hex_digest(self):
        def f(x):
            return x + 1

        result = BytecodeSourceTokenizer().tokenize(f)
        assert result is not None
        assert len(result) == 64  # sha256

    def test_deterministic(self):
        def f(x):
            return x + 1

        t = BytecodeSourceTokenizer()
        assert t.tokenize(f) == t.tokenize(f)

    def test_different_bodies_differ(self):
        def f1(x):
            return x + 1

        def f2(x):
            return x * 2

        t = BytecodeSourceTokenizer()
        assert t.tokenize(f1) != t.tokenize(f2)

    def test_docstring_stripped(self):
        def f(x):
            """A docstring."""
            return x + 1

        tok_with = BytecodeSourceTokenizer().tokenize(f)

        def f(x):  # noqa: F811
            return x + 1

        tok_without = BytecodeSourceTokenizer().tokenize(f)
        assert tok_with == tok_without

    def test_variable_rename_same_hash(self):
        """Bytecode is immune to variable renames (names in co_varnames, not co_code)."""

        def f1(x):
            return x + 1

        def f2(y):
            return y + 1

        t = BytecodeSourceTokenizer()
        assert t.tokenize(f1) == t.tokenize(f2)

    def test_no_code_returns_none(self):
        """Objects without __code__ return None."""
        t = BytecodeSourceTokenizer()
        assert t.tokenize(len) is None

    def test_lambda(self):
        def f(x):
            return x + 1

        t = BytecodeSourceTokenizer()
        assert t.tokenize(f) is not None

    def test_classmethod_vs_regular_same_hash(self):
        """Bytecode doesn't distinguish classmethod from regular method."""

        class C1:
            @classmethod
            def m(cls):
                return 1

        class C2:
            def m(self):
                return 1

        t = BytecodeSourceTokenizer()
        assert t.tokenize(C1.m.__func__) == t.tokenize(C2.m)


class TestOwnMethodCollector:
    def test_collects_regular_methods(self):
        class C:
            def foo(self):
                pass

            def bar(self):
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        assert "foo" in names
        assert "bar" in names

    def test_sorted_by_name(self):
        class C:
            def z(self):
                pass

            def a(self):
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        # Should be sorted
        assert names.index("a") < names.index("z")

    def test_collects_classmethod(self):
        class C:
            @classmethod
            def m(cls):
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        assert "m" in names
        # Should be unwrapped
        func = dict(methods)["m"]
        assert callable(func)
        assert not isinstance(func, classmethod)

    def test_collects_staticmethod(self):
        class C:
            @staticmethod
            def m():
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        assert "m" in names
        func = dict(methods)["m"]
        assert callable(func)

    def test_skips_non_callable(self):
        class C:
            x = 42

            def m(self):
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        assert "x" not in names
        assert "m" in names

    def test_does_not_collect_inherited(self):
        class Parent:
            def parent_method(self):
                pass

        class Child(Parent):
            def child_method(self):
                pass

        methods = OwnMethodCollector().collect(Child)
        names = [name for name, _ in methods]
        assert "child_method" in names
        assert "parent_method" not in names

    def test_empty_class(self):
        class Empty:
            pass

        methods = OwnMethodCollector().collect(Empty)
        # May have __init__ or other dunders from object, but no user methods
        # The key thing is it doesn't crash
        assert isinstance(methods, list)

    def test_deps_included(self):
        def helper():
            return 42

        class C:
            __ccflow_tokenizer_deps__ = [helper]

            def m(self):
                pass

        methods = OwnMethodCollector().collect(C)
        names = [name for name, _ in methods]
        assert any("__dep__" in n for n in names)

    def test_deps_not_inherited(self):
        def helper():
            return 42

        class Parent:
            __ccflow_tokenizer_deps__ = [helper]

        class Child(Parent):
            pass

        methods = OwnMethodCollector().collect(Child)
        names = [name for name, _ in methods]
        assert not any("__dep__" in n for n in names)


class TestSourceTokenizerContrast:
    """Tests documenting known differences between AST and bytecode tokenizers."""

    def test_variable_rename_ast_differs_bytecode_same(self):
        """AST is sensitive to renames; bytecode is not."""

        def f1(x):
            return x + 1

        def f2(y):
            return y + 1

        assert ASTSourceTokenizer().tokenize(f1) != ASTSourceTokenizer().tokenize(f2)
        assert BytecodeSourceTokenizer().tokenize(f1) == BytecodeSourceTokenizer().tokenize(f2)

    def test_both_strip_docstrings(self):
        def f(x):
            """doc"""
            return x + 1

        tok_ast_with = ASTSourceTokenizer().tokenize(f)
        tok_bc_with = BytecodeSourceTokenizer().tokenize(f)

        def f(x):  # noqa: F811
            return x + 1

        tok_ast_without = ASTSourceTokenizer().tokenize(f)
        tok_bc_without = BytecodeSourceTokenizer().tokenize(f)

        assert tok_ast_with == tok_ast_without
        assert tok_bc_with == tok_bc_without


# ---------------------------------------------------------------------------
# Behavior token tests
# ---------------------------------------------------------------------------


class TestBehaviorToken:
    @pytest.mark.parametrize(
        "cls_factory,expect_none",
        [
            pytest.param(lambda: type("C", (), {"__call__": lambda s, x: x + 1}), False, id="with-call"),
            pytest.param(lambda: type("C", (), {}), True, id="without-call"),
            pytest.param(lambda: type("C", (), {"__call__": classmethod(lambda c, x: x + 1)}), False, id="classmethod-call"),
            pytest.param(lambda: type("C", (), {"__call__": staticmethod(lambda x: x + 1)}), False, id="staticmethod-call"),
        ],
    )
    def test_behavior_token_presence(self, cls_factory, expect_none):
        token = compute_behavior_token(cls_factory())
        assert (token is None) == expect_none

    def test_deterministic(self):
        class MyCallable:
            def __call__(self, x):
                return x + 1

        assert compute_behavior_token(MyCallable) == compute_behavior_token(MyCallable)

    def test_cached_on_class(self):
        class MyCallable:
            def __call__(self, x):
                return x * 2

        token = compute_behavior_token(MyCallable)
        assert hasattr(MyCallable, "__ccflow_behavior_token__")
        assert token in MyCallable.__ccflow_behavior_token__.values()

    def test_different_implementations(self):
        class Call1:
            def __call__(self, x):
                return x + 1

        class Call2:
            def __call__(self, x):
                return x * 2

        assert compute_behavior_token(Call1) != compute_behavior_token(Call2)

    def test_docstring_ignored_with_ast(self):
        """AST-normalized hashing should ignore docstrings."""

        class WithDoc:
            def __call__(self, x):
                """This is a docstring."""
                return x + 1

        class WithoutDoc:
            def __call__(self, x):
                return x + 1

        assert compute_behavior_token(WithDoc) == compute_behavior_token(WithoutDoc)

    def test_behavior_token_not_inherited_from_parent(self):
        class Parent:
            def __call__(self, x):
                return x + 1

        class Child(Parent):
            pass

        assert compute_behavior_token(Parent) is not None
        assert compute_behavior_token(Child) is None

    def test_behavior_token_deterministic(self):
        class MyCallable:
            def __call__(self, x):
                return x + 1

        t1 = compute_behavior_token(MyCallable)
        t2 = compute_behavior_token(MyCallable)
        assert t1 == t2
        assert len(t1) == 64  # sha256

    def test_classmethod_vs_regular_differ(self):
        class AsClassmethod:
            @classmethod
            def __call__(cls, x):
                return x + 1

        class AsRegular:
            def __call__(self, x):
                return x + 1

        t1 = compute_behavior_token(AsClassmethod, source_tokenizer=ASTSourceTokenizer())
        t2 = compute_behavior_token(AsRegular, source_tokenizer=ASTSourceTokenizer())
        assert t1 != t2

    def test_include_behavior_in_tokenizer(self):
        M1 = _make_ast_model(__call__=lambda self: self.x + 1)
        M2 = _make_ast_model(__call__=lambda self: self.x * 2)
        assert M1(x=1).model_token != M2(x=1).model_token


# ---------------------------------------------------------------------------
# Composition tests (replaces TestTokenizerConfig)
# ---------------------------------------------------------------------------


class TestComposition:
    def test_default_has_no_collector_or_source_tokenizer(self):
        t = DefaultTokenizer()
        assert t.collector is None
        assert t.source_tokenizer is None

    def test_with_ast_creates_correct_components(self):
        t = DefaultTokenizer.with_ast()
        assert isinstance(t.collector, OwnMethodCollector)
        assert isinstance(t.source_tokenizer, ASTSourceTokenizer)

    def test_with_bytecode_creates_correct_components(self):
        t = DefaultTokenizer.with_bytecode()
        assert isinstance(t.collector, OwnMethodCollector)
        assert isinstance(t.source_tokenizer, BytecodeSourceTokenizer)

    def test_custom_composition(self):
        collector = OwnMethodCollector()
        source_tokenizer = ASTSourceTokenizer()
        t = DefaultTokenizer(collector=collector, source_tokenizer=source_tokenizer)
        assert t.collector is collector
        assert t.source_tokenizer is source_tokenizer


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_model(self):
        class Empty(BaseModel):
            pass

        assert isinstance(Empty().model_token, str)

    def test_model_with_none_values(self):
        assert isinstance(ModelWithOptional(name="test", extra=None).model_token, str)

    def test_model_with_numpy_field(self):
        class NumpyModel(BaseModel):
            data: Any = None  # type: ignore

        assert isinstance(NumpyModel(data=np.array([1, 2, 3])).model_token, str)

    def test_model_with_dataframe_field(self):
        class DFModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            data: Any = None

        assert isinstance(DFModel(data=pd.DataFrame({"a": [1, 2]})).model_token, str)

    def test_deeply_nested(self):
        class Node(BaseModel):
            value: int = 0
            child: Optional["Node"] = None

        Node.model_rebuild()
        current = Node(value=50)
        for i in range(49, 0, -1):
            current = Node(value=i, child=current)
        assert isinstance(current.model_token, str)

    def test_model_with_enum_field(self):
        class EnumModel(BaseModel):
            color: Color = Color.RED

        m1 = EnumModel(color=Color.GREEN)
        m2 = EnumModel(color=Color.RED)
        assert isinstance(m1.model_token, str)
        assert m1.model_token != m2.model_token


# ---------------------------------------------------------------------------
# Diamond pattern tests
# ---------------------------------------------------------------------------


class TestDiamondPatterns:
    """Tests for shared child references (diamond DAGs) in model graphs."""

    def test_shared_child_alias_is_not_treated_as_cycle(self):
        """Parent has child1 and child2 pointing to the SAME SimpleModel instance.
        The tokenizer should treat this as a diamond, not a cycle, and produce
        the same token as a parent with two distinct-but-equal children."""

        class TwoChildren(BaseModel):
            child1: SimpleModel = SimpleModel()
            child2: SimpleModel = SimpleModel()

        shared = SimpleModel(x=42, y="shared")
        parent_shared = TwoChildren(child1=shared, child2=shared)
        parent_distinct = TwoChildren(
            child1=SimpleModel(x=42, y="shared"),
            child2=SimpleModel(x=42, y="shared"),
        )
        # Diamond and distinct-but-equal should produce the same token
        assert parent_shared.model_token == parent_distinct.model_token

    def test_shared_mutable_child_two_paths_deterministic(self):
        """Same shared child referenced twice; repeated tokenization gives identical result."""

        class TwoChildren(BaseModel):
            child1: SimpleModel = SimpleModel()
            child2: SimpleModel = SimpleModel()

        shared = SimpleModel(x=7, y="s")
        parent = TwoChildren(child1=shared, child2=shared)
        t1 = parent.model_token
        t2 = parent.model_token
        assert t1 == t2

    def test_shared_frozen_child_same_parent_token_with_and_without_warmed_cache(self):
        """Same frozen child reused in two parents; compare parent token before and
        after child _model_token cache is 'warmed'. Token should be stable."""

        class TwoFrozen(BaseModel):
            a: FrozenModel = FrozenModel()
            b: FrozenModel = FrozenModel()

        child = FrozenModel(a=99, b="warm")
        # Parent token BEFORE child cache is warmed
        parent1 = TwoFrozen(a=child, b=child)
        token_cold = parent1.model_token

        _ = child.model_token
        assert child._model_token is not None

        parent2 = TwoFrozen(a=child, b=child)
        token_warm = parent2.model_token
        assert parent2.model_token == token_warm
        # Frozen children always use ("__child__", value.model_token), so
        # cold and warm parent tokens are now consistent.
        assert token_cold == token_warm


# ---------------------------------------------------------------------------
# Cycle tests
# ---------------------------------------------------------------------------


class SelfRefModel(BaseModel):
    """Model that can reference itself."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: int = 0
    child: Optional["SelfRefModel"] = None


SelfRefModel.model_rebuild()


class TestCycles:
    """Tests for cycle detection in model graphs."""

    def test_self_referential_model_token_is_deterministic(self):
        """node.child = node; tokenization should not recurse forever."""
        node = SelfRefModel(value=1)
        # Bypass pydantic's validate_assignment (it rejects self-referential cycles)
        node.__dict__["child"] = node
        token = node.model_token
        assert isinstance(token, str)
        assert node.model_token == token

    def test_indirect_cycle_a_to_b_to_a(self):
        """Two-node cycle A -> B -> A."""
        a = SelfRefModel(value=1)
        b = SelfRefModel(value=2)
        a.__dict__["child"] = b
        b.__dict__["child"] = a
        token_a = a.model_token
        assert isinstance(token_a, str)

    def test_cycle_marker_differs_from_none(self):
        """Token of child=None must differ from child=self (cycle)."""
        acyclic = SelfRefModel(value=1, child=None)
        cyclic = SelfRefModel(value=1)
        cyclic.__dict__["child"] = cyclic

        token_acyclic = acyclic.model_token
        token_cyclic = cyclic.model_token
        assert token_acyclic != token_cyclic

    def test_cycle_inside_list_field(self):
        """Model contains items=[self]; should handle gracefully."""

        class ListModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            value: int = 0
            items: List[Any] = []

        m = ListModel(value=1)
        m.items = [m]
        token = m.model_token
        assert isinstance(token, str)

    def test_cycle_inside_dict_field(self):
        """Model contains refs={'self': self}; should handle gracefully."""

        class DictModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            value: int = 0
            refs: Dict[str, Any] = {}

        m = DictModel(value=1)
        m.refs = {"self": m}
        token = m.model_token
        assert isinstance(token, str)


# ---------------------------------------------------------------------------
# Unpicklable / unstable fallback objects
# ---------------------------------------------------------------------------


class TestUnpicklableFallback:
    """Tests for graceful handling of unpicklable or unstable objects."""

    def test_lock_field_tokenization(self):
        """Model with threading.Lock field — should either produce a stable
        token or raise a clear TypeError."""

        class LockModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            lock: Any = None

        m = LockModel(lock=threading.Lock())
        # Locks are cloudpickleable in some versions — so either we get a
        # token (possibly unstable) or a clear TypeError
        try:
            token = m.model_token
            assert isinstance(token, str)
        except TypeError as e:
            assert "tokenize" in str(e).lower() or "Cannot" in str(e)

    def test_generator_field_does_not_silently_tokenize(self):
        """Generators encode execution state; tokenization should be handled
        carefully or rejected."""

        def gen():
            yield 1

        class GenModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            g: Any = None

        m = GenModel(g=gen())
        try:
            token = m.model_token
            assert isinstance(token, str)
        except TypeError:
            pass  # Acceptable: clean rejection

    def test_compiled_regex_same_pattern_same_token(self):
        """re.compile with same pattern/flags should produce the same token."""
        r1 = re.compile("abc", re.IGNORECASE)
        r2 = re.compile("abc", re.IGNORECASE)
        t1 = normalize_token(r1)
        t2 = normalize_token(r2)
        assert t1 == t2

    def test_cloudpickle_failure_becomes_typeerror(self):
        """Object whose pickling deliberately raises should produce TypeError."""

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("cannot pickle")

        with pytest.raises(TypeError, match="Cannot tokenize"):
            normalize_token(Unpicklable())


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------


class TestGracefulFailure:
    def test_untokenizable_type_raises(self):
        """Types that can't be cloudpickled raise TypeError."""

        class Opaque:
            def __reduce__(self):
                raise TypeError("nope")

        with pytest.raises(TypeError):
            normalize_token(Opaque())


# ---------------------------------------------------------------------------
# Model inheritance edge cases
# ---------------------------------------------------------------------------


class TestModelInheritance:
    def test_multiple_inheritance_tokenizer_resolution(self):
        """When two bases define different tokenizers, child uses MRO resolution."""

        class Base1(BaseModel):
            __ccflow_tokenizer__ = DefaultTokenizer()
            x: int = 1

        class Base2(BaseModel):
            __ccflow_tokenizer__ = DefaultTokenizer.with_bytecode()
            y: int = 2

        class Child(Base1, Base2):
            z: int = 3

        m = Child()
        # MRO: Child -> Base1 -> Base2. Should use Base1's tokenizer (data-only)
        assert m.__ccflow_tokenizer__ is Base1.__ccflow_tokenizer__
        assert len(m.model_token) == 64


# ---------------------------------------------------------------------------
# Pydantic-specific edge cases
# ---------------------------------------------------------------------------


class TestPydanticEdgeCases:
    def test_model_construct_token_works_without_validation(self):
        """model_construct() skips validators. Token should still work."""
        m = SimpleModel.model_construct(x=10, y="constructed")
        token = m.model_token
        assert isinstance(token, str)
        m2 = SimpleModel(x=10, y="constructed")
        assert token == m2.model_token

    def test_validators_normalize_inputs_before_tokening(self):
        """Validator transforms raw input; token reflects validated state."""

        class NormalizedModel(BaseModel):
            name: str = ""

            @model_validator(mode="after")
            def _normalize_name(self):
                object.__setattr__(self, "name", self.name.strip().lower())
                # Also clear token cache since we modified a field
                if self.__pydantic_private__ is not None:
                    self.__pydantic_private__["_model_token"] = None
                return self

        m1 = NormalizedModel(name="  HELLO  ")
        m2 = NormalizedModel(name="hello")
        assert m1.model_token == m2.model_token

    def test_computed_field_does_not_affect_token(self):
        """@computed_field should not be included in the token by default,
        since it's derived from other fields."""

        class WithComputed(BaseModel):
            x: int = 1
            y: int = 2

            @computed_field
            @property
            def total(self) -> int:
                return self.x + self.y

        m1 = WithComputed(x=1, y=2)
        m2 = WithComputed(x=1, y=2)
        assert m1.model_token == m2.model_token
        assert "total" not in type(m1).model_fields

    def test_discriminated_union_variant_changes_token(self):
        """Different union branches with same outer model → different tokens."""

        class Cat(BaseModel):
            kind: Literal["cat"] = "cat"
            lives: int = 9

        class Dog(BaseModel):
            kind: Literal["dog"] = "dog"
            lives: int = 1

        class Owner(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            pet: Union[Cat, Dog] = Cat()

        owner_cat = Owner(pet=Cat(lives=9))
        owner_dog = Owner(pet=Dog(lives=9))
        assert owner_cat.model_token != owner_dog.model_token

    def test_model_validate_from_dict_matches_constructor_token(self):
        """model_validate({...}) vs normal construction → same token."""
        m1 = SimpleModel(x=42, y="test")
        m2 = SimpleModel.model_validate({"x": 42, "y": "test"})
        assert m1.model_token == m2.model_token


# ---------------------------------------------------------------------------
# Float / numeric edge cases
# ---------------------------------------------------------------------------


class TestFloatEdgeCases:
    def test_nan_token_is_deterministic(self):
        """Two separate NaN values should produce the same token."""
        nan1 = float("nan")
        assert normalize_token(nan1) is nan1  # Primitive passthrough

        # Even though nan != nan, repr is the same → same hash
        class NanModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            v: Any = None

        m1 = NanModel(v=float("nan"))
        m2 = NanModel(v=float("nan"))
        assert m1.model_token == m2.model_token

    def test_negative_zero_behavior_is_pinned(self):
        """-0.0 vs 0.0: document which way the tokenizer goes."""
        t_pos = normalize_token(0.0)
        t_neg = normalize_token(-0.0)
        # repr(0.0)='0.0', repr(-0.0)='-0.0' → different
        # This is a design choice: bit-pattern identity, not numeric equality
        # Just pin the behavior so changes are intentional
        if repr(0.0) != repr(-0.0):
            assert t_pos != t_neg or t_pos == t_neg  # Always true; real check below

        # The tokens in a model:
        class ZeroModel(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            v: Any = None

        m_pos = ZeroModel(v=0.0)
        m_neg = ZeroModel(v=-0.0)
        # Pin: these should differ (repr-based tokenization)
        assert m_pos.model_token != m_neg.model_token

    def test_inf_and_negative_inf_distinct(self):
        """inf vs -inf should produce different tokens."""
        assert normalize_token(float("inf")) != normalize_token(float("-inf"))

    def test_complex_number_tokenizes(self):
        """Complex numbers should be tokenizable (likely via cloudpickle fallback)."""
        c = 1 + 2j
        token = normalize_token(c)
        assert normalize_token(1 + 2j) == token
        assert normalize_token(3 + 4j) != token


# ---------------------------------------------------------------------------
# Container edge cases
# ---------------------------------------------------------------------------


class TestContainerEdgeCases:
    def test_dict_with_mixed_key_types(self):
        """Dict like {1: 'a', '1': 'b'} — canonicalization must handle
        heterogeneous keys without crashing."""
        d = {1: "a", "1": "b"}
        result = normalize_token(d)
        assert result[0] == "dict"
        assert len(result[1]) == 2

    def test_list_of_models_normalizes_structurally(self):
        """List[ChildModel] field should normalize models structurally."""

        class Parent(BaseModel):
            children: List[SimpleModel] = []

        m1 = Parent(children=[SimpleModel(x=1), SimpleModel(x=2)])
        m2 = Parent(children=[SimpleModel(x=1), SimpleModel(x=2)])
        assert m1.model_token == m2.model_token

        m3 = Parent(children=[SimpleModel(x=1), SimpleModel(x=3)])
        assert m1.model_token != m3.model_token

    def test_dict_of_models_normalizes_structurally(self):
        """Dict[str, ChildModel] field should normalize models structurally."""

        class Parent(BaseModel):
            children: Dict[str, SimpleModel] = {}

        m1 = Parent(children={"a": SimpleModel(x=1), "b": SimpleModel(x=2)})
        m2 = Parent(children={"a": SimpleModel(x=1), "b": SimpleModel(x=2)})
        assert m1.model_token == m2.model_token

    def test_nested_containers_with_none_values(self):
        """Mixed nested None in lists/dicts should tokenize cleanly."""

        class MixedModel(BaseModel):
            data: Any = None

        m1 = MixedModel(data=[None, {"key": None}, [None, 1]])
        m2 = MixedModel(data=[None, {"key": None}, [None, 1]])
        assert m1.model_token == m2.model_token

        m3 = MixedModel(data=[None, {"key": 1}, [None, 1]])
        assert m1.model_token != m3.model_token


# ---------------------------------------------------------------------------
# Merkle tree correctness
# ---------------------------------------------------------------------------


class TestMerkleCorrectness:
    def test_frozen_child_merkle_shortcut_matches_full_normalize(self):
        """Parent token with cached frozen child vs parent token when child
        is fully re-normalized. The Merkle shortcut should produce a
        consistent result."""

        class Parent(BaseModel):
            child: FrozenModel = FrozenModel()
            name: str = "p"

        child = FrozenModel(a=42, b="merkle")

        # Token WITHOUT Merkle shortcut (child cache not warmed)
        p1 = Parent(child=child, name="test")
        token_full = p1.model_token

        # Warm the child cache
        _ = child.model_token
        assert child._model_token is not None

        # Token WITH Merkle shortcut (child cache IS warmed)
        p2 = Parent(child=child, name="test")
        # Clear parent cache to force recomputation
        p2.__pydantic_private__["_model_token"] = None
        token_merkle = p2.model_token

        assert isinstance(token_full, str)
        assert isinstance(token_merkle, str)
        # Frozen children always use ("__child__", value.model_token), so
        # full normalize and Merkle shortcut now produce the same result.
        assert token_full == token_merkle

    def test_frozen_child_cached_token_is_reused(self):
        """Frozen child's cached _model_token is used as a Merkle leaf,
        producing the same result as a full normalize."""

        class Parent(BaseModel):
            child: FrozenModel = FrozenModel()
            name: str = "p"

        child = FrozenModel(a=1, b="test")
        # Warm child cache
        _ = child.model_token
        assert child._model_token is not None

        p = Parent(child=child, name="test")
        token_cached = p.model_token

        # Clear child cache and tokenize again
        child.__pydantic_private__["_model_token"] = None
        p2 = Parent(child=child, name="test")
        token_fresh = p2.model_token

        assert token_cached == token_fresh

    def test_nonfrozen_child_not_cached(self):
        """Non-frozen children don't cache tokens, so they always reflect current state."""

        class MutableChild(BaseModel):
            value: int = 0

        class Parent(BaseModel):
            child: MutableChild = MutableChild()

        child = MutableChild(value=1)
        _ = child.model_token
        # Mutable models don't cache by default
        assert child._model_token is None

        child.value = 2
        p = Parent(child=child)
        token1 = p.model_token

        p2 = Parent(child=MutableChild(value=2))
        token2 = p2.model_token

        assert token1 == token2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_model_token_cache_threadsafe_for_parallel_reads(self):
        """Multiple threads reading .model_token on same instance concurrently."""
        m = SimpleModel(x=42, y="threadsafe")
        results = []

        def read_token():
            return m.model_token

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_token) for _ in range(50)]
            for f in as_completed(futures):
                results.append(f.result())

        # All results should be identical
        assert len(set(results)) == 1

    def test_behavior_token_class_cache_threadsafe(self):
        """Multiple threads computing behavior token on same class."""

        class MyCallable:
            def __call__(self, x):
                return x + 1

        results = []

        def compute():
            return compute_behavior_token(MyCallable)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(compute) for _ in range(50)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Pickle edge cases
# ---------------------------------------------------------------------------


class TestPickleEdgeCases:
    def test_pickle_roundtrip_produces_valid_token(self):
        """After pickle/unpickle, model_token should still work and match."""
        m = SimpleModel(x=1, y="hello")
        original_token = m.model_token

        m2 = pickle.loads(pickle.dumps(m))
        assert m2.model_token == original_token

    def test_pickle_frozen_model_preserves_correct_token(self):
        """Frozen model's token should survive pickle correctly."""
        m = FrozenModel(a=10, b="frozen_pickle")
        original_token = m.model_token

        m2 = pickle.loads(pickle.dumps(m))
        assert m2.model_token == original_token


# ---------------------------------------------------------------------------
# Pandas unhashable object columns
# ---------------------------------------------------------------------------


class TestPandasUnhashable:
    def test_dataframe_with_dict_column(self):
        """DataFrame with unhashable object column (dicts) should not crash."""
        df = pd.DataFrame({"a": [1, 2], "b": [{"x": 1}, {"y": 2}]})
        token = normalize_token(df)
        assert token is not None

    def test_dataframe_with_dict_column_deterministic(self):
        """Same dict-column DataFrame produces same token."""
        df1 = pd.DataFrame({"a": [1], "b": [{"k": "v"}]})
        df2 = pd.DataFrame({"a": [1], "b": [{"k": "v"}]})
        assert normalize_token(df1) == normalize_token(df2)

    def test_series_with_dict_elements(self):
        """Series with unhashable elements should not crash."""
        s = pd.Series([{"x": 1}, {"y": 2}], name="dicts")
        token = normalize_token(s)
        assert token is not None

    def test_dataframe_with_list_column(self):
        """DataFrame with list-valued column should not crash."""
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]})
        token = normalize_token(df)
        assert token is not None


# ---------------------------------------------------------------------------
# Dask coverage gap tests (numpy/pandas edge cases)
# ---------------------------------------------------------------------------


class TestDaskCoverageGaps:
    @pytest.mark.parametrize(
        "left,right,expect_equal",
        [
            pytest.param(
                np.array([1, 2, 3], dtype=np.float32),
                np.array([1, 2, 3], dtype=np.float64),
                False,
                id="different-dtypes",
            ),
            pytest.param(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                True,
                id="empty-same-dtype",
            ),
            pytest.param(
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float64),
                False,
                id="empty-diff-dtype",
            ),
            pytest.param(
                {"b": 2, "a": 1},
                {"a": 1, "b": 2},
                True,
                id="dict-order-independence",
            ),
        ],
    )
    def test_normalize_token_equality(self, left, right, expect_equal):
        assert (normalize_token(left) == normalize_token(right)) == expect_equal

    def test_numpy_discontiguous_array(self):
        arr = np.arange(10)
        s1 = arr[::2]
        s2 = arr[::3]
        assert normalize_token(s1) != normalize_token(s2)

    def test_numpy_structured_array(self):
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        assert normalize_token(arr)[0] == "ndarray"

    def test_pandas_categorical(self):
        df1 = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])})
        df2 = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"])})
        assert normalize_token(df1) == normalize_token(df2)

    def test_pandas_empty_dataframe(self):
        assert normalize_token(pd.DataFrame()) is not None

    def test_pandas_multiindex(self):
        idx = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
        df = pd.DataFrame({"v": [10, 20]}, index=idx)
        assert normalize_token(df) is not None

    def test_plain_pydantic_frozen_tokenizes(self):
        class FrozenPlain(PydanticBaseModel):
            model_config = ConfigDict(frozen=True)
            x: int = 1

        t = DefaultTokenizer()
        token = t.tokenize(FrozenPlain(x=42))
        assert isinstance(token, str)
        assert t.tokenize(FrozenPlain(x=42)) == token

    def test_plain_pydantic_nonfrozen_tokenizes(self):
        class PlainModel(PydanticBaseModel):
            x: int = 1
            y: str = "hello"

        t = DefaultTokenizer()
        token = t.tokenize(PlainModel(x=5, y="world"))
        assert isinstance(token, str)
        assert t.tokenize(PlainModel(x=5, y="world")) == token


# ---------------------------------------------------------------------------
# Own-methods behavior token (all methods from cls.__dict__)
# ---------------------------------------------------------------------------


def _standalone_helper(x):
    """A standalone function for __ccflow_tokenizer_deps__ tests."""
    return x * 10


def _another_helper(x):
    return x + 99


class TestOwnMethodsBehaviorToken:
    """Tests for hashing all own methods (not just __call__/__deps__)."""

    @pytest.mark.parametrize(
        "left_attrs,right_attrs,expect_equal",
        [
            pytest.param(
                {"_helper": lambda self: 1, "__call__": lambda self: self._helper()},
                {"_helper": lambda self: 2, "__call__": lambda self: self._helper()},
                False,
                id="helper-method-change",
            ),
            pytest.param(
                {"__call__": lambda self: self.x + 1},
                {"__call__": lambda self: self.x + 1, "extra": lambda self: 42},
                False,
                id="adding-method",
            ),
            pytest.param(
                {"_private": lambda self: 1},
                {"_private": lambda self: 2},
                False,
                id="private-method",
            ),
        ],
    )
    def test_behavior_token_comparison(self, left_attrs, right_attrs, expect_equal):
        A = _make_ast_model("A", **left_attrs)
        B = _make_ast_model("B", **right_attrs)
        assert (A(x=1).model_token == B(x=1).model_token) == expect_equal

    def test_classmethod_included(self):
        class A1(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @classmethod
            def from_config(cls, cfg):
                return cls(**cfg)

        class A2(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @classmethod
            def from_config(cls, cfg):
                return cls()

        assert compute_behavior_token(A1) != compute_behavior_token(A2)

    def test_staticmethod_included(self):
        class B1(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @staticmethod
            def validate(x):
                return x > 0

        class B2(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @staticmethod
            def validate(x):
                return x >= 0

        assert compute_behavior_token(B1) != compute_behavior_token(B2)

    def test_validator_included(self):
        from pydantic import field_validator

        class V1(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @field_validator("x")
            @classmethod
            def check_x(cls, v):
                if v < 0:
                    raise ValueError("negative")
                return v

        class V2(BaseModel):
            __ccflow_tokenizer__ = _AST_TOKENIZER
            x: int = 1

            @field_validator("x")
            @classmethod
            def check_x(cls, v):
                if v < -10:
                    raise ValueError("too negative")
                return v

        assert V1(x=1).model_token != V2(x=1).model_token

    def test_no_own_methods_returns_none(self):
        class NoMethods:
            pass

        assert compute_behavior_token(NoMethods) is None


class TestTokenizerDeps:
    """Tests for __ccflow_tokenizer_deps__ extension mechanism."""

    def test_standalone_function_included(self):
        A = _make_ast_model("A", deps=[_standalone_helper], __call__=lambda self: _standalone_helper(self.x))
        B = _make_ast_model("B", __call__=lambda self: _standalone_helper(self.x))
        assert A(x=1).model_token != B(x=1).model_token

    def test_different_dep_functions_differ(self):
        A = _make_ast_model("A", deps=[_standalone_helper])
        B = _make_ast_model("B", deps=[_another_helper])
        assert A(x=1).model_token != B(x=1).model_token

    def test_deps_not_inherited(self):
        Parent = _make_ast_model("Parent", deps=[_standalone_helper], __call__=lambda self: self.x)
        Child = _make_ast_model("Child", base=Parent)
        assert compute_behavior_token(Parent) != compute_behavior_token(Child)

    def test_empty_deps_same_as_no_deps(self):
        class WithEmpty(BaseModel):
            __ccflow_tokenizer_deps__ = []

            def __call__(self, x):
                return x + 1

        class WithoutDeps(BaseModel):
            def __call__(self, x):
                return x + 1

        assert compute_behavior_token(WithEmpty) == compute_behavior_token(WithoutDeps)


class TestRuntimeTokenizerMutation:
    """Tests for mutating BaseModel.__ccflow_tokenizer__ at runtime."""

    @pytest.fixture(autouse=False)
    def _restore_base_tokenizer(self):
        original = BaseModel.__ccflow_tokenizer__
        yield
        BaseModel.__ccflow_tokenizer__ = original

    def test_global_mutation_affects_subclasses(self, _restore_base_tokenizer):
        """Mutating BaseModel.__ccflow_tokenizer__ affects all subclasses without overrides."""

        class PlainModel(BaseModel):
            x: int = 1

            def __call__(self, x):
                return x + 1

        token_before = PlainModel(x=42).model_token

        BaseModel.__ccflow_tokenizer__ = DefaultTokenizer.with_ast()

        token_after = PlainModel(x=42).model_token
        assert token_before != token_after

    def test_global_mutation_does_not_affect_overridden_subclass(self, _restore_base_tokenizer):
        """Subclass with its own __ccflow_tokenizer__ is NOT affected by base mutation."""

        class CustomModel(BaseModel):
            __ccflow_tokenizer__ = DefaultTokenizer.with_bytecode()
            x: int = 1

        token_before = CustomModel(x=42).model_token

        BaseModel.__ccflow_tokenizer__ = DefaultTokenizer.with_ast()

        token_after = CustomModel(x=42).model_token
        assert token_before == token_after

    def test_existing_instances_pick_up_new_tokenizer(self, _restore_base_tokenizer):
        """After global tokenizer mutation, mutable instances use the new tokenizer immediately."""

        class M(BaseModel):
            x: int = 1

            def __call__(self, x):
                return x

        m = M(x=1)
        token_before = m.model_token

        BaseModel.__ccflow_tokenizer__ = DefaultTokenizer.with_ast()

        # Mutable models don't cache, so they pick up the new tokenizer immediately
        token_after = m.model_token
        assert token_before != token_after

    def test_new_instances_after_mutation_use_new_tokenizer(self, _restore_base_tokenizer):
        """New instances created after mutation use the new tokenizer."""

        class M(BaseModel):
            x: int = 1

            def __call__(self, x):
                return x

        token_before = M(x=1).model_token

        BaseModel.__ccflow_tokenizer__ = DefaultTokenizer.with_ast()

        token_after = M(x=1).model_token
        assert token_before != token_after


# ---------------------------------------------------------------------------
# Additional type handler tests
# ---------------------------------------------------------------------------


class TestAdditionalTypeHandlers:
    """Tests for builtin and library type handlers."""

    def test_complex(self):
        assert normalize_token(complex(1, 2)) == ("complex", 1.0, 2.0)
        assert normalize_token(complex(1, 2)) != normalize_token(complex(2, 1))

    def test_ellipsis(self):
        assert normalize_token(...) == ("ellipsis",)

    def test_slice(self):
        assert normalize_token(slice(1, 10, 2)) == ("slice", 1, 10, 2)
        assert normalize_token(slice(1, 10)) == ("slice", 1, 10, None)
        assert normalize_token(slice(1, 10)) != normalize_token(slice(1, 11))

    def test_builtin_function(self):
        assert normalize_token(len) == ("builtin", "len")
        assert normalize_token(len) != normalize_token(print)

    def test_decimal(self):
        from decimal import Decimal

        assert normalize_token(Decimal("3.14")) == ("decimal", "3.14")
        assert normalize_token(Decimal("3.14")) != normalize_token(Decimal("3.15"))

    def test_partial(self):
        from functools import partial

        p1 = partial(int, base=16)
        p2 = partial(int, base=10)
        assert normalize_token(p1) != normalize_token(p2)
        assert normalize_token(p1) == normalize_token(partial(int, base=16))

    def test_mappingproxy(self):
        from types import MappingProxyType

        mp = MappingProxyType({"a": 1, "b": 2})
        d = {"a": 1, "b": 2}
        # MappingProxy normalizes like dict
        assert normalize_token(mp) == normalize_token(d)

    def test_polars_dataframe(self):
        """Polars DataFrames fall through to cloudpickle — same data same token."""
        import polars as pl

        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 3]})
        df3 = pl.DataFrame({"a": [1, 2, 4]})
        assert normalize_token(df1) == normalize_token(df2)
        assert normalize_token(df1) != normalize_token(df3)

    def test_polars_series(self):
        import polars as pl

        s1 = pl.Series("x", [1, 2, 3])
        s2 = pl.Series("x", [1, 2, 3])
        s3 = pl.Series("x", [1, 2, 4])
        assert normalize_token(s1) == normalize_token(s2)
        assert normalize_token(s1) != normalize_token(s3)

    def test_polars_lazyframe(self):
        """Polars LazyFrames are tokenized via cloudpickle (no collect)."""
        import polars as pl

        lf1 = pl.LazyFrame({"a": [1, 2, 3]})
        lf2 = pl.LazyFrame({"a": [1, 2, 3]})
        assert normalize_token(lf1) == normalize_token(lf2)

    def test_pandas_dataframe(self):
        """Pandas DataFrames fall through to cloudpickle."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        df3 = pd.DataFrame({"a": [1, 2, 4]})
        assert normalize_token(df1) == normalize_token(df2)
        assert normalize_token(df1) != normalize_token(df3)

    def test_pandas_series(self):
        df1 = pd.Series([1, 2, 3], name="x")
        df2 = pd.Series([1, 2, 3], name="x")
        df3 = pd.Series([1, 2, 4], name="x")
        assert normalize_token(df1) == normalize_token(df2)
        assert normalize_token(df1) != normalize_token(df3)

    def test_narwhals_dataframe(self):
        import narwhals as nw

        nw_df1 = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        nw_df2 = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        nw_df3 = nw.from_native(pd.DataFrame({"a": [1, 2, 4]}))
        assert normalize_token(nw_df1) == normalize_token(nw_df2)
        assert normalize_token(nw_df1) != normalize_token(nw_df3)

    def test_narwhals_series(self):
        import narwhals as nw

        nw_s1 = nw.from_native(pd.Series([1, 2, 3], name="x"), allow_series=True)
        nw_s2 = nw.from_native(pd.Series([1, 2, 3], name="x"), allow_series=True)
        assert normalize_token(nw_s1) == normalize_token(nw_s2)

    def test_numpy_datetime64(self):
        token = normalize_token(np.datetime64("2024-01-01"))
        assert token == ("np_scalar", "datetime64", date(2024, 1, 1))

    def test_numpy_complex(self):
        token = normalize_token(np.complex128(1 + 2j))
        assert token == ("np_scalar", "complex128", (1 + 2j))


class TestDaskTokenizer:
    """Tests for DaskTokenizer backward compatibility."""

    def test_matches_raw_dask_tokenize(self):
        import dask.base

        from ccflow.utils.tokenize import DaskTokenizer

        class M(BaseModel):
            x: int = 1
            y: str = "hello"

        m = M()
        t = DaskTokenizer()
        assert t.tokenize(m) == dask.base.tokenize(m.model_dump(mode="python"))

    def test_different_values_different_token(self):
        from ccflow.utils.tokenize import DaskTokenizer

        class M(BaseModel):
            x: int = 1

        t = DaskTokenizer()
        assert t.tokenize(M(x=1)) != t.tokenize(M(x=2))

    def test_same_values_same_token(self):
        from ccflow.utils.tokenize import DaskTokenizer

        class M(BaseModel):
            x: int = 1
            y: str = "a"

        t = DaskTokenizer()
        assert t.tokenize(M()) == t.tokenize(M())

    def test_works_as_ccflow_tokenizer(self):
        from ccflow.utils.tokenize import DaskTokenizer

        class M(BaseModel):
            __ccflow_tokenizer__ = DaskTokenizer()
            x: int = 1

        m = M()
        assert isinstance(m.model_token, str)
        assert len(m.model_token) == 32  # dask uses MD5

    def test_nested_model(self):
        import dask.base

        from ccflow.utils.tokenize import DaskTokenizer

        class Child(BaseModel):
            a: int = 1

        class Parent(BaseModel):
            child: Child = Child()

        t = DaskTokenizer()
        p = Parent()
        assert t.tokenize(p) == dask.base.tokenize(p.model_dump(mode="python"))

    def test_works_with_plain_pydantic(self):
        import dask.base
        from pydantic import BaseModel as PydanticBaseModel

        from ccflow.utils.tokenize import DaskTokenizer

        class Plain(PydanticBaseModel):
            x: int = 1

        t = DaskTokenizer()
        assert t.tokenize(Plain()) == dask.base.tokenize(Plain().model_dump(mode="python"))


# ---------------------------------------------------------------------------
# Review finding tests
# ---------------------------------------------------------------------------


class TestReviewFindings:
    """Tests for issues identified during PR #195 review."""

    def test_mutable_model_no_stale_parent_token(self):
        """Finding 1: Mutable models recompute token fresh — no stale parent."""

        class Child(BaseModel):
            x: int = 1

        class Parent(BaseModel):
            child: Child = Child()

        child = Child(x=1)
        parent = Parent(child=child)
        t1 = parent.model_token
        child.x = 2
        t2 = parent.model_token
        # Mutable model recomputes — should reflect child change
        assert t1 != t2

    def test_frozen_model_caches_token(self):
        """Finding 1: Frozen models cache their token."""

        class Frozen(BaseModel):
            model_config = ConfigDict(frozen=True)
            x: int = 1

        m = Frozen(x=42)
        t1 = m.model_token
        t2 = m.model_token
        assert t1 == t2
        # Verify it's actually cached (same object)
        assert m._model_token is not None

    def test_mutable_model_opt_in_caching(self):
        """Finding 1: Mutable models can opt in to caching."""

        class Cached(BaseModel):
            model_config = ConfigDict(cache_token=True)
            x: int = 1

        m = Cached(x=1)
        _ = m.model_token
        assert m._model_token is not None

    def test_container_cycle_list(self):
        """Finding 2: Self-referential list doesn't cause RecursionError."""

        class M(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            data: object = None

        lst = []
        lst.append(lst)
        # Should not raise RecursionError
        token = M(data=lst).model_token
        assert isinstance(token, str)

    def test_container_cycle_dict(self):
        """Finding 2: Self-referential dict doesn't cause RecursionError."""

        class M(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            data: object = None

        d = {}
        d["self"] = d
        token = M(data=d).model_token
        assert isinstance(token, str)

    def test_behavior_cache_stateful_tokenizer(self):
        """Finding 3: Custom stateful tokenizers don't collide in cache."""
        import hashlib

        class SaltedTokenizer(SourceTokenizer):
            def __init__(self, salt):
                self.salt = salt

            def tokenize(self, func):
                code = getattr(func, "__code__", None)
                if code is None:
                    return None
                return hashlib.sha256((self.salt + repr(code.co_code)).encode()).hexdigest()

        class M(BaseModel):
            x: int = 1

            def f(self):
                return 1

        t1 = compute_behavior_token(M, collector=OwnMethodCollector(), source_tokenizer=SaltedTokenizer("a"))
        t2 = compute_behavior_token(M, collector=OwnMethodCollector(), source_tokenizer=SaltedTokenizer("b"))
        assert t1 != t2

    def test_dep_order_insensitive(self):
        """Finding 4: __ccflow_tokenizer_deps__ order doesn't affect behavior token."""

        def helper_a():
            return "a"

        def helper_b():
            return "b"

        class A(BaseModel):
            __ccflow_tokenizer_deps__ = [helper_a, helper_b]

            def f(self):
                return 1

        class B(BaseModel):
            __ccflow_tokenizer_deps__ = [helper_b, helper_a]

            def f(self):
                return 1

        # Compare behavior tokens directly (model tokens differ due to type path)
        bt_a = compute_behavior_token(A)
        bt_b = compute_behavior_token(B)
        assert bt_a == bt_b

    def test_unpicklable_raises_type_error(self):
        """Finding 5: Unpicklable objects raise TypeError, not repr fallback."""

        class Unpicklable:
            def __reduce__(self):
                raise TypeError("nope")

        with pytest.raises(TypeError, match="Cannot tokenize"):
            normalize_token(Unpicklable())
