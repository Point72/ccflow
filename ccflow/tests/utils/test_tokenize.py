"""Tests for tokenize helpers used by cache keys."""

import enum as _enum
import re
from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel as _PlainPydantic

from ccflow.callable import CallableModel, ContextBase, EvaluatorBase, ModelEvaluationContext
from ccflow.context import NullContext
from ccflow.evaluators.common import cache_key
from ccflow.result import GenericResult
from ccflow.utils.tokenize import compute_behavior_token, compute_cache_token, compute_data_token, normalize_token, tokenize


class TestComputeDataToken:
    def test_deterministic(self):
        value = {"a": [1, 2], "b": ("x", 3)}

        assert compute_data_token(value) == compute_data_token(value)

    def test_different_values_different_tokens(self):
        assert compute_data_token({"x": 1}) != compute_data_token({"x": 2})


class TestComputeCacheToken:
    def test_deterministic(self):
        class Helper:
            def f(self):
                return 1

        token1 = compute_cache_token(data_values=[{"x": 1}], behavior_classes=[Helper])
        token2 = compute_cache_token(data_values=[{"x": 1}], behavior_classes=[Helper])
        assert token1 == token2

    def test_data_changes_token(self):
        class Helper:
            def f(self):
                return 1

        token1 = compute_cache_token(data_values=[{"x": 1}], behavior_classes=[Helper])
        token2 = compute_cache_token(data_values=[{"x": 2}], behavior_classes=[Helper])
        assert token1 != token2

    def test_behavior_changes_token(self):
        class HelperA:
            def f(self):
                return 1

        class HelperB:
            def f(self):
                return 2

        token1 = compute_cache_token(data_values=[{"x": 1}], behavior_classes=[HelperA])
        token2 = compute_cache_token(data_values=[{"x": 1}], behavior_classes=[HelperB])
        assert token1 != token2


class TestComputeBehaviorToken:
    def test_returns_sha256_hex(self):
        class M:
            def f(self):
                return 1

        token = compute_behavior_token(M)
        assert isinstance(token, str)
        assert len(token) == 64

    def test_deterministic(self):
        class M:
            def f(self):
                return 1

        assert compute_behavior_token(M) == compute_behavior_token(M)

    def test_different_bytecode_different_token(self):
        class A:
            def f(self):
                return 1

        class B:
            def f(self):
                return 2

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_same_logic_same_token(self):
        class A:
            def f(self):
                return 42

        class B:
            def f(self):
                return 42

        assert compute_behavior_token(A) == compute_behavior_token(B)

    def test_none_for_no_methods(self):
        class Empty:
            x = 1

        assert compute_behavior_token(Empty) is None

    def test_cached_on_class(self):
        class M:
            def f(self):
                return 1

        token = compute_behavior_token(M)
        assert M.__ccflow_tokenizer_cache__ == token
        # Second call returns cached value
        assert compute_behavior_token(M) is token

    def test_docstring_ignored(self):
        class A:
            def f(self):
                """This is a docstring."""
                return 1

        class B:
            def f(self):
                """Different docstring."""
                return 1

        assert compute_behavior_token(A) == compute_behavior_token(B)

    def test_comments_ignored(self):
        """Comments don't exist in bytecode, so they're inherently ignored."""

        class A:
            def f(self):
                # a comment
                return 1

        class B:
            def f(self):
                # different comment
                return 1

        assert compute_behavior_token(A) == compute_behavior_token(B)

    def test_constants_matter(self):
        class A:
            def f(self):
                return "hello"

        class B:
            def f(self):
                return "world"

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_defaults_matter(self):
        class A:
            def f(self, x=1):
                return x

        class B:
            def f(self, x=2):
                return x

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_kwdefaults_matter(self):
        class A:
            def f(self, *, x=1):
                return x

        class B:
            def f(self, *, x=2):
                return x

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_closure_values_matter(self):
        def make_model(value):
            class M:
                def f(self):
                    return value

            return M

        assert compute_behavior_token(make_model(1)) != compute_behavior_token(make_model(2))


class TestMethodCollection:
    def test_includes_regular_methods(self):
        class M:
            def method_a(self):
                return 1

            def method_b(self):
                return 2

        token = compute_behavior_token(M)
        assert token is not None

    def test_includes_staticmethod(self):
        class A:
            @staticmethod
            def helper():
                return 1

        class B:
            @staticmethod
            def helper():
                return 2

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_includes_classmethod(self):
        class A:
            @classmethod
            def factory(cls):
                return cls()

        class B:
            @classmethod
            def factory(cls):
                return None

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_method_order_insensitive(self):
        """Alphabetical sorting means definition order doesn't matter."""

        class A:
            def beta(self):
                return 2

            def alpha(self):
                return 1

        class B:
            def alpha(self):
                return 1

            def beta(self):
                return 2

        assert compute_behavior_token(A) == compute_behavior_token(B)

    def test_skips_ccflow_internal_attrs(self):
        """Attributes starting with __ccflow_ are skipped."""

        class A:
            __ccflow_tokenizer_deps__ = []

            def f(self):
                return 1

        class B:
            def f(self):
                return 1

        assert compute_behavior_token(A) == compute_behavior_token(B)


def _helper_add(x):
    return x + 1


def _helper_mul(x):
    return x * 2


class TestDeps:
    def test_deps_included(self):
        class NoDeps:
            def f(self):
                return 1

        class WithDeps:
            __ccflow_tokenizer_deps__ = [_helper_add]

            def f(self):
                return 1

        assert compute_behavior_token(NoDeps) != compute_behavior_token(WithDeps)

    def test_dep_order_insensitive(self):
        class A:
            __ccflow_tokenizer_deps__ = [_helper_add, _helper_mul]

            def f(self):
                return 1

        class B:
            __ccflow_tokenizer_deps__ = [_helper_mul, _helper_add]

            def f(self):
                return 1

        assert compute_behavior_token(A) == compute_behavior_token(B)

    def test_dep_change_changes_token(self):
        def v1():
            return 1

        def v2():
            return 2

        class A:
            __ccflow_tokenizer_deps__ = [v1]

            def f(self):
                return 1

        class B:
            __ccflow_tokenizer_deps__ = [v2]

            def f(self):
                return 1

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_class_dep_included(self):
        class HelperA:
            def f(self):
                return 1

        class HelperB:
            def f(self):
                return 2

        class A:
            __ccflow_tokenizer_deps__ = [HelperA]

            def f(self):
                return 1

        class B:
            __ccflow_tokenizer_deps__ = [HelperB]

            def f(self):
                return 1

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_subclass_deps_extend_inherited_deps(self):
        def base_a():
            return 1

        def base_b():
            return 2

        def sub_dep():
            return 3

        class BaseA:
            __ccflow_tokenizer_deps__ = [base_a]

            def f(self):
                return 1

        class BaseB:
            __ccflow_tokenizer_deps__ = [base_b]

            def f(self):
                return 1

        class SubA(BaseA):
            __ccflow_tokenizer_deps__ = [sub_dep]

        class SubB(BaseB):
            __ccflow_tokenizer_deps__ = [sub_dep]

        assert compute_behavior_token(SubA) != compute_behavior_token(SubB)

    def test_recursive_class_deps_raise(self):
        class A:
            def f(self):
                return 1

        class B:
            def g(self):
                return 2

        A.__ccflow_tokenizer_deps__ = [B]
        B.__ccflow_tokenizer_deps__ = [A]

        with pytest.raises(TypeError, match="Recursive __ccflow_tokenizer_deps__ class dependency"):
            compute_behavior_token(A)


class TestCacheKeyIntegration:
    def test_callable_model_includes_behavior(self):
        """cache_key for a CallableModel includes the behavior hash."""
        from ccflow import Flow

        class MyModel(CallableModel):
            x: int = 1

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.x + 1)

        class MyModelV2(CallableModel):
            x: int = 1

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.x + 2)

        key1 = cache_key(MyModel(x=1))
        key2 = cache_key(MyModelV2(x=1))
        # Same config, different __call__ → different cache key
        assert key1 != key2

    def test_same_callable_same_key(self):
        from ccflow import Flow

        class MyModel(CallableModel):
            x: int = 1

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.x)

        key1 = cache_key(MyModel(x=1))
        key2 = cache_key(MyModel(x=1))
        assert key1 == key2

    def test_different_config_different_key(self):
        from ccflow import Flow

        class MyModel(CallableModel):
            x: int = 1

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.x)

        key1 = cache_key(MyModel(x=1))
        key2 = cache_key(MyModel(x=2))
        assert key1 != key2

    def test_context_base_no_behavior(self):
        """ContextBase with no custom methods has no behavior token — still works."""

        class MyContext(ContextBase):
            value: int = 1

        key = cache_key(MyContext(value=1))
        assert isinstance(key, bytes)

    def test_helper_default_arg_changes_key(self):
        from ccflow import Flow

        class A(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.helper())

            def helper(self, x=1):
                return x

        class B(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=self.helper())

            def helper(self, x=2):
                return x

        assert cache_key(A()) != cache_key(B())

    def test_class_dep_changes_key(self):
        from ccflow import Flow

        class HelperA:
            def f(self):
                return 1

        class HelperB:
            def f(self):
                return 2

        class A(CallableModel):
            __ccflow_tokenizer_deps__ = [HelperA]

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=1)

        class B(CallableModel):
            __ccflow_tokenizer_deps__ = [HelperB]

            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=1)

        assert cache_key(A()) != cache_key(B())

    def test_opaque_evaluator_behavior_changes_key(self):
        from ccflow import Flow

        class MyModel(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=1)

        class OpaqueA(EvaluatorBase):
            tag: str = "same"

            def __call__(self, context: ModelEvaluationContext):
                return context()

        class OpaqueB(EvaluatorBase):
            tag: str = "same"

            def __call__(self, context: ModelEvaluationContext):
                result = context()
                return result

        inner = ModelEvaluationContext(model=MyModel(), context=NullContext())
        key1 = cache_key(ModelEvaluationContext(model=OpaqueA(), context=inner))
        key2 = cache_key(ModelEvaluationContext(model=OpaqueB(), context=inner))
        assert key1 != key2


class TestDecoratorUnwrapping:
    def test_flow_call_different_impls_differ(self):
        """@Flow.call wrappers are unwrapped — different implementations hash differently."""
        from ccflow import Flow

        class A(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=1)

        class B(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=2)

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_flow_call_same_impl_same(self):
        """Same @Flow.call implementation produces the same token."""
        from ccflow import Flow

        class A(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=42)

        class B(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult:
                return GenericResult(value=42)

        assert compute_behavior_token(A) == compute_behavior_token(B)


class TestInheritedMethods:
    def test_inherited_call_included(self):
        """Subclass that inherits __call__ from parent picks up parent's method."""

        class Base:
            def __call__(self):
                return 1

        class Sub(Base):
            pass

        # Sub inherits __call__, so it should have a behavior token
        assert compute_behavior_token(Sub) is not None
        assert compute_behavior_token(Sub) == compute_behavior_token(Base)

    def test_override_changes_token(self):
        """Subclass overriding a method gets a different token."""

        class Base:
            def __call__(self):
                return 1

        class Sub(Base):
            def __call__(self):
                return 2

        assert compute_behavior_token(Sub) != compute_behavior_token(Base)

    def test_subclass_cache_independent(self):
        """Parent and subclass don't share __ccflow_tokenizer_cache__."""

        class Base:
            def f(self):
                return 1

        class Sub(Base):
            def g(self):
                return 2

        t_base = compute_behavior_token(Base)
        t_sub = compute_behavior_token(Sub)
        # Sub has additional method g, so tokens differ
        assert t_base != t_sub
        # But base's cached token is unaffected
        assert compute_behavior_token(Base) == t_base


class _Color(_enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class TestNormalizeTokenPrimitives:
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
            (timedelta(hours=1, minutes=30), ("timedelta", 0, 5400, 0)),
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
        assert normalize_token(Path("/tmp/test.txt")) == ("path", "/tmp/test.txt")

    def test_pure_path(self):
        assert normalize_token(PurePosixPath("/tmp/test.txt")) == ("path", "/tmp/test.txt")

    def test_enum(self):
        assert normalize_token(_Color.RED) == ("enum", _Color.__module__, "_Color", "RED")

    def test_nested_collections(self):
        data = {"key": [1, (2, 3)]}
        assert normalize_token(data) == ("dict", (("key", ("list", (1, ("tuple", (2, 3))))),))


class TestNormalizeTokenAdditionalBuiltins:
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
        # len is a builtin with __self__ = builtins module
        assert normalize_token(len) == ("builtin", "builtins", "len", ("module", "builtins"))
        assert normalize_token(len) != normalize_token(print)

    def test_decimal(self):
        assert normalize_token(Decimal("3.14")) == ("decimal", "3.14")
        assert normalize_token(Decimal("3.14")) != normalize_token(Decimal("3.15"))

    def test_partial(self):
        p1 = partial(int, base=16)
        p2 = partial(int, base=10)
        assert normalize_token(p1) != normalize_token(p2)
        assert normalize_token(p1) == normalize_token(partial(int, base=16))

    def test_mappingproxy(self):
        mp = MappingProxyType({"a": 1, "b": 2})
        # MappingProxy is tagged distinctly from dict so a proxy is never confused with the same-keyed dict
        assert normalize_token(mp) == ("mappingproxy", (("a", 1), ("b", 2)))
        assert normalize_token(mp) != normalize_token({"a": 1, "b": 2})


class TestNormalizeTokenFunctionsAndTypes:
    def test_function(self):
        def my_func(x):
            return x + 1

        result = normalize_token(my_func)
        assert result[0] == "__function__"
        assert isinstance(result[1], str)
        assert len(result[1]) == 64

    def test_function_deterministic(self):
        def my_func(x):
            return x + 1

        assert normalize_token(my_func) == normalize_token(my_func)

    def test_type(self):
        assert normalize_token(int) == ("type", "builtins.int")

    def test_pydantic_basemodel(self):
        class PlainPydantic(_PlainPydantic):
            x: int = 1

        obj = PlainPydantic(x=5)
        result = normalize_token(obj)
        assert result[0] == "pydantic"
        assert "PlainPydantic" in result[1]
        # Same data → same canonical form
        assert normalize_token(PlainPydantic(x=5)) == normalize_token(PlainPydantic(x=5))
        assert normalize_token(PlainPydantic(x=5)) != normalize_token(PlainPydantic(x=6))


class TestNormalizeTokenNumpyPandas:
    def test_numpy_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = normalize_token(arr)
        assert result[0] == "ndarray"
        assert result[1] == "int64"
        assert result[2] == (3,)
        assert normalize_token(arr) == normalize_token(np.array([1, 2, 3], dtype=np.int64))

    def test_numpy_different_data(self):
        assert normalize_token(np.array([1, 2, 3])) != normalize_token(np.array([1, 2, 4]))

    def test_numpy_different_dtype(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float64)
        assert normalize_token(a) != normalize_token(b)

    def test_numpy_empty_same_dtype(self):
        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        assert normalize_token(a) == normalize_token(b)

    def test_numpy_empty_diff_dtype(self):
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float64)
        assert normalize_token(a) != normalize_token(b)

    def test_numpy_discontiguous_array(self):
        arr = np.arange(10)
        assert normalize_token(arr[::2]) != normalize_token(arr[::3])

    def test_numpy_structured_array(self):
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        assert normalize_token(arr)[0] == "ndarray"

    def test_numpy_scalar(self):
        s = np.int64(42)
        assert normalize_token(s) == ("np_scalar", "int64", 42)

    def test_numpy_datetime64(self):
        token = normalize_token(np.datetime64("2024-01-01"))
        assert token == ("np_scalar", "datetime64", date(2024, 1, 1))

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        assert normalize_token(ts) == ("pd_timestamp", ts.isoformat())

    def test_pandas_dataframe_via_cloudpickle(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        df3 = pd.DataFrame({"a": [1, 2, 4]})
        assert normalize_token(df1) == normalize_token(df2)
        assert normalize_token(df1) != normalize_token(df3)

    def test_pandas_series_via_cloudpickle(self):
        s1 = pd.Series([1, 2, 3], name="x")
        s2 = pd.Series([1, 2, 3], name="x")
        s3 = pd.Series([1, 2, 4], name="x")
        assert normalize_token(s1) == normalize_token(s2)
        assert normalize_token(s1) != normalize_token(s3)


class TestNormalizeTokenCloudpickleFallback:
    def test_compiled_regex_same_pattern_same_token(self):
        r1 = re.compile("abc", re.IGNORECASE)
        r2 = re.compile("abc", re.IGNORECASE)
        assert normalize_token(r1) == normalize_token(r2)

    def test_unpicklable_raises_typeerror(self):
        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("cannot pickle")

        with pytest.raises(TypeError, match="Cannot tokenize"):
            normalize_token(Unpicklable())

    def test_arbitrary_object_falls_back(self):
        class Foo:
            def __init__(self, x):
                self.x = x

        result = normalize_token(Foo(1))
        assert result[0] == "__cloudpickle__"
        assert isinstance(result[1], str)


class TestNormalizeTokenContainerEdgeCases:
    def test_dict_with_mixed_key_types(self):
        d = {1: "a", "1": "b"}
        result = normalize_token(d)
        assert result[0] == "dict"
        assert len(result[1]) == 2

    def test_dict_order_independence(self):
        assert normalize_token({"b": 2, "a": 1}) == normalize_token({"a": 1, "b": 2})

    def test_inf_and_negative_inf_distinct(self):
        assert normalize_token(float("inf")) != normalize_token(float("-inf"))


class TestTokenizePublicAPI:
    def test_tokenize_returns_hex_string(self):
        token = tokenize({"a": [1, 2, 3]})
        assert isinstance(token, str)
        assert len(token) == 64  # sha256 hex

    def test_tokenize_deterministic(self):
        assert tokenize({"a": 1, "b": 2}) == tokenize({"b": 2, "a": 1})

    def test_tokenize_distinguishes(self):
        assert tokenize({"a": 1}) != tokenize({"a": 2})


class TestNormalizeTokenDeterminismRegressions:
    _ADDR_RE = re.compile(r"0x[0-9a-fA-F]{4,}")

    def _assert_portable(self, canonical):
        r = repr(canonical)
        assert "<code object" not in r, r
        assert "at 0x" not in r, r
        assert not self._ADDR_RE.search(r), r

    def test_closure_captured_value_changes_token(self):
        def make(n):
            def inner(x):
                return x + n

            return inner

        assert normalize_token(make(1)) != normalize_token(make(2))
        assert tokenize(make(1)) != tokenize(make(2))

    def test_closure_same_captured_value_same_token(self):
        def make(n):
            def inner(x):
                return x + n

            return inner

        assert tokenize(make(7)) == tokenize(make(7))

    def test_function_with_comprehension_has_no_memory_addresses(self):
        def f():
            return [x * 2 for x in range(3)]

        self._assert_portable(normalize_token(f.__code__))

    def test_nested_def_has_no_memory_addresses(self):
        def outer():
            def inner(x):
                return x + 1

            return inner

        self._assert_portable(normalize_token(outer.__code__))

    def test_lambda_inside_function_has_no_memory_addresses(self):
        def f():
            return list(map(lambda x: x * 2, range(3)))

        self._assert_portable(normalize_token(f.__code__))

    def test_genexpr_has_no_memory_addresses(self):
        def f():
            return sum(x for x in range(3))

        self._assert_portable(normalize_token(f.__code__))

    def test_function_canonical_form_is_portable(self):
        def f():
            return [x * 2 for x in range(3)]

        self._assert_portable(normalize_token(f))

    def test_function_with_comprehension_in_process_stable(self):
        # End-to-end smoke test: same source → same digest within one process.
        # Cross-process stability follows from the canonical form containing
        # no memory addresses (asserted above).
        def f():
            return [x * 2 for x in range(3)]

        assert tokenize(f) == tokenize(f)

    def test_object_dtype_ndarray_equal_content_equal_token(self):
        a = np.array(["a", "b", "c"], dtype=object)
        b = np.array(["a", "b", "c"], dtype=object)
        assert normalize_token(a) == normalize_token(b)

    def test_object_dtype_ndarray_different_content_different_token(self):
        a = np.array(["a", "b", "c"], dtype=object)
        b = np.array(["a", "b", "d"], dtype=object)
        assert normalize_token(a) != normalize_token(b)

    def test_object_dtype_ndarray_routes_through_tolist(self):
        a = np.array(["a", "b", "c"], dtype=object)
        # Object-dtype arrays must recurse via normalize_token(tolist()) rather
        # than tobytes(), which would embed PyObject* pointers.
        assert normalize_token(a) == ("ndarray", str(a.dtype), a.shape, normalize_token(a.tolist()))

    def test_object_dtype_ndarray_canonical_form_is_portable(self):
        a = np.array(["a", "b", "c"], dtype=object)
        r = repr(normalize_token(a))
        assert "0x" not in r or not re.search(r"0x[0-9a-fA-F]{4,}", r), r

    def test_object_dtype_ndarray_with_dict_elements(self):
        a = np.array([{"k": 1}, {"k": 2}], dtype=object)
        b = np.array([{"k": 1}, {"k": 2}], dtype=object)
        assert normalize_token(a) == normalize_token(b)


class _Stateful:
    def __init__(self, x):
        self.x = x

    def m(self):
        pass

    @classmethod
    def cm(cls):
        pass


class TestBoundMethods:
    _ADDR_RE = re.compile(r"0x[0-9a-fA-F]{4,}")

    def test_bound_method_distinct_instances_differ(self):
        a, b = _Stateful(1), _Stateful(2)
        assert tokenize(a.m) != tokenize(b.m)

    def test_bound_method_same_instance_state_matches(self):
        assert tokenize(_Stateful(1).m) == tokenize(_Stateful(1).m)

    def test_classmethod_via_class_eq_via_instance(self):
        assert tokenize(_Stateful.cm) == tokenize(_Stateful(99).cm)

    def test_bound_method_canonical_form_is_portable(self):
        a = _Stateful(1)
        r = repr(normalize_token(a.m))
        assert "<code object" not in r
        assert not self._ADDR_RE.search(r), r

    def test_classmethod_canonical_form_is_portable(self):
        r = repr(normalize_token(_Stateful.cm))
        assert not self._ADDR_RE.search(r), r

    def test_method_wrapper_tokenizes(self):
        canonical = normalize_token({}.__init__)
        assert canonical[0] == "__method_wrapper__"
        assert canonical[1] == "__init__"


class TestOrderedDict:
    def test_different_order_different_token(self):
        od1 = OrderedDict([("a", 1), ("b", 2)])
        od2 = OrderedDict([("b", 2), ("a", 1)])
        assert tokenize(od1) != tokenize(od2)

    def test_same_order_same_token(self):
        od1 = OrderedDict([("a", 1), ("b", 2)])
        od2 = OrderedDict([("a", 1), ("b", 2)])
        assert tokenize(od1) == tokenize(od2)

    def test_ordereddict_distinct_from_plain_dict(self):
        od = OrderedDict([("a", 1), ("b", 2)])
        d = {"a": 1, "b": 2}
        assert tokenize(od) != tokenize(d)

    def test_canonical_form_preserves_insertion_order(self):
        od = OrderedDict([("z", 1), ("a", 2)])
        canonical = normalize_token(od)
        assert canonical[0] == "__ordereddict__"
        assert canonical[1] == (("z", 1), ("a", 2))


class TestTokenizeVariadic:
    def test_multiple_args_distinguish_from_tuple(self):
        assert tokenize(1, 2) != tokenize((1, 2))

    def test_kwargs_distinguish_from_positional(self):
        assert tokenize(x=1) != tokenize(1)

    def test_kwargs_order_independent(self):
        assert tokenize(a=1, b=2) == tokenize(b=2, a=1)

    def test_args_order_matters(self):
        assert tokenize(1, 2) != tokenize(2, 1)

    def test_single_arg_returns_hex(self):
        token = tokenize(42)
        assert isinstance(token, str)
        assert len(token) == 64

    def test_zero_args_is_deterministic(self):
        assert tokenize() == tokenize()


class TestCycleDetection:
    def test_self_referential_dict(self):
        d: dict = {}
        d["self"] = d
        token = tokenize(d)
        assert isinstance(token, str)
        assert len(token) == 64

    def test_self_referential_list(self):
        lst: list = []
        lst.append(lst)
        token = tokenize(lst)
        assert isinstance(token, str)

    def test_indirect_cycle_dict_list(self):
        d: dict = {}
        lst: list = [d]
        d["x"] = lst
        assert isinstance(tokenize(d), str)

    def test_self_referential_pydantic(self):
        from typing import Any

        from pydantic import ConfigDict

        from ccflow import BaseModel as CcflowBaseModel

        class Node(CcflowBaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            ref: Any = None

        n = Node()
        n.ref = n
        assert isinstance(tokenize(n), str)

    def test_shared_subobject_not_treated_as_cycle(self):
        sub = [1, 2, 3]
        parent = [sub, sub]
        canonical = normalize_token(parent)
        # Both list elements should produce the same non-cycle canonical form
        assert canonical[0] == "list"
        first, second = canonical[1]
        assert first == second
        assert first[0] == "list"
        assert "__cycle__" not in repr(canonical)

    def test_shared_dict_sibling_not_treated_as_cycle(self):
        sub = {"k": "v"}
        parent = {"a": sub, "b": sub}
        canonical = normalize_token(parent)
        assert "__cycle__" not in repr(canonical)

    def test_self_referential_object_dtype_ndarray(self):
        a = np.empty(1, dtype=object)
        a[0] = a
        # Must not infinite-loop; tolist() returns a fresh list each call.
        token = tokenize(a)
        assert isinstance(token, str)

    def test_cycle_canonical_form_marker(self):
        d: dict = {}
        d["self"] = d
        canonical = normalize_token(d)
        assert "__cycle__" in repr(canonical)

    def test_visited_set_isolated_between_calls(self):
        # After a cycle in one call, a subsequent unrelated call must not
        # see ghost ids from the prior visited set.
        d: dict = {}
        d["self"] = d
        tokenize(d)
        # Fresh value with same id() reuse pattern should tokenize normally
        normal = {"a": 1}
        assert "__cycle__" not in repr(normalize_token(normal))


class TestAdversarialFixes:
    """Regression tests for fixes from a follow-up adversarial review of the native engine."""

    def test_code_includes_attribute_names(self):
        # `self.x` vs `self.y` differ only in co_names; without including co_names the two classes'
        # methods would hash identically.
        class A:
            def m(self):
                return self.x

        class B:
            def m(self):
                return self.y

        assert compute_behavior_token(A) != compute_behavior_token(B)

    def test_code_includes_local_variable_names(self):
        def f1():
            apples = 1
            return apples

        def f2():
            oranges = 1
            return oranges

        assert tokenize(f1) != tokenize(f2)

    def test_code_includes_signature_arity(self):
        def one_pos(a, /, b):
            return a + b

        def two_pos(a, b, /):
            return a + b

        # Same body, same names — only co_posonlyargcount differs.
        assert tokenize(one_pos) != tokenize(two_pos)

    def test_builtin_distinguishes_module(self):
        import cmath
        import math

        assert tokenize(math.sin) != tokenize(cmath.sin)

    def test_builtin_distinguishes_bound_instance(self):
        a, b = [1, 2, 3], [9, 9, 9]
        assert tokenize(a.append) != tokenize(b.append)
        # Same content list ⇒ same token (structural, not identity).
        assert tokenize([1, 2, 3].append) == tokenize([1, 2, 3].append)

    def test_pydantic_extras_change_token(self):
        from pydantic import BaseModel as PB, ConfigDict

        class M(PB):
            model_config = ConfigDict(extra="allow")
            x: int = 1

        m_no_extras = M(x=1)
        m_with_extras = M(x=1, extra_field="hello")
        assert tokenize(m_no_extras) != tokenize(m_with_extras)

    def test_pydantic_overridden_iter_does_not_hide_fields(self):
        from pydantic import BaseModel as PB

        class Hidden(PB):
            x: int = 1

            def __iter__(self):
                return iter(())

        # Even though __iter__ yields nothing, fields must still be tokenized via model_fields.
        assert tokenize(Hidden(x=1)) != tokenize(Hidden(x=2))

    def test_mappingproxy_preserves_iteration_order(self):
        mp1 = MappingProxyType(OrderedDict([("a", 1), ("b", 2)]))
        mp2 = MappingProxyType(OrderedDict([("b", 2), ("a", 1)]))
        assert tokenize(mp1) != tokenize(mp2)

    def test_masked_array_includes_mask(self):
        ma1 = np.ma.array([1, 2, 3], mask=[0, 0, 1])
        ma2 = np.ma.array([1, 2, 3], mask=[0, 1, 0])
        ma3 = np.ma.array([1, 2, 3], mask=[0, 0, 1])
        assert tokenize(ma1) != tokenize(ma2)
        assert tokenize(ma1) == tokenize(ma3)

    def test_masked_array_includes_fill_value(self):
        ma1 = np.ma.array([1, 2, 3], mask=[0, 0, 1], fill_value=99)
        ma2 = np.ma.array([1, 2, 3], mask=[0, 0, 1], fill_value=42)
        assert tokenize(ma1) != tokenize(ma2)

    def test_compile_first_const_not_stripped(self):
        # `_normalize_code` must NOT strip the first const, because for a `compile(..., "exec")`
        # block the first const is real program data, not a docstring slot.
        c1 = compile("x = 'foo'", "<test>", "exec")
        c2 = compile("x = 'bar'", "<test>", "exec")
        assert tokenize(c1) != tokenize(c2)

    def test_function_docstring_still_stripped(self):
        # Conversely, _hash_function_bytecode must still ignore docstrings on function bodies.
        def with_doc():
            """A docstring."""
            return 1

        def without_doc():
            return 1

        # Wrapping in classes lets compute_behavior_token surface the difference (or lack thereof).
        WithDoc = type("X", (object,), {"f": with_doc})
        WithoutDoc = type("X", (object,), {"f": without_doc})
        assert compute_behavior_token(WithDoc) == compute_behavior_token(WithoutDoc)

    def test_slice_recurses_on_bounds(self):
        class Marker:
            pass

        m = Marker()
        canon = normalize_token(slice(m, None, None))
        # The Marker should have been normalized via the type/cloudpickle dispatch — no raw repr leaks
        # like "<...Marker object at 0x...>" should appear.
        assert "object at 0x" not in repr(canon)

    def test_slice_primitive_bounds_stable(self):
        assert tokenize(slice(1, 2, 3)) == tokenize(slice(1, 2, 3))
        assert tokenize(slice(1, 2, 3)) != tokenize(slice(1, 2, 4))

    def test_timedelta_microsecond_precision(self):
        td1 = timedelta(days=10**8, microseconds=1)
        td2 = timedelta(days=10**8, microseconds=2)
        # total_seconds() loses microsecond precision at large day counts due to float rounding;
        # the (days, seconds, microseconds) decomposition keeps them distinct.
        assert tokenize(td1) != tokenize(td2)

    def test_enum_includes_module(self):
        E1 = _enum.Enum("Color", {"RED": 1}, module="pkg.one")
        E2 = _enum.Enum("Color", {"RED": 1}, module="pkg.two")
        assert tokenize(E1.RED) != tokenize(E2.RED)
