"""Tests for tokenize helpers used by cache_key()."""

from ccflow.callable import CallableModel, ContextBase, EvaluatorBase, ModelEvaluationContext
from ccflow.context import NullContext
from ccflow.evaluators.common import cache_key
from ccflow.result import GenericResult
from ccflow.utils.tokenize import compute_behavior_token, compute_data_token

# ---------------------------------------------------------------------------
# Data token
# ---------------------------------------------------------------------------


class TestComputeDataToken:
    def test_deterministic(self):
        value = {"a": [1, 2], "b": ("x", 3)}

        assert compute_data_token(value) == compute_data_token(value)

    def test_different_values_different_tokens(self):
        assert compute_data_token({"x": 1}) != compute_data_token({"x": 2})


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


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
        assert M.__behavior_token_cache__ == token
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


# ---------------------------------------------------------------------------
# Method collection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dependencies (__ccflow_tokenizer_deps__)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Integration with cache_key()
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decorator unwrapping (Flow.call, etc.)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MRO / inherited methods
# ---------------------------------------------------------------------------


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
        """Parent and subclass don't share __behavior_token_cache__."""

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
