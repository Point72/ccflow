"""Dependency annotation markers for Flow.model.

This module provides:
- Dep: Annotation marker for dependency parameters that can accept CallableModel
- DepOf: Shorthand for Annotated[Union[T, CallableModel], Dep()]
"""

from typing import TYPE_CHECKING, Annotated, Callable, Optional, Type, TypeVar, Union, get_args, get_origin

from .base import ContextBase

if TYPE_CHECKING:
    from .callable import CallableModel

__all__ = ("Dep", "DepOf")

T = TypeVar("T")

# Lazy reference to CallableModel to avoid circular import
_CallableModel = None


def _get_callable_model():
    """Lazily import CallableModel to avoid circular imports."""
    global _CallableModel
    if _CallableModel is None:
        from .callable import CallableModel

        _CallableModel = CallableModel
    return _CallableModel


class _DepOfMeta(type):
    """Metaclass that makes DepOf[ContextType, ResultType] work."""

    def __getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "DepOf requires 2 type arguments: DepOf[ContextType, ResultType]. "
                "Use ... for ContextType to inherit from parent: DepOf[..., ResultType]"
            )
        context_type, result_type = item
        CallableModel = _get_callable_model()

        if context_type is ...:
            # DepOf[..., ResultType] - inherit context from parent
            return Annotated[Union[result_type, CallableModel], Dep()]
        else:
            # DepOf[ContextType, ResultType] - explicit context type
            return Annotated[Union[result_type, CallableModel], Dep(context_type=context_type)]


class DepOf(metaclass=_DepOfMeta):
    """
    Shorthand for Annotated[Union[ResultType, CallableModel], Dep(context_type=...)].

    Follows Callable convention: DepOf[InputContext, OutputResult]

    For class fields, accepts either:
    - The result type directly (pre-computed value)
    - A CallableModel that produces the result type (resolved at call time)

    Usage:
        # Inherit context type from parent model (most common)
        data: DepOf[..., GenericResult[dict]]

        # Explicit context type validation
        data: DepOf[DateRangeContext, GenericResult[dict]]

    At call time, if the field contains a CallableModel, it will be automatically
    resolved using __deps__ and the resolved value will be accessible via self.field_name.

    For dependencies with transforms, define them in __deps__:
        def __deps__(self, context):
            transformed_ctx = context.model_copy(update={...})
            return [(self.data, [transformed_ctx])]
    """

    pass


def _is_compatible_type(actual: Type, expected: Type) -> bool:
    """Check if actual type is compatible with expected type.

    Handles generic types like GenericResult[pl.DataFrame] where issubclass
    would raise TypeError.

    Args:
        actual: The actual type to check
        expected: The expected type to match against

    Returns:
        True if actual is compatible with expected
    """
    # Handle None/empty types
    if actual is None or expected is None:
        return actual is expected

    # Get origins for generic types
    actual_origin = get_origin(actual) or actual
    expected_origin = get_origin(expected) or expected

    # Check if origins are compatible
    try:
        if not (isinstance(actual_origin, type) and isinstance(expected_origin, type)):
            return False
        if not issubclass(actual_origin, expected_origin):
            return False
    except TypeError:
        # issubclass can fail for certain types
        return False

    # Check generic args if present
    actual_args = get_args(actual)
    expected_args = get_args(expected)

    if expected_args and actual_args:
        if len(actual_args) != len(expected_args):
            return False
        return all(_is_compatible_type(a, e) for a, e in zip(actual_args, expected_args))

    return True


class Dep:
    """
    Annotation marker for dependency parameters.

    Marks a parameter as accepting either the declared type or a CallableModel
    that produces that type. Supports optional context transform and
    construction-time type validation.

    Usage:
        # No transform, no explicit validation (uses parent's context_type)
        prices: Annotated[GenericResult[pl.DataFrame], Dep()]

        # With transform
        prices: Annotated[GenericResult[pl.DataFrame], Dep(
            transform=lambda ctx: ctx.model_copy(update={"start": ctx.start - timedelta(days=1)})
        )]

        # With explicit context_type validation
        prices: Annotated[GenericResult[pl.DataFrame], Dep(
            context_type=DateRangeContext,
            transform=lambda ctx: ctx.model_copy(update={"start": ctx.start - timedelta(days=1)})
        )]

        # Cross-context dependency (transform changes context type)
        sim_data: Annotated[GenericResult[pl.DataFrame], Dep(
            context_type=SimulationContext,
            transform=date_to_simulation_context
        )]
    """

    def __init__(
        self,
        transform: Optional[Callable[..., ContextBase]] = None,
        context_type: Optional[Type[ContextBase]] = None,
    ):
        """
        Args:
            transform: Optional function to transform context before calling dependency.
                       Signature: (context) -> transformed_context
            context_type: Expected context_type of the dependency CallableModel.
                          If None, defaults to the parent model's context_type.
                          Validated at construction time when a CallableModel is passed.
        """
        self.transform = transform
        self.context_type = context_type

    def apply(self, context: ContextBase) -> ContextBase:
        """Apply the transform to a context, or return unchanged if no transform."""
        if self.transform is not None:
            return self.transform(context)
        return context

    def validate_dependency(
        self,
        value: "CallableModel",  # noqa: F821
        expected_result_type: Type,
        parent_context_type: Type[ContextBase],
        param_name: str,
    ) -> None:
        """
        Validate a CallableModel dependency at construction time.

        Args:
            value: The CallableModel being passed as a dependency
            expected_result_type: The result type from the Annotated type hint
            parent_context_type: The context_type of the parent model
            param_name: Name of the parameter (for error messages)

        Raises:
            TypeError: If context_type or result_type don't match
        """
        # Import here to avoid circular import
        from .callable import CallableModel

        if not isinstance(value, CallableModel):
            return  # Not a CallableModel, skip validation

        # Determine expected context type
        expected_ctx = self.context_type if self.context_type is not None else parent_context_type

        # Validate context_type - the dependency's context_type should be compatible
        # with what we'll pass to it (expected_ctx)
        dep_context_type = value.context_type
        try:
            if not issubclass(expected_ctx, dep_context_type):
                raise TypeError(
                    f"Dependency '{param_name}': expected context_type compatible with "
                    f"{dep_context_type.__name__}, but will pass {expected_ctx.__name__}"
                )
        except TypeError:
            # issubclass can fail for certain types, try alternate check
            if expected_ctx != dep_context_type:
                raise TypeError(f"Dependency '{param_name}': context_type mismatch - expected {dep_context_type}, got {expected_ctx}")

        # Validate result_type using the generic-safe comparison
        # If expected_result_type is Union[T, CallableModel], extract T for validation
        dep_result_type = value.result_type
        actual_expected_type = expected_result_type

        # Handle Union[T, CallableModel] from DepOf expansion
        if get_origin(expected_result_type) is Union:
            union_args = get_args(expected_result_type)
            # Filter out CallableModel from the union
            non_callable_types = [t for t in union_args if t is not CallableModel]
            if non_callable_types:
                actual_expected_type = non_callable_types[0]

        if not _is_compatible_type(dep_result_type, actual_expected_type):
            raise TypeError(
                f"Dependency '{param_name}': expected result_type compatible with "
                f"{actual_expected_type}, but got CallableModel with result_type {dep_result_type}"
            )

    def __repr__(self):
        parts = []
        if self.transform is not None:
            parts.append(f"transform={self.transform}")
        if self.context_type is not None:
            parts.append(f"context_type={self.context_type.__name__}")
        return f"Dep({', '.join(parts)})" if parts else "Dep()"

    def __eq__(self, other):
        if not isinstance(other, Dep):
            return False
        return self.transform == other.transform and self.context_type == other.context_type

    def __hash__(self):
        # Make Dep hashable for use in sets/dicts
        return hash((id(self.transform), self.context_type))


def extract_dep(annotation) -> tuple:
    """Extract Dep from Annotated[T, Dep(...)] or DepOf[ContextType, T].

    When multiple Dep annotations exist (e.g., from nested Annotated that flattens),
    returns the LAST one, which represents the outermost user annotation.

    Args:
        annotation: A type annotation, possibly Annotated with Dep

    Returns:
        Tuple of (base_type, Dep instance or None)
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        # Find the LAST Dep - nested Annotated flattens, so outer annotation comes last
        last_dep = None
        for metadata in args[1:]:
            if isinstance(metadata, Dep):
                last_dep = metadata
        if last_dep is not None:
            return base_type, last_dep
    return annotation, None
