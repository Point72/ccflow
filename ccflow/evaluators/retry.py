from typing import List

from pydantic import Field, field_validator
from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
from ..exttypes import PyObjectPath
from ..utils.retry import RetryPolicy

__all__ = [
    "RetryEvaluator",
]


class RetryEvaluator(EvaluatorBase, RetryPolicy):
    """Evaluator that retries the evaluation of a callable model on failure.

    This is the cross-cutting ("how to run") way to add retries: it wraps the evaluation of the models in its scope and is configured at runtime via
    ``FlowOptions`` / ``FlowOptionsOverride``. The retry mechanics come from :class:`~ccflow.utils.retry.RetryPolicy`.

    The evaluator is *transparent*: a successful evaluation returns exactly the same value as evaluating the wrapped context directly, so it does
    not affect cache keys or dependency-graph deduplication.

    Concurrency / parallelism:
        This evaluator keeps **no** mutable retry bookkeeping on the instance; all per-call state (attempt counter, elapsed time, last exception)
        lives in local variables. The same ``RetryEvaluator`` instance can therefore be shared safely across threads and is safe to combine with
        evaluators that execute callables in parallel (e.g. a Ray or Celery evaluator).

        All fields are primitives or serializable ``PyObjectPath`` references, so the evaluator can be pickled and shipped to remote workers.
        Where the retry happens depends on composition order:

        - Place ``RetryEvaluator`` *inside* a parallel evaluator to retry each task independently on the worker that runs it.
        - Place ``RetryEvaluator`` *outside* a parallel evaluator to retry the whole parallel dispatch as a single unit.

    Selecting which models to retry:
        By default every model evaluated through this evaluator is eligible for retry. When the evaluator is applied as part of a single global
        chain (e.g. combined with logging/caching), use ``include_model_types`` and/or ``exclude_model_types`` to limit retries to specific
        ``CallableModel`` types. Non-selected models are passed straight through (evaluated once, no retry). For coarser targeting you can instead
        scope the evaluator with ``FlowOptionsOverride(models=..., model_types=...)`` or a per-call ``_options={"evaluator": ...}`` override, or
        pin it on an individual model via its ``meta.options``. To attach a retry policy to a single specific model as part of the graph itself,
        use :class:`~ccflow.models.retry.RetryModel` instead.
    """

    include_model_types: List[PyObjectPath] = Field(
        default_factory=list,
        description="If non-empty, only models that are instances of one of these types are retried; all others are evaluated once and passed through.",
    )
    exclude_model_types: List[PyObjectPath] = Field(
        default_factory=list,
        description="Models that are instances of one of these types are never retried (evaluated once and passed through). Takes precedence over include_model_types.",
    )

    @field_validator("include_model_types", "exclude_model_types")
    @classmethod
    def _validate_model_types(cls, value: List[PyObjectPath]) -> List[PyObjectPath]:
        for path in value:
            if not isinstance(path.object, type):
                raise ValueError(f"{path} does not resolve to a type")
        return value

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def _selects(self, context: ModelEvaluationContext) -> bool:
        """Whether the model in ``context`` is eligible for retry."""
        model = context.model
        exclude = tuple(path.object for path in self.exclude_model_types)
        if exclude and isinstance(model, exclude):
            return False
        if self.include_model_types:
            include = tuple(path.object for path in self.include_model_types)
            return isinstance(model, include)
        return True

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        if not self._selects(context):
            return context()
        name = context.model.meta.name or context.model.__class__.__name__
        return self._run_with_retry(context, name=name, detail=f"{context.fn} on {context.context}")
