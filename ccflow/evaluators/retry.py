from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
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
        By default every model evaluated through this evaluator is eligible for retry. To limit retries to specific models, scope the evaluator with
        ``FlowOptionsOverride(models=..., model_types=...)`` or a per-call ``_options={"evaluator": ...}`` override, or pin it on an individual model
        via its ``meta.options``. To attach a retry policy to a single specific model as part of the graph itself, use
        :class:`~ccflow.models.retry.RetryModel` instead.
    """

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        name = context.model.meta.name or context.model.__class__.__name__
        return self._run_with_retry(context, name=name, detail=f"{context.fn} on {context.context}")
