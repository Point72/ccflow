from typing import Generic

from ..callable import CallableModelType, ContextType, Flow, ResultType, WrapperModel
from ..utils.retry import RetryPolicy

__all__ = [
    "RetryModel",
]


class RetryModel(WrapperModel[CallableModelType], RetryPolicy, Generic[CallableModelType]):
    """A callable model that wraps another model and retries it on failure.

    This is the structural ("what the graph is") way to add retries: unlike :class:`~ccflow.evaluators.retry.RetryEvaluator`, it is a first-class
    ``CallableModel`` node, so a retry policy can be declared statically in config/registries (e.g. ``RetryModel(model=${fetch}, max_attempts=5)``)
    and shows up explicitly in the graph and serialization. It inherits the wrapped model's ``context_type`` / ``result_type`` (from
    ``WrapperModel``) and reuses the retry mechanics from :class:`~ccflow.utils.retry.RetryPolicy`.

    The wrapped model is invoked through its own flow decorator, so any evaluators configured for it (logging, caching, etc.) still apply on each
    attempt.
    """

    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        name = self.model.meta.name or self.model.__class__.__name__
        return self._run_with_retry(lambda: self.model(context), name=name, detail=f"__call__ on {context}")
