import logging
import random
import time
from typing import List, Optional, Union

from pydantic import Field, field_validator
from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
from ..exttypes import PyObjectPath

__all__ = [
    "RetryError",
    "RetryEvaluator",
]

log = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when a ``RetryEvaluator`` exhausts all attempts and ``reraise`` is ``False``.

    The original exception that caused the final failure is available both as the
    ``__cause__`` of this error (via ``raise ... from``) and as the ``last_exception``
    attribute.
    """

    def __init__(self, message: str, attempts: int, last_exception: BaseException):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class RetryEvaluator(EvaluatorBase):
    """Evaluator that retries the evaluation of a callable model on failure.

    Retry behaviour is controlled by a stop condition (``max_attempts`` and/or
    ``max_delay``) and a wait strategy that supports exponential backoff with
    optional jitter, inspired by `tenacity <https://github.com/jd/tenacity>`_.

    The evaluator is *transparent*: a successful evaluation returns exactly the
    same value as evaluating the wrapped context directly, so it does not affect
    cache keys or dependency-graph deduplication.

    Concurrency / parallelism:
        This evaluator keeps **no** mutable retry bookkeeping on the instance; all
        per-call state (attempt counter, elapsed time, last exception) lives in
        local variables. The same ``RetryEvaluator`` instance can therefore be
        shared safely across threads and is safe to combine with evaluators that
        execute callables in parallel (e.g. a Ray or Celery evaluator).

        All fields are primitives or serializable ``PyObjectPath`` references, so
        the evaluator can be pickled and shipped to remote workers. Where the
        retry happens depends on composition order:

        - Place ``RetryEvaluator`` *inside* a parallel evaluator to retry each
          task independently on the worker that runs it.
        - Place ``RetryEvaluator`` *outside* a parallel evaluator to retry the
          whole parallel dispatch as a single unit.
    """

    max_attempts: int = Field(default=3, ge=1, description="Maximum number of attempts (including the first) before giving up.")
    max_delay: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional cap on the total time (in seconds) spent waiting between retries. Once a further wait would exceed this budget, no more retries are attempted.",
    )
    retry_exceptions: List[PyObjectPath] = Field(
        default_factory=lambda: [PyObjectPath.validate(Exception)],
        description="Exception types that trigger a retry. An exception is retried if it is an instance of any of these.",
    )
    no_retry_exceptions: List[PyObjectPath] = Field(
        default_factory=list,
        description="Exception types that are never retried, even if they match ``retry_exceptions``. Takes precedence over ``retry_exceptions``.",
    )
    wait_initial: float = Field(default=0.0, ge=0.0, description="Base delay (in seconds) used for the first retry / exponential backoff.")
    wait_multiplier: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff multiplier applied per attempt: delay = wait_initial * multiplier^(attempt-1)."
    )
    wait_max: Optional[float] = Field(
        default=None, ge=0.0, description="Optional cap on the delay (in seconds) for any single retry, applied before jitter."
    )
    wait_jitter: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum random jitter (in seconds) added to each delay, drawn uniformly from [0, wait_jitter]. Helps avoid thundering-herd retries.",
    )
    reraise: bool = Field(
        default=True, description="If True, re-raise the last underlying exception on failure. If False, raise a RetryError that wraps it."
    )
    log_level: int = Field(default=logging.WARNING, description="Log level used to report each retry attempt.")

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: Union[int, str]) -> int:
        """Allow the log level to be specified as a name (e.g. "INFO") in addition to an int."""
        if isinstance(v, str):
            level = logging.getLevelName(v.upper())
            if not isinstance(level, int):
                raise ValueError(f"Invalid log level: {v}")
            return level
        return v

    @field_validator("retry_exceptions", "no_retry_exceptions")
    @classmethod
    def _validate_exception_types(cls, value: List[PyObjectPath]) -> List[PyObjectPath]:
        for path in value:
            obj = path.object
            if not (isinstance(obj, type) and issubclass(obj, Exception)):
                raise ValueError(f"{path} does not resolve to an Exception subclass")
        return value

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def _should_retry(self, exc: BaseException) -> bool:
        no_retry = tuple(path.object for path in self.no_retry_exceptions)
        if no_retry and isinstance(exc, no_retry):
            return False
        retry = tuple(path.object for path in self.retry_exceptions)
        return isinstance(exc, retry)

    def _compute_delay(self, attempt: int, jitter_value: Optional[float] = None) -> float:
        """Compute the delay (in seconds) before the retry following ``attempt`` (1-based)."""
        delay = self.wait_initial * (self.wait_multiplier ** (attempt - 1))
        if self.wait_max is not None:
            delay = min(delay, self.wait_max)
        if self.wait_jitter:
            jitter_value = random.uniform(0.0, self.wait_jitter) if jitter_value is None else jitter_value
            delay += jitter_value
        return max(delay, 0.0)

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        model_name = context.model.meta.name or context.model.__class__.__name__
        total_wait = 0.0
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return context()
            except Exception as exc:
                if not self._should_retry(exc):
                    raise
                last_exception = exc
                if attempt >= self.max_attempts:
                    break
                delay = self._compute_delay(attempt)
                if self.max_delay is not None and total_wait + delay > self.max_delay:
                    break
                log.log(
                    self.log_level,
                    "[%s]: Attempt %d/%d of %s on %s failed with %s: %s. Retrying in %.3fs.",
                    model_name,
                    attempt,
                    self.max_attempts,
                    context.fn,
                    context.context,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                    total_wait += delay

        message = f"Retry attempts exhausted after {attempt} attempt(s) for {model_name}."
        assert last_exception is not None  # The loop only breaks/exits after at least one caught exception.
        if self.reraise:
            raise last_exception
        raise RetryError(message, attempts=attempt, last_exception=last_exception) from last_exception
