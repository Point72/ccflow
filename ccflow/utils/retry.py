import logging
import random
import time
from typing import Callable, List, Optional, Union

from pydantic import Field, field_validator

from ..base import BaseModel, ResultType
from ..exttypes import PyObjectPath

__all__ = [
    "RetryError",
    "RetryPolicy",
]

log = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when a retry helper exhausts all attempts and ``reraise`` is ``False``.

    The original exception that caused the final failure is available both as the ``__cause__`` of this error (via ``raise ... from``) and as the
    ``last_exception`` attribute.
    """

    def __init__(self, message: str, attempts: int, last_exception: BaseException):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class RetryPolicy(BaseModel):
    """Shared configuration and logic for retrying a callable with exponential backoff and optional jitter.

    This is the single source of truth for retry behaviour, reused by both ``RetryEvaluator`` (a cross-cutting evaluator that wraps how models
    are evaluated) and ``RetryModel`` (a structural wrapper that makes "retry this specific model" a first-class node in the graph). It is
    inspired by `tenacity <https://github.com/jd/tenacity>`_.

    Retry behaviour is controlled by a stop condition (``max_attempts`` and/or ``max_delay``) and a wait strategy that supports exponential backoff
    with optional jitter (``wait_initial``, ``wait_multiplier``, ``wait_max`` and ``wait_jitter``). Which exceptions trigger a retry is controlled
    by ``retry_exceptions`` / ``no_retry_exceptions``.
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

    def _run_with_retry(self, attempt_fn: Callable[[], ResultType], name: str, detail: str) -> ResultType:
        """Run ``attempt_fn`` with retries according to this policy.

        Args:
            attempt_fn: A zero-argument callable that performs a single attempt and returns the result.
            name: A short name (e.g. the model name) used in log/error messages.
            detail: A description of the call (e.g. ``"__call__ on <context>"``) used in log messages.
        """
        total_wait = 0.0
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return attempt_fn()
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
                    "[%s]: Attempt %d/%d (%s) failed with %s: %s. Retrying in %.3fs.",
                    name,
                    attempt,
                    self.max_attempts,
                    detail,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                    total_wait += delay

        message = f"Retry attempts exhausted after {attempt} attempt(s) for {name}."
        assert last_exception is not None  # The loop only breaks/exits after at least one caught exception.
        if self.reraise:
            raise last_exception
        raise RetryError(message, attempts=attempt, last_exception=last_exception) from last_exception
