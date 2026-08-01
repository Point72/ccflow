import logging
import random
import secrets
import time
from collections.abc import Callable

from pydantic import Field, field_validator

from ..base import BaseModel, ResultType
from ..exttypes import PyObjectPath
from .reporting import Reporter, ReportEvent, ReportPhase, current_run_id, current_span_depth, current_span_id

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
    max_delay: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional cap on the total time (in seconds) spent waiting between retries. Once a further wait would exceed this budget, no more retries are attempted.",
    )
    retry_exceptions: list[PyObjectPath] = Field(
        default_factory=lambda: [PyObjectPath.validate(Exception)],
        description="Exception types that trigger a retry. An exception is retried if it is an instance of any of these.",
    )
    no_retry_exceptions: list[PyObjectPath] = Field(
        default_factory=list,
        description="Exception types that are never retried, even if they match ``retry_exceptions``. Takes precedence over ``retry_exceptions``.",
    )
    wait_initial: float = Field(default=0.0, ge=0.0, description="Base delay (in seconds) used for the first retry / exponential backoff.")
    wait_multiplier: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff multiplier applied per attempt: delay = wait_initial * multiplier^(attempt-1)."
    )
    wait_max: float | None = Field(
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
    reporter: Reporter | None = Field(
        default=None,
        description="Optional reporting sink that receives retry lifecycle events (failure, retry, success, give-up). See ccflow.utils.reporting.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: int | str) -> int:
        """Allow the log level to be specified as a name (e.g. "INFO") in addition to an int."""
        if isinstance(v, str):
            level = logging.getLevelName(v.upper())
            if not isinstance(level, int):
                raise ValueError(f"Invalid log level: {v}")  # noqa: TRY004
            return level
        return v

    @field_validator("retry_exceptions", "no_retry_exceptions")
    @classmethod
    def _validate_exception_types(cls, value: list[PyObjectPath]) -> list[PyObjectPath]:
        for path in value:
            obj = path.object
            if not (isinstance(obj, type) and issubclass(obj, Exception)):
                raise ValueError(f"{path} does not resolve to an Exception subclass")  # noqa: TRY004
        return value

    def _should_retry(self, exc: BaseException) -> bool:
        no_retry = tuple(path.object for path in self.no_retry_exceptions)
        if no_retry and isinstance(exc, no_retry):
            return False
        retry = tuple(path.object for path in self.retry_exceptions)
        return isinstance(exc, retry)

    def _compute_delay(self, attempt: int, jitter_value: float | None = None) -> float:
        """Compute the delay (in seconds) before the retry following ``attempt`` (1-based)."""
        delay = self.wait_initial * (self.wait_multiplier ** (attempt - 1))
        if self.wait_max is not None:
            delay = min(delay, self.wait_max)
        if self.wait_jitter:
            jitter_value = random.uniform(0.0, self.wait_jitter) if jitter_value is None else jitter_value
            delay += jitter_value
        return max(delay, 0.0)

    def _emit_retry(
        self,
        phase: ReportPhase,
        *,
        name: str,
        detail: str,
        span_id: str | None,
        attempt: int,
        exc: BaseException | None = None,
        delay: float | None = None,
        reason: str | None = None,
    ) -> None:
        """Emit a single retry-lifecycle event to the configured reporter (no-op if unset)."""
        if self.reporter is None or span_id is None:
            return
        extra: dict = {}
        if delay is not None:
            extra["delay"] = delay
        if reason is not None:
            extra["reason"] = reason
        parent_span_id = current_span_id()
        parent_depth = current_span_depth()
        depth = parent_depth + 1 if parent_depth is not None else 0
        try:
            self.reporter.emit(
                ReportEvent(
                    phase=phase,
                    model_name=name,
                    fn="__call__",
                    context_repr=detail,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    run_id=current_run_id(),
                    depth=depth,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    exception_type=type(exc).__name__ if exc is not None else None,
                    exception_message=str(exc) if exc is not None else None,
                    extra=extra,
                )
            )
        except Exception:
            # Retry reporting is best-effort: a broken sink must never change retry behaviour.
            log.exception("Reporter %r failed to emit %s retry event; continuing.", type(self.reporter).__name__, phase.value)

    def _run_with_retry(self, attempt_fn: Callable[[], ResultType], name: str, detail: str) -> ResultType:
        """Run ``attempt_fn`` with retries according to this policy.

        Args:
            attempt_fn: A zero-argument callable that performs a single attempt and returns the result.
            name: A short name (e.g. the model name) used in log/error messages.
            detail: A description of the call (e.g. ``"__call__ on <context>"``) used in log messages.
        """
        total_wait = 0.0
        last_exception: Exception | None = None
        budget_exceeded = False
        span_id = secrets.token_hex(8) if self.reporter is not None else None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = attempt_fn()
            except Exception as exc:
                self._emit_retry(ReportPhase.ERROR, name=name, detail=detail, span_id=span_id, attempt=attempt, exc=exc)
                if not self._should_retry(exc):
                    self._emit_retry(ReportPhase.GIVE_UP, name=name, detail=detail, span_id=span_id, attempt=attempt, exc=exc, reason="non-retryable")
                    raise
                last_exception = exc
                if attempt >= self.max_attempts:
                    break
                delay = self._compute_delay(attempt)
                if self.max_delay is not None and total_wait + delay > self.max_delay:
                    budget_exceeded = True
                    break
                self._emit_retry(ReportPhase.RETRY, name=name, detail=detail, span_id=span_id, attempt=attempt, exc=exc, delay=delay)
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
            else:
                self._emit_retry(ReportPhase.SUCCESS, name=name, detail=detail, span_id=span_id, attempt=attempt)
                return result

        if budget_exceeded:
            message = f"Retry stopped after {attempt} attempt(s) for {name}: max_delay budget exceeded."
            reason = "max_delay budget exceeded"
        else:
            message = f"Retry attempts exhausted after {attempt}/{self.max_attempts} attempt(s) for {name}."
            reason = "attempts exhausted"
        assert last_exception is not None  # The loop only breaks/exits after at least one caught exception.
        self._emit_retry(ReportPhase.GIVE_UP, name=name, detail=detail, span_id=span_id, attempt=attempt, exc=last_exception, reason=reason)
        if self.reraise:
            raise last_exception
        raise RetryError(message, attempts=attempt, last_exception=last_exception) from last_exception
