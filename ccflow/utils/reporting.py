"""Shared core for *reporting* (telemetry / observability) of callable model evaluation.

This module is the single source of truth for reporting behaviour, reused by the reporting
evaluators (:mod:`ccflow.evaluators.reporting`) and reporting models (:mod:`ccflow.models.reporting`),
exactly as :mod:`ccflow.utils.retry` is shared by ``RetryEvaluator`` / ``RetryModel``.

Reporting is *telemetry about the evaluation itself* -- which model ran, on what context, how long it
took, how it relates to other evaluations in the graph (parent/child spans) and whether it failed.
It is strictly **transparent**: it never changes the value returned by the wrapped evaluation.

The building blocks are:

* :class:`ReportEvent` -- a structured, serializable lifecycle record.
* :class:`Reporter` -- a pluggable sink for events (in-memory, logging, composite, vendor-specific).
* :class:`ReportingPolicy` and its signal-specific subclasses (:class:`TracingPolicy`,
  :class:`MetricsPolicy`, :class:`AlertsPolicy`) -- the orchestration mixins that wrap a single
  evaluation in a span and emit events at each lifecycle phase.
"""

import itertools
import logging
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pprint import pformat
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from ..base import BaseModel, ResultType

if TYPE_CHECKING:
    from opentelemetry.trace import Span

__all__ = [
    "ReportPhase",
    "AlertPriority",
    "ReportEvent",
    "ReportContext",
    "Reporter",
    "NoOpReporter",
    "InMemoryReporter",
    "LoggingReporter",
    "CompositeReporter",
    "UIReporter",
    "ReportingStateStore",
    "NodeState",
    "current_span_id",
    "current_span_depth",
    "current_run_id",
    "run_scope",
    "FormatConfig",
    "ReportingPolicy",
    "LoggingPolicy",
    "TracingPolicy",
    "MetricsPolicy",
    "AlertsPolicy",
    "OpenTelemetryTracingPolicy",
    "OpenTelemetryMetricsPolicy",
]

log = logging.getLogger(__name__)


class ReportPhase(str, Enum):
    """Lifecycle phase of a reported evaluation.

    The first group are *node* phases (a single model evaluation) and are emitted by the reporting
    policies/evaluators/models during normal flows. The second group are *run* / *graph* scoped phases
    used by the UI/observability layer to frame an overall run and its discovered graph.

    .. note::

        ``RUN_STARTED`` / ``RUN_FINISHED`` / ``GRAPH_DISCOVERED`` are **reserved** for an explicit
        run/graph observer layer and are *not* emitted by the current evaluator/model paths.
        :class:`ReportingStateStore` already knows how to fold them so that the consuming layer can be
        added without a breaking change. ``QUEUED`` / ``SKIPPED`` are emitted today by
        :class:`~ccflow.evaluators.reporting.DryRunEvaluator`.
    """

    START = "START"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    RETRY = "RETRY"
    GIVE_UP = "GIVE_UP"
    END = "END"
    # UI / observability scoped phases
    QUEUED = "QUEUED"
    SKIPPED = "SKIPPED"
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    GRAPH_DISCOVERED = "GRAPH_DISCOVERED"


class AlertPriority(str, Enum):
    """Severity / priority tag for alerts, following the common ``P1``-``P5`` convention.

    ``P1`` is the most severe (page now), ``P5`` the least (informational).
    """

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"


# *****************************************************************************
# Span correlation
#
# A module-level ContextVar tracks the *current* span so that nested evaluations
# (e.g. a RetryModel wrapping a CallableModel, or dependencies evaluated within a
# parent) link together into a span tree. ContextVars are per-thread / per-async
# context, so this preserves the thread-safety guarantee of the evaluators.
# *****************************************************************************

_CURRENT_SPAN: ContextVar[Optional[Tuple[str, int]]] = ContextVar("ccflow_report_span", default=None)
_CURRENT_RUN: ContextVar[Optional[str]] = ContextVar("ccflow_report_run", default=None)


def current_span_id() -> Optional[str]:
    """Return the span id of the currently-active reporting span, or ``None``."""
    current = _CURRENT_SPAN.get()
    return current[0] if current else None


def current_span_depth() -> Optional[int]:
    """Return the depth of the currently-active reporting span, or ``None`` if there is none.

    Useful for events emitted outside the :meth:`ReportingPolicy._run_with_reporting` flow (e.g. retry
    lifecycle events) so they can nest one level below the active span (``current_span_depth() + 1``).
    """
    current = _CURRENT_SPAN.get()
    return current[1] if current else None


def current_run_id() -> Optional[str]:
    """Return the id of the currently-active run, or ``None``.

    A *run* scopes all events emitted while a root evaluation is in flight, so a UI can group events
    belonging to one logical execution (multiple runs may be live concurrently across threads).
    """
    return _CURRENT_RUN.get()


def _new_span_id() -> str:
    return secrets.token_hex(8)


@contextmanager
def run_scope(run_id: Optional[str] = None) -> Iterator[str]:
    """Bind a ``run_id`` for the duration of the ``with`` block.

    Events emitted inside the block (via any reporting policy) are tagged with this run id. If no id is
    supplied a fresh one is generated. Nested ``run_scope`` calls reuse the outermost run id unless an
    explicit id is given, so a single root evaluation maps to a single run.
    """
    if run_id is None:
        existing = _CURRENT_RUN.get()
        run_id = existing if existing is not None else _new_span_id()
    token = _CURRENT_RUN.set(run_id)
    try:
        yield run_id
    finally:
        _CURRENT_RUN.reset(token)


@dataclass
class ReportContext:
    """Live, per-call state for a single reported evaluation.

    This object is created fresh inside :meth:`ReportingPolicy._run_with_reporting` and never stored
    on the policy instance, so a single evaluator/model can be shared safely across threads.
    """

    model_name: str
    model_type: str
    fn: str
    context_repr: str
    span_id: str
    parent_span_id: Optional[str]
    depth: int
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _start_perf: float = field(default_factory=time.perf_counter)
    extra: Dict[str, Any] = field(default_factory=dict)

    def elapsed(self) -> float:
        """Seconds elapsed since the context was created."""
        return time.perf_counter() - self._start_perf


class ReportEvent(BaseModel):
    """A structured, serializable record of a single lifecycle event for an evaluation."""

    phase: ReportPhase = Field(description="The lifecycle phase this event represents.")
    model_name: str = Field(description="The name of the model being evaluated (meta.name or class name).")
    model_type: str = Field("", description="The fully-qualified type of the model being evaluated.")
    fn: str = Field("__call__", description="The function being evaluated.")
    context_repr: str = Field("", description="A bounded repr of the context the model is evaluated on.")
    span_id: str = Field(description="Unique id of this evaluation span.")
    parent_span_id: Optional[str] = Field(None, description="Span id of the enclosing evaluation, if any.")
    run_id: Optional[str] = Field(None, description="Id of the run this event belongs to, if a run scope is active.")
    depth: int = Field(0, ge=0, description="Nesting depth of this span in the evaluation tree.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the event was emitted.")
    duration: Optional[float] = Field(None, ge=0.0, description="Elapsed seconds for the evaluation (set on terminal phases).")
    attempt: Optional[int] = Field(None, ge=1, description="Attempt number, for retry lifecycle events.")
    max_attempts: Optional[int] = Field(None, ge=1, description="Maximum attempts, for retry lifecycle events.")
    exception_type: Optional[str] = Field(None, description="Type name of the exception, for ERROR/GIVE_UP events.")
    exception_message: Optional[str] = Field(None, description="Message of the exception, for ERROR/GIVE_UP events.")
    priority: Optional[AlertPriority] = Field(None, description="Alert priority, for alert events.")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional sink-specific metadata.")


# *****************************************************************************
# Reporters (pluggable sinks)
# *****************************************************************************


class Reporter(BaseModel):
    """A pluggable sink that consumes :class:`ReportEvent` objects.

    The naming convention for concrete reporters mirrors the publishers: ``<Where>Reporter``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def emit(self, event: ReportEvent) -> None:
        """Consume a single event. The base implementation is a no-op."""


class NoOpReporter(Reporter):
    """A reporter that discards all events."""


class InMemoryReporter(Reporter):
    """A reporter that collects events in memory, for testing and introspection.

    The collected events can be used to reconstruct the span tree of an evaluation.
    """

    # Private so it does not affect tokenization/serialization of the reporter itself.
    _events: List[ReportEvent] = PrivateAttr(default_factory=list)

    @property
    def events(self) -> List[ReportEvent]:
        """The collected events, in emission order."""
        return list(self._events)

    def emit(self, event: ReportEvent) -> None:
        self._events.append(event)

    def clear(self) -> None:
        """Discard all collected events."""
        self._events.clear()

    def __deepcopy__(self, memo):
        # Share the same buffer when the framework deep-copies the reporter (e.g. inside the
        # ModelEvaluationContext), mirroring MemoryCacheEvaluator.
        return self


class LoggingReporter(Reporter):
    """A reporter that writes events to a :mod:`logging` logger."""

    logger_name: str = Field("ccflow.reporting", description="Name of the logger to write events to.")
    log_level: int = Field(logging.INFO, description="Level at which events are logged.")

    def emit(self, event: ReportEvent) -> None:
        logging.getLogger(self.logger_name).log(
            self.log_level,
            "[%s] %s %s on %s (span=%s parent=%s depth=%d)%s",
            event.model_name,
            event.phase.value,
            event.fn,
            event.context_repr,
            event.span_id,
            event.parent_span_id,
            event.depth,
            f" :: {event.exception_type}: {event.exception_message}" if event.exception_type else "",
        )


class CompositeReporter(Reporter):
    """A reporter that fans events out to a list of child reporters."""

    reporters: List[Reporter] = Field(default_factory=list, description="Child reporters to fan events out to.")

    def emit(self, event: ReportEvent) -> None:
        for reporter in self.reporters:
            # Isolate each child: a broken sink must not prevent the others from receiving the event,
            # nor fail the user's computation.
            try:
                reporter.emit(event)
            except Exception:
                log.exception("Child reporter %r failed to emit %s event; continuing.", type(reporter).__name__, event.phase.value)


class UIReporter(Reporter):
    """A thread-safe, bounded reporter that buffers events for a UI / observability consumer.

    Events are appended to a bounded :class:`collections.deque`; when full, the oldest events are
    dropped (the UI is expected to poll :meth:`drain` regularly and tolerate gaps). This decouples the
    hot evaluation path from any (possibly slow) transport: the producing thread only does a cheap
    append under a lock, and the consumer drains on its own cadence.
    """

    maxlen: int = Field(10_000, ge=1, description="Maximum number of buffered events before the oldest are dropped.")

    _buffer: "deque[ReportEvent]" = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._buffer = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def emit(self, event: ReportEvent) -> None:
        with self._lock:
            self._buffer.append(event)

    def drain(self) -> List[ReportEvent]:
        """Atomically remove and return all currently-buffered events, in emission order."""
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events

    def __deepcopy__(self, memo):
        # Share the same buffer when the framework deep-copies the reporter (e.g. inside the
        # ModelEvaluationContext), mirroring InMemoryReporter.
        return self


@dataclass
class NodeState:
    """Folded UI state for a single node (one ``span_id``), reconstructed from its events."""

    span_id: str
    model_name: str
    model_type: str
    fn: str
    context_repr: str
    parent_span_id: Optional[str]
    run_id: Optional[str]
    depth: int
    phase: ReportPhase
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    duration: Optional[float] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    priority: Optional["AlertPriority"] = None


class ReportingStateStore:
    """Folds a stream of :class:`ReportEvent` objects into per-node state for a UI.

    The store is *event-sourced*: feeding it the same events (e.g. drained from a :class:`UIReporter`)
    in order reconstructs the current state of every node, keyed by ``span_id``. This is intentionally
    transport-agnostic so it can run in-process or behind a socket.

    Phase folding preserves the *outcome*: once a node reaches a terminal outcome
    (SUCCESS / ERROR / GIVE_UP / SKIPPED) a later transient phase (e.g. a stray START) never clobbers
    it. ``END`` marks completion but is **not** an outcome, so it never replaces a recorded outcome --
    it only merges metadata (e.g. duration). The one allowed terminal transition is ``ERROR -> RETRY``,
    so a retry span (which emits ERROR for a failed attempt, then RETRY) shows the live retry phase
    rather than getting stuck on the intermediate error.
    """

    # Terminal *outcomes* of a node. END is intentionally excluded: it signals completion, not outcome.
    _OUTCOME = frozenset(
        {
            ReportPhase.SUCCESS,
            ReportPhase.ERROR,
            ReportPhase.GIVE_UP,
            ReportPhase.SKIPPED,
        }
    )

    def __init__(self) -> None:
        self.nodes: Dict[str, NodeState] = {}
        self.runs: Dict[str, ReportPhase] = {}

    def _next_phase(self, current: ReportPhase, incoming: ReportPhase) -> ReportPhase:
        """Decide the node phase after folding ``incoming`` onto ``current``."""
        # END marks completion but is not an outcome: never let it hide a recorded outcome.
        if incoming == ReportPhase.END and current in self._OUTCOME:
            return current
        # Once an outcome is recorded, only allow ERROR -> RETRY (a failed attempt that will retry).
        if current in self._OUTCOME and incoming not in self._OUTCOME:
            if not (current == ReportPhase.ERROR and incoming == ReportPhase.RETRY):
                return current
        return incoming

    def apply(self, event: ReportEvent) -> None:
        """Fold a single event into the store."""
        if event.phase in (ReportPhase.RUN_STARTED, ReportPhase.RUN_FINISHED):
            if event.run_id is not None:
                self.runs[event.run_id] = event.phase
            return
        node = self.nodes.get(event.span_id)
        if node is None:
            self.nodes[event.span_id] = NodeState(
                span_id=event.span_id,
                model_name=event.model_name,
                model_type=event.model_type,
                fn=event.fn,
                context_repr=event.context_repr,
                parent_span_id=event.parent_span_id,
                run_id=event.run_id,
                depth=event.depth,
                phase=event.phase,
                attempt=event.attempt,
                max_attempts=event.max_attempts,
                duration=event.duration,
                exception_type=event.exception_type,
                exception_message=event.exception_message,
                priority=event.priority,
            )
            return
        # Fold the phase (preserving outcome), but always merge available metadata so e.g. the
        # duration carried on a trailing END is recorded even when the outcome phase is kept.
        node.phase = self._next_phase(node.phase, event.phase)
        if event.attempt is not None:
            node.attempt = event.attempt
        if event.max_attempts is not None:
            node.max_attempts = event.max_attempts
        if event.duration is not None:
            node.duration = event.duration
        if event.exception_type is not None:
            node.exception_type = event.exception_type
            node.exception_message = event.exception_message
        if event.priority is not None:
            node.priority = event.priority

    def apply_all(self, events: List[ReportEvent]) -> None:
        """Fold a batch of events into the store, in order."""
        for event in events:
            self.apply(event)

    def roots(self) -> List[NodeState]:
        """Return the nodes that have no parent within the store (the run roots)."""
        return [n for n in self.nodes.values() if n.parent_span_id is None or n.parent_span_id not in self.nodes]

    def children(self, span_id: str) -> List[NodeState]:
        """Return the nodes whose parent is ``span_id``."""
        return [n for n in self.nodes.values() if n.parent_span_id == span_id]


# *****************************************************************************
# Reporting policies (orchestration mixins)
# *****************************************************************************


class ReportingPolicy(BaseModel):
    """Shared configuration and orchestration for reporting on a single evaluation.

    Subclasses specialise the *signal* (tracing, metrics, alerts) by overriding the lifecycle hooks
    (:meth:`_on_start`, :meth:`_on_success`, :meth:`_on_error`, :meth:`_on_end`) and/or the span
    context manager :meth:`_span`. The base implementation emits :class:`ReportEvent` objects to the
    configured :attr:`reporter`, and always returns the wrapped result unchanged (transparency).
    """

    reporter: Optional[Reporter] = Field(None, description="Sink that events are emitted to. If None, the policy is a transparent pass-through.")
    capture_context_repr: bool = Field(True, description="Whether to include a bounded repr of the context in events.")
    max_context_repr: int = Field(200, ge=0, description="Maximum length of the captured context repr.")

    def _emit(self, event: ReportEvent) -> None:
        if self.reporter is None:
            return
        try:
            self.reporter.emit(event)
        except Exception:
            # Reporting is transparent: a broken sink must never change the evaluation result.
            log.exception("Reporter %r failed to emit %s event; continuing.", type(self.reporter).__name__, event.phase.value)

    def _context_repr(self, context: Any) -> str:
        if not self.capture_context_repr or self.max_context_repr <= 0:
            return ""
        text = repr(context)
        if len(text) > self.max_context_repr:
            text = text[: self.max_context_repr - 1] + "\u2026"
        return text

    def _event(self, ctx: ReportContext, phase: ReportPhase, **kwargs: Any) -> ReportEvent:
        return ReportEvent(
            phase=phase,
            model_name=ctx.model_name,
            model_type=ctx.model_type,
            fn=ctx.fn,
            context_repr=ctx.context_repr,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            run_id=_CURRENT_RUN.get(),
            depth=ctx.depth,
            **kwargs,
        )

    # -- lifecycle hooks (override in subclasses) ----------------------------

    def _on_start(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        self._emit(self._event(ctx, ReportPhase.START))

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        self._emit(self._event(ctx, ReportPhase.SUCCESS, duration=ctx.elapsed()))

    def _on_error(self, ctx: ReportContext, span: "Optional[Span]", exc: BaseException) -> None:
        self._emit(
            self._event(
                ctx,
                ReportPhase.ERROR,
                duration=ctx.elapsed(),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
        )

    def _on_end(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        self._emit(self._event(ctx, ReportPhase.END, duration=ctx.elapsed()))

    @contextmanager
    def _span(self, ctx: ReportContext) -> Iterator["Optional[Span]"]:
        """Context manager that is active for the duration of the wrapped evaluation.

        Subclasses (e.g. OpenTelemetry) override this to open a real span; the base implementation
        yields ``None`` and relies on the lifecycle hooks for emission.
        """
        yield None

    # -- orchestration -------------------------------------------------------

    def _make_context(self, *, model_name: str, model_type: str, fn: str, context: Any) -> ReportContext:
        parent = _CURRENT_SPAN.get()
        parent_span_id = parent[0] if parent else None
        depth = (parent[1] + 1) if parent else 0
        return ReportContext(
            model_name=model_name,
            model_type=model_type,
            fn=fn,
            context_repr=self._context_repr(context),
            span_id=_new_span_id(),
            parent_span_id=parent_span_id,
            depth=depth,
        )

    def _run_with_reporting(
        self,
        attempt_fn: Callable[[], ResultType],
        *,
        model_name: str,
        model_type: str,
        fn: str,
        context: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ResultType:
        """Run ``attempt_fn`` once, wrapped in a reporting span. Returns its result unchanged.

        Args:
            attempt_fn: A zero-argument callable that performs the evaluation and returns the result.
            model_name: Short name of the model (meta.name or class name).
            model_type: Fully-qualified type name of the model.
            fn: The function being evaluated (e.g. ``"__call__"``).
            context: The context the model is evaluated on (used for a bounded repr).
            extra: Optional per-call metadata stashed on the :class:`ReportContext` for use by hooks
                (e.g. the live model / context / options needed by :class:`LoggingPolicy`).
        """
        ctx = self._make_context(model_name=model_name, model_type=model_type, fn=fn, context=context)
        if extra:
            ctx.extra.update(extra)
        token = _CURRENT_SPAN.set((ctx.span_id, ctx.depth))
        try:
            with self._span(ctx) as span:
                self._on_start(ctx, span)
                try:
                    result = attempt_fn()
                except BaseException as exc:
                    self._on_error(ctx, span, exc)
                    raise
                else:
                    self._on_success(ctx, span, result)
                    return result
                finally:
                    self._on_end(ctx, span)
        finally:
            _CURRENT_SPAN.reset(token)


# *****************************************************************************
# Logging policy (the default, back-compat reporting signal)
# *****************************************************************************


class FormatConfig(BaseModel):
    """Configuration for formatting the result of the evaluation.

    This is used by the :class:`LoggingPolicy` (and ``LoggingEvaluator``) to control how the result
    is formatted when ``log_result=True``.
    """

    arrow_as_polars: bool = Field(
        False,
        description="Whether to convert pyarrow tables to polars tables for formatting, as arrow formatting does not work well with large tables or provide control over options",
    )
    pformat_config: Dict[str, Any] = Field({}, description="pformat config to use for formatting data")
    polars_config: Dict[str, Any] = Field({}, description="polars config to use for formatting polars frames")
    pandas_config: Dict[str, Any] = Field({}, description="pandas config to use for formatting pandas objects")


class LoggingPolicy(ReportingPolicy):
    """Reporting policy that logs the start/end (and optionally the result) of each evaluation.

    This lifts the behaviour of the original ``LoggingEvaluator`` into a reusable mixin so it can be
    shared by both ``LoggingEvaluator`` (cross-cutting) and ``LoggingModel`` (structural). The logging
    is done *inline in the lifecycle hooks* to preserve the exact historical output and ``FormatConfig``
    behaviour; the hooks also call ``super()`` so that, if a :attr:`~ReportingPolicy.reporter` is
    configured, structured :class:`ReportEvent` objects are emitted in addition to the log lines.

    The live model / context / options are read from ``ctx.extra`` (populated by the evaluator/model),
    so per-call ``log_level`` / ``verbose`` overrides from ``FlowOptions`` continue to work.
    """

    log_level: int = Field(logging.DEBUG, description="The log level for start/end of evaluation")
    verbose: bool = Field(True, description="Whether to output the model definition as part of logging")
    log_result: bool = Field(False, description="Whether to log the result of the evaluation")
    format_config: FormatConfig = Field(FormatConfig(), description="Configuration for formatting the result of the evaluation if log_result=True")

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: Any) -> int:
        """Validate that the log level is a valid logging level."""
        if isinstance(v, str):
            return getattr(logging, v.upper(), "")
        return v

    def _log_level(self, ctx: ReportContext) -> int:
        options = ctx.extra.get("options") or {}
        return options.get("log_level", self.log_level)

    def _verbose(self, ctx: ReportContext) -> bool:
        options = ctx.extra.get("options") or {}
        return options.get("verbose", self.verbose)

    def _on_start(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        log_level = self._log_level(ctx)
        raw_context = ctx.extra.get("raw_context")
        log.log(log_level, "[%s]: Start evaluation of %s on %s.", ctx.model_name, ctx.fn, raw_context)
        if self._verbose(ctx):
            log.log(log_level, "[%s]: %s", ctx.model_name, ctx.extra.get("model"))
        super()._on_start(ctx, span)

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        if self.log_result and result is not None:
            log.log(
                self._log_level(ctx),
                self._format_result(result),
                ctx.model_name,
                ctx.fn,
                ctx.extra.get("raw_context"),
            )
        super()._on_success(ctx, span, result)

    def _on_end(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        log.log(
            self._log_level(ctx),
            "[%s]: End evaluation of %s on %s (time elapsed: %s).",
            ctx.model_name,
            ctx.fn,
            ctx.extra.get("raw_context"),
            timedelta(seconds=ctx.elapsed()),
        )
        super()._on_end(ctx, span)

    def _format_result(self, result: ResultType) -> str:
        """Handle formatting of the result, returning a ``log``-style format string."""
        # Add special formatting for eager table/data frame types embedded in the results
        import pyarrow as pa

        result_dict = result.model_dump(by_alias=True)
        for k, v in result_dict.items():
            try:
                if self.format_config.arrow_as_polars and isinstance(v, pa.Table):
                    import polars as pl  # Only import polars if needed

                    result_dict[k] = pl.from_arrow(v)
            except TypeError:
                pass

        if self.format_config.polars_config:  # Control formatting of polars tables if set
            import polars as pl  # Only import polars if needed

            polars_context = pl.Config(**self.format_config.polars_config)
        else:
            polars_context = nullcontext()

        if self.format_config.pandas_config:  # Control formatting of pandas tables if set
            import pandas as pd

            pandas_context = pd.option_context(*itertools.chain.from_iterable(self.format_config.pandas_config.items()))
        else:
            pandas_context = nullcontext()

        with polars_context, pandas_context:
            msg_str = "[%s]: Result of %s on %s:\n"
            return f"{msg_str}{pformat(result_dict, **self.format_config.pformat_config)}"


class TracingPolicy(ReportingPolicy):
    """Reporting policy specialised for distributed *tracing* (spans).

    The generic implementation emits START/SUCCESS/ERROR/END events that describe a span; concrete
    backends (e.g. :class:`OpenTelemetryTracingPolicy`) override :meth:`_span` to open real spans.
    """


class MetricsPolicy(ReportingPolicy):
    """Reporting policy specialised for *metrics* (counters / latency histograms).

    The generic implementation records an evaluation count and a latency measurement (in the event
    ``extra``); concrete backends override the hooks to push to a metrics backend.
    """

    metric_prefix: str = Field("ccflow.model", description="Prefix applied to emitted metric names.")

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        self._emit(
            self._event(
                ctx,
                ReportPhase.SUCCESS,
                duration=ctx.elapsed(),
                extra={"metric": f"{self.metric_prefix}.success", "value": 1, "latency_seconds": ctx.elapsed()},
            )
        )

    def _on_error(self, ctx: ReportContext, span: "Optional[Span]", exc: BaseException) -> None:
        self._emit(
            self._event(
                ctx,
                ReportPhase.ERROR,
                duration=ctx.elapsed(),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                extra={"metric": f"{self.metric_prefix}.error", "value": 1, "latency_seconds": ctx.elapsed()},
            )
        )

    def _on_end(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        self._emit(
            self._event(
                ctx,
                ReportPhase.END,
                duration=ctx.elapsed(),
                extra={"metric": f"{self.metric_prefix}.latency", "value": ctx.elapsed()},
            )
        )


class AlertsPolicy(ReportingPolicy):
    """Reporting policy specialised for *alerting*.

    Emits a prioritised alert event when an evaluation fails (and, optionally, when it succeeds again
    after previously failing). Concrete backends (PagerDuty / Opsgenie / JSM / ...) override the hooks
    to route alerts to an on-call system.
    """

    priority: AlertPriority = Field(AlertPriority.P3, description="Priority tag applied to emitted alerts (P1 = most severe).")
    alert_on_error: bool = Field(True, description="Whether to emit an alert when an evaluation fails.")
    alert_on_success: bool = Field(False, description="Whether to emit a (recovery) alert when an evaluation succeeds.")

    def _on_start(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        # Alerts are only interesting on terminal phases.
        pass

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        if self.alert_on_success:
            self._emit(self._event(ctx, ReportPhase.SUCCESS, duration=ctx.elapsed(), priority=self.priority))

    def _on_error(self, ctx: ReportContext, span: "Optional[Span]", exc: BaseException) -> None:
        if self.alert_on_error:
            self._emit(
                self._event(
                    ctx,
                    ReportPhase.ERROR,
                    duration=ctx.elapsed(),
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    priority=self.priority,
                )
            )

    def _on_end(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        pass


# *****************************************************************************
# OpenTelemetry policies (real, optional-dependency backends)
# *****************************************************************************


def _require_opentelemetry():
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "OpenTelemetry reporting requires the 'opentelemetry-api' package. Install it with `pip install opentelemetry-api`."
        ) from exc
    return trace


def _require_opentelemetry_metrics():
    try:
        from opentelemetry import metrics
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "OpenTelemetry metrics reporting requires the 'opentelemetry-api' package. Install it with `pip install opentelemetry-api`."
        ) from exc
    return metrics


class OpenTelemetryTracingPolicy(TracingPolicy):
    """Tracing policy backed by OpenTelemetry spans.

    Opens a real OpenTelemetry span around each evaluation, records exceptions, and sets the span
    status. ``opentelemetry-api`` is an optional dependency, imported lazily on first use.
    """

    tracer_name: str = Field("ccflow", description="Name passed to ``opentelemetry.trace.get_tracer``.")
    span_name_prefix: str = Field("ccflow", description="Prefix for generated span names (``<prefix>.<model_name>``).")

    _tracer: Any = PrivateAttr(None)

    def _get_tracer(self):
        if self._tracer is None:
            trace = _require_opentelemetry()
            self._tracer = trace.get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def _span(self, ctx: ReportContext) -> Iterator["Optional[Span]"]:
        tracer = self._get_tracer()
        with tracer.start_as_current_span(f"{self.span_name_prefix}.{ctx.model_name}") as span:
            span.set_attribute("ccflow.model_name", ctx.model_name)
            span.set_attribute("ccflow.model_type", ctx.model_type)
            span.set_attribute("ccflow.fn", ctx.fn)
            span.set_attribute("ccflow.depth", ctx.depth)
            yield span

    def _on_start(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        super()._on_start(ctx, span)

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        trace = _require_opentelemetry()
        if span is not None:
            span.set_status(trace.Status(trace.StatusCode.OK))
        super()._on_success(ctx, span, result)

    def _on_error(self, ctx: ReportContext, span: "Optional[Span]", exc: BaseException) -> None:
        trace = _require_opentelemetry()
        if span is not None:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
        super()._on_error(ctx, span, exc)


class OpenTelemetryMetricsPolicy(MetricsPolicy):
    """Metrics policy backed by OpenTelemetry counters and histograms.

    Records a success/error counter and a latency histogram per evaluation. ``opentelemetry-api`` is
    an optional dependency, imported lazily on first use.
    """

    meter_name: str = Field("ccflow", description="Name passed to ``opentelemetry.metrics.get_meter``.")

    _meter: Any = PrivateAttr(None)
    _success_counter: Any = PrivateAttr(None)
    _error_counter: Any = PrivateAttr(None)
    _latency: Any = PrivateAttr(None)

    def _ensure_instruments(self):
        if self._meter is None:
            metrics = _require_opentelemetry_metrics()
            self._meter = metrics.get_meter(self.meter_name)
            self._success_counter = self._meter.create_counter(f"{self.metric_prefix}.success")
            self._error_counter = self._meter.create_counter(f"{self.metric_prefix}.error")
            self._latency = self._meter.create_histogram(f"{self.metric_prefix}.latency", unit="s")

    def _on_success(self, ctx: ReportContext, span: "Optional[Span]", result: ResultType) -> None:
        self._ensure_instruments()
        attrs = {"model_name": ctx.model_name, "fn": ctx.fn}
        self._success_counter.add(1, attrs)
        self._latency.record(ctx.elapsed(), attrs)
        super()._on_success(ctx, span, result)

    def _on_error(self, ctx: ReportContext, span: "Optional[Span]", exc: BaseException) -> None:
        self._ensure_instruments()
        attrs = {"model_name": ctx.model_name, "fn": ctx.fn, "exception_type": type(exc).__name__}
        self._error_counter.add(1, attrs)
        self._latency.record(ctx.elapsed(), attrs)
        super()._on_error(ctx, span, exc)

    def _on_end(self, ctx: ReportContext, span: "Optional[Span]") -> None:
        # The histogram is recorded on success/error; avoid double-counting here.
        ReportingPolicy._on_end(self, ctx, span)
