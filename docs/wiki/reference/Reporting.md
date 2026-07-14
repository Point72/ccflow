# Reporting

`ccflow` reporting records model-evaluation lifecycle events without changing successful results. Reporting can be attached across an evaluation scope with an evaluator or to one graph node with a wrapper model.

## Events

`ReportEvent` is the serializable event type. Its fields are:

| Field                                 | Type                      | Meaning                                           |
| ------------------------------------- | ------------------------- | ------------------------------------------------- |
| `phase`                               | `ReportPhase`             | Lifecycle phase represented by the event.         |
| `model_name`                          | `str`                     | Model metadata name or class name.                |
| `model_type`                          | `str`                     | Fully qualified model type.                       |
| `fn`                                  | `str`                     | Evaluated function; defaults to `__call__`.       |
| `context_repr`                        | `str`                     | Bounded representation of the evaluation context. |
| `span_id`                             | `str`                     | Identifier for this evaluation.                   |
| `parent_span_id`                      | `Optional[str]`           | Enclosing evaluation identifier.                  |
| `run_id`                              | `Optional[str]`           | Active run identifier.                            |
| `depth`                               | `int`                     | Evaluation-tree depth.                            |
| `timestamp`                           | `datetime`                | Event creation time in UTC.                       |
| `duration`                            | `Optional[float]`         | Elapsed seconds for terminal phases.              |
| `attempt`, `max_attempts`             | `Optional[int]`           | Retry lifecycle metadata.                         |
| `exception_type`, `exception_message` | `Optional[str]`           | Failure metadata.                                 |
| `priority`                            | `Optional[AlertPriority]` | Alert priority from `P1` through `P5`.            |
| `extra`                               | `dict[str, Any]`          | Signal- or sink-specific metadata.                |

Node phases are `START`, `SUCCESS`, `ERROR`, `RETRY`, `GIVE_UP`, `END`, `QUEUED`, and `SKIPPED`. `RUN_STARTED`, `RUN_FINISHED`, and `GRAPH_DISCOVERED` are reserved for run and graph observers and are not emitted by evaluator or model reporting paths.

`run_scope(run_id=None)` associates events with a logical run. Nested scopes reuse the outer run identifier unless an explicit identifier is supplied.

## Reporters

Reporters consume `ReportEvent` instances through `emit(event)`.

| Reporter            | Behavior                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `Reporter`          | Base no-op sink.                                                                                         |
| `NoOpReporter`      | Explicitly discards events.                                                                              |
| `InMemoryReporter`  | Retains events in emission order; `clear()` empties the buffer.                                          |
| `LoggingReporter`   | Writes structured lifecycle events to a Python logger.                                                   |
| `CompositeReporter` | Fans events out to child reporters and isolates child failures.                                          |
| `UIReporter`        | Stores events in a thread-safe bounded buffer; `drain()` atomically returns and removes buffered events. |

`ReportingPolicy` accepts these common fields:

| Field                  | Default | Meaning                                                             |
| ---------------------- | ------- | ------------------------------------------------------------------- |
| `reporter`             | `None`  | Event sink. With no sink, structured event construction is skipped. |
| `capture_context_repr` | `True`  | Includes a bounded context representation in events.                |
| `max_context_repr`     | `200`   | Maximum context-representation length.                              |

Reporter failures do not alter model results or replace model exceptions.

## Evaluators

Reporting evaluators are runtime, cross-cutting configuration:

- `ReportingEvaluator` emits standard lifecycle events.
- `TracingReportingEvaluator` emits span-oriented lifecycle events.
- `MetricsReportingEvaluator` adds count and latency metric metadata.
- `AlertsReportingEvaluator` emits priority-tagged failure events and optional recovery events.
- `OpenTelemetryTracingReportingEvaluator` creates OpenTelemetry spans.
- `OpenTelemetryMetricsReportingEvaluator` records OpenTelemetry counters and histograms.
- `OpenTelemetryEvaluator` aliases `OpenTelemetryTracingReportingEvaluator`.

These evaluators are transparent: successful evaluations return the wrapped result unchanged.

`DryRunEvaluator`, defined in `ccflow.evaluators.dry_run`, is separate from the reporting evaluator hierarchy. It discovers the dependency graph without executing model bodies and composes a `ReportingPolicy` through its `reporting` field. With a configured reporter it emits `QUEUED` and `SKIPPED` events. Its default unvalidated synthetic result is for planning only; setting `synthetic_result=False` executes the wrapped evaluation after planning.

## Reporting models

Reporting models attach reporting structurally to one wrapped model:

- `ReportingModel`
- `LoggingModel`
- `TracingReportingModel`
- `MetricsReportingModel`
- `AlertsReportingModel`
- `OpenTelemetryTracingReportingModel`
- `OpenTelemetryMetricsReportingModel`
- `OpenTelemetryModel`, an alias for `OpenTelemetryTracingReportingModel`

These types are first-class graph nodes and inherit the wrapped model's context and result types.

## OpenTelemetry

OpenTelemetry tracing and metrics use the `opentelemetry-api` package. The tracing policy accepts `tracer_name`; the metrics policy accepts `meter_name` and `metric_prefix`. SDK configuration and exporters remain application concerns.

## State projection

`ReportingStateStore` folds events into per-run and per-node state for observability consumers. `roots()` returns nodes without a parent in the store, and `children(span_id)` returns direct descendants. `UIReporter` supplies the bounded event buffer used to feed this projection.
