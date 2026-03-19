#!/usr/bin/env python
"""
Evaluator Demo: Caching & Execution Strategies
===============================================

Shows how to change execution behavior (caching, graph evaluation, logging)
WITHOUT changing user code. The same @Flow.model functions work with any
evaluator stack — you just configure it at the top level.

Key insight: "default lazy" is an evaluator concern, not a wiring concern.
Users write plain functions and wire them by passing outputs as inputs.
The evaluator layer controls how they execute.

Demonstrates:
    1. Default execution (eager, no caching) — diamond dep calls load twice
    2. MemoryCacheEvaluator — deduplicates shared deps in a diamond
    3. GraphEvaluator + Cache — topological evaluation + deduplication
    4. LoggingEvaluator — adds tracing around every model call
    5. Per-model opt-out — @Flow.model(cacheable=False) overrides global

Run with: python examples/evaluator_demo.py
"""

from __future__ import annotations

import logging
import sys

# Suppress default debug logging from ccflow evaluators for clean demo output
logging.disable(logging.DEBUG)

from ccflow import Flow, FlowOptionsOverride  # noqa: E402
from ccflow.evaluators.common import (  # noqa: E402
    GraphEvaluator,
    LoggingEvaluator,
    MemoryCacheEvaluator,
    MultiEvaluator,
)

# =============================================================================
# Plain @Flow.model functions — no evaluator concerns in the code
# =============================================================================

call_counts: dict[str, int] = {}


def _track(name: str) -> None:
    call_counts[name] = call_counts.get(name, 0) + 1


@Flow.model
def load_data(x: int, source: str = "warehouse") -> list:
    """Load raw data. Expensive — we want to avoid calling this twice."""
    _track("load_data")
    return [x, x * 2, x * 3]


@Flow.model
def compute_sum(data: list) -> int:
    """Branch A: sum the data."""
    _track("compute_sum")
    return sum(data)


@Flow.model
def compute_max(data: list) -> int:
    """Branch B: max of the data."""
    _track("compute_max")
    return max(data)


@Flow.model
def combine(sum_result: int, max_result: int) -> dict:
    """Combine results from both branches."""
    _track("combine")
    return {"sum": sum_result, "max": max_result, "total": sum_result + max_result}


@Flow.model(cacheable=False)
def volatile_timestamp(seed: int) -> str:
    """Explicitly non-cacheable — always re-executes even with global caching."""
    _track("volatile_timestamp")
    from datetime import datetime

    return datetime.now().isoformat()


# =============================================================================
# Wire the pipeline — diamond dependency on load_data
#
#   load_data ──┬── compute_sum ──┐
#               └── compute_max ──┴── combine
# =============================================================================

shared = load_data(source="prod")
branch_a = compute_sum(data=shared)
branch_b = compute_max(data=shared)
pipeline = combine(sum_result=branch_a, max_result=branch_b)


def run() -> dict:
    call_counts.clear()
    result = pipeline.flow.compute(x=5)
    loads = call_counts.get("load_data", 0)
    print(f"  Result: {result.value}")
    print(f"  load_data called: {loads}x | total model calls: {sum(call_counts.values())}")
    return result.value


# =============================================================================
# Demo 1: Default — no evaluator
# =============================================================================

print("=" * 70)
print("1. Default (eager, no caching)")
print("   load_data is called TWICE — once per branch")
print("=" * 70)
run()

# =============================================================================
# Demo 2: MemoryCacheEvaluator — deduplicates shared deps
# =============================================================================

print()
print("=" * 70)
print("2. MemoryCacheEvaluator (global override)")
print("   load_data is called ONCE — second branch hits cache")
print("=" * 70)
with FlowOptionsOverride(options={"evaluator": MemoryCacheEvaluator(), "cacheable": True}):
    run()

# =============================================================================
# Demo 3: Cache + GraphEvaluator — topological order + deduplication
# =============================================================================

print()
print("=" * 70)
print("3. GraphEvaluator + MemoryCacheEvaluator")
print("   Evaluates in dependency order: load_data → branches → combine")
print("=" * 70)
evaluator = MultiEvaluator(evaluators=[MemoryCacheEvaluator(), GraphEvaluator()])
with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
    run()

# =============================================================================
# Demo 4: Logging — trace every model call
# =============================================================================

print()
print("=" * 70)
print("4. LoggingEvaluator + MemoryCacheEvaluator")
print("   Adds timing/tracing around every evaluation")
print("=" * 70)

# Re-enable logging for this demo (use stdout so log lines interleave with print correctly)
logging.disable(logging.NOTSET)
logging.basicConfig(level=logging.INFO, format="    LOG: %(message)s", stream=sys.stdout)

evaluator = MultiEvaluator(evaluators=[LoggingEvaluator(log_level=logging.INFO), MemoryCacheEvaluator()])
with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
    run()

# Suppress again for clean output
logging.disable(logging.DEBUG)
logging.getLogger().handlers.clear()

# =============================================================================
# Demo 5: Per-model opt-out — cacheable=False overrides global
# =============================================================================

print()
print("=" * 70)
print("5. Per-model opt-out: @Flow.model(cacheable=False)")
print("   volatile_timestamp always re-executes despite global cacheable=True")
print("=" * 70)

ts = volatile_timestamp(seed=0)

with FlowOptionsOverride(options={"evaluator": MemoryCacheEvaluator(), "cacheable": True}):
    call_counts.clear()
    r1 = ts.flow.compute(seed=0)
    r2 = ts.flow.compute(seed=0)
    print(f"  Call 1: {r1.value}")
    print(f"  Call 2: {r2.value}")
    print(f"  volatile_timestamp called: {call_counts.get('volatile_timestamp', 0)}x")
    print(f"  (Same result? {r1.value == r2.value} — called twice, timestamps may differ)")
