"""Functional calculator stages built with the ``@Flow.model`` API.

Each calculator reads its input operands from the runtime context and exposes
its own configuration as ordinary bound parameters. This is the split that makes
the functions interchangeable from configuration: the context carries *what to
compute on*, while the bound fields carry *how to compute*.
"""

from pydantic import Field

from ccflow import ContextBase, Flow, FromContext

__all__ = ("Numbers", "add", "lower_gap", "mean", "power", "rounded", "scale", "tail_ratio", "upper_gap")


class Numbers(ContextBase):
    """Runtime input for the calculators: the operands to compute on."""

    values: list[float] = Field(default_factory=list)


@Flow.model(context_type=Numbers)
def add(values: FromContext[list[float]], offset: float = 0.0) -> float:
    """Sum the input values, then add a configurable offset."""
    return sum(values) + offset


@Flow.model(context_type=Numbers)
def scale(values: FromContext[list[float]], factor: float = 1.0) -> float:
    """Sum the input values, then multiply by a configurable factor."""
    return sum(values) * factor


@Flow.model(context_type=Numbers)
def power(values: FromContext[list[float]], exponent: float = 2.0) -> float:
    """Raise each input value to a configurable exponent and sum the results."""
    return sum(value**exponent for value in values)


@Flow.model
def rounded(value: float, digits: int = 2) -> float:
    """Round an upstream calculator's result to a configurable precision.

    ``value`` is a regular parameter, so it can be bound to a literal or fed by
    another calculator model wired in through configuration.
    """
    return round(value, digits)


@Flow.model(context_type=Numbers)
def mean(values: FromContext[list[float]]) -> float:
    """Average of the input values."""
    return sum(values) / len(values)


@Flow.model(context_type=Numbers)
def upper_gap(values: FromContext[list[float]], center: float) -> float:
    """How far the largest value sits above ``center`` (typically the mean)."""
    return max(values) - center


@Flow.model(context_type=Numbers)
def lower_gap(values: FromContext[list[float]], center: float) -> float:
    """How far the smallest value sits below ``center`` (typically the mean)."""
    return center - min(values)


@Flow.model
def tail_ratio(upper: float, lower: float) -> float:
    """Ratio of the upper gap to the lower gap.

    ``upper`` and ``lower`` are regular parameters fed by ``upper_gap`` and
    ``lower_gap``, which both depend on ``mean`` — a diamond whose shared node
    the graph evaluator computes once.
    """
    return upper / lower
