"""Node results and the provider interfaces that let each node run standalone.

Every node's output is a `ResultBase`, which is what the ccflow publishers serialize. Each node also
declares its upstream by *interface* rather than by concrete type: an advocate needs something that
yields a `PoemResult`, not specifically a `ProduceModel`. That is what allows a node to be run on its
own against a previously published file instead of re-running everything above it.

This module deliberately omits ``from __future__ import annotations``: ccflow resolves the
``__call__`` context type by ``issubclass``, and postponed evaluation leaves it a string.
"""

from pydantic import Field

from ccflow import BasePublisher, CallableModel, Flow, NullContext, ResultBase

from .contracts import Critique, Poem, Verdict

__all__ = (
    "CritiqueProvider",
    "CritiqueResult",
    "PoemProvider",
    "PoemResult",
    "PublishingModel",
    "VerdictProvider",
    "VerdictResult",
)

_NULL_CONTEXT = NullContext()


class PoemResult(ResultBase):
    """The produced artifact."""

    poem: Poem


class CritiqueResult(ResultBase):
    """One advocate's case."""

    critique: Critique


class VerdictResult(ResultBase):
    """The arbiter's decision, with everything it decided over."""

    verdict: Verdict
    poem: Poem
    critiques: list[Critique]


class PublishingModel(CallableModel):
    """A node that can optionally write its result somewhere via a ccflow publisher."""

    publisher: BasePublisher | None = Field(default=None, description="Where to write this node's result. Null writes nothing.")
    enforce_checks: bool = Field(
        default=True,
        description="Whether mechanical constraints are enforced in code. Turning this off leaves the judgement rubric in place but stops counting "
        "anything, which is only appropriate for wiring smoke tests.",
    )

    def _publish(self, result: ResultBase) -> ResultBase:
        if self.publisher is not None:
            self.publisher.data = result
            self.publisher()
        return result


class PoemProvider(PublishingModel):
    """Anything that yields the artifact: the live producer, or a reader over a published file."""

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> PoemResult:
        raise NotImplementedError


class CritiqueProvider(PublishingModel):
    """Anything that yields one advocate's case: the live advocate, or a reader over a file."""

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> CritiqueResult:
        raise NotImplementedError


class VerdictProvider(PublishingModel):
    """Anything that yields a decision: a single arbitration, or several reconciled into one."""

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> VerdictResult:
        raise NotImplementedError
