"""Readers that stand in for a live node by loading its published output.

These are what make the single-node tasks work. An advocate declares its upstream as a
``PoemProvider``, so it cannot tell whether the poem came from a model or from a file written by an
earlier run — and pinning the artifact to a file is the only way to re-judge *the same* poem under a
different model or rubric, since model output is not reproducible.

This module deliberately omits ``from __future__ import annotations``: ccflow resolves the
``__call__`` context type by ``issubclass``, and postponed evaluation leaves it a string.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ccflow import Flow, NullContext

from .contracts import Critique, Poem
from .results import CritiqueProvider, CritiqueResult, PoemProvider, PoemResult

__all__ = ("CritiqueFileSource", "PoemFileSource")

_NULL_CONTEXT = NullContext()


def _read(path: str, key: str, contract: type[BaseModel]) -> BaseModel:
    """Load `path`, accepting either a bare contract or the result wrapper that contains it."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"No published output at {file}. Run the upstream task with a publisher first.")
    payload = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and key in payload:
        payload = payload[key]
    return contract.model_validate(payload)


class PoemFileSource(PoemProvider):
    """Read a poem published by an earlier run."""

    path: str = Field(description="Path to the JSON written by the produce task.")

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> PoemResult:
        return self._publish(PoemResult(poem=_read(self.path, "poem", Poem)))


class CritiqueFileSource(CritiqueProvider):
    """Read one advocate's case published by an earlier run."""

    path: str = Field(description="Path to the JSON written by an advocate task.")

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> CritiqueResult:
        return self._publish(CritiqueResult(critique=_read(self.path, "critique", Critique)))
