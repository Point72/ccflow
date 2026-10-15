"""Constraints decided in code rather than by asking a model.

Constraints split into two kinds, and conflating them is how a graph ends up trusting a judgement
about a countable fact. An arbiter asked whether a poem respected a line budget has to *count*, and
a model that miscounts produces a confident verdict resting on a false premise.

Everything here is the mechanical kind: a line budget, a claim cap, whether a quotation actually
occurs in the text it was supposedly taken from. Judgement criteria -- clarity, originality, whether
a case is well argued -- stay in the rubric, where a model reasons about them.

A violation is raised through the agent's output validator, so it consumes the same repair budget as
a schema failure rather than introducing a second failure path.
"""

import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

__all__ = (
    "ClaimsWithinLimit",
    "EvidenceAppearsInArtifact",
    "LinesWithinLimit",
    "OutputCheck",
    "PositionAsAssigned",
    "VerdictWellFormed",
    "run_checks",
)


class OutputCheck(BaseModel, ABC):
    """A constraint decidable without asking a model.

    Abstract so a subclass that misspells ``violations`` fails at class definition rather than at the
    moment a constraint should have been enforced.
    """

    @abstractmethod
    def violations(self, output) -> list[str]:
        """Every way ``output`` breaks this constraint, phrased so a model can act on it."""


def run_checks(checks, output) -> list[str]:
    """Every violation across every check, so one re-prompt can address them all."""
    return [problem for check in checks for problem in check.violations(output)]


def _normalize(text: str) -> str:
    """Collapse whitespace, drop surrounding quotes and trailing ellipsis, lowercase."""
    text = text.strip().strip("\"'\u201c\u201d\u2018\u2019")
    text = re.sub(r"[.\u2026]+$", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _poem_lines(poem) -> list[str]:
    return [line for stanza in poem.stanzas for line in stanza.splitlines() if line.strip()]


class LinesWithinLimit(OutputCheck):
    """The produced artifact must not exceed the line budget its specification declared."""

    limit: int = Field(ge=1, description="Maximum number of non-empty lines.")

    def violations(self, output) -> list[str]:
        count = len(_poem_lines(output))
        if count > self.limit:
            return [f"the poem has {count} non-empty lines but max_lines is {self.limit}"]
        return []


class ClaimsWithinLimit(OutputCheck):
    """An advocate must respect the claim cap its rubric declared."""

    limit: int = Field(ge=1, description="Maximum number of claims.")

    def violations(self, output) -> list[str]:
        count = len(output.claims)
        if count > self.limit:
            return [f"you made {count} claims but max_claims is {self.limit}; keep only the {self.limit} strongest"]
        return []


class EvidenceAppearsInArtifact(OutputCheck):
    """Every quoted string must actually occur in the text the advocate was shown.

    This proves a quotation is real. It cannot tell whether the quotation supports the claim -- that
    remains a judgement, and is the arbiter's job.
    """

    artifact_text: str = Field(description="Everything the advocate was shown, which is what it may legitimately quote.")
    required: bool = Field(default=True, description="Whether a claim must quote at all.")

    def violations(self, output) -> list[str]:
        haystack = _normalize(self.artifact_text)
        problems = []
        for claim in output.claims:
            quoted = claim.evidence.strip()
            if not quoted:
                if self.required:
                    problems.append(f"the claim {claim.statement!r} quotes nothing; every claim must quote the artifact")
                continue
            if _normalize(quoted) not in haystack:
                problems.append(f"the quotation {quoted!r} does not appear in the artifact; quote it exactly or drop the claim")
        return problems


class PositionAsAssigned(OutputCheck):
    """An advocate must report the side it was assigned, not the side it came to prefer."""

    expected: str = Field(description="The assigned position.")

    def violations(self, output) -> list[str]:
        if output.position.strip().lower() != self.expected.strip().lower():
            return [f"you reported position {output.position!r} but were assigned {self.expected!r}"]
        return []


class VerdictWellFormed(OutputCheck):
    """The verdict must be about the positions that were actually argued."""

    positions: list[str] = Field(description="Positions placed before the arbiter.")
    allow_no_decision: bool = Field(default=True, description="Whether a null winner is permitted.")

    def violations(self, output) -> list[str]:
        expected = {p.strip().lower() for p in self.positions}
        scored = {s.position.strip().lower() for s in output.scores}
        problems = []
        if missing := expected - scored:
            problems.append(f"no score for {sorted(missing)}; score every position argued")
        if extra := scored - expected:
            problems.append(f"scored {sorted(extra)}, which nobody argued; score only {sorted(expected)}")
        if output.winner is None:
            if not self.allow_no_decision:
                problems.append("this rubric requires a decision, so winner cannot be null")
        elif output.winner.strip().lower() not in expected:
            problems.append(f"winner {output.winner!r} is not one of the positions argued, {sorted(expected)}")
        return problems
