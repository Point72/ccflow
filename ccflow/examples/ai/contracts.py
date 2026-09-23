"""Typed hand-off contracts for the poem review graph.

Every edge in the graph is one of these models. A node never hands free text to the next node: it
hands an instance of a declared contract, and the model that produced it was held to that contract's
schema.

The field descriptions are load-bearing. They are serialized into the JSON schema the language model
is constrained by, so they are written as instructions to the model rather than as notes to a reader.

This module deliberately omits ``from __future__ import annotations``: ccflow resolves a callable
model's context type by ``issubclass``, and postponed evaluation leaves the annotation a string.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

__all__ = (
    "ArbitrationBrief",
    "ArbitrationRubric",
    "ArtifactT",
    "Claim",
    "Critique",
    "CritiqueBrief",
    "CritiqueRubric",
    "Lessons",
    "Poem",
    "PoemSpec",
    "PositionScore",
    "RebuttalBrief",
    "Verdict",
)

#: The artifact under production and debate. Generic so the same graph can be pointed at a different
#: kind of artifact by swapping this type, without touching any node.
ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


class PoemSpec(BaseModel):
    """What the producing agent is asked to write."""

    theme: str = Field(description="Subject the poem must be about.")
    form: str = Field(default="free verse", description="Poetic form, e.g. 'sonnet', 'haiku', 'free verse'.")
    max_lines: int = Field(default=24, ge=1, description="Hard upper bound on total lines across all stanzas.")
    constraints: list[str] = Field(default_factory=list, description="Additional requirements the poem must satisfy.")


class Poem(BaseModel):
    """The produced artifact."""

    title: str = Field(description="Title of the poem.")
    stanzas: list[str] = Field(description="Stanzas in order. Each entry is one stanza, lines separated by newlines.")
    notes: str = Field(default="", description="Brief note on the choices made, for the critics to engage with.")


class CritiqueRubric(BaseModel):
    """How an advocate is told to build its case. Configuration, not prose buried in a prompt."""

    criteria: list[str] = Field(description="Dimensions to argue on, e.g. 'imagery', 'adherence to form'.")
    max_claims: int = Field(default=4, ge=1, description="Maximum number of claims to make.")
    require_evidence: bool = Field(default=True, description="Whether every claim must quote the artifact.")


class Lessons(BaseModel):
    """Standing corrections carried between runs, so a mistake made once is not made again.

    Scoped by role because most lessons are not universal: a rule about which words to avoid belongs
    to whoever writes the poem, not to the arbiter scoring the arguments about it. A lesson applied
    to the wrong role is worse than no lesson, because it spends attention and can mislead.

    Lessons are guidance, not enforcement. Anything decidable mechanically belongs in ``checks``,
    where it is measured rather than asked for.
    """

    general: list[str] = Field(default_factory=list, description="Applies to every role.")
    by_role: dict[str, list[str]] = Field(default_factory=dict, description="Extra guidance keyed by role name: producer, pro, con, arbiter.")

    def for_role(self, role: str) -> list[str]:
        """General lessons first, then any specific to this role."""
        return [*self.general, *self.by_role.get(role, [])]


class Claim(BaseModel):
    """A single argued point."""

    statement: str = Field(description="The claim being made, in one sentence.")
    evidence: str = Field(description="Direct quotation from the artifact supporting the claim.")
    criterion: str = Field(description="Which rubric criterion this claim addresses.")
    weight: float = Field(ge=0.0, le=1.0, description="How much this claim should count, 0 to 1.")


class Critique(BaseModel):
    """An advocate's case for its assigned position."""

    position: str = Field(description="The position argued, exactly as assigned.")
    claims: list[Claim] = Field(description="The claims making up the case, strongest first.")
    summary: str = Field(description="One-paragraph summary of the case.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the case, 0 to 1.")


class ArbitrationRubric(BaseModel):
    """How the arbiter is told to decide."""

    criteria: list[str] = Field(description="What makes an argument strong, in priority order.")
    allow_no_decision: bool = Field(default=True, description="Whether 'undecided' is permitted. Forcing a binary on a genuine tie produces noise.")


class PositionScore(BaseModel):
    """The arbiter's score for one position."""

    position: str = Field(description="Position being scored.")
    score: float = Field(ge=0.0, le=1.0, description="Strength of this position's case, 0 to 1.")
    reasoning: str = Field(description="Why this score.")


class Verdict(BaseModel):
    """The arbiter's decision."""

    winner: str | None = Field(description="Winning position, or null if genuinely undecided.")
    scores: list[PositionScore] = Field(description="One score per position considered.")
    rationale: str = Field(description="Why the winner won, referencing specific claims.")
    decisive: bool = Field(description="False when the positions were close enough that the decision could go either way.")


class CritiqueBrief(BaseModel, Generic[ArtifactT]):
    """Everything an advocate needs: the artifact, the position to argue, and the rubric."""

    artifact: ArtifactT
    position: str = Field(description="Position this advocate must argue, regardless of its own view.")
    rubric: CritiqueRubric


class RebuttalBrief(BaseModel, Generic[ArtifactT]):
    """What an advocate needs to revise its case after reading the opposing one."""

    artifact: ArtifactT
    position: str = Field(description="Position this advocate must still argue. Revising is not conceding.")
    rubric: CritiqueRubric
    own_case: Critique = Field(description="This advocate's first-round case, to be revised.")
    opposing_case: Critique = Field(description="The case now being answered.")


class ArbitrationBrief(BaseModel, Generic[ArtifactT]):
    """Everything the arbiter needs: the artifact, every critique, and the decision rubric."""

    artifact: ArtifactT
    critiques: list[Critique]
    rubric: ArbitrationRubric
