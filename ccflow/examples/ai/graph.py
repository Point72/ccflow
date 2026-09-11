"""The agent graph: produce a poem, argue both sides of it, adjudicate.

Node wiring is model composition, not sequential Python. Each node holds its upstream as a *field*,
declares it in ``__deps__`` so the evaluator can schedule and cache it, and invokes it in
``__call__``.

Upstreams are declared by interface (``PoemProvider``, ``CritiqueProvider``), so any node can be run
on its own against a published file instead of the live node above it. That is what makes the
single-node tasks possible without a second copy of the graph.

Every context here is ``NullContext``. That is deliberate and is what makes the graph legal:
``__deps__`` must be resolvable *before* the dependency runs, so a context cannot carry an upstream
node's output. Data flows through composition instead, and the cache evaluator configured in
``config/base.yaml`` is what stops ``produce`` running once per advocate — both advocates call the
same model instance with the same context, so the second call is served from cache.

Two behaviours exist because they were measured, not because they seemed prudent. Replaying a fixed
set of cases through the arbiter with only the *presentation* varied moved the outcome: most
decisions followed whichever case was shown first, and naming the positions shifted them again. So
positions are blinded by default, self-rated confidence is withheld, and
``CounterbalancedArbitrateModel`` judges both orders and reports only what survives both.

This module deliberately omits ``from __future__ import annotations``: ccflow resolves the
``__call__`` context type by ``issubclass``, and postponed evaluation leaves it a string.
"""

import re
from functools import reduce

from pydantic import Field

from ccflow import Flow, GraphDepList, NullContext

from .checks import ClaimsWithinLimit, EvidenceAppearsInArtifact, LinesWithinLimit, PositionAsAssigned, VerdictWellFormed
from .contracts import (
    ArbitrationBrief,
    ArbitrationRubric,
    Critique,
    CritiqueBrief,
    CritiqueRubric,
    Poem,
    PoemSpec,
    PositionScore,
    RebuttalBrief,
    Verdict,
)
from .results import CritiqueProvider, CritiqueResult, PoemProvider, PoemResult, PublishingModel, VerdictProvider, VerdictResult
from .session import AgentSession, AgentSessionContext

__all__ = (
    "AdvocateModel",
    "ArbitrateModel",
    "CounterbalancedArbitrateModel",
    "DebateModel",
    "ProduceModel",
    "RebuttalModel",
)

_NULL_CONTEXT = NullContext()


def _advocate_checks(poem: Poem, position: str, rubric: CritiqueRubric) -> list:
    """Mechanical constraints on any case about `poem`, whether first round or rebuttal."""
    # The quotable text is everything the advocate was shown, not just the stanzas: the brief carries
    # the title and notes too, and quoting them is legitimate.
    quotable = " ".join([poem.title, *poem.stanzas, poem.notes])
    return [
        ClaimsWithinLimit(limit=rubric.max_claims),
        PositionAsAssigned(expected=position),
        EvidenceAppearsInArtifact(artifact_text=quotable, required=rubric.require_evidence),
    ]


class ProduceModel(PoemProvider):
    """Produce the poem from its specification."""

    session: AgentSession
    spec: PoemSpec

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> PoemResult:
        checks = [LinesWithinLimit(limit=self.spec.max_lines)] if self.enforce_checks else []
        result = self.session(AgentSessionContext(prompt=self.spec.model_dump_json(indent=2), checks=checks))
        return self._publish(PoemResult(poem=result.output))


class AdvocateModel(CritiqueProvider):
    """Argue one assigned position about the produced poem.

    The position is assigned, not chosen: an advocate argues its side regardless of its own view,
    which is what makes the pair adversarial rather than two independent opinions.
    """

    session: AgentSession
    produce: PoemProvider
    position: str = Field(description="Position to argue, e.g. 'pro' or 'con'.")
    rubric: CritiqueRubric

    @Flow.deps
    def __deps__(self, context: NullContext) -> GraphDepList:
        return [(self.produce, [NullContext()])]

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> CritiqueResult:
        poem = self.produce(NullContext()).poem
        brief = CritiqueBrief[Poem](artifact=poem, position=self.position, rubric=self.rubric)
        checks = _advocate_checks(poem, self.position, self.rubric) if self.enforce_checks else []
        result = self.session(AgentSessionContext(prompt=brief.model_dump_json(indent=2), checks=checks))
        return self._publish(CritiqueResult(critique=result.output))


class RebuttalModel(CritiqueProvider):
    """Revise one advocate's case after it has read the opposing one.

    A first-round case is written blind, so both sides can spend claims on points the other has
    already answered. Rebuttal is where a case earns its keep: a claim that has been met should be
    dropped or defended rather than repeated.

    The output is a `Critique` like any other, so a rebuttal round drops in wherever a first-round
    advocate would go, including underneath a counterbalanced arbiter.
    """

    session: AgentSession
    own: CritiqueProvider = Field(description="This advocate's first-round case.")
    opponent: CritiqueProvider = Field(description="The case being answered.")
    artifact: PoemProvider
    position: str = Field(description="Position to keep arguing. Revising is not conceding.")
    rubric: CritiqueRubric

    @Flow.deps
    def __deps__(self, context: NullContext) -> GraphDepList:
        return [(self.own, [NullContext()]), (self.opponent, [NullContext()]), (self.artifact, [NullContext()])]

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> CritiqueResult:
        poem = self.artifact(NullContext()).poem
        brief = RebuttalBrief[Poem](
            artifact=poem,
            position=self.position,
            rubric=self.rubric,
            own_case=self.own(NullContext()).critique,
            opposing_case=self.opponent(NullContext()).critique,
        )
        checks = _advocate_checks(poem, self.position, self.rubric) if self.enforce_checks else []
        # Withheld for the same reason the arbiter does not see it: a self-rating is a claim about a
        # case, not evidence for it, and anchoring a revision on one invites deference.
        prompt = brief.model_dump_json(indent=2, exclude={"own_case": {"confidence"}, "opposing_case": {"confidence"}})
        result = self.session(AgentSessionContext(prompt=prompt, checks=checks))
        return self._publish(CritiqueResult(critique=result.output))


class ArbitrateModel(VerdictProvider):
    """Weigh the advocates' cases and decide."""

    session: AgentSession
    advocates: list[CritiqueProvider] = Field(min_length=1, description="Cases to judge. Not limited to two.")
    rubric: ArbitrationRubric
    artifact: PoemProvider = Field(
        description="The poem being judged: the producer the advocates read, or a file reader when arbitration runs alone."
    )
    blind_positions: bool = Field(
        default=True,
        description="Hide position names behind neutral labels while judging. Measured on fixed cases, naming the positions shifts the winner.",
    )

    @Flow.deps
    def __deps__(self, context: NullContext) -> GraphDepList:
        return [(advocate, [NullContext()]) for advocate in self.advocates] + [(self.artifact, [NullContext()])]

    def _blind(self, critiques: list[Critique]) -> tuple[list[Critique], dict[str, str]]:
        """Return the cases as the arbiter sees them, and a map from shown label back to position."""
        if not self.blind_positions:
            return critiques, {c.position: c.position for c in critiques}
        labels = [f"case_{i}" for i in range(1, len(critiques) + 1)]
        shown = [c.model_copy(update={"position": label}) for label, c in zip(labels, critiques)]
        return shown, dict(zip(labels, (c.position for c in critiques)))

    @staticmethod
    def _unblind(verdict: Verdict, real: dict[str, str]) -> Verdict:
        # Tolerant of how the arbiter rewrites a label in prose: "case_1", "Case 1", "case-1".
        separator = r"[\s_-]*"
        patterns = [(r"\b" + re.escape(label).replace("_", separator) + r"\b", position) for label, position in real.items()]

        def restore(text: str) -> str:
            return reduce(lambda t, kv: re.sub(kv[0], kv[1], t, flags=re.IGNORECASE), patterns, text)

        # Matched the way `VerdictWellFormed` validates, so a label the check accepted cannot then
        # fail to resolve here and reach the caller unrestored.
        by_label = {label.strip().lower(): position for label, position in real.items()}

        def position_of(name: str) -> str:
            return by_label.get(name.strip().lower(), name)

        scores = [s.model_copy(update={"position": position_of(s.position), "reasoning": restore(s.reasoning)}) for s in verdict.scores]
        winner = position_of(verdict.winner) if verdict.winner else None
        return verdict.model_copy(update={"scores": scores, "winner": winner, "rationale": restore(verdict.rationale)})

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> VerdictResult:
        critiques: list[Critique] = [advocate(NullContext()).critique for advocate in self.advocates]
        poem = self.artifact(NullContext()).poem
        shown, real = self._blind(critiques)
        brief = ArbitrationBrief[Poem](artifact=poem, critiques=shown, rubric=self.rubric)
        checks = (
            [VerdictWellFormed(positions=[c.position for c in shown], allow_no_decision=self.rubric.allow_no_decision)] if self.enforce_checks else []
        )
        # An advocate's self-rated confidence is a claim about its case, not evidence for it, and an
        # arbiter was observed discounting the case that rated itself lower. Withhold it.
        prompt = brief.model_dump_json(indent=2, exclude={"critiques": {"__all__": {"confidence"}}})
        result = self.session(AgentSessionContext(prompt=prompt, checks=checks))
        return self._publish(VerdictResult(verdict=self._unblind(result.output, real), poem=poem, critiques=critiques))


class CounterbalancedArbitrateModel(VerdictProvider):
    """Judge the same cases in both presentation orders and report only what survives both.

    Presentation order measurably moves the winner, so a single-order verdict confounds the strength
    of a case with the slot it happened to occupy. Agreement across both orders is the weakest claim
    worth reporting; disagreement is recorded as undecided rather than resolved, because a winner
    that depends on presentation order is an artifact of the harness.

    Both arbiters read the same advocate models, so the cases are generated once and only the
    arbitration is repeated: counterbalancing costs one extra call, not a second graph.
    """

    forward: ArbitrateModel = Field(description="Arbiter seeing the advocates in configured order.")
    reverse: ArbitrateModel = Field(description="Arbiter seeing the same advocates reversed.")

    @Flow.deps
    def __deps__(self, context: NullContext) -> GraphDepList:
        return [(self.forward, [NullContext()]), (self.reverse, [NullContext()])]

    @staticmethod
    def _reconcile(first: Verdict, second: Verdict) -> Verdict:
        agreed = first.winner == second.winner
        by_position: dict[str, list[PositionScore]] = {}
        for verdict, label in ((first, "shown in configured order"), (second, "shown reversed")):
            for score in verdict.scores:
                by_position.setdefault(score.position, []).append(score.model_copy(update={"reasoning": f"[{label}] {score.reasoning}"}))
        scores = [
            PositionScore(position=position, score=sum(s.score for s in seen) / len(seen), reasoning="\n\n".join(s.reasoning for s in seen))
            for position, seen in by_position.items()
        ]
        if not agreed:
            headline = (
                f"The orders disagreed -- {first.winner} in configured order, {second.winner} reversed -- so this is "
                "recorded as undecided. A winner that depends on presentation order is an artifact of the harness."
            )
        elif first.winner is None:
            headline = "Both presentation orders left this undecided."
        else:
            headline = f"Both presentation orders chose {first.winner}."
        return Verdict(
            winner=first.winner if agreed else None,
            scores=scores,
            rationale=f"{headline}\n\n[configured order] {first.rationale}\n\n[reversed order] {second.rationale}",
            # Agreement alone is not decisiveness: both arbiters must also have found their own call clear.
            decisive=agreed and first.decisive and second.decisive,
        )

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> VerdictResult:
        first, second = self.forward(NullContext()), self.reverse(NullContext())
        return self._publish(VerdictResult(verdict=self._reconcile(first.verdict, second.verdict), poem=first.poem, critiques=first.critiques))


class DebateModel(PublishingModel):
    """The task handoff: run the whole graph and return the verdict."""

    arbiter: VerdictProvider = Field(description="A single arbitration, or several reconciled into one.")

    @Flow.deps
    def __deps__(self, context: NullContext) -> GraphDepList:
        return [(self.arbiter, [NullContext()])]

    @Flow.call
    def __call__(self, context: NullContext = _NULL_CONTEXT) -> VerdictResult:
        return self._publish(self.arbiter(NullContext()))
