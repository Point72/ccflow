"""Tests for the mechanical constraints in ``ccflow.ai.checks``.

These are pure functions over already-built contract instances, so nothing here calls a model.
"""

from unittest import TestCase

from ccflow.ai import (
    Claim,
    ClaimsWithinLimit,
    Critique,
    EvidenceAppearsInArtifact,
    LinesWithinLimit,
    OutputCheck,
    Poem,
    PositionAsAssigned,
    PositionScore,
    Verdict,
    VerdictWellFormed,
    run_checks,
)

POEM = Poem(title="First Light", stanzas=["the water turns\nfoam at the rocks", "a buoy blinks"], notes="four stanzas, no end rhyme")
QUOTABLE = " ".join([POEM.title, *POEM.stanzas, POEM.notes])


def _critique(position="pro", evidence="foam at the rocks", claims=None):
    return Critique(
        position=position,
        claims=claims if claims is not None else [Claim(statement="the imagery is concrete", evidence=evidence, criterion="imagery", weight=0.5)],
        summary="summary",
        confidence=0.5,
    )


class TestLinesWithinLimit(TestCase):
    def test_within_budget_passes(self):
        self.assertEqual(LinesWithinLimit(limit=3).violations(POEM), [])

    def test_over_budget_reports_both_numbers(self):
        problems = LinesWithinLimit(limit=2).violations(POEM)
        self.assertEqual(len(problems), 1)
        self.assertIn("3", problems[0])
        self.assertIn("2", problems[0])

    def test_blank_lines_are_not_counted(self):
        poem = Poem(title="t", stanzas=["a\n\n\nb"], notes="")
        self.assertEqual(LinesWithinLimit(limit=2).violations(poem), [])


class TestClaimsWithinLimit(TestCase):
    def test_within_cap_passes(self):
        self.assertEqual(ClaimsWithinLimit(limit=1).violations(_critique()), [])

    def test_over_cap_is_reported(self):
        critique = _critique(claims=[Claim(statement=f"s{i}", evidence="a buoy blinks", criterion="imagery", weight=0.5) for i in range(3)])
        self.assertEqual(len(ClaimsWithinLimit(limit=2).violations(critique)), 1)


class TestEvidenceAppearsInArtifact(TestCase):
    def test_a_real_quotation_passes(self):
        self.assertEqual(EvidenceAppearsInArtifact(artifact_text=QUOTABLE).violations(_critique(evidence="a buoy blinks")), [])

    def test_an_invented_quotation_is_caught(self):
        check = EvidenceAppearsInArtifact(artifact_text=QUOTABLE)
        self.assertEqual(len(check.violations(_critique(evidence="gulls stitch the air"))), 1)

    def test_the_title_and_notes_are_quotable(self):
        """An advocate is shown the whole brief, so quoting the title or notes is legitimate."""
        check = EvidenceAppearsInArtifact(artifact_text=QUOTABLE)
        self.assertEqual(check.violations(_critique(evidence="First Light")), [])
        self.assertEqual(check.violations(_critique(evidence="no end rhyme")), [])

    def test_surrounding_quotes_and_trailing_ellipsis_are_tolerated(self):
        check = EvidenceAppearsInArtifact(artifact_text=QUOTABLE)
        self.assertEqual(check.violations(_critique(evidence='"the water turns..."')), [])

    def test_missing_evidence_is_caught_when_required(self):
        self.assertEqual(len(EvidenceAppearsInArtifact(artifact_text=QUOTABLE, required=True).violations(_critique(evidence=""))), 1)

    def test_missing_evidence_is_allowed_when_not_required(self):
        self.assertEqual(EvidenceAppearsInArtifact(artifact_text=QUOTABLE, required=False).violations(_critique(evidence="")), [])


class TestPositionAsAssigned(TestCase):
    def test_the_assigned_position_passes(self):
        self.assertEqual(PositionAsAssigned(expected="pro").violations(_critique(position="pro")), [])

    def test_a_different_position_is_caught(self):
        self.assertEqual(len(PositionAsAssigned(expected="con").violations(_critique(position="pro"))), 1)

    def test_case_and_padding_do_not_matter(self):
        self.assertEqual(PositionAsAssigned(expected="pro").violations(_critique(position="  PRO ")), [])


class TestVerdictWellFormed(TestCase):
    def _verdict(self, winner="pro", positions=("pro", "con")):
        return Verdict(
            winner=winner,
            scores=[PositionScore(position=p, score=0.5, reasoning="r") for p in positions],
            rationale="r",
            decisive=True,
        )

    def test_a_well_formed_verdict_passes(self):
        self.assertEqual(VerdictWellFormed(positions=["pro", "con"]).violations(self._verdict()), [])

    def test_a_missing_score_is_caught(self):
        self.assertEqual(len(VerdictWellFormed(positions=["pro", "con"]).violations(self._verdict(positions=("pro",)))), 1)

    def test_a_score_for_an_unargued_position_is_caught(self):
        self.assertEqual(len(VerdictWellFormed(positions=["pro", "con"]).violations(self._verdict(positions=("pro", "con", "other")))), 1)

    def test_a_winner_nobody_argued_is_caught(self):
        self.assertEqual(len(VerdictWellFormed(positions=["pro", "con"]).violations(self._verdict(winner="other"))), 1)

    def test_a_null_winner_is_allowed_by_default(self):
        self.assertEqual(VerdictWellFormed(positions=["pro", "con"]).violations(self._verdict(winner=None)), [])

    def test_a_null_winner_is_caught_when_the_rubric_forbids_it(self):
        check = VerdictWellFormed(positions=["pro", "con"], allow_no_decision=False)
        self.assertEqual(len(check.violations(self._verdict(winner=None))), 1)


class TestRunChecks(TestCase):
    def test_every_violation_is_reported_at_once(self):
        """One re-prompt should be able to fix everything, so checks do not short-circuit."""
        problems = run_checks(
            [PositionAsAssigned(expected="pro"), EvidenceAppearsInArtifact(artifact_text="nothing matches")],
            _critique(position="con", evidence="never written"),
        )
        self.assertEqual(len(problems), 2)

    def test_no_checks_means_no_violations(self):
        self.assertEqual(run_checks([], _critique()), [])


class TestBaseCheckIsAbstract(TestCase):
    def test_the_base_check_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            OutputCheck()

    def test_a_subclass_that_misspells_the_method_fails_at_construction(self):
        """The failure must land where the mistake is, not inside `run_checks` mid-enforcement."""

        class Misspelled(OutputCheck):
            def violation(self, output) -> list[str]:  # not `violations`
                return []

        with self.assertRaises(TypeError):
            Misspelled()

    def test_a_correct_subclass_works(self):
        class Always(OutputCheck):
            def violations(self, output) -> list[str]:
                return ["nope"]

        self.assertEqual(run_checks([Always()], POEM), ["nope"])
