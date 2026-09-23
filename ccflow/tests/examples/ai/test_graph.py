"""Tests for graph wiring, presentation neutrality, role-scoped lessons, and the file sources.

Everything here runs on pydantic-ai's TestModel or on pure functions, so no credentials or network
access are needed.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from ccflow import NullContext
from ccflow.examples.ai import (
    AdvocateModel,
    AgentSession,
    ArbitrateModel,
    ArbitrationRubric,
    Claim,
    CounterbalancedArbitrateModel,
    Critique,
    CritiqueFileSource,
    CritiqueRubric,
    Lessons,
    Poem,
    PoemFileSource,
    PoemSpec,
    PositionScore,
    ProduceModel,
    RebuttalModel,
    Verdict,
    load_agent_profile,
    profile_search_roots,
)

POEM = Poem(title="First Light", stanzas=["the water turns\nfoam at the rocks"], notes="no end rhyme")


def _session(role, profile=None, contract=None):
    return AgentSession(
        role=role,
        profile_name=profile or role,
        command="do the thing",
        output_contract=contract or ("ccflow.examples.ai.contracts.Poem" if role == "producer" else "ccflow.examples.ai.contracts.Critique"),
    )


def _graph():
    produce = ProduceModel(session=_session("producer"), spec=PoemSpec(theme="the sea"), enforce_checks=False)
    rubric = CritiqueRubric(criteria=["imagery"])
    advocates = [
        AdvocateModel(session=_session(position), produce=produce, position=position, rubric=rubric, enforce_checks=False)
        for position in ("pro", "con")
    ]
    return ArbitrateModel(
        session=_session("arbiter", contract="ccflow.examples.ai.contracts.Verdict"),
        advocates=advocates,
        artifact=produce,
        rubric=ArbitrationRubric(criteria=["evidence"]),
        enforce_checks=False,
    )


def _critique(position, confidence=0.5):
    return Critique(
        position=position,
        claims=[Claim(statement="s", evidence="foam at the rocks", criterion="imagery", weight=0.5)],
        summary="summary",
        confidence=confidence,
    )


def _verdict(winner, pro, con, decisive):
    return Verdict(
        winner=winner,
        scores=[
            PositionScore(position="pro", score=pro, reasoning="pro said things"),
            PositionScore(position="con", score=con, reasoning="con said things"),
        ],
        rationale=f"{winner} won",
        decisive=decisive,
    )


class TestPersonas(TestCase):
    def test_every_role_has_a_persona(self):
        for name in ("producer", "pro", "con", "arbiter", "steelman", "skeptic"):
            with self.subTest(name=name):
                self.assertTrue(load_agent_profile(name).instructions)

    def test_the_frontmatter_is_stripped_from_the_instructions(self):
        profile = load_agent_profile("producer")
        self.assertNotIn("description:", profile.instructions)
        self.assertTrue(profile.description)

    def test_a_missing_persona_names_where_it_looked(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_agent_profile("no_such_persona")
        self.assertIn("CCFLOW_AI_AGENTS_PATH", str(caught.exception))

    def test_the_bundled_directory_is_searched(self):
        self.assertTrue(any(root.name == "personas" for root in profile_search_roots()))

    def test_the_advocate_personas_are_symmetric(self):
        """Giving one side a rule the other lacks is a thumb on the scale that is hard to see."""
        for rule in ("mutually consistent", "is not fabrication"):
            with self.subTest(rule=rule):
                self.assertIn(rule, load_agent_profile("pro").instructions)
                self.assertIn(rule, load_agent_profile("con").instructions)


class TestDependencyWiring(TestCase):
    def setUp(self):
        self.arbiter = _graph()

    def test_an_advocate_depends_on_the_producer(self):
        advocate = self.arbiter.advocates[0]
        self.assertEqual([model for model, _ in advocate.__deps__(NullContext())], [advocate.produce])

    def test_the_arbiter_depends_on_every_advocate_and_the_artifact(self):
        deps = [model for model, _ in self.arbiter.__deps__(NullContext())]
        for advocate in self.arbiter.advocates:
            self.assertIn(advocate, deps)
        self.assertIn(self.arbiter.artifact, deps)

    def test_declared_contexts_are_resolvable_without_running_anything(self):
        """__deps__ must be answerable before its dependency runs, which is why contexts are null."""
        for _, contexts in self.arbiter.__deps__(NullContext()):
            self.assertEqual(contexts, [NullContext()])

    def test_both_advocates_reference_one_production(self):
        pro, con = self.arbiter.advocates
        self.assertEqual(pro.produce, con.produce)
        self.assertEqual(pro.produce, self.arbiter.artifact)

    def test_a_rebuttal_depends_on_both_cases_and_the_artifact(self):
        pro, con = self.arbiter.advocates
        rebuttal = RebuttalModel(
            session=_session("pro"),
            own=pro,
            opponent=con,
            artifact=pro.produce,
            position="pro",
            rubric=CritiqueRubric(criteria=["imagery"]),
        )
        self.assertEqual([model for model, _ in rebuttal.__deps__(NullContext())], [pro, con, pro.produce])


class TestPresentationNeutrality(TestCase):
    """Guards against the arbiter being swayed by how a case is presented rather than its content."""

    def setUp(self):
        self.arbiter = _graph()
        self.critiques = [_critique("pro", 0.8), _critique("con", 0.5)]

    def test_blinding_replaces_position_names_with_slot_labels(self):
        shown, real = self.arbiter.model_copy(update={"blind_positions": True})._blind(self.critiques)
        self.assertEqual([c.position for c in shown], ["case_1", "case_2"])
        self.assertEqual(real, {"case_1": "pro", "case_2": "con"})

    def test_blinding_leaves_the_cases_otherwise_untouched(self):
        shown, _ = self.arbiter.model_copy(update={"blind_positions": True})._blind(self.critiques)
        self.assertEqual([c.summary for c in shown], ["summary", "summary"])
        self.assertEqual([c.position for c in self.critiques], ["pro", "con"])

    def test_opting_out_leaves_the_names_alone(self):
        shown, real = self.arbiter.model_copy(update={"blind_positions": False})._blind(self.critiques)
        self.assertEqual([c.position for c in shown], ["pro", "con"])
        self.assertEqual(real, {"pro": "pro", "con": "con"})

    def test_the_verdict_comes_back_in_real_positions(self):
        verdict = Verdict(
            winner="case_2",
            scores=[
                PositionScore(position="case_1", score=0.4, reasoning="case_1 asserted"),
                PositionScore(position="case_2", score=0.7, reasoning="case_2 quoted"),
            ],
            rationale="case_2 beat case_1",
            decisive=True,
        )
        restored = ArbitrateModel._unblind(verdict, {"case_1": "pro", "case_2": "con"})
        self.assertEqual(restored.winner, "con")
        self.assertEqual([s.position for s in restored.scores], ["pro", "con"])
        self.assertEqual(restored.rationale, "con beat pro")

    def test_a_relabelled_case_is_restored_however_it_was_written(self):
        """Observed live: an arbiter rewrites 'case_1' as 'Case 1' in prose, which exact matching missed."""
        for written in ("case_1", "Case 1", "case-1", "CASE_1"):
            with self.subTest(written=written):
                verdict = Verdict(
                    winner="case_1",
                    scores=[PositionScore(position="case_1", score=0.6, reasoning=f"{written} quoted well.")],
                    rationale=f"{written} edges ahead.",
                    decisive=True,
                )
                restored = ArbitrateModel._unblind(verdict, {"case_1": "pro", "case_2": "con"})
                self.assertEqual(restored.rationale, "pro edges ahead.")
                self.assertEqual(restored.scores[0].reasoning, "pro quoted well.")

    def test_a_recased_label_resolves_in_the_structured_fields(self):
        """`VerdictWellFormed` accepts a position case-insensitively, so restoring must match the same way."""
        verdict = Verdict(
            winner="Case_1",
            scores=[PositionScore(position="CASE_1", score=0.6, reasoning="fine")],
            rationale="fine",
            decisive=True,
        )
        restored = ArbitrateModel._unblind(verdict, {"case_1": "pro", "case_2": "con"})
        self.assertEqual(restored.winner, "pro")
        self.assertEqual(restored.scores[0].position, "pro")

    def test_restoring_does_not_maul_unrelated_words(self):
        verdict = Verdict(
            winner="case_1",
            scores=[PositionScore(position="case_1", score=0.6, reasoning="The case is well made.")],
            rationale="A staircase and a suitcase are not labels.",
            decisive=True,
        )
        restored = ArbitrateModel._unblind(verdict, {"case_1": "pro", "case_2": "con"})
        self.assertEqual(restored.rationale, "A staircase and a suitcase are not labels.")
        self.assertEqual(restored.scores[0].reasoning, "The case is well made.")


class TestCounterbalancedArbitration(TestCase):
    """A winner is reported only when it survives both presentation orders."""

    def test_agreement_keeps_the_winner(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict("pro", 0.8, 0.6, True), _verdict("pro", 0.7, 0.5, True))
        self.assertEqual(merged.winner, "pro")
        self.assertTrue(merged.decisive)

    def test_disagreement_is_recorded_as_undecided(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict("pro", 0.8, 0.6, True), _verdict("con", 0.5, 0.7, True))
        self.assertIsNone(merged.winner)
        self.assertFalse(merged.decisive)
        self.assertIn("disagreed", merged.rationale)

    def test_agreement_on_a_close_call_is_still_not_decisive(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict("pro", 0.6, 0.58, True), _verdict("pro", 0.6, 0.59, False))
        self.assertEqual(merged.winner, "pro")
        self.assertFalse(merged.decisive)

    def test_scores_are_averaged_across_orders(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict("pro", 0.8, 0.6, True), _verdict("pro", 0.6, 0.4, True))
        self.assertEqual({s.position: round(s.score, 6) for s in merged.scores}, {"pro": 0.7, "con": 0.5})

    def test_both_orders_reasoning_is_retained(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict("pro", 0.8, 0.6, True), _verdict("pro", 0.7, 0.5, True))
        pro = next(s for s in merged.scores if s.position == "pro")
        self.assertIn("configured order", pro.reasoning)
        self.assertIn("reversed", pro.reasoning)

    def test_agreeing_that_it_is_undecided_reads_as_undecided(self):
        merged = CounterbalancedArbitrateModel._reconcile(_verdict(None, 0.6, 0.6, False), _verdict(None, 0.6, 0.6, False))
        self.assertIsNone(merged.winner)
        self.assertNotIn("chose None", merged.rationale)
        self.assertTrue(merged.rationale.startswith("Both presentation orders left this undecided."))


class TestLessonScoping(TestCase):
    """A lesson reaches the roles it names and no others."""

    def test_general_lessons_reach_every_role(self):
        lessons = Lessons(general=["be plain"])
        self.assertEqual(lessons.for_role("producer"), ["be plain"])
        self.assertEqual(lessons.for_role("arbiter"), ["be plain"])

    def test_role_lessons_reach_only_that_role(self):
        lessons = Lessons(by_role={"producer": ["avoid 'seam'"]})
        self.assertEqual(lessons.for_role("producer"), ["avoid 'seam'"])
        self.assertEqual(lessons.for_role("con"), [])

    def test_general_comes_before_role_specific(self):
        self.assertEqual(Lessons(general=["g"], by_role={"pro": ["p"]}).for_role("pro"), ["g", "p"])

    def test_an_unknown_role_gets_only_the_general_ones(self):
        self.assertEqual(Lessons(general=["g"], by_role={"pro": ["p"]}).for_role("nobody"), ["g"])

    def test_lessons_reach_the_model_instructions(self):
        session = AgentSession(
            role="producer",
            profile_name="producer",
            command="do the thing",
            output_contract="ccflow.examples.ai.contracts.Poem",
            lessons=Lessons(general=["be plain"], by_role={"producer": ["avoid 'seam'"]}),
        )
        self.assertIn("be plain", session.instructions)
        self.assertIn("avoid 'seam'", session.instructions)

    def test_another_roles_lesson_stays_out_of_the_instructions(self):
        session = AgentSession(
            role="producer",
            profile_name="producer",
            command="do the thing",
            output_contract="ccflow.examples.ai.contracts.Poem",
            lessons=Lessons(by_role={"arbiter": ["fluency is not argument"]}),
        )
        self.assertNotIn("fluency is not argument", session.instructions)

    def test_a_role_keeps_its_lessons_when_the_persona_is_swapped(self):
        """`role` is separate from `profile_name` so swapping personas cannot silently drop lessons."""
        session = AgentSession(
            role="pro",
            profile_name="steelman",
            command="argue",
            output_contract="ccflow.examples.ai.contracts.Critique",
            lessons=Lessons(by_role={"pro": ["quote what you praise"]}),
        )
        self.assertIn("quote what you praise", session.instructions)
        self.assertIn("Steelman", session.instructions)

    def test_no_lessons_leaves_the_instructions_unchanged(self):
        common = {"role": "producer", "profile_name": "producer", "command": "go", "output_contract": "ccflow.examples.ai.contracts.Poem"}
        self.assertEqual(AgentSession(**common).instructions, AgentSession(**common, lessons=Lessons()).instructions)


class TestFileSources(TestCase):
    """Reading a published artifact is what pins it, so the same poem can be re-judged."""

    def test_a_published_result_is_read_back(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "poem.json"
            path.write_text(json.dumps({"poem": POEM.model_dump()}))
            self.assertEqual(PoemFileSource(path=str(path))().poem, POEM)

    def test_a_bare_contract_is_also_accepted(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "poem.json"
            path.write_text(POEM.model_dump_json())
            self.assertEqual(PoemFileSource(path=str(path))().poem, POEM)

    def test_a_critique_is_read_back(self):
        critique = _critique("pro")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "critique_pro.json"
            path.write_text(json.dumps({"critique": critique.model_dump()}))
            self.assertEqual(CritiqueFileSource(path=str(path))().critique, critique)

    def test_a_missing_file_says_what_to_run(self):
        with self.assertRaises(FileNotFoundError) as caught:
            PoemFileSource(path="/nonexistent/poem.json")()
        self.assertIn("publisher", str(caught.exception))

    def test_a_file_source_substitutes_for_the_live_producer(self):
        """The advocate declares a `PoemProvider`, so it cannot tell the difference."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "poem.json"
            path.write_text(json.dumps({"poem": POEM.model_dump()}))
            advocate = AdvocateModel(
                session=_session("pro"),
                produce=PoemFileSource(path=str(path)),
                position="pro",
                rubric=CritiqueRubric(criteria=["imagery"]),
                enforce_checks=False,
            )
            self.assertTrue(advocate().critique.claims)


class TestGraphRunsOffline(TestCase):
    """The whole graph must run on TestModel, so the example needs no credentials."""

    def test_the_graph_produces_a_verdict(self):
        result = _graph()(NullContext())
        self.assertIsInstance(result.verdict, Verdict)
        self.assertIsInstance(result.poem, Poem)
        self.assertEqual(len(result.critiques), 2)

    def test_the_arbiter_judged_the_artifact_that_was_produced(self):
        arbiter = _graph()
        self.assertEqual(arbiter(NullContext()).poem, arbiter.advocates[0].produce(NullContext()).poem)
