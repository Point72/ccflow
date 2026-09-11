"""Tests for the bundled config groups.

Group options are discovered from the filesystem and parametrized, so a new variant is covered the
moment it is added.
"""

from pathlib import Path
from unittest import TestCase

from ccflow import ModelRegistry
from ccflow.examples import ai
from ccflow.examples.ai import (
    AdvocateModel,
    ArbitrateModel,
    ArbitrationRubric,
    CounterbalancedArbitrateModel,
    CritiqueFileSource,
    CritiqueRubric,
    DebateModel,
    PoemFileSource,
    ProduceModel,
    RebuttalModel,
    load_config,
)

GROUPS = ("model", "profile", "agents", "rubrics", "lessons", "spec", "checks", "publisher", "task")


def _config_dir() -> Path:
    return Path(ai.__file__).resolve().parent / "config"


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (_config_dir() / group).glob("*.yaml"))


ALL_OPTIONS = [(group, option) for group in GROUPS for option in _options(group)]


def _registry(*overrides):
    return load_config(overrides=["checks=off", *overrides])


class RegistryTestCase(TestCase):
    """The root registry is global, so start and finish with a clean one.

    Without this these tests pass alone and fail in a suite, in both directions: entries left by
    another module change what `/task` resolves to here, and entries left here leak into everything
    that runs afterwards.
    """

    def setUp(self):
        ModelRegistry.root().clear()

    def tearDown(self):
        ModelRegistry.root().clear()


class TestGroupLayout(RegistryTestCase):
    def test_every_group_exists(self):
        for group in GROUPS:
            with self.subTest(group=group):
                self.assertTrue((_config_dir() / group).is_dir())

    def test_every_group_offers_a_choice(self):
        for group in GROUPS:
            with self.subTest(group=group):
                self.assertGreaterEqual(len(_options(group)), 2, f"{group} has no alternative to its default")

    def test_every_option_composes_and_instantiates(self):
        for group, option in ALL_OPTIONS:
            with self.subTest(group=group, option=option):
                self.assertIsNotNone(_registry(f"{group}={option}")["/task"])


class TestTasks(RegistryTestCase):
    def test_the_default_task_runs_the_whole_graph(self):
        task = _registry()["/task"]
        self.assertIsInstance(task, DebateModel)
        self.assertIsInstance(task.arbiter, ArbitrateModel)

    def test_the_produce_task_is_a_single_node(self):
        self.assertIsInstance(_registry("task=produce")["/task"], ProduceModel)

    def test_the_single_node_tasks_read_published_files(self):
        advocate = _registry("task=advocate")["/task"]
        self.assertIsInstance(advocate, AdvocateModel)
        self.assertIsInstance(advocate.produce, PoemFileSource)

        arbiter = _registry("task=arbitrate")["/task"]
        self.assertTrue(all(isinstance(a, CritiqueFileSource) for a in arbiter.advocates))
        self.assertIsInstance(arbiter.artifact, PoemFileSource)

    def test_the_advocate_task_name_and_persona_follow_the_position(self):
        advocate = _registry("task=advocate", "position=con")["/task"]
        self.assertEqual(advocate.position, "con")
        self.assertEqual(advocate.session.profile_name, "con")

    def test_the_whole_graph_shares_one_production(self):
        """Configure these differently and the advocates argue about different poems."""
        task = _registry()["/task"]
        pro, con = task.arbiter.advocates
        self.assertEqual(pro.produce, con.produce)
        self.assertEqual(pro.produce, task.arbiter.artifact)

    def test_the_counterbalanced_task_judges_both_orders(self):
        arbiter = _registry("task=counterbalanced")["/task"].arbiter
        self.assertIsInstance(arbiter, CounterbalancedArbitrateModel)
        self.assertEqual([a.position for a in arbiter.forward.advocates], ["pro", "con"])
        self.assertEqual([a.position for a in arbiter.reverse.advocates], ["con", "pro"])

    def test_both_orders_judge_the_same_cases(self):
        arbiter = _registry("task=counterbalanced")["/task"].arbiter
        forward = sorted(arbiter.forward.advocates, key=lambda a: a.position)
        reverse = sorted(arbiter.reverse.advocates, key=lambda a: a.position)
        self.assertEqual(forward, reverse)

    def test_the_rebuttal_task_has_each_advocate_answer_the_other(self):
        arbiter = _registry("task=rebuttal")["/task"].arbiter
        pro, con = arbiter.advocates
        self.assertIsInstance(pro, RebuttalModel)
        self.assertEqual(pro.own, con.opponent)
        self.assertEqual(con.own, pro.opponent)
        self.assertNotEqual(pro.own, pro.opponent)

    def test_the_rebuttal_command_differs_from_the_opening_one(self):
        pro = _registry("task=rebuttal")["/task"].arbiter.advocates[0]
        self.assertNotEqual(pro.session.command, pro.own.session.command)

    def test_the_strongest_task_combines_both_defences(self):
        arbiter = _registry("task=rebuttal_counterbalanced")["/task"].arbiter
        self.assertIsInstance(arbiter, CounterbalancedArbitrateModel)
        self.assertTrue(all(isinstance(a, RebuttalModel) for a in arbiter.forward.advocates))


class TestGroupsReachEveryNode(RegistryTestCase):
    def test_the_model_group_reaches_every_session(self):
        task = _registry("model=anthropic")["/task"]
        advocates = task.arbiter.advocates
        models = {task.arbiter.session.model, advocates[0].produce.session.model} | {a.session.model for a in advocates}
        self.assertEqual(models, {"anthropic:claude-sonnet-4-0"})

    def test_the_default_model_needs_no_credentials(self):
        self.assertEqual(_registry()["/task"].arbiter.session.model, "test")

    def test_the_profile_group_assigns_a_persona_per_role(self):
        advocates = _registry()["/task"].arbiter.advocates
        self.assertEqual([a.session.profile_name for a in advocates], ["pro", "con"])

    def test_the_steelman_profile_swaps_both_advocate_personas(self):
        advocates = _registry("profile=steelman")["/task"].arbiter.advocates
        self.assertEqual([a.session.profile_name for a in advocates], ["steelman", "skeptic"])

    def test_the_roles_are_unchanged_by_a_persona_swap(self):
        advocates = _registry("profile=steelman")["/task"].arbiter.advocates
        self.assertEqual([a.session.role for a in advocates], ["pro", "con"])

    def test_the_strict_rubric_tightens_both_rubrics(self):
        task = _registry("rubrics=strict")["/task"]
        self.assertFalse(task.arbiter.rubric.allow_no_decision)
        self.assertTrue(all(a.rubric.max_claims == 2 for a in task.arbiter.advocates))

    def test_the_haiku_spec_tightens_the_line_budget(self):
        task = _registry("spec=haiku", "task=produce")["/task"]
        self.assertEqual(task.spec.max_lines, 3)
        self.assertEqual(task.spec.form, "haiku")

    def test_the_checks_group_reaches_every_node(self):
        task = _registry()["/task"]
        nodes = [task.arbiter, *task.arbiter.advocates, task.arbiter.advocates[0].produce]
        self.assertEqual({n.enforce_checks for n in nodes}, {False})

    def test_checks_are_enforced_by_default(self):
        """The safe default must survive, so pointing at a real model cannot silently lose it."""
        task = load_config()["/task"]
        self.assertTrue(task.arbiter.enforce_checks)

    def test_the_concise_agents_variant_inherits_every_session(self):
        # `load_config` mutates the one root registry, so each value must be read before reloading.
        default = load_config(overrides=["checks=off", "agents=poem"])["/task"].arbiter.session
        default_command, contract = default.command, default.output_contract
        concise = load_config(overrides=["checks=off", "agents=poem_concise"])["/task"].arbiter.session
        self.assertNotEqual(concise.command, default_command)
        self.assertEqual(concise.output_contract, contract)

    def test_a_lesson_reaches_the_role_it_names(self):
        producer = _registry("lessons=house_style")["/task"].arbiter.advocates[0].produce.session
        self.assertTrue(any("seam" in lesson for lesson in producer.lessons.for_role("producer")))
        self.assertIn("seam", producer.instructions)

    def test_a_producer_lesson_does_not_reach_the_arbiter(self):
        arbiter = _registry("lessons=house_style")["/task"].arbiter.session
        self.assertFalse(any("seam" in lesson for lesson in arbiter.lessons.for_role("arbiter")))

    def test_a_single_role_book_leaves_other_roles_untouched(self):
        task = _registry("lessons=producer_only")["/task"]
        self.assertTrue(task.arbiter.advocates[0].produce.session.lessons.for_role("producer"))
        self.assertEqual(task.arbiter.session.lessons.for_role("arbiter"), [])

    def test_positions_are_blinded_by_default(self):
        self.assertTrue(_registry()["/task"].arbiter.blind_positions)

    def test_rubrics_coerce_from_plain_yaml(self):
        """Rubrics are plain mappings, not registry entries; the typed fields must coerce them."""
        task = _registry()["/task"]
        self.assertIsInstance(task.arbiter.rubric, ArbitrationRubric)
        for advocate in task.arbiter.advocates:
            self.assertIsInstance(advocate.rubric, CritiqueRubric)


class TestTasksRun(RegistryTestCase):
    def test_each_whole_graph_task_returns_a_verdict_over_two_cases(self):
        for task_name in ("end_to_end", "counterbalanced", "rebuttal", "rebuttal_counterbalanced"):
            with self.subTest(task=task_name):
                result = _registry(f"task={task_name}")["/task"]()
                self.assertEqual(len(result.critiques), 2)
                self.assertTrue(result.poem.title)
