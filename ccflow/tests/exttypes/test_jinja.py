from unittest import TestCase

from ccflow.exttypes.jinja import JinjaTemplate


class TestJinjaTemplate(TestCase):
    def test_template(self):
        v = "My {{foo|lower}}"
        t = JinjaTemplate(v)
        self.assertEqual(t.template.render(foo="FOO"), "My foo")
        self.assertEqual(JinjaTemplate.validate(v), t)

    def test_bad(self):
        v = "My {{"
        self.assertRaises(ValueError, JinjaTemplate.validate, v)

    def test_deepcopy(self):
        # Pydantic models sometimes require deep copy, and this can pose problems
        # for Jinja templates if they are stored on the object, i.e. see https://github.com/pallets/jinja/issues/758
        from copy import deepcopy

        v = "My {{foo|lower}}"
        t = JinjaTemplate(v)

        # First access the template
        t.template
        # Then attempt the copy
        self.assertEqual(deepcopy(t), t)
