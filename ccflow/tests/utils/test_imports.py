import subprocess
import sys
from unittest import TestCase

from ccflow.utils.imports import import_or_install


class TestImportOrInstall(TestCase):
    def setUp(self):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pip-install-test"])

    def tearDown(self):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pip-install-test"])

    def test_already_installed(self):
        with self.assertLogs("ccflow.utils.imports", level="INFO") as cm:
            import_or_install(import_name="ray", pip_name="ray")
            self.assertEqual(cm.output, ["INFO:ccflow.utils.imports:ray is already installed"])

    def test_not_installed_no_source(self):
        self.assertRaises(ValueError, import_or_install, "foo123")

    def test_not_installed_both_sources(self):
        self.assertRaises(ValueError, import_or_install, "foo123", "pip_foo_123", "conda-foo-123")

    def test_not_installed_pip(self):
        with self.assertLogs("ccflow.utils.imports", level="INFO") as cm:
            import_or_install(import_name="pip_install_test", pip_name="pip-install-test")
            self.assertEqual(
                cm.output,
                [
                    "INFO:ccflow.utils.imports:pip installed pip-install-test",
                ],
            )

    def test_not_installed_pip_unavailable(self):
        with self.assertLogs("ccflow.utils.imports", level="ERROR") as cm:
            self.assertRaises(ImportError, import_or_install, "foo123", "pip_foo_123")
            self.assertEqual(len(cm.output), 1)
            expected_prefix = "ERROR:ccflow.utils.imports:Error pip installing pip_foo_123"
            self.assertEqual(
                cm.output[0][: len(expected_prefix)],
                expected_prefix,
            )

    def test_not_installed_conda_unavailable(self):
        with self.assertLogs("ccflow.utils.imports", level="ERROR") as cm:
            self.assertRaises(ImportError, import_or_install, "foo123", None, "conda_foo_123")
            self.assertEqual(len(cm.output), 1)
            expected_prefix = "ERROR:ccflow.utils.imports:Error conda installing conda_foo_123"
            self.assertEqual(
                cm.output[0][: len(expected_prefix)],
                expected_prefix,
            )
