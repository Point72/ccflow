from pathlib import Path, PureWindowsPath
from unittest import TestCase


class TestPackagePaths(TestCase):
    def test_package_paths_are_valid_on_windows(self):
        package = Path(__file__).resolve().parents[1]
        reserved = [str(path.relative_to(package)) for path in package.rglob("*") if PureWindowsPath(path.name).is_reserved()]
        self.assertEqual(reserved, [])
