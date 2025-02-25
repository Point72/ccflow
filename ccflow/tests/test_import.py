import subprocess
import sys
from unittest import TestCase


class TestImports(TestCase):
    def test(self):
        # Assert that importing ccflow doesn't cause expensive modules to be imported.
        res = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
        if res.returncode != 0:
            raise AssertionError(res.stderr)


if __name__ == "__main__":
    import ccflow

    _ = ccflow
    expensive_imports = [
        "ray",
        "deltalake",
        "emails",
        "matplotlib",
        "mlflow",
        "plotly",
        "pyarrow.dataset",
        # These aren't necessarily expensive, just things we don't want to import.
        "cexprtk",
    ]
    for m in expensive_imports:
        if m in sys.modules:
            raise AssertionError(f"{m} was imported!")
