import subprocess
import sys
import textwrap
from unittest import TestCase


class TestImports(TestCase):
    def test(self):
        # Assert that importing ccflow doesn't cause expensive modules to be imported.
        res = subprocess.run([sys.executable, __file__], capture_output=True, check=False, text=True)
        if res.returncode != 0:
            raise AssertionError(res.stderr)

    def test_without_hydra(self):
        # hydra-core (and omegaconf, which it brings in) is optional; only config loading should need it.
        code = textwrap.dedent(
            """
            import importlib.util
            import sys

            class _Uninstalled:
                def find_spec(self, name, path=None, target=None):
                    if name.partition(".")[0] in ("hydra", "lerna", "omegaconf"):
                        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

            sys.meta_path.insert(0, _Uninstalled())

            import ccflow

            registry = ccflow.ModelRegistry(name="test")
            registry.add("model", ccflow.BaseModel())
            assert isinstance(registry["model"], ccflow.BaseModel)

            lazy = ccflow.LazyRegistry(name="lazy", model={"_target_": "ccflow.BaseModel"}, sub={"inner": {"_target_": "ccflow.BaseModel"}})
            assert not lazy.is_loaded("model")
            assert "sub/inner" in lazy

            for load in (lambda: lazy["model"], lambda: registry.load_config({"m": {"_target_": "ccflow.BaseModel"}})):
                try:
                    load()
                except ModuleNotFoundError as e:
                    assert e.name == "hydra", e
                else:
                    raise AssertionError("expected hydra to be required")

            try:
                import ccflow.utils.hydra  # noqa: F401
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError("ccflow.utils.hydra imported without hydra")

            if importlib.util.find_spec("panel") and importlib.util.find_spec("panel_material_ui"):
                import ccflow.ui.panel  # noqa: F401
            if importlib.util.find_spec("spaday") and importlib.util.find_spec("starlette"):
                from ccflow.ui.spaday import serve_registry

                serve_registry(registry, run=False)
            """
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False, text=True)
        if res.returncode != 0:
            raise AssertionError(res.stderr)

    def test_without_pandas(self):
        # pandas is optional; frequencies, relative dates and chunking must work without it.
        code = textwrap.dedent(
            """
            import sys
            from datetime import date

            class _Uninstalled:
                def find_spec(self, name, path=None, target=None):
                    if name.partition(".")[0] == "pandas":
                        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

            sys.meta_path.insert(0, _Uninstalled())

            from ccflow import DateContext, Frequency
            from ccflow.publishers import PandasFilePublisher  # noqa: F401
            from ccflow.publishers.file import PandasFilePublisher as _P  # noqa: F401
            from ccflow.utils.chunker import dates_to_chunks

            assert Frequency.validate("3mo") == "1q"
            assert isinstance(DateContext(date="-1d").date, date)
            assert dates_to_chunks(date(2022, 4, 6), date(2022, 4, 7), "1mo") == [(date(2022, 4, 1), date(2022, 4, 30))]
            """
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False, text=True)
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
        "pyarrow",
        "pyarrow.dataset",
        # These aren't necessarily expensive, just things we don't want to import.
        "cexprtk",
    ]
    for m in expensive_imports:
        if m in sys.modules:
            raise AssertionError(f"{m} was imported!")
