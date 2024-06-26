import importlib
import logging
import sys

__all__ = (
    "import_or_install",
    "lazy_import",
)
log = logging.getLogger(__name__)


def import_or_install(import_name: str, pip_name: str = None, conda_name: str = None) -> None:
    """
    Convenience function to import package if available. otherwise install from either pip or conda
    :param import_name: Name of the package on import statement
    :type import_name: str
    :param pip_name: Name of the package for pip install
    :type pip_name: str or None
    :param conda_name: Name of the package for conda install
    :type conda_name: str or None
    :return: None
    :raises subprocess.CalledProcessError: if package is not available and cannot be installed
    :raises ValueError: if package is not available and no pip or conda package specified
    """
    import subprocess
    import sys

    try:
        importlib.import_module(import_name)
        log.info("%s is already installed", import_name)
    except ImportError:
        if not (conda_name or pip_name):
            raise ValueError(f"{import_name} is not installed and no pip or conda package provided")

        if conda_name and pip_name:
            raise ValueError("Provide only one of the pip or conda package to install")

        if conda_name:
            try:
                subprocess.check_call([sys.executable, "conda", "install", "-y", conda_name])
                log.info("conda installed %s", conda_name)
            except subprocess.CalledProcessError:
                log.exception("Error conda installing %s", conda_name)

        if pip_name:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                log.info("pip installed %s", pip_name)
            except subprocess.CalledProcessError:
                log.exception("Error pip installing %s", pip_name)

    importlib.import_module(import_name)


def lazy_import(name: str):
    """
    https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    """
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
