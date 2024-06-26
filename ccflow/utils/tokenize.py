# ruff: noqa: F401
try:
    from dask.base import normalize_token, tokenize
except ImportError:
    from .dask_tokenize import normalize_token, tokenize
