from typing import get_args

import pytest
from polars.testing import assert_frame_equal

from ccflow import ModelRegistry
from ccflow.examples.tpch import TPCHTable, load_config


@pytest.fixture(scope="module")
def registry():
    # Load the TPC-H example registry from its YAML. We override the scale
    # factor on the single shared backend; that one override flows through
    # to all table/answer/query entries because they all reference
    # ``/tpch/backend``.
    load_config(overrides=["tpch.backend.scale_factor=0.1"], overwrite=True)
    return ModelRegistry.root()


@pytest.mark.parametrize("table", get_args(TPCHTable))
def test_tpch_table_provider(registry, table):
    provider = registry[f"/table/{table}"]
    out = provider()
    assert out is not None
    assert len(out.df) > 0


@pytest.mark.parametrize("query_id", range(1, 23))
def test_tpch_answer_provider(registry, query_id):
    provider = registry[f"/answer/Q{query_id}"]
    out = provider()
    assert out is not None
    assert len(out.df) > 0


@pytest.mark.parametrize("query_id", range(1, 23))
def test_tpch_query(registry, query_id):
    query = registry[f"/query/Q{query_id}"]
    answer = registry[f"/answer/Q{query_id}"]
    out = query()
    expected = answer()
    assert_frame_equal(out.df.to_polars(), expected.df.to_polars(), check_dtypes=False)
