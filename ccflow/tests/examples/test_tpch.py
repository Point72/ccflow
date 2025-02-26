from typing import get_args

import pytest
from polars.testing import assert_frame_equal

from ccflow.examples.tpch import TPCHAnswerGenerator, TPCHDataGenerator, TPCHQueryContext, TPCHQueryRunner, TPCHTable, TPCHTableContext


@pytest.fixture(scope="module")
def scale_factor():
    return 0.1


@pytest.fixture(scope="module")
def tpch_answer_generator(scale_factor):
    return TPCHAnswerGenerator(scale_factor=scale_factor)


@pytest.fixture(scope="module")
def tpch_data_generator(scale_factor):
    return TPCHDataGenerator(scale_factor=scale_factor)


@pytest.mark.parametrize("query_id", range(1, 23))
def test_tpch_answer_generation(tpch_answer_generator, query_id):
    context = TPCHQueryContext(query_id=query_id)
    out = tpch_answer_generator(context)
    assert out is not None
    assert len(out.df) > 0


@pytest.mark.parametrize("table", get_args(TPCHTable))
def test_tpch_data_generation(tpch_data_generator, table):
    context = TPCHTableContext(table=table)
    out = tpch_data_generator(context)
    assert out is not None
    assert len(out.df) > 0


@pytest.mark.parametrize("query_id", range(1, 23))
def test_tpch_queries(tpch_answer_generator, tpch_data_generator, query_id):
    runner = TPCHQueryRunner(table_provider=tpch_data_generator)
    context = TPCHQueryContext(query_id=query_id)
    answer = tpch_answer_generator(context)
    out = runner(context)
    assert out is not None
    assert answer is not None
    assert_frame_equal(out.df.to_polars(), answer.df.to_polars(), check_dtype=False)
