import pandas as pd

from bodo.submit.spawner import submit_jit
from bodo.tests.conftest import datapath_util
from bodo.tests.utils import pytest_spawn_mode

pytestmark = pytest_spawn_mode


CUSTOMER_TABLE_PATH = datapath_util("tpch-test_data/parquet/customer.parquet")


@submit_jit
def read_customer_table():
    df = pd.read_parquet(CUSTOMER_TABLE_PATH)
    return df


@submit_jit
def aggregate_customer_data(df):
    max_acctbal = (
        df[df["C_ACCTBAL"] > 1000.0]
        .groupby(["C_MKTSEGMENT", "C_NATIONKEY"])
        .agg({"C_ACCTBAL": "max"})
    )
    return max_acctbal


@submit_jit
def jit_fn_that_calls_other_jit_fns():
    """Function decorated with @submit_jit that calls other functions that are
    also decorated"""
    df = read_customer_table()
    max_acctbal = aggregate_customer_data(df)
    print("Max acctbal:")
    print(max_acctbal)


def test_recursive():
    """Test that a JIT function that calls another JIT function in spawn mode is
    supported"""
    jit_fn_that_calls_other_jit_fns()
