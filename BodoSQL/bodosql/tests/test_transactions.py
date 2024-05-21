"""
Test transaction support in BodoSQL. Right now, transactions are no-op, 
so we just test them as is. In the future, they need to be tied to the
storage / catalog, and the tests will need to be updated and moved
accordingly.
"""

import pandas as pd
import pytest

import bodosql
from bodo.tests.utils import check_func
from bodosql.tests.utils import assert_equal_par


@pytest.mark.parametrize("extension", ["", "WORK", "TRANSACTION"])
def test_begin(extension, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    query = f"BEGIN {extension}"
    bc = bodosql.BodoSQLContext()
    py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})

    # execute_ddl Version
    bodo_output = bc.execute_ddl(query)
    assert_equal_par(bodo_output, py_output)

    # Jit Version
    # Intentionally returns replicated output
    check_func(impl, (bc, query), py_output=py_output, only_seq=True)

    # Python Version
    bodo_output = bc.sql(query)
    assert_equal_par(bodo_output, py_output)


@pytest.mark.parametrize("extension", ["", "WORK"])
def test_commit(extension, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    query = f"COMMIT {extension}"
    bc = bodosql.BodoSQLContext()
    py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})

    # execute_ddl Version
    bodo_output = bc.execute_ddl(query)
    assert_equal_par(bodo_output, py_output)

    # Jit Version
    # Intentionally returns replicated output
    check_func(impl, (bc, query), py_output=py_output, only_seq=True)

    # Python Version
    bodo_output = bc.sql(query)
    assert_equal_par(bodo_output, py_output)


@pytest.mark.parametrize("extension", ["", "WORK"])
def test_rollback(extension, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    query = f"BEGIN {extension}"
    bc = bodosql.BodoSQLContext()
    py_output = pd.DataFrame({"STATUS": [f"Statement executed successfully."]})

    # execute_ddl Version
    bodo_output = bc.execute_ddl(query)
    assert_equal_par(bodo_output, py_output)

    # Jit Version
    # Intentionally returns replicated output
    check_func(impl, (bc, query), py_output=py_output, only_seq=True)

    # Python Version
    bodo_output = bc.sql(query)
    assert_equal_par(bodo_output, py_output)


def test_begin_commit(memory_leak_check):
    def impl(bc):
        bc.sql("BEGIN")
        df = bc.sql("SELECT 1 as test")
        bc.sql("COMMIT")
        return df

    bc = bodosql.BodoSQLContext()
    py_output = pd.DataFrame({"TEST": pd.Series([1], dtype="int32")})

    check_func(impl, (bc,), py_output=py_output)
