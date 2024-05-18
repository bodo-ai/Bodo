from mpi4py import MPI

import bodo
from bodo.tests.utils import _test_equal_guard, gen_unique_table_id, reduce_sum


def gen_unique_id(name_prefix: str) -> str:
    path = None
    if bodo.get_rank() == 0:
        path = gen_unique_table_id(name_prefix)
    path = MPI.COMM_WORLD.bcast(path)
    return path


def test_equal_par(bodo_output, py_output):
    passed = _test_equal_guard(
        bodo_output,
        py_output,
    )
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Parallel test failed"
