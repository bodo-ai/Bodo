"""Tests I/O for HDF5 Files"""

import h5py
import numpy as np
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    pytest_mark_one_rank,
)

if bodo.test_compiler:
    import numba
    from numba.core.ir_utils import build_definitions, find_callname

    from bodo.tests.utils import get_start_end
    from bodo.utils.testing import ensure_clean
    from bodo.utils.utils import is_call_assign

pytestmark = [pytest.mark.hdf5, pytest.mark.slow]


@pytest.mark.smoke
def test_h5_read_seq(datapath, memory_leak_check):
    def test_impl(fname):
        f = h5py.File(fname, "r")
        X = f["points"][:]
        f.close()
        return X

    # passing function name as value to test value-based dispatch
    fname = datapath("lr.hdf5")
    check_func(test_impl, (fname,), only_seq=True)


def test_h5_read_const_infer_seq(datapath, memory_leak_check):
    fname = datapath("")

    def test_impl():
        p = fname + "lr"
        f = h5py.File(p + ".hdf5", "r")
        s = "po"
        X = f[s + "ints"][:]
        f.close()
        return X

    check_func(test_impl, (), only_seq=True)


def test_h5_read_parallel(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:]
        Y = f["responses"][:]
        f.close()
        return X.sum() + Y.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl(), decimal=2)
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest.mark.skip(
    "H5py bug breaks boolean arrays, https://github.com/h5py/h5py/issues/1847"
)
def test_h5_filter(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        b = np.arange(11) % 3 == 0
        X = f["test"][b, :, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 4  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice1(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["test"][:, 1:, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 11  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice2(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:, 1]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 101  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_read_group(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        X = g1["data"][:]
        f.close()
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func() == test_impl()


def test_h5_file_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        s = 0
        for gname in f.keys():
            X = f[gname]["data"][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.types.int64[:]})
    assert bodo_func() == test_impl()
    # test using locals for typing
    bodo_func = bodo.jit(test_impl, locals={"X": bodo.types.int64[:]})
    assert bodo_func() == test_impl()


def test_h5_group_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        s = 0
        for dname in g1.keys():
            X = g1[dname][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.types.int64[:]})
    assert bodo_func() == test_impl()


@pytest_mark_one_rank
@pytest.mark.smoke
def test_h5_write(memory_leak_check):
    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        dset1 = f.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(
            test_impl, returns_maybe_distributed=False, args_maybe_distributed=False
        )(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


@pytest_mark_one_rank
def test_h5_group_write(memory_leak_check):
    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        g1 = f.create_group("AA")
        g2 = g1.create_group("BB")
        dset1 = g2.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(test_impl)(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["AA"]["BB"]["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


@pytest.mark.slow
def test_h5_remove_dead(datapath, memory_leak_check):
    """make sure dead hdf5 read calls are removed properly"""
    from bodo.tests.utils import DeadcodeTestPipeline

    fname = datapath("lr.hdf5")

    def impl():
        f = h5py.File(fname, "r")
        f["points"][:, :]
        f.close()

    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    bodo_func()
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    fir._definitions = build_definitions(fir.blocks)
    for stmt in fir.blocks[0].body:
        assert not (
            is_call_assign(stmt)
            and find_callname(fir, stmt.value) == ("h5read", "bodo.io.h5_api")
        )


def test_h5_write_parallel():
    fname = "lr_w.hdf5"

    def test_impl(N, D):
        points = np.ones((N, D))
        responses = np.arange(N) + 1.0
        f = h5py.File(fname, "w")
        dset1 = f.create_dataset("points", (N, D), dtype="f8")
        dset1[:] = points
        dset2 = f.create_dataset("responses", (N,), dtype="f8")
        dset2[:] = responses
        f.close()

    N = 101
    D = 10
    bodo_func = bodo.jit(test_impl)
    with ensure_clean(fname):
        bodo_func(N, D)
        f = h5py.File("lr_w.hdf5", "r")
        X = f["points"][:]
        Y = f["responses"][:]
        f.close()
        np.testing.assert_almost_equal(X, np.ones((N, D)))
        np.testing.assert_almost_equal(Y, np.arange(N) + 1.0)


def test_h5_write_group():
    def test_impl(n, fname):
        arr = np.arange(n)
        n = len(arr)
        f = h5py.File(fname, "w")
        g1 = f.create_group("G")
        dset1 = g1.create_dataset("data", (n,), dtype="i8")
        dset1[:] = arr
        f.close()

    n = 101
    arr = np.arange(n)
    fname = "test_group.hdf5"
    bodo_func = bodo.jit(test_impl)
    with ensure_clean(fname):
        bodo_func(n, fname)
        f = h5py.File(fname, "r")
        X = f["G"]["data"][:]
        f.close()
        np.testing.assert_almost_equal(X, arr)
