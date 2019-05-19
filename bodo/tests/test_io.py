import unittest
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import h5py
import pyarrow.parquet as pq
import bodo
from bodo.utils.testing import ensure_clean
from bodo.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_rank,
    get_start_end)


kde_file = 'kde.parquet'


def test_pq_pandas_date(datapath):
    fname = datapath('pandas_dt.pq')
    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


def test_pq_spark_date(datapath):
    fname = datapath('sdf_dt.pq')
    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({'DT64': df.DT64, 'col2': df.DATE})

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())

def test_h5_read_seq(datapath):
    fname = datapath("lr.hdf5")
    def test_impl():
        f = h5py.File(fname, "r")
        X = f['points'][:]
        f.close()
        return X

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_allclose(bodo_func(), test_impl())


def test_h5_read_const_infer_seq(datapath):
    fname = datapath("")
    def test_impl():
        p = fname + 'lr'
        f = h5py.File(p + ".hdf5", "r")
        s = 'po'
        X = f[s + 'ints'][:]
        f.close()
        return X

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_allclose(bodo_func(), test_impl())


def test_h5_read_parallel(datapath):
    fname = datapath("lr.hdf5")
    def test_impl():
        f = h5py.File(fname, "r")
        X = f['points'][:]
        Y = f['responses'][:]
        f.close()
        return X.sum() + Y.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl(), decimal=2)
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


def test_h5_filter(datapath):
    fname = datapath("h5_test_filter.h5")
    def test_impl():
        f = h5py.File(fname, "r")
        b = np.arange(11) % 3 == 0
        X = f['test'][b,:,:,:]
        f.close()
        return X

    bodo_func = bodo.jit(locals={'X:return': 'distributed'})(test_impl)
    n = 4  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_read_group(datapath):
    fname = datapath("test_group_read.hdf5")
    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f['G']
        X = g1['data'][:]
        f.close()
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func() == test_impl()


def test_h5_file_keys(datapath):
    fname = datapath("test_group_read.hdf5")
    def test_impl():
        f = h5py.File(fname, "r")
        s = 0
        for gname in f.keys():
            X = f[gname]['data'][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={'X': bodo.int64[:]})
    assert bodo_func() == test_impl()
    # test using locals for typing
    bodo_func = bodo.jit(test_impl, locals={'X': bodo.int64[:]})
    assert bodo_func() == test_impl()


def test_h5_group_keys(datapath):
    fname = datapath("test_group_read.hdf5")
    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f['G']
        s = 0
        for dname in g1.keys():
            X = g1[dname][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={'X': bodo.int64[:]})
    assert bodo_func() == test_impl()


class TestIO(unittest.TestCase):

    def setUp(self):
        if get_rank() == 0:

            # test_csv_cat1
            data = ("2,B,SA\n"
                    "3,A,SBC\n"
                    "4,C,S123\n"
                    "5,B,BCD\n")

            with open("csv_data_cat1.csv", "w") as f:
                f.write(data)

            # test_csv_single_dtype1
            data = ("2,4.1\n"
                    "3,3.4\n"
                    "4,1.3\n"
                    "5,1.1\n")

            with open("csv_data_dtype1.csv", "w") as f:
                f.write(data)

            # test_np_io1
            n = 111
            A = np.random.ranf(n)
            A.tofile("np_file1.dat")

    @unittest.skip("fix collective create dataset")
    def test_h5_write_parallel(self):
        def test_impl(N, D):
            points = np.ones((N,D))
            responses = np.arange(N)+1.0
            f = h5py.File("lr_w.hdf5", "w")
            dset1 = f.create_dataset("points", (N,D), dtype='f8')
            dset1[:] = points
            dset2 = f.create_dataset("responses", (N,), dtype='f8')
            dset2[:] = responses
            f.close()

        N = 101
        D = 10
        bodo_func = bodo.jit(test_impl)
        bodo_func(N, D)
        f = h5py.File("lr_w.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]
        f.close()
        np.testing.assert_almost_equal(X, np.ones((N,D)))
        np.testing.assert_almost_equal(Y, np.arange(N)+1.0)

    @unittest.skip("fix collective create dataset and group")
    def test_h5_write_group(self):
        def test_impl(n, fname):
            arr = np.arange(n)
            n = len(arr)
            f = h5py.File(fname, "w")
            g1 = f.create_group("G")
            dset1 = g1.create_dataset("data", (n,), dtype='i8')
            dset1[:] = arr
            f.close()

        n = 101
        arr = np.arange(n)
        fname = "test_group.hdf5"
        bodo_func = bodo.jit(test_impl)
        bodo_func(n, fname)
        f = h5py.File(fname, "r")
        X = f['G']['data'][:]
        f.close()
        np.testing.assert_almost_equal(X, arr)

    def test_pq_read(self):
        def test_impl():
            t = pq.read_table('kde.parquet')
            df = t.to_pandas()
            X = df['points']
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_read_global_str1(self):
        def test_impl():
            df = pd.read_parquet(kde_file)
            X = df['points']
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_read_freevar_str1(self):
        kde_file2 = 'kde.parquet'
        def test_impl():
            df = pd.read_parquet(kde_file2)
            X = df['points']
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pd_read_parquet(self):
        def test_impl():
            df = pd.read_parquet('kde.parquet')
            X = df['points']
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.two.values=='foo'
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str_with_nan_seq(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values=='foo'
            return A

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())

    def test_pq_str_with_nan_par(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str_with_nan_par_multigroup(self):
        def test_impl():
            df = pq.read_table('example2.parquet').to_pandas()
            A = df.five.values=='foo'
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_bool(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.three.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.one.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_float_no_nan(self):
        def test_impl():
            df = pq.read_table('example.parquet').to_pandas()
            return df.four.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_csv1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
            )
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_keys1(self):
        def test_impl():
            dtype = {'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int}
            return pd.read_csv("csv_data1.csv",
                names=dtype.keys(),
                dtype=dtype,
            )
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_const_dtype1(self):
        def test_impl():
            dtype = {'A': 'int', 'B': 'float64', 'C': 'float', 'D': 'int64'}
            return pd.read_csv("csv_data1.csv",
                names=dtype.keys(),
                dtype=dtype,
            )
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv")

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv")
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_skip1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},
                skiprows=2,
            )
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer_skip1(self):
        def test_impl():
            return pd.read_csv("csv_data_infer1.csv", skiprows=2)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer_skip_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_infer1.csv", skiprows=2,
                names=['A', 'B', 'C', 'D'])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_rm_dead1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int},)
            return df.B.values
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_csv_date1(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int},
                parse_dates=[2])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_str1(self):
        def test_impl():
            return pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':np.float, 'D':np.int})
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_str_parallel1(self):
        def test_impl():
            df = pd.read_csv("csv_data_date1.csv",
                names=['A', 'B', 'C', 'D'],
                dtype={'A':np.int, 'B':np.float, 'C':str, 'D':np.int})
            return (df.A.sum(), df.B.sum(), (df.C == '1966-11-13').sum(),
                    df.D.sum())
        bodo_func = bodo.jit(locals={'df:return': 'distributed'})(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_usecols1(self):
        def test_impl():
            return pd.read_csv("csv_data1.csv",
                names=['C'],
                dtype={'C':np.float},
                usecols=[2],
            )
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_cat1(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            return df.C2
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(), test_impl(), check_names=False)

    def test_csv_cat2(self):
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C', 'D'])
            df = pd.read_csv("csv_data_cat1.csv",
                names=['C1', 'C2', 'C3'],
                dtype={'C1':np.int, 'C2': ct_dtype, 'C3':str},
            )
            return df
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_single_dtype1(self):
        def test_impl():
            df = pd.read_csv("csv_data_dtype1.csv",
                names=['C1', 'C2'],
                dtype=np.float64,
            )
            return df
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_write_csv1(self):
        def test_impl(df, fname):
            df.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        n = 111
        df = pd.DataFrame({'A': np.arange(n)})
        hp_fname = 'test_write_csv1_hpat.csv'
        pd_fname = 'test_write_csv1_pd.csv'
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(df, hp_fname)
            test_impl(df, pd_fname)
            pd.testing.assert_frame_equal(
                pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({'A': np.arange(n)})
            df.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = 'test_write_csv1_hpat_par.csv'
        pd_fname = 'test_write_csv1_pd_par.csv'
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(n, hp_fname)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)
            if get_rank() == 0:
                test_impl(n, pd_fname)
                pd.testing.assert_frame_equal(
                    pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_np_io1(self):
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())

    def test_np_io2(self):
        # parallel version
        def test_impl():
            A = np.fromfile("np_file1.dat", np.float64)
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_np_io3(self):
        def test_impl(A):
            if get_rank() == 0:
                A.tofile("np_file_3.dat")

        bodo_func = bodo.jit(test_impl)
        n = 111
        A = np.random.ranf(n)
        bodo_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)

    def test_np_io4(self):
        # parallel version
        def test_impl(n):
            A = np.arange(n)
            A.tofile("np_file_3.dat")

        bodo_func = bodo.jit(test_impl)
        n = 111
        A = np.arange(n)
        bodo_func(n)
        B = np.fromfile("np_file_3.dat", np.int64)
        np.testing.assert_almost_equal(A, B)


if __name__ == "__main__":
    unittest.main()
