# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Old tests for Series values which can be useful since they are different and may
expose corner cases.
"""
import os
import unittest

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    get_start_end,
)
from bodo.utils.typing import BodoError

_cov_corr_series = [
    (pd.Series(x), pd.Series(y))
    for x, y in [
        ([np.nan, -2.0, 3.0, 9.1], [np.nan, -2.0, 3.0, 5.0]),
        # TODO(quasilyte): more intricate data for complex-typed series.
        # Some arguments make assert_almost_equal fail.
        # Functions that yield mismatching results: _column_corr_impl and _column_cov_impl.
        (
            [complex(-2.0, 1.0), complex(3.0, 1.0)],
            [complex(-3.0, 1.0), complex(2.0, 1.0)],
        ),
        ([complex(-2.0, 1.0), complex(3.0, 1.0)], [1.0, -2.0]),
        ([1.0, -4.5], [complex(-4.5, 1.0), complex(3.0, 1.0)]),
    ]
]


@pytest.mark.slow
class TestSeries(unittest.TestCase):
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({"A": [1, 2, 3]})
            return (df.A == 1).sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n)})
            return (df.A == 2).sum()

        n = 11
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({"A": ["a", "b", "c"]})
            return (df.A == "a").sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({"A": A})
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_index1(self):
        def test_impl(A):
            return A

        S = pd.Series([3, 5, 6], ["a", "b", "c"], name="A")
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(S), test_impl(S), check_index_type=False
        )

    def test_series_attr1(self):
        def test_impl(A):
            return A.size

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_argsort1(self):
        def test_impl(A):
            return A.argsort()

        n = 11
        np.random.seed(0)
        A = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2, 3]).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_astype_str1(self):
        def test_impl(A):
            return A.astype(str)

        n = 11
        S = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_astype_str2(self):
        def test_impl(A):
            return A.astype(str)

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_shape1(self):
        def test_impl(A):
            return A.shape

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A.copy(), 0), test_impl(df.A.copy(), 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series3(self):
        def test_impl(A, i):
            S = pd.Series(A)
            S[i] = 100

        n = 11
        A = np.arange(n)
        A1 = A.copy()
        A2 = A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1, A2)

    def test_setitem_series_bool1(self):
        def test_impl(A):
            A[A > 3] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1)
        test_impl(A2)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series_bool2(self):
        def test_impl(A, B):
            A[A > 3] = B[A > 3]

        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, df.B)
        test_impl(A2, df.B)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_getitem_series_str1(self):
        def test_impl(A, i):
            return A[i]

        df = pd.DataFrame({"A": ["aa", "bb", "cc"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_op1(self):
        def test_impl(A, i):
            return A + A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 0), test_impl(df.A, 0), check_names=False
        )

    def test_series_op2(self):
        def test_impl(A, i):
            return A + i

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op3(self):
        def test_impl(A, i):
            A += i
            return A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A.copy(), 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op4(self):
        def test_impl(A):
            return A.add(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op5(self):
        def test_impl(A):
            return A.pow(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op6(self):
        def test_impl(A, B):
            return A.eq(B)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_op7(self):
        def test_impl(A):
            return -A

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_inplace_binop_array(self):
        def test_impl(A, B):
            A += B
            return A

        n = 11
        A = np.arange(n) ** 2.0  # TODO: use 2 for test int casting
        B = pd.Series(np.ones(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(A.copy(), B), test_impl(A, B))

    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    def test_series_fusion2(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

    def test_series_len(self):
        def test_impl(A, i):
            return len(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(["1", "2", "3"])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_series_list_str_unbox1(self):
        def test_impl(A):
            return A.iloc[0]

        S = pd.Series([["aa", "b"], ["ccc"], []])
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))
        # call twice to test potential refcount errors
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_np_typ_call_replace(self):
        # calltype replacement is tricky for np.typ() calls since variable
        # type can't provide calltype
        def test_impl(i):
            return np.int32(i)

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(1), test_impl(1))

    def test_series_ufunc1(self):
        def test_impl(A, i):
            return np.isinf(A).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A, 1), test_impl(df.A, 1))

    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame(
                {
                    "one": np.array([-1, np.nan, 2.5]),
                    "two": ["foo", "bar", "baz"],
                    "three": [True, False, True],
                }
            )
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(
            test_impl(), bodo_func(), check_dtype=False, check_column_type=False
        )

    @unittest.skip("needs empty_like typing fix in npydecl.py")
    def test_series_empty_like(self):
        def test_impl(A):
            return np.empty_like(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertTrue(isinstance(bodo_func(df.A), np.ndarray))

    def test_series_fillna1(self):
        def test_impl(A):
            return A.fillna(5.0)

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False
        )

    def test_series_fillna_str1(self):
        def test_impl(A):
            return A.fillna("dd")

        df = pd.DataFrame({"A": ["aa", "b", None, "ccc"]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False, check_dtype=False
        )

    @pytest.mark.skipif(
        bodo.hiframes.boxing._use_dict_str_type,
        reason="not supported for dict string type",
    )
    def test_series_fillna_str_inplace1(self):
        def test_impl(A):
            A.fillna("dd", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2), check_dtype=False)
        # TODO: handle string array reflection
        # bodo_func(S1)
        # test_impl(S2)
        # np.testing.assert_array_equal(S1, S2)

    @pytest.mark.skipif(
        bodo.hiframes.boxing._use_dict_str_type,
        reason="not supported for dict string type",
    )
    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2), check_dtype=False)

    def test_series_dropna_float1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str_parallel1(self):
        def test_impl(A):
            B = A.dropna()
            return (B == "gg").sum()

        S1 = pd.Series(["aa", "b", None, "ccc", "dd", "gg"])
        bodo_func = bodo.jit(distributed_block=["A"])(test_impl)
        start, end = get_start_end(len(S1))
        # TODO: gatherv
        self.assertEqual(bodo_func(S1[start:end]), test_impl(S1))

    def test_series_sum1(self):
        def test_impl(S):
            return S.sum()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 0
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_sum2(self):
        def test_impl(S):
            return (S + S).sum()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_prod1(self):
        def test_impl(S):
            return S.prod()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 1
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_count1(self):
        def test_impl(S):
            return S.count()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(["aa", "bb", np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_mean1(self):
        def test_impl(S):
            return S.mean()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_min1(self):
        def test_impl(S):
            return S.min()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_max1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_value_counts(self):
        def test_impl(S):
            return S.value_counts()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["AA", "BB", "C", "AA", "C", "AA"])
        pd.testing.assert_series_equal(
            bodo_func(S), test_impl(S), check_index_type=False
        )

    def test_series_dist_input1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(distributed_block={"S"})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(bodo_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_tuple_input1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        s_tup = (S, 1, S2)
        self.assertEqual(bodo_func(s_tup), test_impl(s_tup))

    @unittest.skip("pending handling of build_tuple in dist pass")
    def test_series_tuple_input_dist1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(distributed=["s_tup"])(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        start, end = get_start_end(n)
        s_tup = (S, 1, S2)
        h_s_tup = (S[start:end], 1, S2[start:end])
        self.assertEqual(bodo_func(h_s_tup), test_impl(s_tup))

    def test_series_concat1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        S2 = pd.Series([6.0, 7.0])
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_concat_str1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2])

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series(["aa", "bb", np.nan, "", "GGG"])
        S2 = pd.Series(["1", "12", "", np.nan, "A"])
        # TODO: handle index in concat
        pd.testing.assert_series_equal(
            bodo_func(S1, S2),
            test_impl(S1, S2),
            check_dtype=False,
        )

    def test_series_cov1(self):
        def test_impl(S1, S2):
            return S1.cov(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_corr1(self):
        def test_impl(S1, S2):
            return S1.corr(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()

        S = pd.Series(["aa", "abc", "c", "cccd"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_str2str(self):
        str2str_methods = (
            "capitalize",
            "lower",
            "lstrip",
            "rstrip",
            "strip",
            "swapcase",
            "title",
            "upper",
        )
        for method in str2str_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            # XXX: \t support pending Numba #4188
            # S = pd.Series([' \tbbCD\t ', 'ABC', ' mCDm\t', 'abc'])
            S = pd.Series([" bbCD ", "ABC", " mCDm ", np.nan, "abc"])
            check_func(test_impl, (S,))

    def test_series_strip_with_args(self):
        strip_methods = (
            "lstrip",
            "rstrip",
            "strip",
        )
        for method in strip_methods:
            func_text = "def test_impl(S, to_strip):\n"
            func_text += "  return S.str.{}(to_strip)\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            S = pd.Series(
                [
                    "\n \tbbCD\t ",
                    "\tABC ",
                    " mCDm\t",
                    "\nabc\n",
                    " bbCD ",
                    "ABC",
                    " mCDm ",
                    np.nan,
                    "abc",
                ]
            )
            check_func(test_impl, (S, " "))
            check_func(test_impl, (S, "\t"))
            check_func(test_impl, (S, "\t\n "))

    def test_series_str2bool(self):
        str2bool_methods = (
            "isalnum",
            "isalpha",
            "isdigit",
            "isspace",
            "isupper",
            "islower",
            "istitle",
            "isnumeric",
            "isdecimal",
        )
        for method in str2bool_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            S = pd.Series(
                [" 1aB ", "982", "ABC", "  ", np.nan, "abc", "Hi There", "100.20"]
            )
            check_func(test_impl, (S,))

    def test_series_append1(self):
        def test_impl(S, other):
            return S.append(other).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        # Test single series
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_append2(self):
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        S3 = pd.Series([1.0])
        # Test series tuple
        np.testing.assert_array_equal(bodo_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_isna1(self):
        def test_impl(S):
            return S.isna()

        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        check_func(test_impl, (S,))

    def test_series_isnull1(self):
        def test_impl(S):
            return S.isnull()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_notna1(self):
        def test_impl(S):
            return S.notna()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_str_isna1(self):
        def test_impl(S):
            return S.isna()

        S = pd.Series(["aa", None, "AB", "ABC", "c", "cccd"])
        check_func(test_impl, (S,))

    def test_series_nlargest1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_default1(self):
        def test_impl(S):
            return S.nlargest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_nan1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_nsmallest1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_default1(self):
        def test_impl(S):
            return S.nsmallest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_nan1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_default1(self):
        def test_impl(S):
            return S.head()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_index1(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_head_index2(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], ["a", "ab", "abc", "c", "f", "hh", ""])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_index_type=False)

    def test_series_median1(self):
        def test_impl(S):
            return S.median()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))
        # odd size
        m = 101
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_median_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.median()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_series_argsort_parallel(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.argsort().values

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(A):
            return A.idxmin()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_idxmax1(self):
        def test_impl(A):
            return A.idxmax()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values1(self):
        def test_impl(A):
            return A.sort_values()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values_index1(self):
        def test_impl(A, B):
            S = pd.Series(A, B)
            return S.sort_values()

        n = 11
        np.random.seed(0)
        # TODO: support passing Series with Index
        # S = pd.Series(np.random.ranf(n), np.random.randint(0, 100, n))
        A = np.random.ranf(n)
        B = np.random.ranf(n)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_sort_values_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            S = df.points
            return S.sort_values()

        check_func(test_impl, (), only_seq=True, sort_output=True)

    def test_series_shift_default1(self):
        def test_impl(S):
            return S.shift()

        # testing for correct default values
        def test_unsup_1(S):
            return S.shift(freq=None, axis=0, fill_value=None)

        # testing when integer default value is unsupported
        def test_unsup_2(S):
            return S.shift(axis=1)

        # testing when nonetype default value is unsupported
        def test_unsup_3(S):
            return S.shift(freq=1)

        bodo_func = bodo.jit(test_impl)
        bodo_func_1 = bodo.jit(test_unsup_1)
        bodo_func_2 = bodo.jit(test_unsup_2)
        bodo_func_3 = bodo.jit(test_unsup_3)
        S = pd.Series([np.nan, 2.0, 3.0, 5.0, np.nan, 6.0, 7.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
        pd.testing.assert_series_equal(test_impl(S), bodo_func_1(S))
        with pytest.raises(BodoError, match="parameter only supports default value"):
            bodo_func_2(S)
        with pytest.raises(BodoError, match="parameter only supports default value"):
            bodo_func_3(S)


if __name__ == "__main__":
    unittest.main()
