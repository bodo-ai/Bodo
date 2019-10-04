import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


def test_len():
    def test_impl(S):
        return S.str.len()

    S = pd.Series([' bbCD ', 'ABC', ' mCDm ', np.nan, 'abcffcc', '', 'A'],
        [4, 3, 5, 1, 0, -3, 2], name='A')
    check_func(test_impl, (S,), check_dtype=False)


def test_split():
    def test_impl(S):
        return S.str.split(',')

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    S = pd.Series(['ABCC', 'ABBD', 'AA', 'C,ABB, D', 'B,B,CC'],
        [3, 5, 1, 0, 2], name='A')
    # TODO: support distributed
    check_func(test_impl, (S,), False)


def test_get():
    def test_impl(S):
        B = S.str.split(',')
        return B.str.get(1)

    # TODO: support and test NA
    S = pd.Series(['AB,CC', 'C,ABB,D', 'LLL,JJ', 'C,D', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    # TODO: support distributed
    check_func(test_impl, (S,), False)


def test_replace_regex():
    def test_impl(S):
        return S.str.replace('AB*', 'EE', regex=True)

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'CCD', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_replace_noregex():
    def test_impl(S):
        return S.str.replace('AB', 'EE', regex=False)

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'AA', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_contains_regex():
    def test_impl(S):
        return S.str.contains('AB*', regex=True)

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'CCD', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_contains_noregex():
    def test_impl(S):
        return S.str.contains('AB', regex=False)

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'AA', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_count_noflag():
    def test_impl(S):
        return S.str.count('A')

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'AA', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_count_flag():
    def test_impl(S):
        return S.str.count('A', re.IGNORECASE)

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'AA', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))


def test_find():
    def test_impl(S):
        return S.str.find('AB')

    S = pd.Series(['ABCC', 'CABBD', np.nan, 'AA', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
    check_func(test_impl, (S,), check_dtype=False)


def test_center():
    def test_impl(S):
        return S.str.center(5, '*')

    S = pd.Series(['ABCDDC', 'ABBD', 'AA', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_ljust():
    def test_impl(S):
        return S.str.ljust(5, '*')

    S = pd.Series(['ABCDDC', 'ABBD', 'AA', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_rjust():
    def test_impl(S):
        return S.str.rjust(5, '*')

    S = pd.Series(['ABCDDC', 'ABBD', 'AA', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_zfill():
    def test_impl(S):
        return S.str.zfill(10)

    S = pd.Series(['ABCDDCABABAAB', 'ABBD', 'AA', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_startswith():
    def test_impl(S):
        return S.str.startswith("AB")

    S = pd.Series(['ABCC', 'ABBD', 'AA', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_endswith():
    def test_impl(S):
        return S.str.startswith("AB")

    S = pd.Series(['AB', 'ABB', 'BAAB', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))


def test_isupper():
    def test_impl(S):
        return S.str.isupper()

    S = pd.Series(['AB', 'ABb', 'abab', 'C,ABB, D', np.nan],
        [3, 5, 1, 0, 2], name='A')
    check_func(test_impl, (S,))
    
