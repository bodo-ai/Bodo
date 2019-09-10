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
    S = pd.Series(['AB,CC', 'C,ABB,D', 'LLL,JJ', 'C,D', 'C,ABB,D'],
        [4, 3, 5, 1, 0], name='A')
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

def test_startswith():
    def test_impl(S):
        return S.str.startswith("AB")

    S = pd.Series(['ABCC', 'ABBD', 'AA', 'C,ABB, D'],
        [3, 5, 1, 0], name='A')
    check_func(test_impl, (S,))
