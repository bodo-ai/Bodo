# -*- coding: utf-8 -*-

import unittest
import bodo
import numpy as np
import pandas as pd
import glob
import gc
import re
import pyarrow.parquet as pq
from bodo.libs.str_arr_ext import StringArray
from bodo.libs.str_ext import unicode_to_std_str, std_str_to_unicode
import pytest


@pytest.mark.parametrize('ind', [
    np.array([True, True, False, False, False, True]),
    np.array([0, 1, 3, 4]),
    slice(1, 5),
])
def test_string_array_getitem_na(ind):
    def impl(S, index):
        return S.iloc[index]

    bodo_func = bodo.jit(impl)
    S = pd.Series(['A', np.nan, 'CC', 'DD', np.nan, 'ABC'])
    # ind = slice(0, 3)
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind))
    pd.testing.assert_series_equal(impl(S, ind), bodo_func(S, ind))


class TestString(unittest.TestCase):
    def test_pass_return(self):
        def test_impl(_str):
            return _str
        bodo_func = bodo.jit(test_impl)
        # pass single string and return
        arg = 'test_str'
        self.assertEqual(bodo_func(arg), test_impl(arg))
        # pass string list and return
        arg = ['test_str1', 'test_str2']
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_const(self):
        def test_impl():
            return 'test_str'
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_str2str(self):
        str2str_methods = ['capitalize', 'casefold', 'lower', 'lstrip',
            'rstrip', 'strip', 'swapcase', 'title', 'upper']
        for method in str2str_methods:
            func_text = "def test_impl(_str):\n"
            func_text += "  return _str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            test_impl = loc_vars['test_impl']
            bodo_func = bodo.jit(test_impl)
            arg = ' \tbbCD\t '
            self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_equality(self):
        def test_impl(_str):
            return (_str=='test_str')
        bodo_func = bodo.jit(test_impl)
        arg = 'test_str'
        self.assertEqual(bodo_func(arg), test_impl(arg))
        def test_impl(_str):
            return (_str!='test_str')
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_concat(self):
        def test_impl(_str):
            return (_str+'test_str')
        bodo_func = bodo.jit(test_impl)
        arg = 'a_'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_split(self):
        def test_impl(_str):
            return _str.split('/')
        bodo_func = bodo.jit(test_impl)
        arg = 'aa/bb/cc'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_replace(self):
        def test_impl(_str):
            return _str.replace('/', ';')
        bodo_func = bodo.jit(test_impl)
        arg = 'aa/bb/cc'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_getitem_int(self):
        def test_impl(_str):
            return _str[3]
        bodo_func = bodo.jit(test_impl)
        arg = 'aa/bb/cc'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_int_cast(self):
        def test_impl(_str):
            return int(_str)
        bodo_func = bodo.jit(test_impl)
        arg = '12'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_float_cast(self):
        def test_impl(_str):
            return float(_str)
        bodo_func = bodo.jit(test_impl)
        arg = '12.2'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_string_str_cast(self):
        def test_impl(a):
            return str(a)
        bodo_func = bodo.jit(test_impl)
        for arg in [np.int32(45), 43, np.float32(1.4), 4.5]:
            py_res = test_impl(arg)
            h_res = bodo_func(arg)
            # XXX: use startswith since bodo output can have extra characters
            self.assertTrue(h_res.startswith(py_res))

    def test_re_sub(self):
        def test_impl(_str):
            p = re.compile('ab*')
            return p.sub('ff', _str)
        bodo_func = bodo.jit(test_impl)
        arg = 'aabbcc'
        self.assertEqual(bodo_func(arg), test_impl(arg))

    def test_regex_std(self):
        def test_impl(_str, _pat):
            return bodo.libs.str_ext.contains_regex(_str, bodo.libs.str_ext.compile_regex(_pat))
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func('What does the fox say', r'd.*(the |fox ){2}'), True)
        self.assertEqual(bodo_func('What does the fox say', r'[kz]u*'), False)


    def test_replace_regex_std(self):
        def test_impl(_str, pat, val):
            s = unicode_to_std_str(_str)
            e = bodo.libs.str_ext.compile_regex(unicode_to_std_str(pat))
            val = unicode_to_std_str(val)
            out = bodo.libs.str_ext.str_replace_regex(s, e, val)
            return std_str_to_unicode(out)

        _str = 'What does the fox say'
        pat = r'd.*(the |fox ){2}'
        val = 'does the cat '
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(_str, pat, val),
            _str.replace(re.compile(pat).search(_str).group(), val))

    def test_replace_noregex_std(self):
        def test_impl(_str, pat, val):
            s = unicode_to_std_str(_str)
            e = unicode_to_std_str(pat)
            val = unicode_to_std_str(val)
            out = bodo.libs.str_ext.str_replace_noregex(s, e, val)
            return std_str_to_unicode(out)

        _str = 'What does the fox say'
        pat = 'does the fox'
        val = 'does the cat'
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(_str, pat, val),
            _str.replace(pat, val))


    # string array tests
    def test_string_array_constructor(self):
        # create StringArray and return as list of strings
        def test_impl():
            return StringArray(['ABC', 'BB', 'CDEF'])
        bodo_func = bodo.jit(test_impl)
        self.assertTrue(np.array_equal(bodo_func(), ['ABC', 'BB', 'CDEF']))

    def test_string_array_comp(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A=='ABC'
            return B.sum()
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), 1)

    def test_string_series(self):
        def test_impl(ds):
            rs = ds == 'one'
            return ds, rs
        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({'A': [1,2,3]*33, 'B': ['one', 'two', 'three']*33})
        ds, rs = bodo_func(df.B)
        gc.collect()
        self.assertTrue(isinstance(ds, pd.Series) and isinstance(rs, pd.Series))
        self.assertTrue(ds[0] == 'one' and ds[2] == 'three' and rs[0] == True and rs[2] == False)

    def test_string_array_bool_getitem(self):
        def test_impl():
            A = StringArray(['ABC', 'BB', 'CDEF'])
            B = A=='ABC'
            C = A[B]
            return len(C) == 1 and C[0] == 'ABC'
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), True)

    def test_string_NA_box(self):
        def test_impl():
            df = pq.read_table('bodo/tests/data/example.parquet').to_pandas()
            return df.five
        bodo_func = bodo.jit(test_impl)
        # XXX just checking isna() since Pandas uses None in this case
        # instead of nan for some reason
        np.testing.assert_array_equal(bodo_func().isna(), test_impl().isna())

    # test utf8 decode
    def test_decode_empty1(self):
        def test_impl(S):
            return S[0]
        bodo_func = bodo.jit(test_impl)
        S = pd.Series([''])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_single_ascii_char1(self):
        def test_impl(S):
            return S[0]
        bodo_func = bodo.jit(test_impl)
        S = pd.Series(['A'])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_ascii1(self):
        def test_impl(S):
            return S[0]
        bodo_func = bodo.jit(test_impl)
        S = pd.Series(['Abc12', 'bcd', '345'])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_unicode1(self):
        def test_impl(S):
            return S[0], S[1], S[2]
        bodo_func = bodo.jit(test_impl)
        S = pd.Series(['¡Y tú quién te crees?', '🐍⚡', '大处着眼，小处着手。',])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_decode_unicode2(self):
        # test strings that start with ascii
        def test_impl(S):
            return S[0], S[1], S[2]
        bodo_func = bodo.jit(test_impl)
        S = pd.Series(['abc¡Y tú quién te crees?', 'dd2🐍⚡', '22 大处着眼，小处着手。',])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_encode_unicode1(self):
        def test_impl():
            return pd.Series(['¡Y tú quién te crees?', '🐍⚡', '大处着眼，小处着手。',])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    @unittest.skip("TODO: explore np array of strings")
    def test_box_np_arr_string(self):
        def test_impl(A):
            return A[0]
        bodo_func = bodo.jit(test_impl)
        A = np.array(['AA', 'B'])
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_glob(self):
        def test_impl():
            glob.glob("*py")

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_set_string(self):
        def test_impl():
            s = bodo.libs.set_ext.init_set_string()
            s.add('ff')
            for v in s:
                pass
            return v

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_dict_string(self):
        def test_impl():
            s = bodo.libs.dict_ext.dict_unicode_type_unicode_type_init()
            s['aa'] = 'bb'
            return s['aa'], ('aa' in s)

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), ('bb', True))


if __name__ == "__main__":
    unittest.main()
