"""
PYTEST_DONT_REWRITE
"""
import unittest
import bodo

# using separate file for assert tests to avoid pytest rewrite errors
class AssertTest(unittest.TestCase):

    def test_assert(self):
        # make sure assert in an inlined function works
        def g(a):
            assert a==0

        hpat_g = bodo.jit(g)
        def f():
            hpat_g(0)

        hpat_f = bodo.jit(f)
        hpat_f()
